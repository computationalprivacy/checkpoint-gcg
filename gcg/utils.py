import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Union
from copy import deepcopy

import fastchat
import torch
import transformers
from fastchat.conversation import get_conv_template
from transformers import AutoModelForCausalLM, AutoTokenizer

from gcg.eval_input import EvalInput
from gcg.types import PrefixCache


logger = logging.getLogger(__name__)


class Role(Enum):
    USER = 1
    ASSISTANT = 2
    SYSTEM = 3


@dataclass
class Message:
    role: Role
    content: str

    def __str__(self):
        return f"[{self.role.name.title()}]: {self.content}"

    @staticmethod
    def serialize(messages, user_only=False):
        if not isinstance(messages, list):
            messages = [messages]
        if user_only:
            messages = [
                {"role": m.role.name, "content": m.content}
                for m in messages
                if m.role == Role.USER
            ]
        else:
            messages = [{"role": m.role.name, "content": m.content} for m in messages]
        return messages

    @staticmethod
    def unserialize(messages: Union[dict, List[dict]]):
        if not isinstance(messages, list):
            messages = [messages]
        messages = [Message(Role[m["role"]], m["content"]) for m in messages]
        return messages


class SuffixManager:
    """Suffix manager for adversarial suffix generation."""

    valid_templates = (
        "secalign_llama-3",
        "secalign_mistral",
        "secalign_qwen2",
        "metasecalign",
        "safety-tuned-llama",
        "llama-3",
        "llama-2",
        "vicuna_v1.1",
        "mistral",
        "chatgpt",
        "completion",
        "raw",
        "tinyllama",
        "struq",
        "bipia",
    )

    def __init__(self, *, tokenizer, use_system_instructions, conv_template):
        """Initialize suffix manager.

        Args:
            tokenizer: Tokenizer for model.
            use_system_instructions: Whether to use system instructions.
            conv_template: Conversation template.
        """
        self.tokenizer = tokenizer
        self.use_system_instructions = use_system_instructions
        self.conv_template = conv_template
        self.is_tiktoken = not isinstance(tokenizer, AutoTokenizer)
        logger.info(
            "SuffixManager initialized with conv_template=%s, is_tiktoken=%s, "
            "use_system_instructions=%s",
            self.conv_template.name,
            self.is_tiktoken,
            use_system_instructions,
        )

        self.sep_tokens = self.tokenizer(
            self.conv_template.sep, add_special_tokens=False
        ).input_ids

        self.num_tok_sep = len(self.sep_tokens)

        if self.conv_template.name == "chatgpt":
            # Space is subsumed by following token in GPT tokenizer
            assert self.conv_template.sep == " ", self.conv_template.sep
            self.num_tok_sep = 0
        elif self.conv_template.name == "llama-3":
            # FastChat adds <|eot_id|> after each message, but it's not sep.
            # Not exactly sure why, but not we need to manually set
            # self.num_tok_sep because sep is just "".
            # https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#L167
            self.num_tok_sep = 1
        elif self.conv_template.name == "bipia":
            self.num_tok_sep = 2
        self.num_tok_sep2 = 0
        if self.conv_template.sep2 not in (None, ""):
            self.num_tok_sep2 = len(
                self.tokenizer(
                    self.conv_template.sep2, add_special_tokens=False
                ).input_ids
            )
        if self.conv_template.stop_str not in (None, ""):
            self.num_tok_sep2 += 1

        print("num_tok_sep:", self.num_tok_sep)
        print("num_tok_sep2:", self.num_tok_sep2)

    @torch.no_grad()
    def get_input_ids(
        self,
        messages: list[Message],
        adv_suffix: str | None = None,
        target: str | None = None,
        static_only: bool = False,
    ) -> tuple[torch.Tensor, slice, slice, slice]:
        """Turn messages into token ids. Run once for attack step.

        Compute token ids for given messages and target, along with slices
        tracking positions of important tokens.

        Args:
            messages: Messages in the conversation.
            adv_suffix: Current adversarial suffix.
            target: Current target output for model.
            static_only: If True, only return token ids for static tokens.

        Returns:
            input_ids: Token ids for messages and target.
            optim_slice: Slice of input_ids corresponding to tokens to optimize.
            target_slice: Slice of input_ids corresponding to target.
            loss_slice: Slice of input_ids corresponding to loss.
        """
        if self.conv_template.name not in self.valid_templates:
            raise NotImplementedError(
                f"{self.conv_template.name} is not implemented! Please use one "
                f"of {self.valid_templates}"
            )
        self.conv_template.messages = []
        if self.conv_template.name not in ["secalign_llama-3", "secalign_mistral", "secalign_qwen2", "metasecalign"]:
            if messages[0].content:
                self.conv_template.set_system_message(messages[0].content)

        user_msg = messages[1].content
        if len(messages) <= 2:
            self.conv_template.append_message(
                self.conv_template.roles[0], messages[1].content
            )  # user rules
        else:
            if not self.use_system_instructions:
                self.conv_template.append_message(
                    self.conv_template.roles[0], messages[1].content
                )  # user rules
                self.conv_template.append_message(
                    self.conv_template.roles[1], messages[2].content
                )  # asst response
                user_msg = messages[3].content
            # user msg
            self.conv_template.append_message(self.conv_template.roles[0], user_msg)

        sep = deepcopy(self.conv_template.sep)
        self.conv_template.sep = ""
        toks = (
            self.tokenizer(self.conv_template.get_prompt()).input_ids
            + self.tokenizer(" ", add_special_tokens=False).input_ids
            + self.sep_tokens
        )

        num_static_tokens = len(toks)

        if user_msg:
            num_static_tokens -= self.num_tok_sep
        elif self.conv_template.name == "vicuna_v1.1":
            pass
        else:
            num_static_tokens -= self.num_tok_sep2

        static_input_ids = torch.tensor(toks[:num_static_tokens])
        if static_only:
            self.conv_template.sep = sep
            return static_input_ids

        # It seems that we do not need toks and self.conv_template after this function
        # Thus, we can calculate toks by adding (user_msg, adv_suffix, '\n\n', self.conv_template.roles[1]) tokens directly
        # instead of asking self.tokenizer to do self.tokenizer(self.conv_template.get_prompt()).input_ids

        toks = (
            self.tokenizer(self.conv_template.get_prompt()).input_ids
            + self.tokenizer(" ", add_special_tokens=False).input_ids
            + self.tokenizer(adv_suffix, add_special_tokens=False).input_ids
            + self.sep_tokens
        )
        optim_slice = slice(num_static_tokens, len(toks) - self.num_tok_sep)

        toks = (
            self.tokenizer(self.conv_template.get_prompt()).input_ids
            + self.tokenizer(" ", add_special_tokens=False).input_ids
            + self.tokenizer(adv_suffix, add_special_tokens=False).input_ids
            + self.sep_tokens
            + self.tokenizer(
                self.conv_template.roles[1], add_special_tokens=False
            ).input_ids
            + self.tokenizer("\n", add_special_tokens=False).input_ids
        )
        assistant_role_slice = slice(optim_slice.stop, len(toks))

        toks = (
            self.tokenizer(self.conv_template.get_prompt()).input_ids
            + self.tokenizer(" ", add_special_tokens=False).input_ids
            + self.tokenizer(adv_suffix, add_special_tokens=False).input_ids
            + self.sep_tokens
            + self.tokenizer(
                self.conv_template.roles[1], add_special_tokens=False
            ).input_ids
            + self.tokenizer("\n" + target, add_special_tokens=False).input_ids
            + self.tokenizer(
                self.tokenizer.eos_token, add_special_tokens=False
            ).input_ids
        )

        target_slice = slice(assistant_role_slice.stop, len(toks) - self.num_tok_sep2)
        loss_slice = slice(
            assistant_role_slice.stop - 1, len(toks) - self.num_tok_sep2 - 1
        )

        # Don't need final sep tokens
        input_ids = torch.tensor(toks[: target_slice.stop])
        self.conv_template.sep = sep
        return input_ids, optim_slice, target_slice, loss_slice

    @torch.no_grad()
    def gen_eval_inputs(
        self,
        messages: list[Message],
        suffix: str,
        target: str,
        num_fixed_tokens: int = 0,
        max_target_len: int | None = None,
    ) -> EvalInput:
        """Generate inputs for evaluation. Run once for each sample

        Returns:
            eval_inputs: Inputs for evaluation.
        """
        suffix_ids = self.tokenizer(
            suffix, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        suffix_ids.requires_grad_(False)
        suffix_ids.squeeze_(0)

        out = self.get_input_ids(messages, suffix, target)
        orig_input_ids, optim_slice, target_slice, loss_slice = out

        if max_target_len is not None:
            # Adjust target slice to be at most max_target_len
            end = min(target_slice.stop, target_slice.start + max_target_len)
            target_slice = slice(target_slice.start, end)
            loss_slice = slice(loss_slice.start, end - 1)
        # Offset everything to ignore static tokens which are processed separately
        orig_input_ids = orig_input_ids[num_fixed_tokens:]
        optim_slice = slice(
            optim_slice.start - num_fixed_tokens,
            optim_slice.stop - num_fixed_tokens,
        )
        target_slice = slice(
            target_slice.start - num_fixed_tokens,
            target_slice.stop - num_fixed_tokens,
        )
        loss_slice = slice(
            loss_slice.start - num_fixed_tokens,
            loss_slice.stop - num_fixed_tokens,
        )
        target_ids = orig_input_ids[target_slice]
        assert target_ids.ndim == 1
        target_ids.requires_grad_(False)

        eval_input = EvalInput(
            suffix_ids=suffix_ids,
            dynamic_input_ids=orig_input_ids,
            target_ids=target_ids,
            optim_slice=optim_slice,
            target_slice=target_slice,
            loss_slice=loss_slice,
        )
        return eval_input


def _simple_template(messages: list[Message]):
    texts = [
        "The following is a conversation between a user and an AI assistant. Please respond to the user as the assistant."
    ]
    for m in messages:
        texts.append(f"{m.role.name.title()}>{m.content}")
    texts.append(f"{Role.ASSISTANT.name.title()}>")
    return "\n".join(texts)


def build_prompt(
    messages: list[Message],
    template_name: str | None = None,
    return_openai_chat_format: bool = False,
):
    if template_name is None:
        return _simple_template(messages)

    conv = get_conv_template(template_name)
    for m in messages:
        if m.role == Role.SYSTEM and m.content:
            conv.set_system_message(m.content)
        elif m.role == Role.USER:
            conv.append_message(conv.roles[0], m.content)
        elif m.role == Role.ASSISTANT:
            conv.append_message(conv.roles[1], m.content)

    # Append assistant response if user message is the last message
    if messages[-1].role == Role.USER:
        conv.append_message(conv.roles[1], None)

    if return_openai_chat_format:
        return conv.to_openai_api_messages()
    return conv.get_prompt()


def batchify_kv_cache(prefix_cache, batch_size):
    batch_prefix_cache = []
    for k, v in prefix_cache:
        batch_prefix_cache.append(
            (k.repeat(batch_size, 1, 1, 1), v.repeat(batch_size, 1, 1, 1))
        )
    return batch_prefix_cache


def get_nonascii_toks(tokenizer, device="cpu") -> torch.Tensor:
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    non_ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        try:
            tok = tokenizer.decode([i], clean_up_tokenization_spaces=False)
        except:
            # GPT tokenizer throws an error for some tokens
            # pyo3_runtime.PanicException: no entry found for key
            continue
        if not is_ascii(tok):
            non_ascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        non_ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        non_ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        non_ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        non_ascii_toks.append(tokenizer.unk_token_id)
    non_ascii_toks = list(set(non_ascii_toks))

    return torch.tensor(non_ascii_toks, device=device)


def get_prefix_cache(
    suffix_manager: SuffixManager,
    model,
    tokenizer,
    messages: list[Message],
) -> PrefixCache:
    static_input_ids = suffix_manager.get_input_ids(messages, static_only=True)
    static_input_str = tokenizer.decode(
        static_input_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    logger.info("Fixed prefix: %s", static_input_str)
    num_static_tokens = len(static_input_ids)
    logger.info("Fixing the first %d tokens as prefix", num_static_tokens)
    logger.info("Caching prefix...")
    device = model.device if hasattr(model, "device") else model.module.device
    with torch.no_grad():
        embed_layer = model.get_input_embeddings()
        input_embeds = embed_layer(static_input_ids.to(device)).unsqueeze(0)
        outputs = model(inputs_embeds=input_embeds, use_cache=True)
        prefix_cache = outputs.past_key_values
    return prefix_cache, num_static_tokens
