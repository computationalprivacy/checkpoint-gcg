# Checkpoint-GCG: Auditing and Attacking Fine-Tuning-Based Prompt Injection Defenses
This repository contains the source code for the paper "[Checkpoint-GCG: Auditing and Attacking Fine-Tuning-Based Prompt Injection Defenses](https://arxiv.org/abs/2505.15738)" by Xiaoxue Yang*, Bozhidar Stevanoski*, Matthieu Meeus, and Yves-Alexandre de Montjoye (* denotes equal contribution).

## Abstract
Large language models (LLMs) are increasingly deployed in real-world applications ranging from chatbots to agentic systems, where they are expected to process untrusted data and follow trusted instructions. Failure to distinguish between the two poses significant security risks, exploited by prompt injection attacks, which inject malicious instructions into the data to control model outputs. Model-level defenses have been proposed to mitigate prompt injection attacks. These defenses fine-tune LLMs to ignore injected instructions in untrusted data. We introduce Checkpoint-GCG, a white-box attack against fine-tuning-based defenses. Checkpoint-GCG enhances the Greedy Coordinate Gradient (GCG) attack by leveraging intermediate model checkpoints produced during fine-tuning to initialize GCG, with each checkpoint acting as a stepping stone for the next one to continuously improve attacks. First, we instantiate Checkpoint-GCG to evaluate the robustness of the state-of-the-art defenses in an auditing setup, assuming both (a) full knowledge of the model input and (b) access to intermediate model checkpoints. We show Checkpoint-GCG to achieve up to $96\%$ attack success rate (ASR) against the strongest defense. Second, we relax the first assumption by searching for a universal suffix that would work on unseen inputs, and obtain up to $89.9\%$ ASR against the strongest defense. Finally, we relax both assumptions by searching for a universal suffix that would transfer to similar black-box models and defenses, achieving an ASR of $63.9\%$ against a newly released defended model from Meta. 


## Environment setup
+ Install environment dependencies

    ```
    git clone https://github.com/computationalprivacy/checkpoint-gcg
    cd checkpoint-gcg
    conda create -n checkpoint_gcg python==3.10 
    ```

+ Install package dependencies

  + For fine-tuning and attacking using SecAlign and StruQ (we adopted the requirements in `requirements.txt` in [SecAlign](https://github.com/facebookresearch/SecAlign/tree/main)):

    ```
    pip install -r requirements_secalign_struq.txt
    ```

  + For fine-tuning and attacking using Safety-Tuned LLaMAs (we used the `requirements.txt` from [Safety-Tuned LLaMAs](https://github.com/vinid/safety-tuned-llamas) and installed the listed packages with their latest available versions):

    ```
    pip install -r requirements_safety_tuned_llama.txt
    ```

+ Download data dependencies

    ```
    python setup.py
    ```


## SecAlign 
+ To fine-tune Llama-3-8B-Instruct, Mistral-7B-Instruct, and Qwen2-1.5B-Instruct using SecAlign, run the following respective commands:
  ```
    bash scripts/defense/secalign_llama3instruct.sh
    bash scripts/defense/secalign_mistralinstruct.sh
    bash scripts/defense/secalign_qwen.sh
  ```

## StruQ
+ Similarly, to fine-tune Llama3-8B-Instruct, Mistral-7B-Instruct, and Qwen2-1.5B-Instruct using StruQ, run the following respective commands:
  ```
    bash scripts/defense/struq_llama3instruct.sh
    bash scripts/defense/struq_mistralinstruct.sh
    bash scripts/defense/struq_qwen.sh
  ```


## Safety-Tuned LLaMAs
+ To fine-tune for Safety-Tuned LLaMAs, run the following script, which uses `data/training/saferpaca_Instructions_2000.json` formatted with `data/configs/alpaca.json` as training data. 
    ```
    python safety_llama_finetuning.py
    ```

## Test
+ To run standard GCG and Checkpoint-GCG attacks against defense(s) and model(s), run the following to automatically generate attack `.sh` scripts:

    ```
    python scripts/attack/generate_attack_scripts.py
    ```
+ Run the relevant `.sh` script(s) in `scripts/attack` to launch the desired attacks:

    + Standard GCG vs Checkpoint-GCG
        + Standard GCG shell scripts (directly attacking the final fine-tuned model $\theta_C$) have "direct" in the script filenames 
        + Checkpoint-GCG shell scripts have "checkpoint" in the script filenames, as well as the appropriate checkpoint selection strategy
    + Attacking individual samples vs universal attack
        + Individual-sample attack shell scripts have "individual" in the script filenames
        + Universal attack shell scripts have "universal" in the script filenames 

## Acknowledgements and Citation
This repository is licensed under the MIT License. It builds on [SecAlign](https://github.com/facebookresearch/SecAlign/tree/main) (CC-BY-NC) and [Safety-Tuned LLaMAs](https://github.com/vinid/safety-tuned-llamas) (MIT). We thank the authors for open-sourcing their work.

If you find our work useful in your research, please consider citing:
```
@misc{yang2025checkpointgcgauditingattackingfinetuningbased,
      title={Checkpoint-GCG: Auditing and Attacking Fine-Tuning-Based Prompt Injection Defenses}, 
      author={Xiaoxue Yang and Bozhidar Stevanoski and Matthieu Meeus and Yves-Alexandre de Montjoye},
      year={2025},
      eprint={2505.15738},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2505.15738}, 
}
```