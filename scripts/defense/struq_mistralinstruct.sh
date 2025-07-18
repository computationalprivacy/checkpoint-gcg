python -m torch.distributed.run --nproc_per_node=2 --master_port=29694 train.py   \
--model_name_or_path mistralai/Mistral-7B-Instruct-v0.1      \
--window_size 256 \
--padding_side left \
--data_path data/training/alpaca_data_cleaned.json      \
--output_dir mistralai/Mistral-7B-Instruct-v0.1_Mistral-7B-Instruct-v0.1_NaiveCompletion_2025-05-10-13-41-28      \
--num_train_epochs 3      \
--per_device_train_batch_size 4    \
--per_device_eval_batch_size 4        \
--gradient_accumulation_steps 8       \
--evaluation_strategy "no"      \
--save_strategy "no"       \
--learning_rate 2.5e-6      \
--weight_decay 0.       \
--warmup_ratio 0.03      \
--lr_scheduler_type "cosine"      \
--logging_steps 1      \
--fsdp "full_shard auto_wrap"       \
--fsdp_transformer_layer_cls_to_wrap "MistralDecoderLayer"       \
--bf16 False  \
--fp16 True  \
--tf32 False  \
--save_only_model True  \
--gradient_checkpointing True  \
--attack Mistral-7B-Instruct-v0.1_NaiveCompletion     \
--lr_scale True   \
--downsample True \
--model_max_length 512