CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.run --nproc_per_node=1 --master_port=30535 align.py  \
--model_name_or_path mistralai/Mistral-7B-Instruct-v0.1    \
--data_path data/training/alpaca_data_cleaned.json \
--cache_dir /checkpoint_gcg/CacheDir  \
--evaluation_strategy "no"   \
--save_strategy "steps"             \
--save_steps 1   \
--window_size 256      \
--padding_side left           \
--output_dir mistralai/Mistral-7B-Instruct-v0.1_dpo_NaiveCompletion_2025-04-27-15-02-43  \
--num_train_epochs 3        \
--per_device_train_batch_size 8           \
--gradient_accumulation_steps 8        \
--learning_rate 1.4e-4  \
--lr_scheduler_type "cosine"       \
--logging_steps 1   \
--fsdp "full_shard auto_wrap"      \
--fsdp_transformer_layer_cls_to_wrap "MistralDecoderLayer"    \
--attack NaiveCompletion      \
--lr_scale True      \
--downsample True         \
--alignment dpo     \
--bf16 False  \
--fp16 True  \
--tf32 False  \
--gradient_checkpointing True  \
--model_max_length 512