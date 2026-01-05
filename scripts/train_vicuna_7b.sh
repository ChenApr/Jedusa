torchrun --nproc_per_node=4 medusa/train/train.py \
    --model_name_or_path ../models/vicuna-7b-v1.33 \
    --data_path ../Medusa/vicuna_2048.filtered.json \
    --bf16 True \
    --output_dir test \
    --report_to "wandb" \
    --run_name "medusa-vicuna-7b-training" \
    --logging_dir "./logs" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --medusa_num_heads 3 \
    --medusa_num_layers 1

    # --model_name_or_path lmsys/vicuna-7b-v1.3 \
