NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model /nfs/FM/gongoubo/checkpoints/Qwen/Qwen3-8B-Base \
    --dataset /nfs/FM/gongoubo/cuda_kernel/KernelGYM/drkernel/kernel/scripts/sft/data/parallel_drkernel_minimax_results_sft.parquet \
    --train_type full \
    --max_length 32768 \
    --sequence_parallel_size 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --output_dir ./output_drkernel  \
    --deepspeed zero2 \
    --torch_dtype bfloat16 \
    --fp16 false \
    --bf16 true \
    --save_steps 100000 \
    --eval_steps 100000 \
    --logging_steps 10
