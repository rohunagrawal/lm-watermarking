export HF_HOME=/scratch/ra4951/.cache
export WANDB_CACHE_DIR=/scratch/ra4951/.cache/wandb

python experiments/train_grpo.py \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --dataset-name gsm8k \
    --max-steps 50 \
    --max-samples 64 \
    --batch-size 4 \
    --max-new-tokens 512