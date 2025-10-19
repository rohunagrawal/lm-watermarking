export HF_HOME=/scratch/ra4951/.cache

python experiments/train_grpo.py \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --dataset-name gsm8k \
    --max-steps 10 \
    --max-samples 2 \
    --batch-size 4 \
    --max-new-tokens 512 \
    --use-wandb