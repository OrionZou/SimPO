# export all_proxy=http://127.0.0.1:1081

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tf/orion.zou/huggingface
# export WANDB_DIR=/tf/orion.zou/wandb
export WANDB_MODE=offline
export WANDB_API_KEY=e6ac95c0632c01460f1eecac9c0c89f15ef08066

export PYTHONPATH=.:$PYTHONPATH

# CUDA_VISIBLE_DEVICES=2,3,4,5 nohup accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simpo.py training_configs/llama-3-8b-instruct-simpo-aflow.yaml > llama-3-8b-instruct-simpo-alow-2.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/run_simpo.py training_configs/llama-3-8b-instruct-simpo-aflow.yaml 

# python test_wandb.py
wandb sync  /tf/orion.zou/repos/SimPO/wandb/offline-run-20241201_074324-4kcss4ly