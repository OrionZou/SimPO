# export all_proxy=http://127.0.0.1:1081

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tf/orion.zou/huggingface
# export WANDB_DIR=/tf/orion.zou/wandb
export WANDB_MODE=offline
export WANDB_API_KEY=e6ac95c0632c01460f1eecac9c0c89f15ef08066

export PYTHONPATH=.:$PYTHONPATH

# CUDA_VISIBLE_DEVICES=3,7 nohup accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml --main_process_port 29501 --num_processes 2 scripts/run_simpo.py training_configs/llama-3-8b-instruct-simpo-aflow_v3_2.yaml  > llama-3-8b-instruct-simpo-alow-3.log 2>&1 &

# CUDA_VISIBLE_DEVICES=7 nohup accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml --main_process_port 29501 --num_processes 1 scripts/run_simpo.py training_configs/llama-3-8b-instruct-simpo-aflow_v3_2.yaml  > llama-3-8b-instruct-simpo-alow-3_1.log 2>&1 &

# CUDA_VISIBLE_DEVICES=4  torchrun --master_port 29501 --nproc_per_node 1 scripts/run_simpo.py training_configs/llama-3-8b-instruct-simpo-aflow_v3_2.yaml 
CUDA_VISIBLE_DEVICES=3  nohup accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml --main_process_port 29501  --num_processes 1 scripts/run_simpo.py training_configs/llama-3-8b-instruct-simpo-aflow_v3_2_eval.yaml > llama-3-8b-instruct-simpo-alow-3_2_eval.log 2>&1 &


# CUDA_VISIBLE_DEVICES=2,3,4,5 nohup accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simpo.py training_configs/llama-3-8b-instruct-simpo-aflow.yaml > llama-3-8b-instruct-simpo-alow-2.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/run_simpo.py training_configs/llama-3-8b-instruct-simpo-aflow.yaml 

# CUDA_VISIBLE_DEVICES=2,3,4,5 nohup accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml --main_process_port 29501 --num_processes 4 scripts/run_simpo.py training_configs/llama-3-8b-instruct-simpo-codeUltrafeedback.yaml  > llama-3-8b-instruct-simpo-codeUltrafeedback.log 2>&1 &

# python test_wandb.py
# wandb sync  /tf/orion.zou/repos/SimPO/wandb/offline-run-20250116_090552-91d0pv2c

