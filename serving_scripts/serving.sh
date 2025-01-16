
CUDA_VISIBLE_DEVICES=2 nohup vllm serve /tf/orion.zou/repos/SimPO/outputs/llama-3-8b-instruct-simpo-aflow-3 --dtype auto --api-key token-abc123 --port 8085 --host 0.0.0.0 --gpu_memory_utilization 0.95  --seed 1234  --max-model-len 8192  > vllm_serving.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup vllm serve /tf/orion.zou/repos/SimPO/outputs/llama-3-8b-instruct-simpo-aflow-3 --dtype auto --task reward --api-key token-abc123 --port 8082 --host 0.0.0.0 --gpu_memory_utilization 0.95  --seed 1234 > vllm_serving.log 2>&1 &


# CUDA_VISIBLE_DEVICES=2 nohup python serving_scripts/vllm_serving.py > vllm_serving.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python serving_scripts/rm_serving.py --model /tf/orion.zou/repos/SimPO/outputs/llama-3-8b-instruct-simpo-aflow-3 --port 8086 --host 0.0.0.0 --max_batch_size 32  > rm_serving.log 2>&1 &

# export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
# CUDA_VISIBLE_DEVICES=2 python serving_scripts/vllm_serving.py --model /tf/orion.zou/repos/SimPO/outputs/llama-3-8b-instruct-simpo-aflow-3 --port 8085 --host 0.0.0.0   > vllm_serving.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python serving_scripts/vllm_serving.py --model /tf/orion.zou/repos/SimPO/outputs/llama-3-8b-instruct-simpo-aflow-3 --port 8085 --host 0.0.0.0 