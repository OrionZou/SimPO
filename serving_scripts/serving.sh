
CUDA_VISIBLE_DEVICES=3 nohup vllm serve /tf/orion.zou/repos/SimPO/outputs/llama-3-8b-instruct-simpo-aflow-3 --dtype auto --api-key token-abc123 --port 8085 --host 0.0.0.0 --gpu_memory_utilization 0.95 --tensor-parallel-size 1  --seed 1234  --max-model-len 8192 --max-model-len 8192 --block-size 32 --max-num-seqs 1024 --max-num-batched-tokens 8192   > vllm_serving_8085.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup vllm serve /tf/orion.zou/repos/SimPO/outputs/llama-3-8b-instruct-simpo-aflow-v6 --dtype auto --api-key token-abc123 --port 8092 --host 0.0.0.0 --gpu_memory_utilization 0.95 --tensor-parallel-size 1  --seed 1234  --max-model-len 8192 --block-size 32 --max-num-seqs 1024 --max-num-batched-tokens 8192  > vllm_serving_8092.log 2>&1 &

# export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
# CUDA_VISIBLE_DEVICES=2 python serving_scripts/vllm_serving.py --model /tf/orion.zou/repos/SimPO/outputs/llama-3-8b-instruct-simpo-aflow-3 --port 8085 --host 0.0.0.0   > vllm_serving.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python serving_scripts/vllm_serving.py --model /tf/orion.zou/repos/SimPO/outputs/llama-3-8b-instruct-simpo-aflow-3 --port 8085 --host 0.0.0.0 



# CUDA_VISIBLE_DEVICES=3 nohup python serving_scripts/rm_serving.py --model /tf/orion.zou/repos/SimPO/outputs/llama-3-8b-instruct-simpo-aflow-3 --port 8086 --host 0.0.0.0 --max_batch_size 32  > rm_serving.log 2>&1 &

# CUDA_VISIBLE_DEVICES=4 nohup python serving_scripts/rm_serving.py --model /tf/orion.zou/repos/SimPO/outputs/llama-3-8b-instruct-simpo-aflow-3 --port 8086 --host 0.0.0.0 --max_batch_size 32  > rm_serving_8086.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python serving_scripts/rm_serving.py --model /tf/orion.zou/repos/SimPO/outputs/llama-3-8b-instruct-simpo-aflow-v6 --port 8093 --host 0.0.0.0 --max_batch_size 32  > rm_serving_8093.log 2>&1 &


# CUDA_VISIBLE_DEVICES=4 nohup python serving_scripts/rm_serving.py --model /tf/model/Llama3/Meta-Llama-3-8B-Instruct --port 8086 --host 0.0.0.0 --max_batch_size 32  > rm_serving.log 2>&1 &
