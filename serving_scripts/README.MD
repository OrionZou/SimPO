# RM-SERVING / LLM—SERVING API

### 轻松启动

RM-SERVING 启动
```bash
conda activate vllm-eval
cd /tf/orion.zou/repos/SimPO
CUDA_VISIBLE_DEVICES=3 nohup python serving_scripts/rm_serving.py --model /tf/model/Llama3/Meta-Llama-3-8B-Instruct --port 8086 --host 0.0.0.0 --max_batch_size 32  > rm_serving.log 2>&1 &
```

 --model /tf/model/Llama3/Meta-Llama-3-8B-Instruct 配置RM的路径

 aflow 3.2 训练后的路径： /tf/orion.zou/repos/SimPO/outputs/llama-3-8b-instruct-simpo-aflow-3

VLLM-SERVING 启动
```bash
conda activate vllm-eval
cd /tf/orion.zou/repos/SimPO
CUDA_VISIBLE_DEVICES=2 nohup vllm serve /tf/orion.zou/repos/SimPO/outputs/llama-3-8b-instruct-simpo-aflow-3 --dtype auto --api-key token-abc123 --port 8085 --host 0.0.0.0 --gpu_memory_utilization 0.95  --seed 1234  --max-model-len 8192  > vllm_serving.log 2>&1 &
```