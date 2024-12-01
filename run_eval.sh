export all_proxy=http://127.0.0.1:1081

# export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tf/orion.zou/huggingface
export OPENAI_API_KEY=fk200318-wjpUzP0rOmFb6CYl6K1rcl0sVMqMjSpk
export OPENAI_BASE_URL=https://openai.api2d.net/v1

CUDA_VISIBLE_DEVICES=2,3,4,5 alpaca_eval evaluate_from_model --model_configs /tf/orion.zou/repos/SimPO/eval/alpacaeval2/configs/Llama-3-Instruct-8B-SimPO.yaml --annotators_config alpaca_eval_gpt4_turbo_fn --output_path ./output_eval_origin