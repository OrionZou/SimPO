import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description='vllm reward model config')
    parser.add_argument('--model', type=str,
                       default="/tf/orion.zou/repos/SimPO/outputs/llama-3-8b-instruct-simpo-aflow-3",
                       help="model path")
    parser.add_argument('--text', type=str,
                       default="",
                       help="host")

    return parser.parse_args()

args = parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)

# args.text=''
# print(f"{args.text}")
inputs = tokenizer(args.text, return_tensors="pt", max_length=8192,  padding=True, truncation=True)
# print(inputs)
print(f"token len{(inputs['input_ids'].shape)}")