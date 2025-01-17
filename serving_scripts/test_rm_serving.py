import uvicorn
import asyncio
import argparse
import torch

from typing import List, Optional
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description='vllm reward model config')
    parser.add_argument('--model', type=str,
                       default="/tf/model/Llama3/Meta-Llama-3-8B-Instruct",
                    #    default="/tf/orion.zou/repos/SimPO/outputs/llama-3-8b-instruct-simpo-aflow-3",
                       help="model path")
    parser.add_argument('--max_batch_size', type=int,
                       default=32,
                       help="host")

    return parser.parse_args()

args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_llm():
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # import ipdb;ipdb.set_trace()
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|end_of_text|>'})
    llm = AutoModelForCausalLM.from_pretrained(args.model).to(device) 
    return llm, tokenizer

def llm_score(llm, tokenizer, querys, answers):
    prompts = [query + answer for query, answer in zip(querys, answers)]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(device)
    input_ids = inputs["input_ids"]  # Token IDs
    attention_mask = inputs["attention_mask"]
    
    # import ipdb;ipdb.set_trace()
    # 获取 answers 起始位置
    answers_starts_tensor = torch.tensor(
    [len(ids) for ids in tokenizer(querys, add_special_tokens=False)['input_ids']],
    device=device)

    with torch.no_grad():
        outputs = llm(input_ids=input_ids, attention_mask=attention_mask,use_cache=False)
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]

    # 从 answers 开始对齐目标 token
    logits = logits[:, :-1, :]  # 忽略最后一个 token 的 logits
    target_ids = input_ids[:, 1:]  # 对齐目标 token

    batch_size, seq_len, vocab_size = logits.size()
    # 计算 log_probs
    token_log_probs = logits.log_softmax(dim=-1).gather(2, target_ids.unsqueeze(-1)).squeeze(-1) 

    # 构建 mask，从 answers 的起始位置开始
    batch_size, seq_len, vocab_size = logits.size()
    range_tensor = torch.arange(seq_len, device=device).expand(batch_size, seq_len)  # [batch_size, seq_len]
    mask = range_tensor >= answers_starts_tensor.unsqueeze(1)  # 从 answers 开始

    # 应用 mask 截取 logits 和目标 token
    mask = mask.mul(attention_mask[:,:-1]) 
    # SimPO reward
    beta=2.5
    batch_score = ((token_log_probs * mask).sum(dim=-1) / mask.sum(-1) * beta).tolist()
    # import ipdb;ipdb.set_trace()
    return batch_score

if __name__ == "__main__":
    querys=["Given a square matrix of size N*N given as a list of lists, where each cell is associated with a specific cost. A path is defined as a specific sequence of cells that starts from the top-left cell move only right or down and ends on bottom right cell. We want to find a path with the maximum average over all existing paths. Average is computed as total cost divided by the number of cells visited in the path.\n\ndef maxAverageOfPath(cost):","Average is computed as total cost divided by the number of cells visited in the path.\n\ndef maxAverageOfPath(cost):","Given a square matrix of size N*N given as a list of lists, where each cell is associated with a specific cost. A path is defined as a specific sequence of cells that starts from the top-left cell move only right or down and ends on bottom right cell."]
    answers=[
        "ANALYZE_REQUIREMENTS_PROMPT = \"\"\"\nAnalyze the given problem and identify the key requirements, constraints, and expected behavior of the solution. Provide a concise summary of these aspects to guide the code generation process.\n\"\"\"\n\nGENERATE_CODE_PROMPT = \"\"\"\nGenerate a Python function to solve the given problem. Ensure the function name matches the entry point specified. Include necessary imports and helper functions. Provide a clear and efficient solution. Focus on correctness and optimal performance. Consider the provided requirements in your implementation.\n\"\"\"\n\nVALIDATE_AND_REFINE_PROMPT = \"\"\"\nReview the given solution for the problem. Validate if it meets all the requirements and constraints. If any issues are found, refine the solution to address them. Ensure the refined solution is complete, efficient, and adheres to best coding practices.\n\"\"\"\n\nIMPROVE_CODE_PROMPT = \"\"\"\nThe previous solution failed to pass the tests. Please analyze the error and provide an improved version of the code. Focus on fixing the specific issues mentioned in the error message while maintaining the overall structure and logic of the function. Ensure that your solution is complete and addresses all aspects of the problem, including the provided requirements.\n\"\"\"\nasync def __call__(self, problem: str, entry_point: str):\n        requirements = await self.custom(input=problem, instruction=prompt_custom.ANALYZE_REQUIREMENTS_PROMPT)\n        \n        solutions = []\n        for _ in range(3):\n            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.GENERATE_CODE_PROMPT + f\"\\nRequirements: {requirements['response']}\")\n            solutions.append(solution['response'])\n        \n        # New step: Validate and refine solutions\n        refined_solutions = []\n        for solution in solutions:\n            refined_solution = await self.custom(input=f\"Problem: {problem}\\nRequirements: {requirements['response']}\\nSolution: {solution}\", instruction=prompt_custom.VALIDATE_AND_REFINE_PROMPT)\n            refined_solutions.append(refined_solution['response'])\n        \n        best_solution = await self.sc_ensemble(solutions=refined_solutions, problem=problem)\n        \n        test_result = await self.test(problem=problem, solution=best_solution['response'], entry_point=entry_point)\n        \n        if test_result['result']:\n            return test_result['solution'], self.llm.cost_manager.total_cost\n        else:\n            improved_solution = await self.custom(input=f\"Problem: {problem}\\nRequirements: {requirements['response']}\\nFailed solution: {best_solution['response']}\\nError: {test_result['solution']}\", instruction=prompt_custom.IMPROVE_CODE_PROMPT)\n            return improved_solution['response'], self.llm.cost_manager.total_cost\n\n",
        "improved_solution = await self.custom(input=f\"Problem: {problem}\\nRequirements: {requirements['response']}\\nFailed solution: {best_solution['response']}\\nError: {test_result['solution']}\", instruction=prompt_custom.IMPROVE_CODE_PROMPT)\n            return improved_solution['response'], self.llm.cost_manager.total_cost\n\n",
        "ANALYZE_REQUIREMENTS_PROMPT = \"\"\"\nAnalyze the given problem and identify the key requirements, constraints, and expected behavior of the solution. Provide a concise summary of these aspects to guide the code generation process.\n\"\"\"\n\nGENERATE_CODE_PROMPT = \"\"\"\nGenerate a Python function to solve the given problem. Ensure the function name matches the entry point specified. Include necessary imports and helper functions. Provide a clear and efficient solution. Focus on correctness and optimal performance. Consider the provided requirements in your implementation.\n\"\"\"\n\nVALIDATE_AND_REFINE_PROMPT = \"\"\"\nReview the given solution for the problem. Validate if it meets all the requirements and constraints. If any issues are found, refine the solution to address them. Ensure the refined solution is complete, efficient, and adheres to best coding practices.\n\"\"\"\n\nIMPROVE_CODE_PROMPT = \"\"\"\nThe previous solution failed to pass the tests. Please analyze the error and provide an improved version of the code. Focus on fixing the specific issues mentioned in the error message while maintaining the overall structure and logic of the function. Ensure that your solution is complete and addresses all aspects of the problem, including the provided requirements.\n\"\"\"\nasync def __call__(self, problem: str, entry_point: str):\n        requirements = await self.custom(input=problem, instruction=prompt_custom.ANALYZE_REQUIREMENTS_PROMPT)\n        \n        solutions = []\n        for _ in range(3):\n            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.GENERATE_CODE_PROMPT + f\"\\nRequirements: {requirements['response']}\")\n            solutions.append(solution['response'])\n        \n        # New step: Validate and refine solutions\n        refined_solutions = []\n"
    ]
    llm, tokenizer = init_llm()
    batch_score=llm_score(llm, tokenizer, querys, answers)
    print(batch_score)