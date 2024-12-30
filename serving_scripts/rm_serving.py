import uvicorn
import asyncio
import argparse
import torch

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description='vllm reward model config')
    parser.add_argument('--model', type=str,
                       default="/tf/orion.zou/repos/SimPO/outputs/llama-3-8b-instruct-simpo-aflow-3",
                       help="model path")
    parser.add_argument('--host', type=str,
                       default="0.0.0.0",
                       help="host")
    parser.add_argument('--port', type=int,
                       default=8082,
                       help="host")
    parser.add_argument('--max_batch_size', type=int,
                       default=32,
                       help="host")

    return parser.parse_args()

args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="RM API 服务")

# 定义请求模型
class GenerateRequest(BaseModel):
    dataset_type: str
    queries: List[str]
    codes: List[str]

class ResultScore(BaseModel):
    code: str
    score: float

# 定义响应模型 
class GenerateResponse(BaseModel):
    data: List[ResultScore]
    model_version: str
    error_code: int

async def init_llm():
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = AutoModelForCausalLM.from_pretrained(args.model).to(device) 
    return llm, tokenizer

def llm_score(llm,tokenizer, querys, answers):
    prompts = [query + answer for query, answer in zip(querys, answers)]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
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

# 创建全局 LLM 引擎实例
@app.on_event("startup")
async def startup():
    app.state.llm, app.state.tokenizer = await init_llm()

# 生成接口
@app.post("/v1/reward", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if  not len(request.queries)==len(request.codes):
        return GenerateResponse(
            data=[ResultScore(code=f"len(queries)!=len(codes)={len(request.queries)}!={len(request.codes)}",score=0)],
            model_version=args.model,
            error_code=-1
        ) 

    try:
        # 调用 LLM 引擎生成响应
        resultlist=[]
        
        for i in range(0, len(request.queries), args.max_batch_size):
            if len(request.queries)>args.max_batch_size:
                queries=request.queries[i:i+args.max_batch_size]
                codes=request.codes[i:i+args.max_batch_size]
            else:
                queries=request.queries[i:]
                codes=request.codes[i:]
            batch_score=llm_score(app.state.llm,app.state.tokenizer,queries, codes)
            resultlist.extend([ResultScore(code=code,score=score) for code,score in  zip(codes,batch_score)])

        return GenerateResponse(
            data=resultlist,
            model_version=args.model,
            error_code=0
        )

    except Exception as e:
        return GenerateResponse(
            data=[ResultScore(code=f"inter error",score=0)],
            model_version=args.model,
            error_code=-1
        )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )