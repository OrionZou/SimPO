from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.timeout import TimeoutMiddleware

from pydantic import BaseModel
from typing import List
from vllm import LLM, SamplingParams
import uvicorn
import asyncio
import argparse


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


    return parser.parse_args()

args = parse_args()

# 创建 FastAPI 实例
app = FastAPI()
app.add_middleware(TimeoutMiddleware, timeout=120)

# 创建 vLLM AsyncLLMEngine
def init_engine():
    llm_engine = LLM(
                    model=args.model,
                    tokenizer=args.model,
                    gpu_memory_utilization=0.9,
                    tensor_parallel_size=1,
                    seed=1234,
                    dtype="bfloat16"
                    )
    return llm_engine

llm_engine=init_engine()


# 定义请求和响应模型
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.0
    top_p: float = 0.9
    max_tokens: int = 8192
    stop: List[str] = ["<|end_of_text|>","<|eot_id|>"]

class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    choices: List[dict]
    usage: dict


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty.")
    
    sampling_params=SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
    )
    # 调用 vllm 执行推理
    try:
        output = llm_engine.chat(
            messages=[m.dict() for m in request.messages],
            sampling_params=sampling_params,
        )[0]

        response_data = {
            "id": "chatcmpl-" + str(output.request_id),
            "object": "chat.completion",
            "created": int(output.metrics.first_scheduled_time),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output.outputs[0].text},
                    "finish_reason": output.outputs[0].finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
                "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
            },
        }

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)