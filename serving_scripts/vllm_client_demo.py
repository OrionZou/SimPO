from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "token-abc123"
openai_api_base = "http://server1.zhaojunhua.org:89/v6/v1"
# openai_api_base = "http://localhost:8082/v1"
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model = "/tf/orion.zou/repos/SimPO/outputs/llama-3-8b-instruct-simpo-aflow-3"

input= "Given a square matrix of size N*N given as a list of lists, where each cell is associated with a specific cost. A path is defined as a specific sequence of cells that starts from the top-left cell move only right or down and ends on bottom right cell. We want to find a path with the maximum average over all existing paths. Average is computed as total cost divided by the number of cells visited in the path.\n\ndef maxAverageOfPath(cost):"

def chat_vllm(input:str):
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content":input
        }],
        model=model,
        temperature=0,
        # logprobs=True,
        stop=["<|end_of_text|>","<|eot_id|>"],
        # prompt_logprobs=True
        # stream=True
    )

    print("Chat completion results:")
    print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content

chat_vllm(input)