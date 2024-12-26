from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "token-abc123"
openai_api_base = "http://server1.zhaojunhua.org:89/v3/v1"
# openai_api_base = "http://localhost:8082/v1"
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
print(models)
model = models.data[0].id

chat_completion = client.chat.completions.create(
    messages=[{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "Who won the world series in 2020?"
    }, {
        "role":
        "assistant",
        "content":
        "The Los Angeles Dodgers won the World Series in 2020."
    }, {
        "role": "user",
        "content": "Where was it played?"
    }],
    model=model,
    temperature=0,
    # logprobs=True,
    stop=["<|end_of_text|>","<|eot_id|>"],
    # prompt_logprobs=True
    # stream=True
)

print("Chat completion results:")
print(chat_completion)