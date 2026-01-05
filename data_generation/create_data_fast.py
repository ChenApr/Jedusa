import typer
import json
import asyncio
import httpx
from tqdm import tqdm
from typing_extensions import Annotated

app = typer.Typer()

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def fix_source(source):
    # ShareGPT: [{"from":"human"/"gpt","value":"..."}...]
    if source and source[0].get("from") == "gpt":
        source = source[1:]
    new_source = []
    for item in source:
        role = "assistant" if item.get("from") == "gpt" else "user"
        content = item.get("value", "")
        new_source.append({"role": role, "content": content})
    return new_source

def pick_single_user_turn(messages, pick: str):
    """
    messages: ChatML list (role=user/assistant)
    return: (history_messages, target_user_message)
    为了快：只取一个 user turn 来生成一次 assistant。
    """
    user_idxs = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if not user_idxs:
        return [], None

    if pick == "first":
        idx = user_idxs[0]
    elif pick == "last":
        idx = user_idxs[-1]
    else:
        idx = user_idxs[0]

    history = messages[:idx]   # 可为空
    user_msg = messages[idx]
    return history, user_msg

def build_vicuna_prompt(history, user_msg):
    # 用 Vicuna/FastChat 常见的 "USER:"/"ASSISTANT:" 样式拼 prompt
    # 注意：我们走 /v1/completions，所以 prompt 是纯字符串
    s = SYSTEM_PROMPT
    for m in history:
        if m["role"] == "user":
            s += " USER: " + m["content"].strip()
        elif m["role"] == "assistant":
            s += " ASSISTANT: " + m["content"].strip()
    s += " USER: " + user_msg["content"].strip()
    s += " ASSISTANT:"
    return s

async def one_request(client: httpx.AsyncClient, url: str, model: str, prompt: str,
                      max_tokens: int, temperature: float, top_p: float):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        # 当模型开始“下一轮 USER:”时就停，避免无限长
        "stop": [" USER:"],
    }
    r = await client.post(url, json=payload)
    j = r.json()
    if "choices" not in j:
        return None, j
    text = j["choices"][0].get("text", "")
    return text, None

@app.command()
def main(
    *,
    input_filename: Annotated[str, typer.Option("--input-filename")],
    output_filename: Annotated[str, typer.Option("--output-filename")],
    base_url: Annotated[str, typer.Option("--base-url")] = "http://localhost:8080",
    model: Annotated[str, typer.Option("--model")] = "vicuna",
    concurrency: Annotated[int, typer.Option("--concurrency")] = 128,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens")] = 256,
    temperature: Annotated[float, typer.Option("--temperature")] = 0.0,
    top_p: Annotated[float, typer.Option("--top-p")] = 1.0,
    pick: Annotated[str, typer.Option("--pick")] = "first",  # first/last
):
    completions_url = base_url.rstrip("/") + "/v1/completions"

    async def runner():
        with open(input_filename, "r") as f:
            input_data = json.load(f)

        # 先把 ShareGPT 格式转成 ChatML
        conversations = [fix_source(x["conversations"]) for x in input_data]

        limits = httpx.Limits(
            max_connections=concurrency * 2,
            max_keepalive_connections=concurrency * 2,
            keepalive_expiry=60.0,
        )
        timeout = httpx.Timeout(None)

        sem = asyncio.Semaphore(concurrency)
        out = [None] * len(conversations)

        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:

            async def handle(i: int, conv):
                async with sem:
                    history, user_msg = pick_single_user_turn(conv, pick)
                    if user_msg is None:
                        out[i] = []
                        return
                    prompt = build_vicuna_prompt(history, user_msg)
                    text, err = await one_request(
                        client, completions_url, model, prompt,
                        max_new_tokens, temperature, top_p
                    )
                    if err is not None:
                        # 出错就输出空，避免任务中断
                        print("bad response:", err)
                        out[i] = []
                        return

                    # 形成一个最小可训练对话：history + user + generated assistant
                    new_conv = history + [user_msg, {"role": "assistant", "content": text.strip()}]
                    out[i] = new_conv

            tasks = [asyncio.create_task(handle(i, c)) for i, c in enumerate(conversations)]
            for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                await fut

        with open(output_filename, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    asyncio.run(runner())

if __name__ == "__main__":
    app()