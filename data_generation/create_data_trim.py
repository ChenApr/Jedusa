import typer
import json
from typing_extensions import Annotated
import httpx
from tqdm import tqdm
import asyncio
import transformers

app = typer.Typer()

# ====== 你需要匹配 vLLM server 的 max_model_len ======
MAX_MODEL_LEN = 2048
RESERVE_GEN_TOKENS = 256  # 给assistant生成预留token，防止贴边又超

# ====== 用同一套 tokenizer + chat_template 做长度估计 & 截断 ======
# 这里建议指向你 vLLM 里加载的模型路径（更稳）
TOKENIZER_PATH = "../models/vicuna-7b-v1.33"

# 你的 train.py 里那份 Vicuna v1.5 fallback template（原样粘贴）
VICUNA_V15_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}"
    "{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}"
    "{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}"
    "{% for message in loop_messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if loop.index0 == 0 %}{{ system_message }}{% endif %}"
    "{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}"
    "{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}"
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    TOKENIZER_PATH,
    use_fast=True,
    legacy=False,
)
if tokenizer.chat_template is None:
    tokenizer.chat_template = VICUNA_V15_CHAT_TEMPLATE


def count_tokens(messages) -> int:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    return len(ids)


def trim_messages_to_fit(messages, max_model_len=MAX_MODEL_LEN, reserve=RESERVE_GEN_TOKENS):
    """
    策略：
    1) 保留system（如果有）
    2) 超长就丢最早的 user+assistant 成对轮次，优先保留最近对话
    3) 还超：截断最后一条 user（保留末尾token，通常更接近“有效信息”）
    """
    budget = max_model_len - reserve
    if budget <= 0:
        return messages

    msgs = list(messages)

    # 1) system
    system = []
    if msgs and msgs[0]["role"] == "system":
        system = [msgs[0]]
        msgs = msgs[1:]

    # 2) 丢轮次（每次丢 user+assistant 两条）
    while msgs and count_tokens(system + msgs) > budget:
        if len(msgs) >= 2:
            msgs = msgs[2:]
        else:
            break

    all_msgs = system + msgs

    # 3) 仍然超：截断最后一个 user 的 content
    if all_msgs and count_tokens(all_msgs) > budget:
        last_user_idx = None
        for i in range(len(all_msgs) - 1, -1, -1):
            if all_msgs[i]["role"] == "user":
                last_user_idx = i
                break

        if last_user_idx is not None:
            prefix = all_msgs[:last_user_idx]
            user_msg = dict(all_msgs[last_user_idx])
            suffix = all_msgs[last_user_idx + 1 :]

            other = count_tokens(prefix + [{"role": "user", "content": ""}] + suffix)
            keep = max(0, budget - other)

            content_ids = tokenizer(
                user_msg["content"], add_special_tokens=False
            ).input_ids

            # 保留末尾 keep 个 token
            content_ids = content_ids[-keep:]
            user_msg["content"] = tokenizer.decode(
                content_ids, skip_special_tokens=True
            )
            all_msgs = prefix + [user_msg] + suffix

    return all_msgs


class Conversation:
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append({"role": message["role"], "content": message["content"]})


def fix_source(source):
    # ShareGPT: [{"from": "human"/"gpt", "value": "..."}...]
    if source and source[0]["from"] == "gpt":
        source = source[1:]

    new_source = []
    for item in source:
        role = "assistant" if item["from"] == "gpt" else "user"
        content = item["value"]
        new_source.append({"role": role, "content": content})
    return new_source


async def run_one_turn(conv: Conversation, url: str, model_id: str, client: httpx.AsyncClient) -> bool:
    trimmed = trim_messages_to_fit(conv.messages)
    payload = {"model": model_id, "messages": trimmed}

    resp = await client.post(url, json=payload)

    try:
        content = resp.json()
    except Exception:
        print("bad response(non-json):", resp.status_code, resp.text[:200])
        return False

    if resp.status_code != 200 or "choices" not in content:
        print("bad response:", content)
        return False

    message = content["choices"][0]["message"]
    message.pop("name", None)
    conv.add_message(message)
    return True


async def recreate_conversation(conversation, sem, url, model_id, client):
    async with sem:
        conv = Conversation()
        try:
            # conversation = [{"role":"user",...},{"role":"assistant",...},...]
            # 只取 user turns: [::2]
            for message in conversation[::2]:
                if message["role"] != "user":
                    continue
                conv.add_message(message)
                ok = await run_one_turn(conv, url, model_id, client)
                if not ok:
                    break  # 失败就停，避免 roles 错乱连锁
        except Exception as e:
            print("exception:", e)
        return conv.messages


@app.command()
def main(
    *,
    input_filename: Annotated[str, typer.Option("--input-filename")],
    output_filename: Annotated[str, typer.Option("--output-filename")],
    url: Annotated[str, typer.Option("--url")] = "http://localhost:8080/v1/chat/completions",
    model_id: Annotated[str, typer.Option("--model-id")] = "vicuna",
    concurrency: Annotated[int, typer.Option("--concurrency")] = 16,
):
    sem = asyncio.Semaphore(concurrency)

    async def _main():
        with open(input_filename, "r") as f:
            input_data = json.loads(f.read())

        conversations = [fix_source(x["conversations"]) for x in input_data]

        async with httpx.AsyncClient(timeout=None) as client:
            tasks = [recreate_conversation(c, sem, url, model_id, client) for c in conversations]

            recreated = []
            for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                recreated.append(await fut)

        with open(output_filename, "w") as f:
            json.dump(recreated, f, indent=2, ensure_ascii=False)

    asyncio.run(_main())


if __name__ == "__main__":
    app()