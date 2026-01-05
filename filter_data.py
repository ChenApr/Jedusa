import json
import argparse

def normalize_role(r):
    if r in ("human", "user"): return "user"
    if r in ("gpt", "assistant", "bot"): return "assistant"
    if r == "system": return "system"
    return None

def is_good_conv(conv):
    if not isinstance(conv, list) or len(conv) < 2:
        return False
    # 每条必须有 role/content
    msgs = []
    for m in conv:
        if not isinstance(m, dict): return False
        role = normalize_role(m.get("role") or m.get("from") or m.get("speaker"))
        content = m.get("content") or m.get("value") or m.get("text") or ""
        if role is None or not isinstance(content, str) or len(content.strip()) == 0:
            return False
        if role == "system":
            # system 可保留，但为了简单起见，先丢掉（也可以保留）
            return False
        msgs.append(role)

    # 要求从 user 开始，且严格交替 user/assistant/...
    if msgs[0] != "user":
        return False
    for i in range(1, len(msgs)):
        if msgs[i] == msgs[i-1]:
            return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    args = ap.parse_args()

    data = json.load(open(args.infile, "r"))
    kept = []
    bad = 0
    for x in data:
        if is_good_conv(x):
            kept.append(x)
        else:
            bad += 1

    json.dump(kept, open(args.outfile, "w"), ensure_ascii=False, indent=2)
    print(f"input={len(data)} kept={len(kept)} removed={bad}")

if __name__ == "__main__":
    main()