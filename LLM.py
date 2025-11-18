# LLM_multi.py  (你也可以直接覆蓋原 LLM.py)
import os, requests, base64, json, re, argparse, sys

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def parse_filename(fname):
    match = re.match(r"i(\d+)_f(\d+)", fname)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (-1, -1)

parser = argparse.ArgumentParser()
parser.add_argument("--crops", type=str, required=True)
parser.add_argument("--prompt", type=str, required=True)
args = parser.parse_args()

API_URL = "http://localhost:11434/api/generate"

files = sorted([
    os.path.abspath(os.path.join(args.crops, f))
    for f in os.listdir(args.crops)
    if os.path.isfile(os.path.join(args.crops, f))
], key=lambda x: (parse_filename(os.path.basename(x))[1], parse_filename(os.path.basename(x))[0]))

# ⭐ 由單一 target_id → 多個 target_ids（只做語義判斷，不處理位置）
positive_ids = set()

for idx, f in enumerate(files, 1):
    fname = os.path.basename(f)
    person_id, frame_id = parse_filename(fname)
    print(f"[INFO] ({idx}/{len(files)}) Processing {fname}")

    prompt_text = (
        f"{args.prompt}\n"
        "Respond strictly with a single token: 'yes' if this crop visually matches the described target (clothes, appearance, etc.), otherwise 'no'."
    )

    payload = {
        "model": "qwen2.5vl",
        "prompt": prompt_text,
        "images": [encode_image(f)]
    }

    try:
        resp = requests.post(API_URL, json=payload, stream=True)
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] LLM request failed: {e}")
        continue

    result_text = ""
    for line in resp.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    result_text += data["response"]
            except:
                continue

    result_text = result_text.strip().lower()
    print(f"[DEBUG] LLM response: {result_text}")

    # ⭐ 不再 break；改為全收集
    if "yes" in result_text:
        positive_ids.add(person_id)
        print(result_text)

# 輸出多個 ID
out = {"target_ids": sorted(list(positive_ids))}
print(json.dumps(out, ensure_ascii=False))
sys.exit(0)
