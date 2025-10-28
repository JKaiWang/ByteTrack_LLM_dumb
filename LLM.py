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

target_id = -1

for idx, f in enumerate(files, 1):
    fname = os.path.basename(f)
    person_id, frame_id = parse_filename(fname)
    print(f"[INFO] ({idx}/{len(files)}) Processing {fname}")

    payload = {
        "model": "qwen2.5vl",
        "prompt": f"{args.prompt}\nConfirm whether it meets the requirements. If yes, answer 'yes'. If not, answer 'no'.",
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

    print(f"[DEBUG] LLM response: {result_text.strip()}")

    if "yes" in result_text.lower():
        target_id = person_id
        print(f"[RESULT] Target ID: {target_id}")
        break

print(json.dumps({"target_id": target_id}, ensure_ascii=False))
sys.exit(0)
