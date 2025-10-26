import os, requests, base64, json, re, argparse

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def parse_filename(fname):
    """解析檔名 i{person}_f{frame}.jpg → 回傳 person_id"""
    match = re.match(r"i(\d+)_f(\d+)", fname)
    if match:
        return int(match.group(1)), int(match.group(2))  # (person_id, frame_id)
    return (-1, -1)

parser = argparse.ArgumentParser()
parser.add_argument("--crops", type=str, required=True, help="存放裁切圖片的資料夾")
parser.add_argument("--prompt", type=str, required=True, help="例如：找出穿粉紅色衣服的人")
args = parser.parse_args()

API_URL = "your API"

# 🔑 把檔案按照 frame_id, person_id 排序
files = sorted([
    os.path.abspath(os.path.join(args.crops, f))
    for f in os.listdir(args.crops)
    if os.path.isfile(os.path.join(args.crops, f))
], key=lambda x: (parse_filename(os.path.basename(x))[1],  # frame_id
                    parse_filename(os.path.basename(x))[0]))  # person_id

target_id = -1
for idx, f in enumerate(files, 1):
    fname = os.path.basename(f)
    person_id, frame_id = parse_filename(fname)

    print(f"[INFO] ({idx}/{len(files)}) 檢查 {fname}")

    payload = {
        "model": "qwen2.5vl",  # 確保已經 ollama pull llava
        "prompt": f"{args.prompt}\nconfirm whether it meets the requirements. If yes, answer ""yes"". If not, answer ""no"" and the shirt he wears. 。",
        "images": [encode_image(f)]
    }
    try:
        resp = requests.post(API_URL, json=payload, stream=True)
    except Exception as e:
        print(f"[ERROR] 請求失敗: {e}")
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

    print(f"[DEBUG] LLM 回覆: {result_text.strip()}")

    # ✅ 找到 YES → 鎖定這個 person_id
    if "yes" in result_text.lower():
        target_id = person_id
        print(f"[RESULT] 找到目標: {target_id}")
        break

print(json.dumps({"target_id": target_id}, ensure_ascii=False))
