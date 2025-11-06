# ===============================================
# LLM.py — Qwen2-CLIP embeddings + cosine similarity selection
# ===============================================
import os, json, re, argparse, sys, traceback
from typing import Tuple


def parse_filename(fname: str) -> Tuple[int, int]:
    match = re.match(r"i(\d+)_f(\d+)", fname)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (-1, -1)


parser = argparse.ArgumentParser()
parser.add_argument("--crops", type=str, required=True)
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--threshold", type=float, default=0.30, help="cosine similarity threshold; >= threshold → keep that target id")
parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"], help="inference device")
parser.add_argument("--qwen2-model", type=str, default="openai/clip-vit-large-patch14", help="HF model id for CLIP-like model (e.g., openai/clip-vit-large-patch14)")
# Optional per-ID aggregation
parser.add_argument("--per_id_agg", type=str, choices=["max", "mean"], default="max", help="aggregate multiple crops per id: max or mean")
# Optional batching for image crops
parser.add_argument("--batch_size", type=int, default=32, help="batch size for image feature extraction")
# Optional spatial metadata to inject global/location cues
parser.add_argument("--bbox_meta", type=str, default="", help="optional JSON with boxes: {bboxes:[{id,bbox:[x1,y1,x2,y2]}], width, height}")
parser.add_argument("--alpha", type=float, default=0.85, help="weight for visual cosine vs location prior: score=alpha*sim+(1-alpha)*loc")
args = parser.parse_args()

try:
    if not os.path.isdir(args.crops):
        print(f"[ERROR] Crop folder not found: {args.crops}")
        print(json.dumps({"target_ids": []}, ensure_ascii=False))
        sys.exit(0)

    files = sorted([
        os.path.abspath(os.path.join(args.crops, f))
        for f in os.listdir(args.crops)
        if os.path.isfile(os.path.join(args.crops, f))
    ], key=lambda x: (parse_filename(os.path.basename(x))[1], parse_filename(os.path.basename(x))[0]))

    if not files:
        print(f"[WARN] Crop folder is empty: {args.crops}")
        print(json.dumps({"target_ids": []}, ensure_ascii=False))
        sys.exit(0)

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor  # type: ignore

    # Resolve device object early
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("[WARN] CUDA requested but not available; falling back to CPU.")

# Load model & processor (with fallback to a public CLIP)
    try:
        model = AutoModel.from_pretrained(args.qwen2_model, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(args.qwen2_model, trust_remote_code=True)
        chosen_id = args.qwen2_model
    except Exception as e:
        print(f"[WARN] Failed to load {args.qwen2_model}: {e}")
        fallback_id = "openai/clip-vit-large-patch14"
        print(f"[INFO] Falling back to {fallback_id}")
        model = AutoModel.from_pretrained(fallback_id)
        processor = AutoProcessor.from_pretrained(fallback_id)
        chosen_id = fallback_id
    model = model.to(device)
    model.eval()
    print(f"[INFO] Using CLIP model={chosen_id} on {device}")

    # Text embeddings (support "t1||t2" then take max over prompts later)
    prompts = [p.strip() for p in args.prompt.split("||") if p.strip()]
    with torch.no_grad():
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        if hasattr(model, "get_text_features"):
            text_feat = model.get_text_features(**text_inputs)
        else:
            outputs = model(**text_inputs)
            text_feat = outputs.text_embeds if hasattr(outputs, "text_embeds") else outputs[1]
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    # Build metadata for crops
    metas = []  # (path, pid, fid)
    for f in files:
        fname = os.path.basename(f)
        pid, fid = parse_filename(fname)
        metas.append((f, pid, fid))

    # Extract image features in batches
    all_img_feats = []  # list of [B, D]
    valid_indices = []  # indices in metas corresponding to extracted features
    bs = max(1, int(args.batch_size))
    with torch.no_grad():
        for i in range(0, len(metas), bs):
            batch = metas[i:i+bs]
            imgs = []
            for (path, _, _) in batch:
                try:
                    imgs.append(Image.open(path).convert("RGB"))
                except Exception as e:
                    print(f"[ERROR] Failed to open {os.path.basename(path)}: {e}")
                    imgs.append(None)
            # Filter out failed opens while preserving mapping
            ok_items = [(j, im) for j, im in enumerate(imgs) if im is not None]
            if not ok_items:
                continue
            kept_idx, kept_imgs = zip(*ok_items)
            img_inputs = processor(images=list(kept_imgs), return_tensors="pt", padding=True)
            img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
            if hasattr(model, "get_image_features"):
                img_feat = model.get_image_features(**img_inputs)
            else:
                outputs = model(**img_inputs)
                img_feat = outputs.image_embeds if hasattr(outputs, "image_embeds") else outputs[0]
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            all_img_feats.append(img_feat)
            valid_indices.extend([i + k for k in kept_idx])

    if not all_img_feats:
        print("[WARN] No image features extracted.")
        print(json.dumps({"target_ids": []}, ensure_ascii=False))
        sys.exit(0)

    img_feats = torch.cat(all_img_feats, dim=0)  # [N, D]

    # Compute cosine similarity matrix [N, M]
    sims = img_feats @ text_feat.T  # already normalized => cosine

    # Accumulate similarities per id (max over prompts for each crop)
    sims_by_id = {}
    for row_idx, meta_idx in enumerate(valid_indices):
        _, pid, _ = metas[meta_idx]
        if pid < 0:
            continue  # skip files not matching pattern
        crop_sim = sims[row_idx]
        if crop_sim.ndim == 0:
            s = float(crop_sim.item())
        else:
            s = float(crop_sim.max().item())
        sims_by_id.setdefault(pid, []).append(s)

    # Optional: compute simple location prior from bbox_meta and directional words in prompt
    def _direction_prior_for_id(pid:int) -> float:
        if not args.bbox_meta:
            return 0.5
        try:
            with open(args.bbox_meta, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except Exception:
            return 0.5
        boxes = meta.get('bboxes') or meta.get('boxes') or []
        if not boxes:
            return 0.5
        # infer W,H if not provided
        W = meta.get('width'); H = meta.get('height')
        if W is None or H is None:
            xs2 = [b.get('bbox',[0,0,0,0])[2] for b in boxes if isinstance(b, dict)]
            ys2 = [b.get('bbox',[0,0,0,0])[3] for b in boxes if isinstance(b, dict)]
            W = max(xs2) if xs2 else 1.0
            H = max(ys2) if ys2 else 1.0
        # find box for id
        bb = None
        for b in boxes:
            if int(b.get('id', -1)) == int(pid):
                bb = b.get('bbox', None)
                break
        if not bb or len(bb) < 4:
            return 0.5
        x1,y1,x2,y2 = map(float, bb[:4])
        cx = max(0.0, min(1.0, ((x1+x2)/2.0)/float(W)))
        cy = max(0.0, min(1.0, ((y1+y2)/2.0)/float(H)))
        # parse directions
        p = args.prompt.lower()
        scores = []
        if 'right' in p:
            scores.append(cx)            # closer to 1 on right
        if 'left' in p:
            scores.append(1.0 - cx)      # closer to 1 on left
        if 'top' in p or 'upper' in p:
            scores.append(1.0 - cy)      # top is small y
        if 'bottom' in p or 'lower' in p:
            scores.append(cy)
        if 'center' in p or 'middle' in p:
            scores.append(max(0.0, 1.0 - abs(cx-0.5)*2.0))
        if not scores:
            return 0.5
        # combine by average
        return max(0.0, min(1.0, sum(scores)/len(scores)))

    # Aggregate per id and apply threshold (with optional spatial prior)
    positive_ids = []
    for pid, vals in sims_by_id.items():
        if not vals:
            continue
        agg_sim = max(vals) if args.per_id_agg == "max" else sum(vals)/len(vals)
        loc_prior = _direction_prior_for_id(pid)
        final_score = args.alpha * agg_sim + (1.0 - args.alpha) * loc_prior
        print(f"[AGG] id={pid} sim={agg_sim:.4f} loc={loc_prior:.3f} final={final_score:.4f}")
        if final_score >= args.threshold:
            positive_ids.append(pid)

except Exception as e:
    print(f"[FATAL ERROR] {e}")
    traceback.print_exc()
    positive_ids = []

out = {"target_ids": sorted(list(set(positive_ids)))}
print(json.dumps(out, ensure_ascii=False))
sys.exit(0)
