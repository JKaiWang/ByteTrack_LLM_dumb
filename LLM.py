# ===============================================
# LLM.py — Qwen2-CLIP embeddings + cosine similarity selection
# ===============================================
import os, json, re, argparse, sys, traceback
from typing import Tuple, List, Optional


def parse_filename(fname: str) -> Tuple[int, int]:
    match = re.match(r"i(\d+)_f(\d+)", fname)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (-1, -1)


def select_targets(
    crops_dir: str,
    prompt: str,
    threshold: float = 0.30,
    device: str = "cpu",
    qwen2_model: str = "openai/clip-vit-large-patch14",
    per_id_agg: str = "max",
    batch_size: int = 32,
    bbox_meta: Optional[str] = None,
    alpha: float = 0.85,
    quiet: bool = False
) -> List[int]:
    """
    Select target IDs based on CLIP cosine similarity with prompt.
    
    Args:
        crops_dir: Directory containing cropped images (named i{id}_f{frame}.jpg)
        prompt: Text prompt for target selection
        threshold: Cosine similarity threshold (>= threshold → keep)
        device: "cpu" or "cuda"
        qwen2_model: HuggingFace model ID for CLIP-like model
        per_id_agg: "max" or "mean" - how to aggregate multiple crops per id
        batch_size: Batch size for image feature extraction        bbox_meta: Optional JSON path with bbox metadata
        alpha: Weight for visual cosine vs location prior
    Returns:
        List of target IDs that passed the threshold
    """
    try:
        if not os.path.isdir(crops_dir):
            print(f"[ERROR] Crop folder not found: {crops_dir}")
            return []

        files = sorted([
            os.path.abspath(os.path.join(crops_dir, f))
            for f in os.listdir(crops_dir)
            if os.path.isfile(os.path.join(crops_dir, f))
        ], key=lambda x: (parse_filename(os.path.basename(x))[1], parse_filename(os.path.basename(x))[0]))

        if not files:
            print(f"[WARN] Crop folder is empty: {crops_dir}")
            return []

        import torch
        from PIL import Image
        from transformers import AutoModel, AutoProcessor

        # Resolve device object early
        dev = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        if device == "cuda" and dev.type != "cuda" and not quiet:
            print("[WARN] CUDA requested but not available; falling back to CPU.")

        # Load model & processor
        try:
            model = AutoModel.from_pretrained(qwen2_model, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(qwen2_model, trust_remote_code=True)
            chosen_id = qwen2_model
        except Exception as e:
            if not quiet:
                print(f"[WARN] Failed to load {qwen2_model}: {e}")
            fallback_id = "openai/clip-vit-large-patch14"
            if not quiet:
                print(f"[INFO] Falling back to {fallback_id}")
            model = AutoModel.from_pretrained(fallback_id)
            processor = AutoProcessor.from_pretrained(fallback_id)
            chosen_id = fallback_id
        model = model.to(dev)
        model.eval()
        if not quiet:
            print(f"[INFO] Using CLIP model={chosen_id} on {dev}")

        # Text embeddings
        prompts = [p.strip() for p in prompt.split("||") if p.strip()]
        with torch.no_grad():
            text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(dev) for k, v in text_inputs.items()}
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
        all_img_feats = []
        valid_indices = []
        bs = max(1, int(batch_size))
        with torch.no_grad():
            for i in range(0, len(metas), bs):
                batch = metas[i:i+bs]
                imgs = []
                for (path, _, _) in batch:
                    try:
                        imgs.append(Image.open(path).convert("RGB"))
                    except Exception as e:
                        if not quiet:
                            print(f"[ERROR] Failed to open {os.path.basename(path)}: {e}")
                        imgs.append(None)
                ok_items = [(j, im) for j, im in enumerate(imgs) if im is not None]
                if not ok_items:
                    continue
                kept_idx, kept_imgs = zip(*ok_items)
                img_inputs = processor(images=list(kept_imgs), return_tensors="pt", padding=True)
                img_inputs = {k: v.to(dev) for k, v in img_inputs.items()}
                if hasattr(model, "get_image_features"):
                    img_feat = model.get_image_features(**img_inputs)
                else:
                    outputs = model(**img_inputs)
                    img_feat = outputs.image_embeds if hasattr(outputs, "image_embeds") else outputs[0]
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

                all_img_feats.append(img_feat)
                valid_indices.extend([i + k for k in kept_idx])

        if not all_img_feats:
            if not quiet:
                print("[WARN] No image features extracted.")
            return []

        img_feats = torch.cat(all_img_feats, dim=0)

        # Compute cosine similarity matrix
        sims = img_feats @ text_feat.T

        # Accumulate similarities per id
        sims_by_id = {}
        for row_idx, meta_idx in enumerate(valid_indices):
            _, pid, _ = metas[meta_idx]
            if pid < 0:
                continue
            crop_sim = sims[row_idx]
            if crop_sim.ndim == 0:
                s = float(crop_sim.item())
            else:
                s = float(crop_sim.max().item())
            sims_by_id.setdefault(pid, []).append(s)

        # Location prior helper
        def _direction_prior_for_id(pid: int) -> float:
            if not bbox_meta:
                return 0.5
            try:
                with open(bbox_meta, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception:
                return 0.5
            boxes = meta.get('bboxes') or meta.get('boxes') or []
            if not boxes:
                return 0.5
            W = meta.get('width'); H = meta.get('height')
            if W is None or H is None:
                xs2 = [b.get('bbox',[0,0,0,0])[2] for b in boxes if isinstance(b, dict)]
                ys2 = [b.get('bbox',[0,0,0,0])[3] for b in boxes if isinstance(b, dict)]
                W = max(xs2) if xs2 else 1.0
                H = max(ys2) if ys2 else 1.0
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
            p = prompt.lower()
            scores = []
            if 'right' in p:
                scores.append(cx)
            if 'left' in p:
                scores.append(1.0 - cx)
            if 'top' in p or 'upper' in p:
                scores.append(1.0 - cy)
            if 'bottom' in p or 'lower' in p:
                scores.append(cy)
            if 'center' in p or 'middle' in p:
                scores.append(max(0.0, 1.0 - abs(cx-0.5)*2.0))
            if not scores:
                return 0.5
            return max(0.0, min(1.0, sum(scores)/len(scores)))

        # Aggregate per id and apply threshold
        positive_ids = []
        for pid, vals in sims_by_id.items():
            if not vals:
                continue
            agg_sim = max(vals) if per_id_agg == "max" else sum(vals)/len(vals)
            loc_prior = _direction_prior_for_id(pid)
            final_score = alpha * agg_sim + (1.0 - alpha) * loc_prior
            if not quiet:
                print(f"[AGG] id={pid} sim={agg_sim:.4f} loc={loc_prior:.3f} final={final_score:.4f}")
            if final_score >= threshold:
                positive_ids.append(pid)

        return sorted(list(set(positive_ids)))

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        traceback.print_exc()
        return []


# ============ Main CLI entry point ============
if __name__ == "__main__":
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

    # Use the select_targets function
    positive_ids = select_targets(
        crops_dir=args.crops,
        prompt=args.prompt,
        threshold=args.threshold,
        device=args.device,
        qwen2_model=args.qwen2_model,
        per_id_agg=args.per_id_agg,
        batch_size=args.batch_size,
        bbox_meta=args.bbox_meta if args.bbox_meta else None,
        alpha=args.alpha,
    )

    out = {"target_ids": positive_ids}
    print(json.dumps(out, ensure_ascii=False))
    sys.exit(0)
