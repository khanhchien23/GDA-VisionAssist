"""
Đánh giá LLaVA-1.5-7B trên 100 mẫu Visual Genome (region description).
Pipeline: crop vùng → gửi VLM → mô tả → tính ROUGE-L, CIDEr, Object_Acc, thời gian.

Models:
  1. LLaVA-1.5-7B (llava-hf/llava-1.5-7b-hf)

Requirements: transformers, torch, PIL, tqdm

Usage:
    python scripts/test_multiple_vlms.py --dataset <path> --image-dir <path> [--num-samples 100]
"""

import os
import sys
import json
import torch
import numpy as np
import random
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from typing import Dict, Optional, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.vg_eval_metrics import VGEvaluationMetrics
except ImportError:
    from vg_eval_metrics import VGEvaluationMetrics


# ============================================================================
# VLM CONFIGURATIONS
# ============================================================================

VLM_CONFIGS = [
    {
        "key": "llava_1_5_7b",
        "name": "LLaVA-1.5-7B",
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "use_8bit": True,
        "max_new_tokens": 50,
    },
]


# ============================================================================
# VLM WRAPPER
# ============================================================================

def load_llava(config: dict, device: str) -> Any:
    """Load LLaVA-1.5."""
    from transformers import LlavaProcessor, LlavaForConditionalGeneration

    model_id = config["model_id"]
    processor = LlavaProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    def generate(self, pil_image: Image.Image, prompt: str) -> str:
        # LLaVA 1.5 format: USER: <image>\n{prompt}\nASSISTANT:
        prompt_text = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs = processor(text=prompt_text, images=pil_image, return_tensors="pt")
        first_device = next(model.parameters()).device
        inputs = {k: v.to(first_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=config["max_new_tokens"], do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return processor.decode(gen, skip_special_tokens=True).strip()

    return type("LLaVA", (), {"generate": generate, "model": model, "processor": processor})()


LOADERS = {
    "llava_1_5_7b": load_llava,
}


# ============================================================================
# DATASET & EVALUATION
# ============================================================================

def load_samples(dataset_path: str, image_dir: str, num_samples: int = 100) -> List[dict]:
    """Load Visual Genome region samples."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    flattened = []
    for item in data:
        img_path = item.get("image_path")
        if not img_path:
            img_id = item.get("id") or item.get("image_id")
            if img_id:
                img_path = os.path.join(image_dir, f"{img_id}.jpg")
        if not img_path:
            continue
        for region in item.get("regions", []):
            flattened.append({
                "image_path": img_path,
                "region": region,
                "ground_truth": region.get("phrase", region.get("caption", "")),
            })

    valid = [s for s in flattened if os.path.exists(s["image_path"])]
    if len(valid) > num_samples:
        random.seed(42)
        valid = random.sample(valid, num_samples)
    return valid


def compute_metrics(ref: str, hyp: str) -> dict:
    bleu = VGEvaluationMetrics.compute_bleu(ref, hyp)
    return {
        "BLEU-1": bleu["BLEU-1"],
        "ROUGE-L": VGEvaluationMetrics.compute_rouge_l(ref, hyp),
        "CIDEr": VGEvaluationMetrics.compute_cider_simple(ref, hyp),
        "Object_Acc": VGEvaluationMetrics.compute_object_accuracy(ref, hyp),
    }


def run_vlm_eval(
    vlm,
    samples: List[dict],
    prompt: str,
    device: str,
) -> tuple:
    """Run evaluation for one VLM. Returns (results_list, metrics_agg, avg_time)."""
    results = []
    scores = {"BLEU-1": [], "ROUGE-L": [], "CIDEr": [], "Object_Acc": [], "time": []}
    current_img_path = None
    current_img = None

    for sample in tqdm(samples, desc="Evaluating", leave=True):
        img_path = sample["image_path"]
        region = sample["region"]
        gt = sample["ground_truth"]

        # Load image
        if img_path != current_img_path:
            if not os.path.exists(img_path):
                continue
            current_img = Image.open(img_path).convert("RGB")
            current_img_path = img_path
        if current_img is None:
            continue

        x = max(0, int(region.get("x", 0)))
        y = max(0, int(region.get("y", 0)))
        w = int(region.get("width", 0))
        h = int(region.get("height", 0))
        if w < 10 or h < 10:
            continue

        img_w, img_h = current_img.size
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        cropped = current_img.crop((x, y, x2, y2))

        try:
            t0 = time.perf_counter()
            out = vlm.generate(cropped, prompt)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            print(f"\n  ⚠️ Error: {e}")
            out = ""
            elapsed = 0.0

        m = compute_metrics(gt, out)
        for k in scores:
            if k == "time":
                scores[k].append(elapsed)
            else:
                scores[k].append(m[k])

        results.append({"gt": gt, "pred": out, "metrics": m, "time": elapsed})

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    agg = {k: float(np.mean(v)) if v else 0.0 for k, v in scores.items()}
    avg_time = agg.pop("time", 0.0)
    return results, agg, avg_time


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test LLaVA-1.5-7B on Visual Genome region descriptions")
    parser.add_argument("--dataset", type=str, default="D:/gda/region_descriptions.json",
                        help="Path to region_descriptions.json")
    parser.add_argument("--image-dir", type=str, default="D:/gda/VG_100K",
                        help="Path to VG images directory")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="./reports/vlm_comparison")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print("  🔬 LLaVA-1.5-7B - Region Description (100 samples)")
    print("="*70)
    print(f"  Dataset: {args.dataset}")
    print(f"  Images:  {args.image_dir}")
    print(f"  Device:  {device}")
    print("="*70)

    samples = load_samples(args.dataset, args.image_dir, args.num_samples)
    if not samples:
        print("❌ No valid samples found!")
        return

    print(f"\n✅ Loaded {len(samples)} samples")

    prompt = (
        "Describe the object in this cropped region. "
        "Output a short English noun phrase (3-8 words)."
    )

    all_results = {}
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = VLM_CONFIGS[0]
    key = cfg["key"]
    name = cfg["name"]
    print(f"\n{'='*70}")
    print(f"  📦 [{key}] {name}")
    print("="*70)

    try:
        vlm = load_llava(cfg, device)
        results, agg, avg_time = run_vlm_eval(vlm, samples, prompt, device)

        all_results[key] = {
            "name": name,
            "metrics": agg,
            "avg_time_sec": avg_time,
            "num_samples": len(results),
        }

        del vlm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n  📊 {name}: ROUGE-L={agg['ROUGE-L']:.4f} CIDEr={agg['CIDEr']:.4f} "
              f"ObjAcc={agg['Object_Acc']:.4f} | ⏱️ {avg_time:.2f}s/sample")

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("  📊 EVALUATION SUMMARY - LLaVA-1.5-7B")
    print("="*70)
    if all_results:
        m = all_results[key]["metrics"]
        print(f"  ROUGE-L   : {m['ROUGE-L']:.4f}")
        print(f"  CIDEr     : {m['CIDEr']:.4f}")
        print(f"  Object_Acc: {m['Object_Acc']:.4f}")
        print(f"  BLEU-1    : {m['BLEU-1']:.4f}")
        print(f"  Avg Time  : {all_results[key]['avg_time_sec']:.2f}s/sample")
        print(f"  Samples   : {all_results[key]['num_samples']}")
    print("="*70)

    out_path = os.path.join(args.output_dir, "llava_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved: {out_path}")


if __name__ == "__main__":
    main()