"""
GDA-VisionAssist - Benchmark Script
Đo performance (FPS, latency, VRAM) của hệ thống GDA.
"""

import sys
import os

# Add project root to path so `src` package can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import time
import argparse
import json
from datetime import datetime


def benchmark_vit_encoder(gda, image_rgb, n_runs=10):
    """Benchmark ViT feature extraction"""
    times = []
    
    # Warmup
    for _ in range(3):
        gda._extract_vit_features(image_rgb)
    
    torch.cuda.synchronize()
    
    for _ in range(n_runs):
        start = time.perf_counter()
        features = gda._extract_vit_features(image_rgb)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return {
        "component": "ViT Encoder",
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "output_shape": list(features.shape) if features is not None else None,
    }


def benchmark_segmentation(gda, image_rgb, mask, n_runs=10):
    """Benchmark SETR segmentation"""
    times = []
    image_shape = image_rgb.shape[:2]
    
    # Warmup
    for _ in range(3):
        gda.predict_class_from_region(image_rgb, mask, image_shape)
    
    torch.cuda.synchronize()
    
    for _ in range(n_runs):
        start = time.perf_counter()
        predicted_class, confidence = gda.predict_class_from_region(image_rgb, mask, image_shape)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return {
        "component": "DINOv2 + SETR Segmentation",
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "predicted_class": predicted_class,
        "confidence": float(confidence) if confidence else 0.0,
    }


def benchmark_sam(gda, image_rgb, point, n_runs=5):
    """Benchmark SAM 2 segmentation"""
    times = []
    
    # Warmup
    gda.sam_segmenter.segment_from_point(image_rgb, point)
    
    torch.cuda.synchronize()
    
    for _ in range(n_runs):
        start = time.perf_counter()
        mask = gda.sam_segmenter.segment_from_point(image_rgb, point)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return {
        "component": "SAM 2",
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "mask_pixels": int(mask.sum()) if mask is not None else 0,
    }


def benchmark_full_pipeline(gda, image_rgb, mask, n_runs=5):
    """Benchmark full pipeline"""
    times = []
    
    # Warmup
    gda.process_region(image_rgb, mask)
    
    torch.cuda.synchronize()
    
    for _ in range(n_runs):
        start = time.perf_counter()
        result = gda.process_region(image_rgb, mask, "Đây là gì?")
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return {
        "component": "Full Pipeline",
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "fps": 1.0 / np.mean(times),
    }


def get_gpu_info():
    """Lấy thông tin GPU"""
    if not torch.cuda.is_available():
        return {"device": "CPU", "vram_total": 0, "vram_used": 0}
    
    return {
        "device": torch.cuda.get_device_name(0),
        "vram_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        "vram_used_gb": torch.cuda.memory_allocated(0) / (1024**3),
        "vram_reserved_gb": torch.cuda.memory_reserved(0) / (1024**3),
    }


def main():
    parser = argparse.ArgumentParser(description="GDA-VisionAssist Benchmark")
    parser.add_argument('--n-runs', type=int, default=10, help='Number of benchmark runs')
    parser.add_argument('--image', type=str, default=None, help='Test image path')
    parser.add_argument('--seg-checkpoint', type=str, default='checkpoints/setr_dino_best.pth')
    parser.add_argument('--adaptor-checkpoint', type=str, 
                        default='checkpoints/adaptor_vizwiz/adaptor.pth')
    parser.add_argument('--output', type=str, default=None, help='Save results to JSON')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🏎️  GDA-VisionAssist Benchmark")
    print("=" * 70)
    
    # GPU info
    gpu_info = get_gpu_info()
    print(f"\n🖥️  GPU: {gpu_info['device']}")
    if 'vram_total_gb' in gpu_info:
        print(f"   VRAM: {gpu_info['vram_total_gb']:.1f} GB total")
    
    # Init GDA
    print("\n📦 Loading GDA-VisionAssist...")
    from src.core.gda import GlobalDescriptionAcquisition
    
    gda = GlobalDescriptionAcquisition(
        seg_checkpoint=args.seg_checkpoint,
        adaptor_checkpoint=args.adaptor_checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Post-init GPU info
    gpu_after_init = get_gpu_info()
    if 'vram_used_gb' in gpu_after_init:
        print(f"\n📊 VRAM after init: {gpu_after_init['vram_used_gb']:.2f} GB")
    
    # Prepare test image
    if args.image:
        import cv2
        image = cv2.imread(args.image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print("⚠️ Using random image (use --image for real benchmark)")
    
    h, w = image_rgb.shape[:2]
    point = (w // 2, h // 2)
    
    # Create mask
    mask = gda.sam_segmenter.segment_from_point(image_rgb, point)
    if mask is None or mask.sum() == 0:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h//4:3*h//4, w//4:3*w//4] = 1
    
    # Run benchmarks
    results = []
    
    print(f"\n🏃 Running benchmarks ({args.n_runs} runs each)...\n")
    
    results.append(benchmark_vit_encoder(gda, image_rgb, args.n_runs))
    results.append(benchmark_segmentation(gda, image_rgb, mask, args.n_runs))
    results.append(benchmark_sam(gda, image_rgb, point, min(args.n_runs, 5)))
    results.append(benchmark_full_pipeline(gda, image_rgb, mask, min(args.n_runs, 5)))
    
    # Print results
    print("\n" + "=" * 70)
    print("📊 BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Component':<30} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['component']:<30} {r['mean_ms']:<12.1f} {r['std_ms']:<12.1f} {r['min_ms']:<12.1f}")
    
    if 'fps' in results[-1]:
        print(f"\n⚡ Full Pipeline FPS: {results[-1]['fps']:.1f}")
    
    # Save results
    if args.output:
        output = {
            "timestamp": datetime.now().isoformat(),
            "gpu": gpu_info,
            "gpu_after_init": gpu_after_init,
            "n_runs": args.n_runs,
            "results": results,
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
