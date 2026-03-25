"""
GDA-VisionAssist - Download Models Script
Tải các model cần thiết cho hệ thống GDA.
"""

import os
import sys
import argparse


def download_qwen_model(model_name="Qwen/Qwen2-VL-2B-Instruct"):
    """Tải Qwen2-VL model từ HuggingFace"""
    print(f"\n📦 Đang tải {model_name}...")
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
        
        print("   Downloading processor...")
        AutoProcessor.from_pretrained(model_name)
        
        print("   Downloading tokenizer...")
        AutoTokenizer.from_pretrained(model_name)
        
        print("   Downloading model (this may take a while)...")
        AutoModelForVision2Seq.from_pretrained(model_name)
        
        print(f"✅ {model_name} đã tải xong!")
        return True
    except Exception as e:
        print(f"❌ Lỗi tải {model_name}: {e}")
        return False


def download_sam_model():
    """Tải SAM 2 model"""
    print("\n📦 Đang tải SAM 2...")
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        # SAM 2 sẽ tự tải khi khởi tạo
        print("✅ SAM 2 dependencies ready!")
        return True
    except ImportError:
        print("⚠️ SAM 2 chưa cài. Cài bằng: pip install segment-anything-2")
        return False
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False


def download_dinov2_model(variant="vitb14"):
    """Tải DINOv2 model"""
    print(f"\n📦 Đang tải DINOv2 ({variant})...")
    try:
        import torch
        model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{variant}')
        print(f"✅ DINOv2 {variant} đã tải xong!")
        return True
    except Exception as e:
        print(f"❌ Lỗi tải DINOv2: {e}")
        return False


def check_checkpoints():
    """Kiểm tra các checkpoint đã có"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    
    print("\n📋 Kiểm tra checkpoints:")
    
    required = {
        "SETR Decoder": os.path.join(checkpoints_dir, "setr_dino_best.pth"),
        "Adaptor": os.path.join(checkpoints_dir, "adaptor_vizwiz", "adaptor.pth"),
        "Masked Extractor": os.path.join(checkpoints_dir, "adaptor_vizwiz", "masked_extractor.pth"),
        "Text Decoder": os.path.join(checkpoints_dir, "adaptor_vizwiz", "text_decoder.pth"),
    }
    
    all_found = True
    for name, path in required.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        size = ""
        if exists:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            size = f" ({size_mb:.1f} MB)"
        print(f"  {status} {name}: {os.path.basename(path)}{size}")
        if not exists:
            all_found = False
    
    if not all_found:
        print("\n⚠️ Một số checkpoint chưa có.")
        print("   Bạn cần train model hoặc tải checkpoint từ nguồn bên ngoài.")
    
    return all_found


def main():
    parser = argparse.ArgumentParser(description="Download models for GDA-VisionAssist")
    parser.add_argument('--qwen-only', action='store_true', help='Only download Qwen2-VL')
    parser.add_argument('--dinov2-only', action='store_true', help='Only download DINOv2')
    parser.add_argument('--check', action='store_true', help='Only check existing checkpoints')
    parser.add_argument('--model-name', type=str, 
                        default="Qwen/Qwen2-VL-2B-Instruct",
                        help='Qwen model name')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔧 GDA-VisionAssist - Model Downloader")
    print("=" * 60)
    
    if args.check:
        check_checkpoints()
        return
    
    if args.qwen_only:
        download_qwen_model(args.model_name)
        return
    
    if args.dinov2_only:
        download_dinov2_model()
        return
    
    # Download tất cả
    results = {}
    results['Qwen2-VL'] = download_qwen_model(args.model_name)
    results['DINOv2'] = download_dinov2_model()
    results['SAM 2'] = download_sam_model()
    
    # Check checkpoints
    check_checkpoints()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Tổng kết:")
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
