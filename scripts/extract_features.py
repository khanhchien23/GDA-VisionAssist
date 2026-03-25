"""
Precompute ViT Features - WINDOWS (No Augmentation)
- Extracts features for original images ONLY
- Single-image extraction for stability on 8GB VRAM
- Resumes automatically if interrupted
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoProcessor, AutoModelForVision2Seq

# ============================================================================
# CONFIG - RTX 4060 8GB
# ============================================================================
class Config:
    VIZWIZ_ROOT = r"D:\luu_tam\gda\VizWiz" 
    FEATURES_DIR = r"D:\luu_tam\gda\features"
    IMAGE_SIZE = 336
    MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
    DEVICE = 'cuda'
    EMPTY_CACHE_INTERVAL = 50


# ============================================================================
# FEATURE EXTRACTOR
# ============================================================================
class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        
        print("=" * 60)
        print("🔧 PRECOMPUTE VIT FEATURES (NO AUGMENTATION)")
        print("=" * 60)
        
        print("\n📦 Loading Qwen2-VL...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.processor = AutoProcessor.from_pretrained(config.MODEL_NAME)
        self.vit = self.model.visual
        self.vit.eval()
        
        print("✅ Ready")
    
    def load_image(self, path):
        try:
            img = Image.open(path).convert('RGB')
            return img.resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), Image.LANCZOS)
        except:
            return None
    
    def extract_features(self, img):
        """Extract features for single image."""
        prompt = "<|vision_start|><|image_pad|><|vision_end|>"
        inputs = self.processor(text=[prompt], images=[img], return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            pv = inputs['pixel_values']
            
            if pv.dim() == 2:
                grid_thw = inputs['image_grid_thw']
                feat = self.vit(pv, grid_thw=grid_thw)
                t, h, w = grid_thw[0].tolist()
                return feat.cpu().half(), (h, w)
            else:
                B, C, H, W = pv.shape
                gh, gw = H // 14, W // 14
                grid = torch.tensor([[1, gh, gw]], device=self.device, dtype=torch.long)
                feat = self.vit(pixel_values=pv, grid_thw=grid)
                if isinstance(feat, tuple):
                    feat = feat[0]
                elif hasattr(feat, 'last_hidden_state'):
                    feat = feat.last_hidden_state
                return feat.cpu().half(), (gh, gw)
        
        return None, None
    
    def process_split(self, split):
        data_dir = Path(self.config.VIZWIZ_ROOT)
        
        # Find annotations
        ann_path = data_dir / 'Annotations' / f'{split}.json'
        if not ann_path.exists():
            ann_path = data_dir / f'{split}.json'
        if not ann_path.exists():
            print(f"❌ No {split}.json found")
            return
        
        with open(ann_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # Image directory (handle nested VizWiz/train/train or VizWiz/train)
        img_dir = data_dir / split / split
        if not img_dir.exists():
            img_dir = data_dir / split
        
        out_dir = Path(self.config.FEATURES_DIR) / split
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📂 {split}: {len(annotations)} samples")
        print(f"📁 Images: {img_dir}")
        print(f"📁 Output: {out_dir}")
        
        saved = 0
        errors = 0
        seen = set()
        
        for item in tqdm(annotations, desc=f"{split}"):
            name = item.get('image')
            if not name or name in seen:
                continue
            seen.add(name)
            
            img_path = img_dir / name
            if not img_path.exists():
                continue
            
            base = Path(name).stem
            orig_path = out_dir / f"{base}.pt"
            
            # Skip if exists
            if orig_path.exists():
                saved += 1
                continue
            
            img = self.load_image(str(img_path))
            if img is None:
                continue
            
            # Process Original Only
            try:
                feat, gs = self.extract_features(img)
                if feat is not None:
                    torch.save({'features': feat, 'grid_size': gs}, orig_path)
                    saved += 1
            except:
                errors += 1
            
            if saved % self.config.EMPTY_CACHE_INTERVAL == 0:
                torch.cuda.empty_cache()
        
        print(f"✅ {split}: saved={saved}, errors={errors}")
    
    def run(self):
        for split in ['train', 'val']:
            self.process_split(split)
        print("\n" + "=" * 60)
        print("✅ DONE!")
        print("=" * 60)


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
        try:
            FeatureExtractor(Config).run()
        except KeyboardInterrupt:
            print("\n🛑 Stopped")
        except Exception as e:
            import traceback
            traceback.print_exc()
    else:
        print("❌ CUDA not available")
