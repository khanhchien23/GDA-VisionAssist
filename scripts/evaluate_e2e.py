"""
End-to-End Evaluation Script for Visual Genome Regions
So sánh Full Pipeline (GDA) với Qwen2-VL baseline

Metrics:
- BLEU (1-4): N-gram precision
- ROUGE-L: Longest Common Subsequence
- CIDEr: Consensus-based Image Description Evaluation (Approximate)
- Object Accuracy: Recall of objects in generated description

Datasets:
- Visual Genome Region Descriptions
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Custom Metrics
try:
    from scripts.vg_eval_metrics import VGEvaluationMetrics
except ImportError:
    # Fallback if running from root
    from vg_eval_metrics import VGEvaluationMetrics

# ============================================================================
# EVALUATOR CLASS
# ============================================================================

class VGEvaluator:
    """Evaluator for Visual Genome Regions - Full Pipeline vs Qwen2-VL baseline."""
    
    def __init__(
        self,
        seg_checkpoint: str = "./checkpoints/setr_dino_best.pth",
        adaptor_checkpoint: str = "./checkpoints/adaptor_vizwiz/adaptor.pth",
        device: str = "cuda",
        language: str = "en"
    ):
        self.device = device
        self.seg_checkpoint = seg_checkpoint
        self.adaptor_checkpoint = adaptor_checkpoint
        self.language = language
        self.gda = None
        
    def load_models(self):
        """Load both pipeline and baseline models."""
        print("\n" + "="*70)
        print("🔄 Loading Models for Evaluation")
        print("="*70)
        
        print("\n📦 Loading Full Pipeline (GDA)...")
        from src.core import GlobalDescriptionAcquisition
        
        self.gda = GlobalDescriptionAcquisition(
            seg_checkpoint=self.seg_checkpoint,
            adaptor_checkpoint=self.adaptor_checkpoint,
            device=self.device
        )
        
        print("\n✅ Models loaded successfully")

    def create_region_mask(self, image_shape: Tuple[int, int], region: Dict) -> np.ndarray:
        """Create binary mask from region [x, y, w, h]."""
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        x = max(0, int(region['x']))
        y = max(0, int(region['y']))
        rw = int(region['width'])
        rh = int(region['height'])
        
        # Clip to image boundaries
        x2 = min(w, x + rw)
        y2 = min(h, y + rh)
        
        mask[y:y2, x:x2] = 1
        return mask

    def generate_with_pipeline(self, image: np.ndarray, region: Dict, question: str) -> str:
        """Generate answer using full pipeline (Adaptor + TextDecoder)."""
        try:
            h, w = image.shape[:2]
            
            # Skip regions that are too small (can cause CUDA errors)
            rw = int(region.get('width', 0))
            rh = int(region.get('height', 0))
            if rw < 5 or rh < 5:
                return ""
            
            mask = self.create_region_mask((h, w), region)
            
            # Use process_region with generated mask
            result = self.gda.process_region(image, mask, question)
            return result.get('description', '')
        except Exception as e:
            print(f"\nPipeline error: {e}")
            return ""
    
    def generate_with_qwen_baseline(self, image: np.ndarray, region: Dict, question: str) -> str:
        """
        Generate answer using Qwen2-VL native.
        Note: Qwen2-VL doesn't support mask input natively efficiently without 
        cropping or special prompting. Here we crop the image to the region 
        to simulate 'looking at the region' for the baseline.
        """
        try:
            pil_image = Image.fromarray(image)
            
            # Crop to region
            x = max(0, int(region['x']))
            y = max(0, int(region['y']))
            w = int(region['width'])
            h = int(region['height'])
            
            # Ensure valid crop (minimum 32x32 to avoid CUDA errors)
            if w <= 0 or h <= 0:
                return ""

            cropped_image = pil_image.crop((x, y, x+w, y+h))
            
            # Resize tiny crops to minimum safe size for vision model
            MIN_SIZE = 32
            cw, ch = cropped_image.size
            if cw < MIN_SIZE or ch < MIN_SIZE:
                cropped_image = cropped_image.resize((max(cw, MIN_SIZE), max(ch, MIN_SIZE)), Image.LANCZOS)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": cropped_image},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # Use processor directly
            text = self.gda.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.gda.processor(
                text=[text],
                images=[cropped_image],
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                output_ids = self.gda.full_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )
            
            generated = output_ids[0][inputs.input_ids.shape[1]:]
            response = self.gda.processor.decode(generated, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            print(f"\nQwen baseline error: {e}")
            return ""
    
    def translate_text(self, text: str, to_lang: str = 'vi') -> str:
        """Translate text using Qwen2-VL (text-only mode)."""
        try:
            prompt = f"Translate the following English phrase to Vietnamese: '{text}'\nTranslation:"
            if to_lang == 'en':
                prompt = f"Translate the following Vietnamese phrase to English: '{text}'\nTranslation:"
                
            messages = [{"role": "user", "content": prompt}]
            
            text_input = self.gda.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.gda.processor(
                text=[text_input],
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                output_ids = self.gda.full_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )
            
            generated = output_ids[0][inputs.input_ids.shape[1]:]
            response = self.gda.processor.decode(generated, skip_special_tokens=True)
            return response.strip().replace('"', '').replace("'", "")
            
        except Exception as e:
            print(f"Translation error: {e}")
            return text # Fallback to original

    def evaluate_sample(
        self, 
        image: np.ndarray, 
        region: Dict,
        question: str,
        ground_truth: str
    ) -> Dict:
        """Evaluate a single sample with both methods."""
        import time
        
        start_p = time.time()
        pipeline_output = self.generate_with_pipeline(image, region, question)
        time_p = time.time() - start_p
        
        start_b = time.time()
        baseline_output = self.generate_with_qwen_baseline(image, region, question)
        time_b = time.time() - start_b
        
        def get_metrics(ref, hyp, time_elapsed):
            bleu = VGEvaluationMetrics.compute_bleu(ref, hyp)
            return {
                'BLEU-1': bleu['BLEU-1'],
                'BLEU-4': bleu['BLEU-4'],
                'ROUGE-L': VGEvaluationMetrics.compute_rouge_l(ref, hyp),
                'CIDEr': VGEvaluationMetrics.compute_cider_simple(ref, hyp),
                'Object_Acc': VGEvaluationMetrics.compute_object_accuracy(ref, hyp),
                'Inference_Time': time_elapsed
            }

        pipeline_metrics = get_metrics(ground_truth, pipeline_output, time_p)
        baseline_metrics = get_metrics(ground_truth, baseline_output, time_b)
        
        return {
            'id': region.get('id', 'unknown'),
            'ground_truth': ground_truth,
            'pipeline_output': pipeline_output,
            'baseline_output': baseline_output,
            'pipeline_metrics': pipeline_metrics,
            'baseline_metrics': baseline_metrics
        }
    
    def evaluate_dataset(
        self,
        dataset_path: str,
        image_dir: str,
        num_samples: int = 100,
        output_path: str = "./vg_evaluation_results.json"
    ) -> Dict:
        """Evaluate on Visual Genome dataset."""
        print(f"\n📊 Evaluating on Visual Genome: {dataset_path}")
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return {}

        # Flatten regions mapping: (image_path, region_data)
        # Assuming VG format: [{id: 1, regions: [...]}, ...]
        # Or simplified: [{"image_path": "...", "regions": [...]}]
        
        flattened_samples = []
        
        print("Preparing samples...")
        for item in data:
            # Format 1: eval_dataset.json ( [{image_path: '...', caption: '...'}, ...] )
            if 'image_path' in item and 'caption' in item:
                # Mock a region since this is full-image or specific point
                flattened_samples.append({
                    'image_path': item['image_path'],
                    'region': {'x': 0, 'y': 0, 'width': 500, 'height': 500, 'id': len(flattened_samples)}, # Dummy region
                    'ground_truth': item['caption']
                })
                continue
                
            # Format 2: Standard VG
            # The item might have 'id' directly at the root, 
            # Or the 'image_id' might only be inside the 'regions' list!
            img_id = item.get('id') or item.get('image_id')
            
            regions = item.get('regions', [])
            
            # If root doesn't have id, check the first region
            if not img_id and len(regions) > 0:
                img_id = regions[0].get('image_id')
            
            # Skip if we STILL couldn't find an ID
            if not img_id:
                continue
                
            abs_img_dir = os.path.abspath(image_dir)
            img_path = os.path.join(abs_img_dir, f"{img_id}.jpg")
            
            for region in regions:
                flattened_samples.append({
                    'image_path': img_path,
                    'region': region,
                    'ground_truth': region.get('phrase', region.get('caption', ''))
                })
        
        print(f"Total regions/images found: {len(flattened_samples)}")
        
        # Filter for existing images first
        valid_samples = []
        print("Filtering for existing images...")
        for sample in tqdm(flattened_samples, desc="Checking images"):
            if os.path.exists(sample['image_path']):
                valid_samples.append(sample)
        
        print(f"Valid regions with images: {len(valid_samples)}/{len(flattened_samples)}")
        flattened_samples = valid_samples
        
        if num_samples > 0:
            if len(flattened_samples) > num_samples:
                # Random sample to get variety
                import random
                random.seed(42)
                flattened_samples = random.sample(flattened_samples, num_samples)
            print(f"Evaluating subset: {len(flattened_samples)} samples")
            
        pipeline_scores = {k: [] for k in ['BLEU-1', 'BLEU-4', 'ROUGE-L', 'CIDEr', 'Object_Acc', 'Inference_Time']}
        baseline_scores = {k: [] for k in ['BLEU-1', 'BLEU-4', 'ROUGE-L', 'CIDEr', 'Object_Acc', 'Inference_Time']}
        
        results = []
        skipped = 0
        
        # Cache loaded images to avoid re-opening for same image multiple regions
        current_image_path = None
        current_image_np = None
        
        for sample in tqdm(flattened_samples, desc="Evaluating"):
            try:
                img_path = sample['image_path']
                
                # Load image only if changed
                if img_path != current_image_path:
                    if not os.path.exists(img_path):
                        skipped += 1
                        continue
                    current_image_np = np.array(Image.open(img_path).convert('RGB'))
                    current_image_path = img_path
                
                # DEBUG LOGGING
                tqdm.write(f"Processing: {img_path} (ID: {sample.get('id', 'unknown')})")
                
                # Safety check before processing
                if not torch.cuda.is_available():
                    print("❌ Fatal Error: CUDA is no longer available (Device Assert Triggered). Stopping.")
                    break
                
                if current_image_np is None:
                    skipped += 1
                    continue
                    
                question = "Describe the marked region with a short phrase (e.g., 'a red apple', 'the blue sky'). Do not use full sentences."
                if self.language == 'vi':
                    question = "Hãy mô tả vùng được đánh dấu bằng một cụm từ thật ngắn gọn (ví dụ: 'một quả táo đỏ'). Không dùng câu hoàn chỉnh."
                
                # TRANSLATE GROUND TRUTH IF NEEDED
                ground_truth = sample['ground_truth']
                if self.language == 'vi':
                    # Check cache first or translation needed
                    # ideally we'd cache this but for 100 samples it's fine
                    ground_truth = self.translate_text(ground_truth, to_lang='vi')
                
                result = self.evaluate_sample(
                    current_image_np, 
                    sample['region'], 
                    question, 
                    ground_truth
                )
                
                result['image_path'] = img_path # Add for reference
                results.append(result)
                
                for metric in pipeline_scores:
                    pipeline_scores[metric].append(result['pipeline_metrics'][metric])
                    baseline_scores[metric].append(result['baseline_metrics'][metric])
                    
            except RuntimeError as e:
                if 'CUDA' in str(e) or 'device-side assert' in str(e):
                    tqdm.write(f"🔄❌ CUDA error, resetting GPU state and skipping sample...")
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.synchronize()
                    except:
                        pass
                    # Reset current image to force reload
                    current_image_path = None
                    current_image_np = None
                else:
                    tqdm.write(f"❌ Error: {e}")
                skipped += 1
                continue
            except Exception as e:
                tqdm.write(f"❌ Error processing sample: {e}")
                skipped += 1
                continue


        
        # Summary
        summary = {
            'num_samples': len(results),
            'skipped': skipped,
            'pipeline': {
                metric: float(np.mean(scores)) if scores else 0.0
                for metric, scores in pipeline_scores.items()
            },
            'baseline': {
                metric: float(np.mean(scores)) if scores else 0.0
                for metric, scores in baseline_scores.items()
            }
        }
        
        output = {
            'summary': summary,
            'detailed_results': results[:50],  # Save first 50
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results saved to: {output_path}")
        
        return summary
    
    def print_comparison_table(self, summary: Dict):
        """Print formatted comparison table."""
        print("\n" + "="*80)
        print("📊 VISUAL GENOME EVALUATION RESULTS")
        print("="*80)
        print(f"\nTotal samples evaluated: {summary['num_samples']}")
        print(f"Skipped: {summary.get('skipped', 0)}")
        print("\n" + "-"*80)
        print(f"{'Metric':<20} | {'Full Pipeline':<20} | {'Qwen2-VL Baseline':<20} | {'Diff':<10}")
        print("-"*80)
        
        metrics = ['BLEU-4', 'ROUGE-L', 'CIDEr', 'Object_Acc', 'Inference_Time']
        
        for metric in metrics:
            p_val = summary['pipeline'].get(metric, 0)
            b_val = summary['baseline'].get(metric, 0)
            
            # Format: percentage for some, raw for others?
            # CIDEr is usually 0-10 or more. Others 0-1.
            # Visual Genome usually reported as 0-1 (or %)
            # Let's show raw values 0.00-1.00
            
            diff = p_val - b_val
            if metric == 'Inference_Time':
                # For time, lower is better. Diff is p_val - b_val, so negative is better for pipeline
                diff_str = f"{diff:+.3f}s"
                print(f"{metric:<20} | {p_val:<19.3f}s | {b_val:<19.3f}s | {diff_str}")
            else:
                diff_str = f"{diff:+.3f}"
                print(f"{metric:<20} | {p_val:<20.4f} | {b_val:<20.4f} | {diff_str}")
        
        print("-"*80)
        print("="*80)


def main():
    # ========================================
    # DEFAULT CONFIG
    # ========================================
    DEFAULT_DATASET = "../eval_dataset_100.json" # Dữ liệu Visual Genome JSON
    DEFAULT_IMG_DIR = "../VG_100K" # Thư mục chứa ảnh
    DEFAULT_NUM_SAMPLES = 100  # Reduced for quicker initial testing
    DEFAULT_OUTPUT = "./vg_evaluation_results.json"
    DEFAULT_SEG_CHECKPOINT = "../checkpoints/setr_dino_best.pth"
    DEFAULT_ADAPTOR_CHECKPOINT = "../checkpoints/adaptor_vizwiz/adaptor.pth"
    DEFAULT_LANGUAGE = "en"
    
    parser = argparse.ArgumentParser(description='Visual Genome Evaluation Regions')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET)
    parser.add_argument('--image-dir', type=str, default=DEFAULT_IMG_DIR)
    parser.add_argument('--num-samples', type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT)
    parser.add_argument('--seg-checkpoint', type=str, default=DEFAULT_SEG_CHECKPOINT)
    parser.add_argument('--adaptor-checkpoint', type=str, default=DEFAULT_ADAPTOR_CHECKPOINT)
    parser.add_argument('--language', type=str, default=DEFAULT_LANGUAGE, help='Evaluation language (en/vi)')
    
    args = parser.parse_args()
    
    # ALWAYS resolve relative paths based on script directory
    # This ensures it works regardless of CWD or how the script is launched
    script_dir = Path(__file__).parent
    
    def resolve_path(p):
        """Resolve relative path based on script directory"""
        p = str(p)
        if not os.path.isabs(p):
            return str(script_dir / p)
        return p
    
    args.dataset = resolve_path(args.dataset)
    args.image_dir = resolve_path(args.image_dir)
    args.output = resolve_path(args.output)
    args.seg_checkpoint = resolve_path(args.seg_checkpoint)
    args.adaptor_checkpoint = resolve_path(args.adaptor_checkpoint)
    
    # Fallback: if dataset doesn't exist, try eval_dataset.json
    if not os.path.exists(args.dataset) and os.path.exists(str(script_dir / "../eval_dataset.json")):
        args.dataset = str(script_dir / "../eval_dataset.json")
    
    print(f"\n📁 Dataset: {args.dataset}")
    print(f"🖼️ Images: {args.image_dir}")
    print(f"📊 Samples: {args.num_samples}")
    print(f"🌐 Language: {args.language}")
    
    evaluator = VGEvaluator(
        seg_checkpoint=args.seg_checkpoint,
        adaptor_checkpoint=args.adaptor_checkpoint,
        language=args.language
    )
    
    evaluator.load_models()
    
    summary = evaluator.evaluate_dataset(
        dataset_path=args.dataset,
        image_dir=args.image_dir,
        num_samples=args.num_samples,
        output_path=args.output
    )
    
    evaluator.print_comparison_table(summary)

if __name__ == "__main__":
    main()

