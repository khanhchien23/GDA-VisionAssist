"""
Prepare Evaluation Dataset
Tạo dataset test format cho evaluate_e2e.py

Hỗ trợ:
1. VizWiz dataset
2. Visual Genome (nếu có)
3. Custom images folder
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

def prepare_vizwiz(vizwiz_dir: str, split: str = "val", output_path: str = "eval_dataset.json"):
    """
    Prepare VizWiz dataset for evaluation.
    
    Args:
        vizwiz_dir: Path to VizWiz dataset (contains val/ and Annotations/)
        split: 'val' or 'test'
        output_path: Output JSON path
    """
    print(f"📂 Preparing VizWiz {split} dataset...")
    
    # Load annotations
    ann_path = os.path.join(vizwiz_dir, "Annotations", f"{split}.json")
    if not os.path.exists(ann_path):
        print(f"❌ Annotation file not found: {ann_path}")
        return
    
    with open(ann_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process
    samples = []
    # Check multiple possible image directories (handle nested dirs like val/val/)
    possible_dirs = [
        os.path.join(vizwiz_dir, split),
        os.path.join(vizwiz_dir, split, split),  # Handle val/val/ case
        os.path.join(vizwiz_dir, split, 'images'),
    ]
    
    images_dir = None
    for d in possible_dirs:
        if os.path.exists(d) and any(f.endswith('.jpg') for f in os.listdir(d)[:10]):
            images_dir = d
            break
    
    if images_dir is None:
        print(f"❌ Could not find images directory in {vizwiz_dir}/{split}")
        return
    
    print(f"   📁 Found images in: {images_dir}")
    
    for item in tqdm(data, desc="Processing"):
        image_name = item.get('image') or item.get('file_name')
        if not image_name:
            continue
        
        image_path = os.path.join(images_dir, image_name)
        if not os.path.exists(image_path):
            continue
        
        # Get best answer from multiple annotators
        answers = item.get('answers', [])
        if answers:
            answer_counts = {}
            for ans in answers:
                txt = ans.get('answer', '')
                if txt and txt.lower() != 'unanswerable':
                    answer_counts[txt] = answer_counts.get(txt, 0) + 1
            
            if answer_counts:
                best_answer = max(answer_counts, key=answer_counts.get)
            else:
                best_answer = "Unable to describe"
        else:
            best_answer = "No description available"
        
        samples.append({
            'image_path': image_path,
            'caption': best_answer,
            'question': item.get('question', 'Describe this image')
        })
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created {len(samples)} samples → {output_path}")

def prepare_custom_folder(images_dir: str, output_path: str = "eval_dataset.json"):
    """
    Prepare custom images folder for evaluation.
    No ground truth captions - will use empty reference (for qualitative eval only).
    
    Args:
        images_dir: Path to folder containing images
        output_path: Output JSON path
    """
    print(f"📂 Preparing custom dataset from: {images_dir}")
    
    samples = []
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    for f in Path(images_dir).iterdir():
        if f.suffix.lower() in extensions:
            samples.append({
                'image_path': str(f),
                'caption': '',  # No ground truth
                'question': 'Describe this image in detail'
            })
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created {len(samples)} samples → {output_path}")
    print("⚠️  Note: No ground truth captions - metrics will be for comparison only")

def prepare_visual_genome(vg_dir: str, output_path: str = "eval_dataset.json", num_samples: int = 1000):
    """
    Prepare Visual Genome Regions dataset for evaluation.
    
    Args:
        vg_dir: Path to Visual Genome dataset
        output_path: Output JSON path
        num_samples: Number of samples to use
    """
    print(f"📂 Preparing Visual Genome dataset...")
    
    # Check for region_descriptions.json
    regions_path = os.path.join(vg_dir, "region_descriptions.json")
    if not os.path.exists(regions_path):
        print(f"❌ Region descriptions not found: {regions_path}")
        print("   Download from: https://homes.cs.washington.edu/~ranjay/visualgenome/api.html")
        return
    
    with open(regions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process
    samples = []
    images_dir = os.path.join(vg_dir, "VG_100K")
    images_dir_2 = os.path.join(vg_dir, "VG_100K_2")
    
    for item in tqdm(data[:num_samples], desc="Processing"):
        image_id = item.get('id')
        regions = item.get('regions', [])
        
        if not regions:
            continue
        
        # Find image
        image_path = os.path.join(images_dir, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(images_dir_2, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            continue
        
        # Use first region description as caption
        caption = regions[0].get('phrase', '')
        if not caption:
            continue
        
        samples.append({
            'image_path': image_path,
            'caption': caption,
            'question': 'Describe this image'
        })
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created {len(samples)} samples → {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Prepare Evaluation Dataset')
    parser.add_argument('--source', type=str, required=True, 
                        choices=['vizwiz', 'visual_genome', 'custom'],
                        help='Dataset source')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='eval_dataset.json', help='Output JSON path')
    parser.add_argument('--split', type=str, default='val', help='Dataset split (for VizWiz)')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples')
    
    args = parser.parse_args()
    
    if args.source == 'vizwiz':
        prepare_vizwiz(args.data_dir, args.split, args.output)
    elif args.source == 'visual_genome':
        prepare_visual_genome(args.data_dir, args.output, args.num_samples)
    elif args.source == 'custom':
        prepare_custom_folder(args.data_dir, args.output)

if __name__ == "__main__":
    main()
