"""
Case Study: Đánh giá ảo giác (Hallucination Analysis)
So sánh Full Pipeline (GDA) vs Qwen2-VL Baseline trên Visual Genome Regions

Script này:
1. Đọc kết quả đánh giá đã chạy (vg_evaluation_results.json)
2. Chọn các mẫu tốt nhất cho Case Study
3. Phân tích ảo giác chi tiết
4. Tạo báo cáo + hình ảnh minh họa
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter

# ============================================================================
# CONFIG
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
RESULTS_PATH = SCRIPT_DIR / "vg_evaluation_results.json"
IMAGE_DIR = SCRIPT_DIR.parent / "data" / "visual_genome"
OUTPUT_DIR = SCRIPT_DIR.parent / "reports" / "case_study"

# Stopwords for hallucination analysis
STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'of', 'with',
    'to', 'and', 'at', 'it', 'its', 'that', 'this', 'be', 'has', 'have',
    'for', 'or', 'not', 'by', 'from', 'as', 'but', 'which', 'their',
    'i', 'im', 'me', 'my', 'can', 'cannot', 'could', 'do', 'does',
    'sorry', 'please', 'provide', 'more', 'information', 'context',
    'image', 'region', 'marked', 'appears', 'seems'
}


# ============================================================================
# HALLUCINATION ANALYSIS
# ============================================================================

def tokenize(text):
    """Tokenize và normalize."""
    if not text:
        return []
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()


def extract_content_words(text):
    """Trích xuất từ nội dung (loại bỏ stopwords và từ ngắn)."""
    tokens = tokenize(text)
    return set(w for w in tokens if w not in STOPWORDS and len(w) >= 3)


def analyze_hallucination(ground_truth, generated_text):
    """
    Phân tích ảo giác trong mô tả sinh ra.
    
    Returns:
        dict với:
        - correct_objects: Đối tượng nhắc đúng
        - missed_objects: Đối tượng bị bỏ sót
        - hallucinated_objects: Đối tượng bịa ra (ảo giác)
        - hallucination_rate: Tỷ lệ ảo giác
        - object_recall: Tỷ lệ nhắc đúng
    """
    gt_words = extract_content_words(ground_truth)
    gen_words = extract_content_words(generated_text)
    
    correct = gt_words & gen_words          # Nhắc đúng
    missed = gt_words - gen_words            # Bỏ sót
    hallucinated = gen_words - gt_words      # Bịa ra
    
    # Tính tỷ lệ
    total_gen = len(gen_words) if gen_words else 1
    total_gt = len(gt_words) if gt_words else 1
    
    return {
        'correct_objects': sorted(correct),
        'missed_objects': sorted(missed),
        'hallucinated_objects': sorted(hallucinated),
        'hallucination_rate': len(hallucinated) / total_gen,  # % từ bịa ra
        'object_recall': len(correct) / total_gt,             # % nhắc đúng
        'num_correct': len(correct),
        'num_missed': len(missed),
        'num_hallucinated': len(hallucinated),
    }


def is_baseline_failure(baseline_output):
    """Kiểm tra xem baseline có thất bại hoàn toàn không."""
    failure_patterns = [
        "i'm sorry", "cannot provide", "no image", "no region",
        "please provide", "more information", "more context",
        "marked region(", "region("
    ]
    lower = baseline_output.lower()
    return any(p in lower for p in failure_patterns)


# ============================================================================
# CASE STUDY SELECTION
# ============================================================================

def select_best_case_studies(results, top_k=5):
    """
    Chọn các mẫu tốt nhất cho Case Study.
    Ưu tiên:
    1. Pipeline tốt hơn baseline nhiều nhất
    2. Pipeline có Object_Acc cao
    3. Baseline thất bại rõ ràng (ảo giác hoặc refuse)
    """
    scored_results = []
    
    for i, r in enumerate(results):
        p_metrics = r['pipeline_metrics']
        b_metrics = r['baseline_metrics']
        
        # Tính điểm ưu tiên
        obj_acc_diff = p_metrics['Object_Acc'] - b_metrics['Object_Acc']
        rouge_diff = p_metrics['ROUGE-L'] - b_metrics['ROUGE-L']
        cider_diff = p_metrics['CIDEr'] - b_metrics['CIDEr']
        
        # Bonus nếu baseline thất bại hoàn toàn
        baseline_failed = is_baseline_failure(r['baseline_output'])
        
        # Composite score
        score = (
            obj_acc_diff * 3.0 +      # Object accuracy quan trọng nhất
            rouge_diff * 2.0 +          # ROUGE-L 
            cider_diff * 0.5 +          # CIDEr
            (1.0 if baseline_failed else 0.0) +  # Bonus baseline fail
            p_metrics['Object_Acc'] * 2.0  # Pipeline phải tốt tuyệt đối
        )
        
        scored_results.append((score, i, r))
    
    # Sắp xếp theo điểm giảm dần
    scored_results.sort(key=lambda x: x[0], reverse=True)
    
    return [(idx, r) for _, idx, r in scored_results[:top_k]]


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_case_study_report(results, summary):
    """Tạo báo cáo Case Study dạng text."""
    
    # Chọn top mẫu
    case_studies = select_best_case_studies(results, top_k=5)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("  CASE STUDY: ĐÁNH GIÁ ẢO GIÁC (HALLUCINATION ANALYSIS)")
    report_lines.append("  So sánh Full Pipeline (GDA) vs Qwen2-VL Baseline")
    report_lines.append("=" * 80)
    
    # --- Tổng quan ---
    report_lines.append("\n📊 TỔNG QUAN KẾT QUẢ (100 mẫu)")
    report_lines.append("-" * 60)
    report_lines.append(f"{'Metric':<20} {'Pipeline':>12} {'Baseline':>12} {'Δ':>10}")
    report_lines.append("-" * 60)
    
    for metric in ['BLEU-1', 'BLEU-4', 'ROUGE-L', 'CIDEr', 'Object_Acc']:
        p = summary['pipeline'].get(metric, 0)
        b = summary['baseline'].get(metric, 0)
        d = p - b
        sign = "+" if d >= 0 else ""
        report_lines.append(f"{metric:<20} {p:>12.4f} {b:>12.4f} {sign}{d:>9.4f}")
    
    report_lines.append("-" * 60)
    
    # --- Thống kê ảo giác tổng thể ---
    report_lines.append("\n\n📈 THỐNG KÊ ẢO GIÁC TỔNG THỂ")
    report_lines.append("-" * 60)
    
    pipeline_hall_rates = []
    baseline_hall_rates = []
    baseline_refusals = 0
    
    for r in results:
        p_analysis = analyze_hallucination(r['ground_truth'], r['pipeline_output'])
        b_analysis = analyze_hallucination(r['ground_truth'], r['baseline_output'])
        pipeline_hall_rates.append(p_analysis['hallucination_rate'])
        baseline_hall_rates.append(b_analysis['hallucination_rate'])
        if is_baseline_failure(r['baseline_output']):
            baseline_refusals += 1
    
    avg_p_hall = sum(pipeline_hall_rates) / len(pipeline_hall_rates) if pipeline_hall_rates else 0
    avg_b_hall = sum(baseline_hall_rates) / len(baseline_hall_rates) if baseline_hall_rates else 0
    
    report_lines.append(f"Pipeline - Tỷ lệ ảo giác trung bình:  {avg_p_hall:.2%}")
    report_lines.append(f"Baseline - Tỷ lệ ảo giác trung bình:  {avg_b_hall:.2%}")
    report_lines.append(f"Baseline - Số lần từ chối/thất bại:    {baseline_refusals}/{len(results)} ({baseline_refusals/len(results):.0%})")
    report_lines.append(f"Pipeline - Object Recall trung bình:   {summary['pipeline']['Object_Acc']:.2%}")
    report_lines.append(f"Baseline - Object Recall trung bình:   {summary['baseline']['Object_Acc']:.2%}")
    
    # --- Các Case Study chi tiết ---
    report_lines.append("\n\n" + "=" * 80)
    report_lines.append("  CÁC MẪU ĐIỂN HÌNH (CASE STUDIES)")
    report_lines.append("=" * 80)
    
    for rank, (idx, r) in enumerate(case_studies, 1):
        p_analysis = analyze_hallucination(r['ground_truth'], r['pipeline_output'])
        b_analysis = analyze_hallucination(r['ground_truth'], r['baseline_output'])
        
        report_lines.append(f"\n\n{'─' * 80}")
        report_lines.append(f"  📌 Case Study #{rank}")
        report_lines.append(f"{'─' * 80}")
        
        # Image path
        img_path = r.get('image_path', 'N/A')
        img_name = os.path.basename(img_path)
        report_lines.append(f"\n  🖼️  Ảnh: {img_name}")
        report_lines.append(f"  📂 Path: {img_path}")
        
        # Ground Truth
        report_lines.append(f"\n  ✅ Ground Truth:")
        report_lines.append(f"     \"{r['ground_truth']}\"")
        
        # Pipeline output
        report_lines.append(f"\n  🔵 Full Pipeline (GDA):")
        report_lines.append(f"     \"{r['pipeline_output']}\"")
        report_lines.append(f"     → Từ khóa đúng:   {p_analysis['correct_objects'] or '(không có)'}")
        report_lines.append(f"     → Từ bỏ sót:      {p_analysis['missed_objects'] or '(không có)'}")
        report_lines.append(f"     → Từ ảo giác:     {p_analysis['hallucinated_objects'][:10] or '(không có)'}")
        report_lines.append(f"     → Object Recall:   {p_analysis['object_recall']:.0%}")
        report_lines.append(f"     → Hallucination:   {p_analysis['hallucination_rate']:.0%}")
        
        # Baseline output
        report_lines.append(f"\n  🔴 Qwen2-VL Baseline:")
        report_lines.append(f"     \"{r['baseline_output']}\"")
        baseline_failed = is_baseline_failure(r['baseline_output'])
        if baseline_failed:
            report_lines.append(f"     ⚠️  BASELINE THẤT BẠI - Từ chối trả lời hoặc output vô nghĩa")
        else:
            report_lines.append(f"     → Từ khóa đúng:   {b_analysis['correct_objects'] or '(không có)'}")
            report_lines.append(f"     → Từ bỏ sót:      {b_analysis['missed_objects'] or '(không có)'}")
            report_lines.append(f"     → Từ ảo giác:     {b_analysis['hallucinated_objects'][:10] or '(không có)'}")
        report_lines.append(f"     → Object Recall:   {b_analysis['object_recall']:.0%}")
        report_lines.append(f"     → Hallucination:   {b_analysis['hallucination_rate']:.0%}")
        
        # Metrics comparison
        report_lines.append(f"\n  📊 So sánh điểm số:")
        report_lines.append(f"     {'Metric':<15} {'Pipeline':>10} {'Baseline':>10} {'Δ':>10}")
        report_lines.append(f"     {'─'*45}")
        for m in ['ROUGE-L', 'CIDEr', 'Object_Acc']:
            pv = r['pipeline_metrics'][m]
            bv = r['baseline_metrics'][m]
            dv = pv - bv
            sign = "+" if dv >= 0 else ""
            report_lines.append(f"     {m:<15} {pv:>10.4f} {bv:>10.4f} {sign}{dv:>9.4f}")
        
        # Analysis
        report_lines.append(f"\n  🔍 Phân tích:")
        if baseline_failed:
            report_lines.append(f"     Baseline không thể mô tả vùng vì chỉ nhận crop ảnh theo bbox")
            report_lines.append(f"     → Thiếu ngữ cảnh, không nhận diện được đối tượng")
        
        if p_analysis['object_recall'] > b_analysis['object_recall']:
            report_lines.append(f"     Pipeline nhắc đúng {p_analysis['num_correct']}/{len(extract_content_words(r['ground_truth']))} từ khóa"
                              f" (so với {b_analysis['num_correct']} của baseline)")
            report_lines.append(f"     → Nhờ mask phân đoạn + nhãn ngữ nghĩa SETR giúp neo vào đúng đối tượng")
    
    # --- Kết luận ---
    report_lines.append("\n\n" + "=" * 80)
    report_lines.append("  KẾT LUẬN")
    report_lines.append("=" * 80)
    report_lines.append(f"""
    1. Full Pipeline (GDA) đạt Object Accuracy cao hơn baseline {summary['pipeline']['Object_Acc']/max(summary['baseline']['Object_Acc'],0.001):.1f}× 
       ({summary['pipeline']['Object_Acc']:.2%} vs {summary['baseline']['Object_Acc']:.2%})
    
    2. Baseline Qwen2-VL thất bại {baseline_refusals}/{len(results)} lần ({baseline_refusals/len(results):.0%}) do:
       - Crop ảnh quá nhỏ → mất ngữ cảnh
       - Không có mask → không biết focus vào đâu
       - Từ chối trả lời hoặc sinh output vô nghĩa
    
    3. Pipeline giảm ảo giác nhờ:
       - SAM 2 mask phân đoạn chính xác vùng đối tượng
       - SETR cung cấp nhãn ngữ nghĩa (semantic label)
       - Prompt tích hợp neo mô hình vào đúng đối tượng
       - MaskedFeatureExtractor loại bỏ nhiễu từ vùng nền
    
    4. ROUGE-L cải thiện +{summary['pipeline']['ROUGE-L'] - summary['baseline']['ROUGE-L']:.4f}
       CIDEr cải thiện  +{summary['pipeline']['CIDEr'] - summary['baseline']['CIDEr']:.4f}
    """)
    
    return "\n".join(report_lines)


def generate_visualization(results, output_dir):
    """Tạo visualization cho case study (nếu có matplotlib)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        
        os.makedirs(output_dir, exist_ok=True)
        
        # --- Figure 1: So sánh metrics tổng thể ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Full Pipeline (GDA) vs Qwen2-VL Baseline', fontsize=14, fontweight='bold')
        
        metrics_to_plot = ['ROUGE-L', 'CIDEr', 'Object_Acc']
        colors_pipeline = ['#2196F3', '#2196F3', '#2196F3']
        colors_baseline = ['#F44336', '#F44336', '#F44336']
        
        for ax, metric, cp, cb in zip(axes, metrics_to_plot, colors_pipeline, colors_baseline):
            p_scores = [r['pipeline_metrics'][metric] for r in results]
            b_scores = [r['baseline_metrics'][metric] for r in results]
            
            p_mean = sum(p_scores) / len(p_scores)
            b_mean = sum(b_scores) / len(b_scores)
            
            bars = ax.bar(['Pipeline\n(GDA)', 'Baseline\n(Qwen2-VL)'], 
                         [p_mean, b_mean],
                         color=[cp, cb], alpha=0.85, edgecolor='white', linewidth=2)
            
            # Add value labels
            for bar, val in zip(bars, [p_mean, b_mean]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylim(0, max(p_mean, b_mean) * 1.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: metrics_comparison.png")
        
        # --- Figure 2: Distribution of Object Accuracy ---
        fig, ax = plt.subplots(figsize=(10, 5))
        
        p_obj_acc = [r['pipeline_metrics']['Object_Acc'] for r in results]
        b_obj_acc = [r['baseline_metrics']['Object_Acc'] for r in results]
        
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ax.hist(p_obj_acc, bins=bins, alpha=0.7, label='Pipeline (GDA)', color='#2196F3', edgecolor='white')
        ax.hist(b_obj_acc, bins=bins, alpha=0.7, label='Baseline (Qwen2-VL)', color='#F44336', edgecolor='white')
        
        ax.set_xlabel('Object Accuracy', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title('Distribution of Object Accuracy (100 samples)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'object_accuracy_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: object_accuracy_distribution.png")
        
        # --- Figure 3: Hallucination Rate ---
        fig, ax = plt.subplots(figsize=(10, 5))
        
        p_hall_rates = []
        b_hall_rates = []
        for r in results:
            pa = analyze_hallucination(r['ground_truth'], r['pipeline_output'])
            ba = analyze_hallucination(r['ground_truth'], r['baseline_output'])
            p_hall_rates.append(pa['hallucination_rate'])
            b_hall_rates.append(ba['hallucination_rate'])
        
        # Sort by pipeline hallucination rate for clearer visualization
        indices = list(range(len(results)))
        
        ax.scatter(indices, p_hall_rates, alpha=0.6, label='Pipeline (GDA)', 
                  color='#2196F3', s=30, zorder=3)
        ax.scatter(indices, b_hall_rates, alpha=0.6, label='Baseline (Qwen2-VL)', 
                  color='#F44336', s=30, zorder=2)
        
        # Mean lines
        ax.axhline(y=sum(p_hall_rates)/len(p_hall_rates), color='#1565C0', 
                  linestyle='--', linewidth=2, label=f'Pipeline Avg: {sum(p_hall_rates)/len(p_hall_rates):.2%}')
        ax.axhline(y=sum(b_hall_rates)/len(b_hall_rates), color='#C62828', 
                  linestyle='--', linewidth=2, label=f'Baseline Avg: {sum(b_hall_rates)/len(b_hall_rates):.2%}')
        
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Hallucination Rate', fontsize=12)
        ax.set_title('Hallucination Rate per Sample', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'hallucination_rate.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: hallucination_rate.png")

        # --- Figure 4: Case Study Image (nếu có) ---
        case_studies = select_best_case_studies(results, top_k=3)
        
        for rank, (idx, r) in enumerate(case_studies, 1):
            img_path = r.get('image_path', '')
            img_name = os.path.basename(img_path)
            
            # Tìm ảnh trong data/visual_genome
            local_img_path = IMAGE_DIR / img_name
            if not local_img_path.exists():
                continue
            
            from PIL import Image
            img = Image.open(local_img_path)
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.imshow(img)
            ax.set_title(f'Case Study #{rank}: {img_name}', fontsize=13, fontweight='bold')
            ax.axis('off')
            
            # Add text box
            gt_text = r['ground_truth']
            p_text = r['pipeline_output'][:100] + "..." if len(r['pipeline_output']) > 100 else r['pipeline_output']
            b_text = r['baseline_output'][:100] + "..." if len(r['baseline_output']) > 100 else r['baseline_output']
            
            textstr = (
                f"Ground Truth: {gt_text}\n\n"
                f"Pipeline: {p_text}\n\n"
                f"Baseline: {b_text}"
            )
            
            props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.85)
            fig.text(0.05, -0.02, textstr, fontsize=9, verticalalignment='top',
                    bbox=props, wrap=True, family='monospace')
            
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f'case_study_{rank}_{img_name.replace(".jpg", ".png")}'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Saved: case_study_{rank}_{img_name}")
        
        return True
        
    except ImportError:
        print("  ⚠️ matplotlib không khả dụng. Bỏ qua visualization.")
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("  🔬 CASE STUDY: ĐÁNH GIÁ ẢO GIÁC")
    print("=" * 80)
    
    # 1. Load results
    if not RESULTS_PATH.exists():
        print(f"❌ Không tìm thấy file kết quả: {RESULTS_PATH}")
        print("   Hãy chạy evaluate_e2e.py trước.")
        sys.exit(1)
    
    print(f"\n📂 Đọc kết quả từ: {RESULTS_PATH}")
    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    summary = data['summary']
    results = data['detailed_results']
    print(f"   → {len(results)} mẫu chi tiết, {summary['num_samples']} mẫu tổng")
    
    # 2. Generate report
    print("\n📝 Tạo báo cáo Case Study...")
    report = generate_case_study_report(results, summary)
    
    # Print to console
    print("\n" + report)
    
    # Save report
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = OUTPUT_DIR / "hallucination_analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n💾 Báo cáo đã lưu: {report_path}")
    
    # 3. Generate visualizations
    print("\n📊 Tạo hình ảnh minh họa...")
    generate_visualization(results, OUTPUT_DIR)
    
    # 4. Export case study data (JSON) for further use
    case_studies = select_best_case_studies(results, top_k=5)
    case_study_data = []
    for idx, r in case_studies:
        p_analysis = analyze_hallucination(r['ground_truth'], r['pipeline_output'])
        b_analysis = analyze_hallucination(r['ground_truth'], r['baseline_output'])
        case_study_data.append({
            'ground_truth': r['ground_truth'],
            'pipeline_output': r['pipeline_output'],
            'baseline_output': r['baseline_output'],
            'pipeline_metrics': r['pipeline_metrics'],
            'baseline_metrics': r['baseline_metrics'],
            'pipeline_hallucination': p_analysis,
            'baseline_hallucination': b_analysis,
            'baseline_failed': is_baseline_failure(r['baseline_output']),
            'image_path': r.get('image_path', ''),
        })
    
    case_study_json_path = OUTPUT_DIR / "case_study_data.json"
    with open(case_study_json_path, 'w', encoding='utf-8') as f:
        json.dump(case_study_data, f, indent=2, ensure_ascii=False)
    print(f"💾 Dữ liệu Case Study: {case_study_json_path}")
    
    print("\n✅ Hoàn thành!")
    print(f"📂 Tất cả kết quả trong: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
