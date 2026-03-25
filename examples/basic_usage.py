"""
GDA-VisionAssist - Basic Usage Example
Ví dụ cơ bản về cách sử dụng GDA-VisionAssist.
"""

import cv2
import numpy as np

from src.core.gda import GlobalDescriptionAcquisition


def main():
    """Ví dụ cơ bản: load ảnh → segment → mô tả"""
    
    # 1. Khởi tạo hệ thống GDA
    print("🚀 Đang khởi tạo GDA-VisionAssist...")
    gda = GlobalDescriptionAcquisition(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        seg_checkpoint="checkpoints/setr_dino_best.pth",
        adaptor_checkpoint="checkpoints/adaptor_vizwiz/adaptor.pth",
        device="cuda",
        debug=False
    )
    print("✅ GDA-VisionAssist đã sẵn sàng!\n")
    
    # 2. Load và xử lý ảnh
    image_path = "test_image.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Không thể đọc ảnh: {image_path}")
        print("💡 Hãy cung cấp file ảnh test_image.jpg")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"📷 Ảnh: {image.shape[1]}x{image.shape[0]}")
    
    # 3. Segment vật thể từ điểm click (giả lập click ở giữa ảnh)
    h, w = image.shape[:2]
    click_point = (w // 2, h // 2)
    print(f"🎯 Click point: {click_point}")
    
    mask = gda.sam_segmenter.segment_from_point(
        image_rgb, 
        point=click_point,
        use_iterative=False
    )
    
    if mask is None or mask.sum() == 0:
        print("⚠️ Không segment được vật thể tại điểm này")
        return
    
    area_percent = (mask.sum() / (h * w)) * 100
    print(f"✅ Mask: {mask.sum()} pixels ({area_percent:.1f}%)\n")
    
    # 4. Mô tả vật thể (không có câu hỏi → mô tả tổng quát)
    print("🔄 Đang mô tả vật thể...")
    result = gda.process_region(image_rgb, mask)
    
    print("\n" + "=" * 60)
    print("📊 KẾT QUẢ:")
    if result.get('predicted_class'):
        print(f"  🏷️  Class: {result['predicted_class']} ({result.get('confidence', 0):.0%})")
    print(f"  💬 Mô tả: {result.get('description', 'N/A')}")
    print("=" * 60)
    
    # 5. Hỏi câu hỏi cụ thể
    print("\n🔄 Đang hỏi câu hỏi...")
    result2 = gda.process_region(
        image_rgb, mask, 
        user_query="Vật này màu gì?"
    )
    
    print(f"  ❓ Câu hỏi: Vật này màu gì?")
    print(f"  💬 Trả lời: {result2.get('description', 'N/A')}")
    
    # 6. Hiển thị kết quả
    display = image.copy()
    mask_colored = np.zeros_like(display)
    mask_colored[mask > 0] = [0, 255, 0]  # Green overlay
    display = cv2.addWeighted(display, 0.7, mask_colored, 0.3, 0)
    
    cv2.imshow("GDA Result", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
