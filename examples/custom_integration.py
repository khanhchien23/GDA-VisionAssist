"""
GDA-VisionAssist - Custom Integration Example
Ví dụ tích hợp GDA vào ứng dụng tùy chỉnh.
"""

import cv2
import numpy as np
import threading
import time

from src.core.gda import GlobalDescriptionAcquisition
from src.app.inference_manager import InferenceManager
from src.io import voice


class CustomGDAApp:
    """
    Ví dụ: Tích hợp GDA vào ứng dụng riêng.
    Xử lý ảnh từ thư mục thay vì webcam.
    """
    
    def __init__(self, device="cuda"):
        # Khởi tạo GDA core
        self.gda = GlobalDescriptionAcquisition(
            seg_checkpoint="checkpoints/setr_dino_best.pth",
            adaptor_checkpoint="checkpoints/adaptor_vizwiz/adaptor.pth",
            device=device
        )
        
        # Inference manager cho xử lý nền
        self.inference_manager = InferenceManager(self.gda, maxsize=3)
        self.inference_manager.start()
        
        # TTS (optional)
        self.tts_available = voice.init_tts()
        if self.tts_available:
            self.tts_thread = threading.Thread(
                target=voice.tts_worker, daemon=True
            )
            self.tts_thread.start()
    
    def process_image(self, image_path: str, click_point: tuple, 
                      question: str = None) -> dict:
        """
        Xử lý một ảnh tĩnh.
        
        Args:
            image_path: Đường dẫn ảnh
            click_point: Điểm click (x, y) để segment
            question: Câu hỏi (optional)
            
        Returns:
            Dict chứa kết quả mô tả
        """
        # Load ảnh
        image = cv2.imread(image_path)
        if image is None:
            return {"error": True, "description": f"Không đọc được: {image_path}"}
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Segment
        mask = self.gda.sam_segmenter.segment_from_point(
            image_rgb, point=click_point, use_iterative=False
        )
        
        if mask is None or mask.sum() == 0:
            return {"error": True, "description": "Không tìm thấy vật thể"}
        
        # Xử lý
        result = self.gda.process_region(image_rgb, mask, question)
        return result
    
    def batch_process(self, image_paths: list, click_points: list):
        """
        Xử lý hàng loạt nhiều ảnh.
        
        Args:
            image_paths: Danh sách đường dẫn ảnh
            click_points: Danh sách điểm click tương ứng
        """
        results = []
        
        for i, (path, point) in enumerate(zip(image_paths, click_points)):
            print(f"\n📷 [{i+1}/{len(image_paths)}] {path}")
            result = self.process_image(path, point)
            results.append({
                "image": path,
                "point": point,
                **result
            })
            
            # In kết quả
            if not result.get("error"):
                cls = result.get("predicted_class", "N/A")
                desc = result.get("description", "N/A")
                print(f"  🏷️  {cls}")
                print(f"  💬 {desc}")
            else:
                print(f"  ❌ {result.get('description')}")
        
        return results
    
    def predict_class_only(self, image_path: str, click_point: tuple):
        """
        Chỉ dự đoán class (nhanh hơn full pipeline).
        
        Args:
            image_path: Đường dẫn ảnh
            click_point: Điểm click
            
        Returns:
            Tuple (class_name, confidence)
        """
        image = cv2.imread(image_path)
        if image is None:
            return None, 0.0
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = self.gda.sam_segmenter.segment_from_point(
            image_rgb, point=click_point
        )
        
        if mask is None or mask.sum() == 0:
            return None, 0.0
        
        return self.gda.predict_class_from_region(
            image_rgb, mask, image_rgb.shape[:2]
        )
    
    def cleanup(self):
        """Dọn dẹp tài nguyên"""
        self.inference_manager.stop()
        if self.tts_available:
            voice.speech_queue.put(None)


def main():
    """Demo custom integration"""
    app = CustomGDAApp(device="cuda")
    
    try:
        # Xử lý đơn ảnh
        result = app.process_image(
            "test_image.jpg",
            click_point=(320, 240),
            question="Đây là gì?"
        )
        print(f"\n📊 Result: {result.get('description', 'N/A')}")
        
        # Chỉ dự đoán class
        cls, conf = app.predict_class_only("test_image.jpg", (320, 240))
        print(f"🏷️  Class: {cls} ({conf:.0%})")
        
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
