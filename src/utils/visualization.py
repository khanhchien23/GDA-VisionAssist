"""
GDA-VisionAssist - Visualization Utilities
Các hàm tiện ích cho visualization kết quả.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


# Bảng màu cho 172 classes COCO-Stuff
def generate_colormap(n_classes: int = 172) -> np.ndarray:
    """
    Tạo bảng màu cho segmentation map.
    
    Args:
        n_classes: Số lượng classes
        
    Returns:
        (n_classes, 3) numpy array BGR
    """
    colormap = np.zeros((n_classes, 3), dtype=np.uint8)
    
    for i in range(n_classes):
        r, g, b = 0, 0, 0
        idx = i
        for j in range(8):
            r |= ((idx >> 0) & 1) << (7 - j)
            g |= ((idx >> 1) & 1) << (7 - j)
            b |= ((idx >> 2) & 1) << (7 - j)
            idx >>= 3
        colormap[i] = [b, g, r]  # BGR cho OpenCV
    
    return colormap


# Global colormap
COLORMAP = generate_colormap()


def draw_segmentation_map(seg_map: np.ndarray, 
                           alpha: float = 0.5,
                           background: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Vẽ segmentation map với màu sắc.
    
    Args:
        seg_map: (H, W) class predictions
        alpha: Độ trong suốt nếu có background
        background: (H, W, 3) ảnh nền (optional)
        
    Returns:
        (H, W, 3) ảnh segmentation BGR
    """
    h, w = seg_map.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cls_id in np.unique(seg_map):
        if cls_id < len(COLORMAP):
            colored[seg_map == cls_id] = COLORMAP[cls_id]
    
    if background is not None:
        if background.shape[:2] != (h, w):
            background = cv2.resize(background, (w, h))
        colored = cv2.addWeighted(background, 1 - alpha, colored, alpha, 0)
    
    return colored


def overlay_class_labels(image: np.ndarray, 
                          seg_map: np.ndarray,
                          class_names: List[str],
                          min_area_ratio: float = 0.01,
                          font_scale: float = 0.5) -> np.ndarray:
    """
    Overlay tên class lên ảnh segmentation.
    
    Args:
        image: (H, W, 3) ảnh BGR
        seg_map: (H, W) class predictions
        class_names: Danh sách tên class
        min_area_ratio: Tỷ lệ diện tích tối thiểu để hiển thị label
        font_scale: Kích thước font
        
    Returns:
        Ảnh đã overlay labels
    """
    result = image.copy()
    h, w = seg_map.shape
    total_pixels = h * w
    
    for cls_id in np.unique(seg_map):
        if cls_id == 0 or cls_id >= len(class_names):
            continue
        
        cls_mask = (seg_map == cls_id)
        area_ratio = cls_mask.sum() / total_pixels
        
        if area_ratio < min_area_ratio:
            continue
        
        # Tìm centroid
        y_coords, x_coords = np.where(cls_mask)
        cx = int(x_coords.mean())
        cy = int(y_coords.mean())
        
        # Vẽ label
        label = class_names[cls_id]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        
        # Background cho text
        cv2.rectangle(
            result,
            (cx - tw // 2 - 2, cy - th - 4),
            (cx + tw // 2 + 2, cy + 4),
            (0, 0, 0), -1
        )
        
        cv2.putText(
            result, label,
            (cx - tw // 2, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (255, 255, 255), 1, cv2.LINE_AA
        )
    
    return result


def draw_point_marker(image: np.ndarray, 
                       point: Tuple[int, int],
                       color: Tuple[int, int, int] = (0, 0, 255),
                       radius: int = 5,
                       thickness: int = 2) -> np.ndarray:
    """
    Vẽ marker tại điểm click.
    
    Args:
        image: Ảnh BGR
        point: (x, y) 
        color: Màu (BGR)
        radius: Bán kính
        thickness: Độ dày viền
        
    Returns:
        Ảnh đã vẽ marker
    """
    result = image.copy()
    cv2.circle(result, point, radius, color, thickness)
    cv2.circle(result, point, radius + 3, (255, 255, 255), 1)
    
    # Crosshair
    size = radius + 8
    cv2.line(result, (point[0] - size, point[1]), (point[0] + size, point[1]), color, 1)
    cv2.line(result, (point[0], point[1] - size), (point[0], point[1] + size), color, 1)
    
    return result


def create_comparison_view(original: np.ndarray, 
                            segmented: np.ndarray,
                            result_text: str = "") -> np.ndarray:
    """
    Tạo ảnh so sánh original vs segmented.
    
    Args:
        original: Ảnh gốc BGR
        segmented: Ảnh segmentation BGR
        result_text: Text kết quả (optional)
        
    Returns:
        Ảnh ghép
    """
    h, w = original.shape[:2]
    
    # Resize nếu cần
    if segmented.shape[:2] != (h, w):
        segmented = cv2.resize(segmented, (w, h))
    
    # Ghép ngang
    comparison = np.hstack([original, segmented])
    
    # Thêm text
    if result_text:
        padding = 40
        text_area = np.zeros((padding, comparison.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            text_area, result_text,
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
        comparison = np.vstack([comparison, text_area])
    
    return comparison
