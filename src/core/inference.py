"""
GDA-VisionAssist - Inference Utilities
Các hàm tiện ích cho inference pipeline.
"""

import torch
import numpy as np
import time
from typing import Optional, Tuple, Dict
from PIL import Image


def prepare_image_for_model(image_rgb: np.ndarray, 
                             min_size: int = 56,
                             max_size: int = 1024) -> Image.Image:
    """
    Chuẩn bị ảnh cho model inference.
    
    Args:
        image_rgb: (H, W, 3) numpy array RGB
        min_size: Kích thước tối thiểu
        max_size: Kích thước tối đa
        
    Returns:
        PIL Image đã resize
    """
    pil_image = Image.fromarray(image_rgb)
    w, h = pil_image.size
    
    # Đảm bảo kích thước tối thiểu
    if w < min_size or h < min_size:
        scale = min_size / min(w, h)
        new_w = max(int(w * scale), min_size)
        new_h = max(int(h * scale), min_size)
        pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
    
    # Giới hạn kích thước tối đa
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
    
    return pil_image


def crop_to_mask_region(image_rgb: np.ndarray, 
                         mask: np.ndarray,
                         margin_ratio: float = 0.5,
                         min_margin: int = 60) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Cắt ảnh theo vùng mask với margin.
    
    Args:
        image_rgb: (H, W, 3) numpy array
        mask: (H, W) binary mask
        margin_ratio: Tỷ lệ margin so với kích thước object
        min_margin: Margin tối thiểu (pixels)
        
    Returns:
        (cropped_image, (x_min, y_min, x_max, y_max))
    """
    y_coords, x_coords = np.where(mask > 0)
    
    if len(y_coords) == 0:
        return image_rgb, (0, 0, image_rgb.shape[1], image_rgb.shape[0])
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    h_img, w_img = image_rgb.shape[:2]
    
    # Dynamic margin
    obj_size = max(x_max - x_min, y_max - y_min)
    margin = max(int(obj_size * margin_ratio), min_margin)
    
    x_min_crop = max(0, x_min - margin)
    x_max_crop = min(w_img, x_max + margin)
    y_min_crop = max(0, y_min - margin)
    y_max_crop = min(h_img, y_max + margin)
    
    cropped = image_rgb[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
    bbox = (x_min_crop, y_min_crop, x_max_crop, y_max_crop)
    
    return cropped, bbox


def apply_mask_overlay(image_rgb: np.ndarray, 
                        mask: np.ndarray,
                        color: Tuple[int, int, int] = (0, 255, 0),
                        alpha: float = 0.3,
                        border_color: Tuple[int, int, int] = (255, 0, 0),
                        border_width: int = 2) -> np.ndarray:
    """
    Áp dụng mask overlay lên ảnh (highlight vùng chọn).
    
    Args:
        image_rgb: (H, W, 3) numpy array RGB
        mask: (H, W) binary mask
        color: Màu overlay (RGB)
        alpha: Độ trong suốt overlay
        border_color: Màu viền
        border_width: Độ dày viền
        
    Returns:
        Ảnh đã overlay
    """
    import cv2
    
    result = image_rgb.copy()
    
    # Color overlay
    overlay = np.zeros_like(result)
    overlay[mask > 0] = color
    result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
    
    # Border
    if border_width > 0:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, border_color, border_width)
    
    return result


class InferenceTimer:
    """Context manager để đo thời gian inference."""
    
    def __init__(self, name: str = "", sync_cuda: bool = True):
        self.name = name
        self.sync_cuda = sync_cuda
        self.elapsed = 0.0
        
    def __enter__(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start
        
    def __str__(self):
        return f"{self.name}: {self.elapsed*1000:.1f}ms"


def safe_cuda_cleanup():
    """Dọn dẹp CUDA memory an toàn."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
