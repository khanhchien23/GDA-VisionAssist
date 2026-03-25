"""
SAM 2 Segmenter - Segment Anything Model 2
Sử dụng thư viện gốc sam2 từ Meta cho image segmentation.

Installation:
    pip install sam2

Models available (auto-downloaded from Hugging Face):
    - facebook/sam2-hiera-tiny     (~40MB, fastest)
    - facebook/sam2-hiera-small    (~80MB)
    - facebook/sam2-hiera-base-plus (~160MB)
    - facebook/sam2-hiera-large    (~320MB, best quality)
"""

import torch
import cv2
import numpy as np
from typing import Tuple, Optional, List

# Try to import SAM 2
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("⚠️ SAM 2 not installed. Run: pip install sam2")

# Fallback to SAM 1 if SAM 2 not available
try:
    from transformers import SamModel, SamProcessor
    SAM1_AVAILABLE = True
except ImportError:
    SAM1_AVAILABLE = False


class SAM2Segmenter:
    """
    SAM 2 Segmenter cho image segmentation chất lượng cao.
    
    SAM 2 improvements:
    - 6x faster than SAM 1
    - Better segmentation quality
    - Hiera hierarchical image encoder
    - Memory efficient
    
    Fallback: SAM 1 → GrabCut
    """
    
    # SAM 2 models
    SAM2_MODELS = {
        'tiny': 'facebook/sam2-hiera-tiny',
        'small': 'facebook/sam2-hiera-small',
        'base_plus': 'facebook/sam2-hiera-base-plus',
        'large': 'facebook/sam2-hiera-large',
    }
    
    # SAM 1 fallback models
    SAM1_MODELS = {
        'base': 'facebook/sam-vit-base',
        'large': 'facebook/sam-vit-large',
        'huge': 'facebook/sam-vit-huge',
    }
    
    def __init__(self, 
                 model_size: str = "large",
                 device: str = "cuda"):
        """
        Initialize SAM 2 Segmenter.
        
        Args:
            model_size: 'tiny', 'small', 'base_plus', 'large'
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.model_version = None
        self.predictor = None
        self.sam_model = None
        self.sam_processor = None
        
        # Try SAM 2 first
        if SAM2_AVAILABLE:
            self._load_sam2(model_size)
        elif SAM1_AVAILABLE:
            print("⚠️ SAM 2 not available, using SAM 1...")
            self._load_sam1()
        else:
            print("⚠️ No SAM available, using GrabCut fallback")
            self.model_version = "grabcut"
    
    def _load_sam2(self, model_size: str):
        """Load SAM 2 model."""
        model_name = self.SAM2_MODELS.get(model_size, self.SAM2_MODELS['large'])
        print(f"🎯 Đang tải SAM 2: {model_name}...")
        
        try:
            self.predictor = SAM2ImagePredictor.from_pretrained(model_name)
            
            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                self.predictor.model = self.predictor.model.cuda()
            
            print(f"✅ SAM 2 ready: {model_name}")
            self.model_version = "sam2"
            
        except Exception as e:
            print(f"⚠️ SAM 2 load error: {e}")
            if SAM1_AVAILABLE:
                print("   Falling back to SAM 1...")
                self._load_sam1()
            else:
                self.model_version = "grabcut"
    
    def _load_sam1(self):
        """Load SAM 1 as fallback."""
        model_name = self.SAM1_MODELS['huge']
        print(f"🎯 Đang tải SAM 1: {model_name}...")
        
        try:
            self.sam_model = SamModel.from_pretrained(model_name).to(self.device)
            self.sam_processor = SamProcessor.from_pretrained(model_name)
            self.sam_model.eval()
            
            print(f"✅ SAM 1 ready: {model_name}")
            self.model_version = "sam1"
            
        except Exception as e:
            print(f"⚠️ SAM 1 load error: {e}")
            self.model_version = "grabcut"
    
    @torch.inference_mode()
    def segment_from_point(self, 
                          image_rgb: np.ndarray, 
                          point: Tuple[int, int],
                          use_iterative: bool = False) -> np.ndarray:
        """
        Phân đoạn từ điểm click.
        
        Args:
            image_rgb: (H, W, 3) numpy array, RGB format
            point: (x, y) click coordinate
            use_iterative: Iterative refinement
            
        Returns:
            mask: (H, W) binary mask, uint8
        """
        if self.model_version == "sam2":
            return self._segment_sam2(image_rgb, point, use_iterative)
        elif self.model_version == "sam1":
            return self._segment_sam1(image_rgb, point, use_iterative)
        else:
            return self._grabcut_segment(image_rgb, point)
    
    def _segment_sam2(self, 
                     image_rgb: np.ndarray, 
                     point: Tuple[int, int],
                     use_iterative: bool = False) -> np.ndarray:
        """SAM 2 segmentation."""
        try:
            # Set image
            self.predictor.set_image(image_rgb)
            
            # Prepare point prompts
            point_coords = np.array([[point[0], point[1]]])
            point_labels = np.array([1])  # 1 = foreground
            
            # Predict mask
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )
            
            # Select best mask
            best_idx = scores.argmax()
            mask = masks[best_idx].astype(np.uint8)
            
            # Iterative refinement
            if use_iterative:
                mask = self._refine_sam2(image_rgb, point, mask, logits[best_idx])
            
            self._cleanup()
            return mask
            
        except Exception as e:
            print(f"⚠️ SAM 2 error: {e}")
            return self._grabcut_segment(image_rgb, point)
    
    def _refine_sam2(self, 
                    image_rgb: np.ndarray,
                    point: Tuple[int, int],
                    mask: np.ndarray,
                    mask_logits: np.ndarray) -> np.ndarray:
        """Iterative refinement for SAM 2."""
        # Find boundary points
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return mask
        
        largest_contour = max(contours, key=cv2.contourArea)
        if len(largest_contour) < 5:
            return mask
        
        # Sample boundary points
        indices = np.linspace(0, len(largest_contour)-1, 3, dtype=int)
        boundary_points = [
            [int(largest_contour[idx][0][0]), int(largest_contour[idx][0][1])]
            for idx in indices
        ]
        
        # Multi-point prediction
        all_points = [[point[0], point[1]]] + boundary_points
        point_coords = np.array(all_points)
        point_labels = np.array([1] * len(all_points))
        
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_logits[None, :, :],  # Use previous logits
            multimask_output=False
        )
        
        return masks[0].astype(np.uint8)
    
    def _segment_sam1(self, 
                     image_rgb: np.ndarray, 
                     point: Tuple[int, int],
                     use_iterative: bool = False) -> np.ndarray:
        """SAM 1 segmentation (fallback)."""
        try:
            inputs = self.sam_processor(
                image_rgb,
                input_points=[[[point[0], point[1]]]],
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.sam_model(**inputs)
            
            masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )[0]
            
            scores = outputs.iou_scores.cpu()[0, 0]
            best_idx = scores.argmax().item()
            mask = masks[0, best_idx].numpy().astype(np.uint8)
            
            self._cleanup()
            return mask
            
        except Exception as e:
            print(f"⚠️ SAM 1 error: {e}")
            return self._grabcut_segment(image_rgb, point)
    
    def _grabcut_segment(self, 
                        image_rgb: np.ndarray, 
                        point: Tuple[int, int]) -> np.ndarray:
        """GrabCut fallback."""
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        x, y = point
        rect_size = min(w, h) // 4
        x1 = max(0, x - rect_size // 2)
        y1 = max(0, y - rect_size // 2)
        x2 = min(w, x + rect_size // 2)
        y2 = min(h, y + rect_size // 2)
        rect = (x1, y1, x2 - x1, y2 - y1)
        
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        
        try:
            cv2.grabCut(image_rgb, mask, rect, bgd_model, fgd_model,
                       5, cv2.GC_INIT_WITH_RECT)
            mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        except:
            cv2.circle(mask, (x, y), rect_size // 2, 1, -1)
        
        return mask
    
    def _cleanup(self):
        """Clean GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "version": self.model_version,
            "device": self.device,
            "sam2_available": SAM2_AVAILABLE,
            "sam1_available": SAM1_AVAILABLE
        }
