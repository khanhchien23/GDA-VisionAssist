"""
Tests for src/core module.
Tests GDA core logic, prompt construction, and inference utilities.
"""

import pytest
import numpy as np


class TestPromptConstructor:
    """Test PromptConstructor class."""
    
    def setup_method(self):
        from src.core.prompt import PromptConstructor
        self.pc = PromptConstructor()
    
    def test_detect_question_type_what_is(self):
        """Test nhận diện câu hỏi 'đây là gì'"""
        assert self.pc.detect_question_type("Đây là gì?") == "what_is"
        assert self.pc.detect_question_type("cái này là gì") == "what_is"
        assert self.pc.detect_question_type("what is this") == "what_is"
    
    def test_detect_question_type_color(self):
        """Test nhận diện câu hỏi về màu sắc"""
        assert self.pc.detect_question_type("Vật này màu gì?") == "color"
        assert self.pc.detect_question_type("color of this") == "color"
    
    def test_detect_question_type_describe(self):
        """Test nhận diện câu hỏi mô tả"""
        assert self.pc.detect_question_type("Mô tả vật này") == "describe"
        assert self.pc.detect_question_type("describe this") == "describe"
    
    def test_detect_question_type_general(self):
        """Test câu hỏi không xác định → general_describe"""
        assert self.pc.detect_question_type("xin chào") == "general_describe"
        assert self.pc.detect_question_type(None) == "general_describe"
    
    def test_construct_prompt_no_query(self):
        """Test tạo prompt không có câu hỏi"""
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:300, 200:400] = 1
        
        prompt = self.pc.construct_prompt(mask, predicted_class="laptop")
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "laptop" in prompt
    
    def test_construct_prompt_with_query(self):
        """Test tạo prompt có câu hỏi"""
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:300, 200:400] = 1
        
        prompt = self.pc.construct_prompt(
            mask, predicted_class="laptop", user_query="Màu gì?"
        )
        
        assert "Màu gì?" in prompt
    
    def test_construct_prompt_empty_mask(self):
        """Test prompt với mask trống"""
        mask = np.zeros((480, 640), dtype=np.uint8)
        prompt = self.pc.construct_prompt(mask)
        assert isinstance(prompt, str)
    
    def test_spatial_context_center(self):
        """Test spatial context cho vật ở giữa"""
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[180:300, 250:390] = 1  # Center region
        
        location, area_ratio = self.pc._get_spatial_context(mask)
        
        assert "giữa" in location or "trung tâm" in location
        assert 0.0 < area_ratio < 1.0
    
    def test_spatial_context_top_left(self):
        """Test spatial context cho vật ở góc trên trái"""
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[0:100, 0:100] = 1  # Top-left
        
        location, _ = self.pc._get_spatial_context(mask)
        
        assert "trên" in location
        assert "trái" in location


class TestInferenceUtils:
    """Test inference utility functions."""
    
    def test_prepare_image_for_model(self):
        """Test image preparation"""
        from src.core.inference import prepare_image_for_model
        
        # Small image → resize up
        small_img = np.zeros((30, 30, 3), dtype=np.uint8)
        result = prepare_image_for_model(small_img, min_size=56)
        assert min(result.size) >= 56
        
        # Normal image → unchanged
        normal_img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = prepare_image_for_model(normal_img)
        assert result.size == (640, 480)
    
    def test_crop_to_mask_region(self):
        """Test mask region cropping"""
        from src.core.inference import crop_to_mask_region
        
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:200, 100:200] = 1
        
        cropped, bbox = crop_to_mask_region(image, mask)
        
        assert len(bbox) == 4
        assert cropped.shape[0] > 0
        assert cropped.shape[1] > 0
    
    def test_crop_empty_mask(self):
        """Test cropping với mask trống"""
        from src.core.inference import crop_to_mask_region
        
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = np.zeros((480, 640), dtype=np.uint8)
        
        cropped, bbox = crop_to_mask_region(image, mask)
        assert cropped.shape == image.shape
    
    def test_inference_timer(self):
        """Test InferenceTimer context manager"""
        from src.core.inference import InferenceTimer
        import time
        
        with InferenceTimer("test", sync_cuda=False) as timer:
            time.sleep(0.01)
        
        assert timer.elapsed > 0
        assert "test" in str(timer)
