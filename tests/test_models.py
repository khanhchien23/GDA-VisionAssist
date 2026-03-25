"""
Tests for src/models module.
Tests model architectures, dimensions, and forward passes.
"""

import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    """Return available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class TestSETRSegDecoder:
    """Test SETR Segmentation Decoder."""
    
    def test_naive_decoder_init(self, device):
        """Test SETR-Naive initialization"""
        from src.models.segmentation import SETRSegDecoder
        
        decoder = SETRSegDecoder(
            vit_features_dim=768,
            num_classes=172,
            decoder_type='naive',
            device=device
        )
        assert decoder is not None
    
    def test_pup_decoder_init(self, device):
        """Test SETR-PUP initialization"""
        from src.models.segmentation import SETRSegDecoder
        
        decoder = SETRSegDecoder(
            vit_features_dim=768,
            num_classes=172,
            decoder_type='pup',
            device=device
        )
        assert decoder is not None
    
    def test_forward_shape(self, device):
        """Test forward pass output shape"""
        from src.models.segmentation import SETRSegDecoder
        
        decoder = SETRSegDecoder(
            vit_features_dim=768,
            num_classes=172,
            decoder_type='naive',
            device=device
        ).to(device)
        
        # Simulate DINOv2 features: (B, N_patches, dim)
        # 16x16 = 256 patches
        features = torch.randn(1, 256, 768, device=device)
        
        output = decoder(features, target_size=(480, 640))
        
        assert output.shape[0] == 1       # batch
        assert output.shape[1] == 172     # classes
        assert output.shape[2] == 480     # height
        assert output.shape[3] == 640     # width
    
    def test_forward_dim_mismatch(self, device):
        """Test decoder handles dimension mismatch"""
        from src.models.segmentation import SETRSegDecoder
        
        decoder = SETRSegDecoder(
            vit_features_dim=768,
            num_classes=172,
            decoder_type='naive',
            device=device
        ).to(device)
        
        # Wrong dim (1024 instead of 768) → should auto-create dynamic_proj
        features = torch.randn(1, 256, 1024, device=device)
        
        output = decoder(features, target_size=(224, 224))
        assert output.shape[1] == 172


class TestImprovedVisionLanguageAdaptor:
    """Test Vision-Language Adaptor."""
    
    def test_init(self, device):
        """Test adaptor initialization"""
        from src.models.adaptor import ImprovedVisionLanguageAdaptor
        
        adaptor = ImprovedVisionLanguageAdaptor(
            vision_dim=1536,
            llm_dim=1536,
            num_query_tokens=64
        )
        assert adaptor.query_tokens.shape == (1, 64, 1536)
    
    def test_forward_shape(self, device):
        """Test forward pass output shape"""
        from src.models.adaptor import ImprovedVisionLanguageAdaptor
        
        adaptor = ImprovedVisionLanguageAdaptor(
            vision_dim=1536,
            llm_dim=1536,
            num_query_tokens=64
        ).to(device)
        
        features = torch.randn(1, 100, 1536, device=device)
        output = adaptor(features)
        
        assert output.shape == (1, 64, 1536)
    
    def test_forward_dynamic_proj(self, device):
        """Test adaptor creates dynamic projection for mismatched dims"""
        from src.models.adaptor import ImprovedVisionLanguageAdaptor
        
        adaptor = ImprovedVisionLanguageAdaptor(
            vision_dim=1536,
            llm_dim=1536,
            num_query_tokens=64
        ).to(device)
        
        # Wrong dim → should create dynamic_proj
        features = torch.randn(1, 100, 768, device=device)
        output = adaptor(features)
        
        assert output.shape == (1, 64, 1536)
    
    def test_forward_float16_input(self, device):
        """Test adaptor handles float16 input"""
        if device != "cuda":
            pytest.skip("float16 test requires CUDA")
        
        from src.models.adaptor import ImprovedVisionLanguageAdaptor
        
        adaptor = ImprovedVisionLanguageAdaptor(
            vision_dim=1536, llm_dim=1536
        ).to(device)
        
        features = torch.randn(1, 100, 1536, device=device, dtype=torch.float16)
        output = adaptor(features)
        
        assert output.shape[0] == 1
        assert output.shape[2] == 1536


class TestMaskedFeatureExtractor:
    """Test MaskedFeatureExtractor."""
    
    def test_init(self, device):
        """Test initialization"""
        from src.models.vit_encoder import MaskedFeatureExtractor
        
        extractor = MaskedFeatureExtractor(feature_dim=1536)
        assert extractor is not None


class TestConstants:
    """Test constants module."""
    
    def test_coco_stuff_classes(self):
        """Test COCO-Stuff class list"""
        from src.constants import COCO_STUFF_CLASSES
        
        assert isinstance(COCO_STUFF_CLASSES, list)
        assert len(COCO_STUFF_CLASSES) > 0
        # Should have Vietnamese class names
        assert "người" in COCO_STUFF_CLASSES
        assert "ô tô" in COCO_STUFF_CLASSES
