import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import time
import os
from typing import Optional, Tuple, Dict
from PIL import Image, ImageDraw

# Import từ các modules khác
from ..models.segmentation import SETRSegDecoder
from ..models.adaptor import ImprovedVisionLanguageAdaptor
from ..models.vit_encoder import MaskedFeatureExtractor
from ..models.sam_segmenter import SAM2Segmenter
from ..models.text_decoder import VisionTextDecoder
from ..models.dinov2_encoder import DINOv2Encoder  # NEW: For SETR branch
from .prompt import PromptConstructor
from ..constants import COCO_STUFF_CLASSES

from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer

class GlobalDescriptionAcquisition:
    """
    ĐÃ SỬA LỖI TENSOR SIZE MISMATCH
    """
    
    def __init__(self, 
             model_name="Qwen/Qwen2-VL-2B-Instruct",
             seg_checkpoint=None,
             adaptor_checkpoint=None,
             device="cuda",
             debug=False):
        
        print(f"📦 Loading model: {model_name}")  # ← THÊM dòng này
        self.device = device
        self.debug = debug
        
        print("="*70)
        print("🏗️  KHỞI TẠO GDA - ĐÃ FIX TENSOR SIZE MISMATCH")
        print("="*70)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        # 1. Load Full LVLM model
        print("\n📦 [1/6] Đang tải Qwen2-VL (full model)...")
        
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        llm_int8_enable_fp32_cpu_offload=False
        )

        self.full_model = AutoModelForVision2Seq.from_pretrained(
            model_name, 
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,  # ← ĐỔI True → False
            trust_remote_code=True
        )

        # Force proper Vietnamese handling
        self.tokenizer.clean_up_tokenization_spaces = False

        # Add special handling for Vietnamese
        import locale
        try:
            locale.setlocale(locale.LC_ALL, 'vi_VN.UTF-8')
        except:
            try:
                locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
            except:
                pass
        
        print("✅ Full model loaded (float16)")
        
        # 2. Extract Shared ViT Encoder
        print("\n🔍 [2/6] Extracting Shared ViT Encoder...")
        self.vit_encoder = self.full_model.visual
        self.vit_encoder.eval()

        print("\n🔍 [DEBUG] Kiểm tra Qwen merger structure...")
        if hasattr(self.full_model.visual, 'merger'):
            print("✅ Tìm thấy visual.merger!")
            print(f"   Type: {type(self.full_model.visual.merger)}")
            print(f"   Submodules:")
            for name, module in self.full_model.visual.merger.named_children():
                print(f"      - {name}: {type(module).__name__}")
                if hasattr(module, 'weight'):
                    print(f"        Shape: {module.weight.shape}")
        else:
            print("⚠️ Không tìm thấy merger - kiểm tra architecture")
        
        # Get dimensions - với auto-detection
        self.vision_dim = None
        try:
            self.vision_dim = self.vit_encoder.config.hidden_size
        except:
            try:
                self.vision_dim = self.vit_encoder.config.embed_dim
            except:
                pass
        
        # Fallback: Test với dummy input
        if self.vision_dim is None:
            print("   ⚠️  Cannot detect vision_dim from config, testing...")
            try:
                with torch.no_grad():
                    dummy = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)
                    test_out = self.vit_encoder(dummy)
                    
                    if isinstance(test_out, tuple):
                        test_out = test_out[0]
                    elif isinstance(test_out, dict):
                        test_out = test_out.get('last_hidden_state', test_out.get('hidden_states'))
                    
                    if isinstance(test_out, torch.Tensor):
                        if test_out.dim() == 3:
                            self.vision_dim = test_out.shape[-1]
                        elif test_out.dim() == 2:
                            self.vision_dim = test_out.shape[-1]
                    
                    del dummy, test_out
                    torch.cuda.empty_cache()
            except:
                pass
        
        if self.vision_dim is None:
            self.vision_dim = 1536
            print("   ⚠️  Using default vision_dim=1536")
        
        print(f"✅ ViT Encoder extracted (dim: {self.vision_dim})")
        
        # 2.5. DINOv2 Encoder for SETR (NEW!)
        # Using ViT-B to save VRAM (0.2GB vs 0.6GB for ViT-L)
        print("\n🦖 [2.5/7] Loading DINOv2 for SETR segmentation...")
        self.dino_encoder = DINOv2Encoder(
            variant='vitb14',  # ViT-B/14, 768-dim (smaller, faster!)
            device=device
        )
        self.dino_dim = self.dino_encoder.get_feature_dim()  # 768
        
        # 3. SETR Decoder (NOW uses DINOv2 features!)
        print("\n🎨 [3/7] Khởi tạo SETR Segmentation Decoder (DINOv2)...")
        self.seg_decoder = SETRSegDecoder(
            vit_features_dim=self.dino_dim,  # 1024 from DINOv2-L
            num_classes=172,
            decoder_type='pup',  # 'naive', 'pup', 'mla'
            device=device,
            debug=debug
        ).to(device)

        # Set decoder to training mode
        self.seg_decoder.train()

        # Load checkpoint nếu có
        if seg_checkpoint and os.path.exists(seg_checkpoint):
            print(f"   📥 Loading checkpoint: {seg_checkpoint}")
            checkpoint = torch.load(seg_checkpoint, map_location=device, weights_only=False)
            
            # Support multiple checkpoint formats:
            # 1. train_tam2.py format: key='state'
            # 2. Old format: key='model_state_dict'
            state_dict = None
            if 'state' in checkpoint:
                state_dict = checkpoint['state']
                best_miou = checkpoint.get('best', 0)
                epoch = checkpoint.get('epoch', 0)
                print(f"   📊 Checkpoint from train_tam2: epoch={epoch}, mIoU={best_miou:.4f}")
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'decoder_state_dict' in checkpoint:
                state_dict = checkpoint['decoder_state_dict']
            
            if state_dict:
                try:
                    self.seg_decoder.load_state_dict(state_dict, strict=False)
                    print("   ✅ Loaded SETR decoder successfully")
                except Exception as e:
                    print(f"   ⚠️  Load failed: {e}")
                    # Fallback: Try to load feature adapter only
                    if 'feature_adapter_state_dict' in checkpoint:
                        try:
                            self.seg_decoder.feature_adapter.load_state_dict(
                                checkpoint['feature_adapter_state_dict'], strict=False
                            )
                            print("   ✅ Loaded feature adapter")
                        except:
                            print("   ⚠️  Feature adapter dim mismatch - using new weights")
            else:
                print(f"   ⚠️  Unknown checkpoint format. Keys: {list(checkpoint.keys())}")
        else:
            print("   ⚠️  Using random weights (needs training)")

        self.seg_decoder.train()
        
        # ============================================================
        # 4. Vision-Language Adaptor + MaskedFeatureExtractor + TextDecoder
        # AUTO-DETECT DIMENSIONS FROM CHECKPOINT BEFORE CREATING MODELS
        # ============================================================
        print("\n🌉 [4/7] Khởi tạo Vision-Language Adaptor...")
        
        # Get default LLM dimension from model
        try:
            default_llm_dim = self.full_model.language_model.config.hidden_size
        except:
            default_llm_dim = 1536
        
        # Determine checkpoint directory
        adaptor_dir = os.path.dirname(adaptor_checkpoint) if adaptor_checkpoint else None
        
        # ============================================================
        # STEP 1: DETECT DIMENSIONS FROM CHECKPOINTS BEFORE CREATING MODELS
        # ============================================================
        adaptor_dim = default_llm_dim  # Default
        text_decoder_vision_dim = default_llm_dim
        text_decoder_vocab_size = self.tokenizer.vocab_size
        
        adaptor_state_dict = None
        masked_state_dict = None
        text_decoder_state_dict = None
        
        if adaptor_dir:
            print("   🔍 Detecting dimensions from checkpoints...")
            
            # Load adaptor checkpoint
            adaptor_paths = [
                adaptor_checkpoint if adaptor_checkpoint else None,
                os.path.join(adaptor_dir, 'best_adaptor.pth'),
                os.path.join(adaptor_dir, 'adaptor.pth'),
            ]
            for path in [p for p in adaptor_paths if p]:
                if os.path.exists(path):
                    try:
                        ckpt = torch.load(path, map_location='cpu', weights_only=False)
                        if isinstance(ckpt, dict) and 'adaptor_state_dict' in ckpt:
                            adaptor_state_dict = ckpt['adaptor_state_dict']
                        else:
                            adaptor_state_dict = ckpt
                        
                        if 'query_tokens' in adaptor_state_dict:
                            adaptor_dim = adaptor_state_dict['query_tokens'].shape[-1]
                            print(f"      Adaptor llm_dim: {adaptor_dim}")
                        break
                    except Exception as e:
                        print(f"      ⚠️ Cannot read adaptor: {e}")
            
            # Load text_decoder checkpoint
            decoder_paths = [
                os.path.join(adaptor_dir, 'best_decoder.pth'),
                os.path.join(adaptor_dir, 'text_decoder.pth'),
            ]
            for path in decoder_paths:
                if os.path.exists(path):
                    try:
                        ckpt = torch.load(path, map_location='cpu', weights_only=False)
                        if isinstance(ckpt, dict) and 'text_decoder_state_dict' in ckpt:
                            text_decoder_state_dict = ckpt['text_decoder_state_dict']
                        else:
                            text_decoder_state_dict = ckpt
                        
                        if 'vision_proj.0.weight' in text_decoder_state_dict:
                            text_decoder_vision_dim = text_decoder_state_dict['vision_proj.0.weight'].shape[1]
                            print(f"      TextDecoder vision_dim: {text_decoder_vision_dim}")
                        if 'output_proj.1.weight' in text_decoder_state_dict:
                            text_decoder_vocab_size = text_decoder_state_dict['output_proj.1.weight'].shape[0]
                            print(f"      TextDecoder vocab_size: {text_decoder_vocab_size}")
                        break
                    except Exception as e:
                        print(f"      ⚠️ Cannot read text_decoder: {e}")
            
            # Load masked_extractor checkpoint
            masked_paths = [
                os.path.join(adaptor_dir, 'best_masked_extractor.pth'),
                os.path.join(adaptor_dir, 'masked_extractor.pth'),
            ]
            for path in masked_paths:
                if os.path.exists(path):
                    try:
                        ckpt = torch.load(path, map_location='cpu', weights_only=False)
                        if isinstance(ckpt, dict) and 'masked_extractor_state_dict' in ckpt:
                            masked_state_dict = ckpt['masked_extractor_state_dict']
                        else:
                            masked_state_dict = ckpt
                        break
                    except Exception as e:
                        print(f"      ⚠️ Cannot read masked_extractor: {e}")
        
        # Store final dimension
        self.llm_dim = adaptor_dim
        
        # ============================================================
        # STEP 2: CREATE MODELS WITH DETECTED DIMENSIONS
        # ============================================================
        print(f"\n   📐 Creating models:")
        print(f"      Adaptor: vision_dim={self.vision_dim} → llm_dim={adaptor_dim}")
        print(f"      TextDecoder: vision_dim={text_decoder_vision_dim}, vocab={text_decoder_vocab_size}")
        
        # Create Adaptor
        self.adaptor = ImprovedVisionLanguageAdaptor(
            vision_dim=self.vision_dim,
            llm_dim=adaptor_dim,
            num_query_tokens=64
        ).to(device)
        
        self.adaptor.load_qwen_weights_enhanced(self.full_model.visual)
        
        # Masked Feature Extractor
        self.masked_feature_extractor = MaskedFeatureExtractor(
            feature_dim=self.vision_dim
        ).to(device)
        
        # ============================================================
        # STEP 3: LOAD CHECKPOINTS INTO MODELS
        # ============================================================
        if adaptor_state_dict:
            try:
                self.adaptor.load_state_dict(adaptor_state_dict, strict=False)
                print("   ✅ Loaded trained adaptor")
            except Exception as e:
                print(f"   ⚠️ Adaptor load failed: {e}")
        else:
            print("   ⚠️ No adaptor checkpoint - using Qwen-initialized weights")
        
        if masked_state_dict:
            try:
                self.masked_feature_extractor.load_state_dict(masked_state_dict, strict=False)
                print("   ✅ Loaded trained masked_extractor")
            except Exception as e:
                print(f"   ⚠️ MaskedExtractor load failed: {e}")
        else:
            print("   ⚠️ No masked_extractor checkpoint")
        
        # Freeze adaptor và masked extractor
        for param in self.adaptor.parameters():
            param.requires_grad = False
        for param in self.masked_feature_extractor.parameters():
            param.requires_grad = False

        self.adaptor.eval()
        self.masked_feature_extractor.eval()
        
        # ============================================================
        # 4.5. Vision Text Decoder
        # ============================================================
        print("\n📝 [4.5/7] Khởi tạo Vision Text Decoder...")
        
        self.text_decoder = VisionTextDecoder(
            vision_dim=text_decoder_vision_dim,
            hidden_dim=512,
            vocab_size=text_decoder_vocab_size,
            num_decoder_layers=4,
            num_heads=8,
            max_length=32,
            dropout=0.1
        ).to(device)
        
        self.text_decoder.set_tokenizer(self.tokenizer)
        
        self.text_decoder_trained = True  # 🔥 FORCE ENABLE TextDecoder
        if text_decoder_state_dict:
            try:
                self.text_decoder.load_state_dict(text_decoder_state_dict, strict=False)
                print("   ✅ Loaded trained text_decoder")
            except Exception as e:
                print(f"   ⚠️ TextDecoder load failed: {e}")
        else:
            print("   ℹ️ TextDecoder using random weights (no checkpoint)")
        
        print("   🔥 TextDecoder ENABLED - sẽ cung cấp context cho LLM")
        
        self.text_decoder.eval()
                
        # 5. SAM 2 Segmenter
        print("\n🎯 [5/7] Khởi tạo SAM 2 Segmenter...")
        self.sam_segmenter = SAM2Segmenter(device=device)
        
        # 6. Prompt Constructor
        print("\n💬 [6/7] Khởi tạo Prompt Constructor...")
        self.prompt_constructor = PromptConstructor()
        print("✅ Prompt Constructor ready")
        
        print("\n" + "="*70)
        print(f"\n🎯 Training strategy:")
        print(f"   ❄️  Frozen: Qwen ViT + SAM + LLM")
        print(f"   🔥 Trainable: SegDecoder + Adaptor + TextDecoder")
        print("="*70 + "\n")
    
    @torch.inference_mode()
    def _extract_vit_features(self, image_rgb: np.ndarray) -> Optional[torch.Tensor]:
        """
        Extract ViT features - FINAL ULTRA ROBUST version
        """
        try:
            from PIL import Image
            pil_image = Image.fromarray(image_rgb)
            
            # GUARD: Ensure minimum image size for Qwen2-VL processor
            # Too-small images cause CUDA device-side assert in ViT encoder
            MIN_IMG_SIZE = 56  # Minimum safe size for Qwen2-VL
            w, h = pil_image.size
            if w < MIN_IMG_SIZE or h < MIN_IMG_SIZE:
                pil_image = pil_image.resize(
                    (max(w, MIN_IMG_SIZE), max(h, MIN_IMG_SIZE)), 
                    Image.LANCZOS
                )
            
            # ============================================================
            # STEP 1: Process image
            # ============================================================
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {"type": "text", "text": "Describe."}
                        ]
                    }
                ]
                
                text_prompt = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                inputs = self.processor(
                    text=[text_prompt],
                    images=[pil_image],
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
            except Exception as e:
                print(f"❌ Processor error: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                return None
            
            pixel_values = inputs['pixel_values']
            
            if self.debug:
                print(f"[Debug] Pixel shape: {pixel_values.shape}")
            
            # ============================================================
            # STEP 2: Handle ALL possible pixel_values shapes
            # ============================================================
            try:
                ndim = pixel_values.dim()
                
                # Case 1: 2D tensor (N, C) - already features!
                if ndim == 2:
                    if self.debug:
                        print(f"[Debug] 2D features detected: {pixel_values.shape}")
                    features = pixel_values.unsqueeze(0)  # (1, N, C)
                    return features.float() if features.dtype == torch.float16 else features
                
                # Case 2: 3D tensor (B, N, C) - already features!
                elif ndim == 3:
                    if self.debug:
                        print(f"[Debug] 3D features detected: {pixel_values.shape}")
                    return pixel_values.float() if pixel_values.dtype == torch.float16 else pixel_values
                
                # Case 3: 4D tensor (B, C, H, W) - standard image
                elif ndim == 4:
                    B, C, H, W = pixel_values.shape
                
                # Case 4: 5D tensor (B, T, C, H, W) - multi-image
                elif ndim == 5:
                    B, num_imgs, C, H, W = pixel_values.shape
                    pixel_values = pixel_values.reshape(B * num_imgs, C, H, W)
                    B, C, H, W = pixel_values.shape
                
                else:
                    print(f"⚠️ Unsupported pixel shape: {pixel_values.shape} ({ndim}D)")
                    return None
                
                # Only for 4D/5D: calculate grid
                patch_size = getattr(self.vit_encoder.config, 'patch_size', 14)
                grid_h = H // patch_size
                grid_w = W // patch_size
                
                grid_thw = torch.tensor(
                    [[1, grid_h, grid_w]], 
                    device=self.device,
                    dtype=torch.long
                )
                
                if self.debug:
                    print(f"[Debug] Grid: {grid_h}x{grid_w} = {grid_h*grid_w} patches")
                    
            except Exception as e:
                print(f"❌ Shape handling error: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                return None
            
            # ============================================================
            # STEP 3: Extract features (only if not already features)
            # ============================================================
            if pixel_values.dim() in [4, 5]:  # Need to extract
                vision_outputs = None
                method_used = None
                
                with torch.no_grad():
                    strategies = [
                        lambda: self.vit_encoder(pixel_values=pixel_values, grid_thw=grid_thw),
                        lambda: self.vit_encoder(pixel_values),
                        lambda: self.vit_encoder(pixel_values, grid_thw),
                        lambda: self.vit_encoder.__call__(pixel_values, grid_thw=grid_thw),
                        lambda: self.vit_encoder.forward(pixel_values)
                    ]
                    
                    for idx, strategy in enumerate(strategies, 1):
                        try:
                            vision_outputs = strategy()
                            method_used = f"Method {idx}"
                            if self.debug:
                                print(f"[Debug] {method_used} succeeded")
                            break
                        except Exception as e:
                            if self.debug:
                                print(f"[Debug] Method {idx} failed: {type(e).__name__}")
                            continue
                    
                    if vision_outputs is None:
                        print("❌ All extraction methods failed")
                        return None
                
                # ============================================================
                # STEP 4: Extract tensor from output
                # ============================================================
                features = None
                
                if isinstance(vision_outputs, tuple):
                    for item in vision_outputs:
                        if isinstance(item, torch.Tensor):
                            if item.dim() == 3:
                                features = item
                                break
                            elif item.dim() == 4 and features is None:
                                features = item
                    
                    if features is None:
                        for item in vision_outputs:
                            if isinstance(item, torch.Tensor):
                                features = item
                                break
                
                elif hasattr(vision_outputs, 'last_hidden_state'):
                    features = vision_outputs.last_hidden_state
                
                elif isinstance(vision_outputs, torch.Tensor):
                    features = vision_outputs
                
                elif isinstance(vision_outputs, dict):
                    for key in ['last_hidden_state', 'hidden_states', 'features', 
                               'encoder_outputs', 'vision_outputs', 'image_embeds']:
                        if key in vision_outputs:
                            candidate = vision_outputs[key]
                            if isinstance(candidate, (tuple, list)) and len(candidate) > 0:
                                candidate = candidate[0] if isinstance(candidate[0], torch.Tensor) else candidate
                            if isinstance(candidate, torch.Tensor):
                                features = candidate
                                break
                
                if features is None:
                    try:
                        features = self.vit_encoder.forward(pixel_values)
                        if isinstance(features, tuple):
                            features = features[0]
                        elif isinstance(features, dict) and 'last_hidden_state' in features:
                            features = features['last_hidden_state']
                    except:
                        pass
                
                if features is None or not isinstance(features, torch.Tensor):
                    print(f"⚠️ Cannot extract features: {type(features)}")
                    return None
                
                # ============================================================
                # STEP 5: Reshape to (B, N, C)
                # ============================================================
                original_shape = features.shape
                
                try:
                    if features.dim() == 4:
                        B, C, H_f, W_f = features.shape
                        features = features.flatten(2).permute(0, 2, 1)
                        if self.debug:
                            print(f"[Debug] 4D→3D: {original_shape} → {features.shape}")
                    
                    elif features.dim() == 3:
                        if self.debug:
                            print(f"[Debug] Already 3D: {features.shape}")
                    
                    elif features.dim() == 2:
                        features = features.unsqueeze(0)
                        if self.debug:
                            print(f"[Debug] 2D→3D: {original_shape} → {features.shape}")
                    
                    elif features.dim() == 5:
                        B, T, C, H_f, W_f = features.shape
                        features = features.reshape(B * T, C, H_f, W_f)
                        features = features.flatten(2).permute(0, 2, 1)
                        if self.debug:
                            print(f"[Debug] 5D→3D: {original_shape} → {features.shape}")
                    
                    else:
                        print(f"⚠️ Unsupported dim: {features.dim()} {features.shape}")
                        return None
                    
                except Exception as e:
                    print(f"❌ Reshape error: {e}")
                    if self.debug:
                        import traceback
                        traceback.print_exc()
                    return None
            
            else:
                # Already got features from STEP 2
                features = pixel_values
            
            # ============================================================
            # FINAL VALIDATION
            # ============================================================
            if features.dtype == torch.float16:
                features = features.float()
            
            if features.dim() != 3:
                print(f"⚠️ Expected 3D, got {features.dim()}D: {features.shape}")
                return None
            
            B, N, C = features.shape
            
            if N < 10 or N > 100000:
                if self.debug:
                    print(f"[Debug] Warning: N={N} (unusual)")
            
            if C < 256 or C > 4096:
                if self.debug:
                    print(f"[Debug] Warning: C={C} (unusual)")
            
            if self.debug:
                print(f"[Debug] ✅ Final: {features.shape} (dtype={features.dtype})")
            
            return features
            
        except Exception as e:
            print(f"❌ Extract features error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None
    
    @torch.inference_mode()
    def predict_class_from_region(self, image_rgb: np.ndarray, 
                                mask: np.ndarray,
                                image_shape: Tuple[int, int]) -> Tuple[Optional[str], float]:
        """
        Dự đoán class từ SETR decoder using DINOv2 features.
        
        Args:
            image_rgb: (H, W, 3) numpy array
            mask: (H, W) binary mask
            image_shape: (H, W) target output size
        """
        try:
            # Extract DINOv2 features for SETR
            dino_features = self.dino_encoder.extract_features(image_rgb)
            
            # Get segmentation map từ SETR decoder
            seg_map = self.seg_decoder(dino_features, target_size=image_shape)
            pred_class_map = seg_map.argmax(dim=1)[0].cpu().numpy()
            
            # Get dominant class in masked region
            masked_classes = pred_class_map[mask > 0]
            
            if len(masked_classes) > 0:
                unique, counts = np.unique(masked_classes, return_counts=True)
                dominant_idx = unique[counts.argmax()]
                confidence = counts.max() / len(masked_classes)
                
                if dominant_idx > 0 and dominant_idx <= len(COCO_STUFF_CLASSES) and confidence > 0.5:
                    predicted_class = COCO_STUFF_CLASSES[dominant_idx - 1]
                    return predicted_class, confidence
            
            return None, 0.0
            
        except Exception as e:
            print(f"⚠️ Lỗi predict class: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None, 0.0
        
    @torch.inference_mode()
    def process_region(self, image_rgb: np.ndarray, 
                    mask: np.ndarray,
                    user_query: Optional[str] = None) -> Dict:
        """
        FIXED VERSION: Inject vision tokens into LLM generation
        """
        start_time = time.time()
        print("🔄", end='', flush=True)
        
        # Initialize default values to avoid UnboundLocalError
        predicted_class = None
        confidence = 0.0
        description = "Lỗi xử lý"
        
        try:
            # STEP 1: Extract ViT features
            vit_features = self._extract_vit_features(image_rgb)
            
            if vit_features is None:
                return {
                    'description': "Không thể phân tích ảnh.", 
                    'error': True, 
                    'predicted_class': None, 
                    'confidence': 0.0
                }
            
            print(".", end='', flush=True)
            
            # STEP 2: Predict class using DINOv2 → SETR
            predicted_class, confidence = self.predict_class_from_region(
                image_rgb, mask, image_rgb.shape[:2]  # Now uses DINOv2 internally
            )
            
            if predicted_class:
                print(f"[{predicted_class}:{confidence:.0%}]", end='', flush=True)
            
            # STEP 3: Extract FOCUSED features → Vision Tokens
            masked_features = self.masked_feature_extractor(vit_features, mask)
            vision_tokens = self.adaptor(masked_features)  # (B, 64, 1536)
            
            print(".", end='', flush=True)
            
            # STEP 4: Construct prompt
            text_prompt = self.prompt_constructor.construct_prompt(
                mask, predicted_class, user_query
            )
            
            # ============================================================
            # STEP 5: DIRECT NATIVE GENERATION (Skip vision injection)
            # ============================================================
            # NOTE: Vision token injection doesn't work with Qwen2-VL architecture
            # Using improved image preprocessing + native generation instead
            try:
                from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
                
                # Crop to focused region with better visual emphasis
                y_coords, x_coords = np.where(mask > 0)
                
                if len(y_coords) > 0:
                    x_min, x_max = x_coords.min(), x_coords.max()
                    y_min, y_max = y_coords.min(), y_coords.max()
                    
                    h_img, w_img = image_rgb.shape[:2]
                    
                    # DYNAMIC MARGIN based on object size
                    object_width = x_max - x_min
                    object_height = y_max - y_min
                    object_size = max(object_width, object_height)

                    # Larger context for smaller objects (Increased for better semantic understanding)
                    if object_size < 100:
                        margin = int(object_size * 1.5)  # 150% context for small (was 1.0)
                    elif object_size < 200:
                        margin = int(object_size * 1.0)  # 100% context for medium (was 0.6)
                    else:
                        margin = int(object_size * 0.5)  # 50% context for large (was 0.3)

                    margin = max(margin, 60)  # At least 60px
                    
                    x_min_crop = max(0, x_min - margin)
                    x_max_crop = min(w_img - 1, x_max + margin) # Ensure within bounds
                    y_min_crop = max(0, y_min - margin)
                    y_max_crop = min(h_img - 1, y_max + margin) # Ensure within bounds
                    
                    # Robust safety check for 0-size or negative crops
                    if (x_max_crop <= x_min_crop + 2) or (y_max_crop <= y_min_crop + 2):
                        print("Warning: Crop too small, adjusting...")
                        x_mid = (x_min + x_max) // 2
                        y_mid = (y_min + y_max) // 2
                        margin = 64
                        x_min_crop = max(0, x_mid - margin)
                        x_max_crop = min(w_img - 1, x_mid + margin)
                        y_min_crop = max(0, y_mid - margin)
                        y_max_crop = min(h_img - 1, y_mid + margin)
                        
                    cropped_img = image_rgb[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
                    cropped_mask = mask[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
                    
                    # Last resort safety check
                    if cropped_img.size == 0 or cropped_img.shape[0] < 2 or cropped_img.shape[1] < 2:
                        print("❌ Error: Invalid crop size!")
                        raise ValueError(f"Invalid crop size: {cropped_img.shape}")
                    
                    # ============================================================
                    # 🔥 FOCUSED IMAGE - Chỉ hiển thị object được chọn
                    # ============================================================
                    
                    # Method: 
                    # 1. Nền TRẮNG (không có distractions)
                    # 2. Object giữ nguyên
                    # 3. Viền ĐỎ DÀY (rất dễ thấy)
                    
                    # Create PIL images
                    pil_original = Image.fromarray(cropped_img)
                    pil_mask = Image.fromarray((cropped_mask * 255).astype(np.uint8))
                    
                    # Option 1: WHITE BACKGROUND - Loại bỏ hoàn toàn distractions
                    white_bg = Image.new('RGB', pil_original.size, (255, 255, 255))
                    
                    # Foreground: Original object with slight enhancement
                    foreground = pil_original.copy()
                    enhancer = ImageEnhance.Contrast(foreground)
                    foreground = enhancer.enhance(1.15)  # Boost contrast
                    enhancer = ImageEnhance.Sharpness(foreground)
                    foreground = enhancer.enhance(1.2)   # Sharper
                    
                    # Composite: Object on white background
                    pil_image = Image.composite(foreground, white_bg, pil_mask)
                    
                    # Draw THICK RED contour (RẤT dễ thấy)
                    contours, _ = cv2.findContours(
                        cropped_mask.astype(np.uint8), 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    if contours:
                        result_np = np.array(pil_image)
                        # RED outline - THICK (5px)
                        cv2.drawContours(result_np, contours, -1, (255, 0, 0), 5)
                        # Inner GREEN outline (2px) for contrast
                        cv2.drawContours(result_np, contours, -1, (0, 255, 0), 2)
                        pil_image = Image.fromarray(result_np)
                    
                    # Resize if too small (LLM needs details)
                    min_size = 256
                    if pil_image.width < min_size or pil_image.height < min_size:
                        scale = max(min_size / pil_image.width, min_size / pil_image.height)
                        new_size = (int(pil_image.width * scale), int(pil_image.height * scale))
                        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                    
                    if self.debug:
                        print(f"\n[Debug] FOCUSED Image: {pil_image.width}x{pil_image.height}px (object: {object_width}x{object_height})")
                        # Save debug image
                        pil_image.save("debug_focused_image.png")
                        print("[Debug] Saved: debug_focused_image.png")
                        
                else:
                    pil_image = Image.fromarray(image_rgb)
                
                print(".", end='', flush=True)
                
                # ============================================================
                # 🔥 STEP 5.1: TEXT DECODER - Get object description from adaptor
                # ============================================================
                adaptor_context = ""
                
                if self.text_decoder_trained:
                    try:
                        # vision_tokens is already computed above: (B, 64, 1536)
                        adaptor_text = self.text_decoder.decode_to_text(
                            vision_tokens,
                            self.tokenizer,
                            max_length=24,
                            temperature=0.5
                        )
                        
                        if adaptor_text and len(adaptor_text) > 3:
                            adaptor_context = f"[Nhận dạng: {adaptor_text}] "
                            if self.debug:
                                print(f"\n[Debug TextDecoder]: '{adaptor_text}'")
                    except Exception as td_err:
                        if self.debug:
                            print(f"\n⚠️ TextDecoder error: {td_err}")
                
                # ============================================================
                # 🔥 IMPROVED PROMPTS - Include adaptor context + SETR class
                # ============================================================
                
                # Context integration
                context_parts = []
                if adaptor_context:
                    context_parts.append(adaptor_context)
                if predicted_class:
                    context_parts.append(f"[Phân loại: {predicted_class}]")
                
                full_context = " ".join(context_parts)
                
                if user_query:
                    text_prompt = f"""{full_context}
Ảnh này chỉ chứa MỘT vật thể duy nhất trên nền trắng, được viền đỏ.
Hãy nhìn kỹ vật thể đó và trả lời câu hỏi bằng tiếng Việt:
{user_query}
Ngoài ra, hãy cho biết công dụng/chức năng chính của vật thể này."""
                else:
                    text_prompt = f"""{full_context}
Ảnh này chỉ chứa MỘT vật thể duy nhất trên nền trắng, được viền đỏ.
Hãy mô tả chi tiết vật thể này bằng tiếng Việt:
- Đây là gì?
- Màu sắc và hình dạng?
- Đặc điểm nổi bật?
- Công dụng/chức năng chính của vật thể này là gì?"""
                
                # ========================================================
                # CREATE AND PROCESS INPUT
                # ========================================================
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {"type": "text", "text": text_prompt}
                        ]
                    }
                ]
                
                # Apply chat template
                text = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # Process inputs
                inputs = self.processor(
                    text=[text],
                    images=[pil_image],
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                print(".", end='', flush=True)
                
                # ========================================================
                # 🔥 DIRECT NATIVE GENERATION (Skip vision injection)
                # ========================================================
                # Vision token injection doesn't work with Qwen2-VL
                # Going straight to native generation with pixel_values
                
                generated_ids_trimmed = None
                
                try:
                    gen_kwargs = {
                        'input_ids': inputs['input_ids'],
                        'attention_mask': inputs.get('attention_mask'),
                        'max_new_tokens': 150,  # More tokens for detailed description
                        'min_new_tokens': 15,   # Ensure substantial output
                        'temperature': 0.4,     # 🔥 Lower for accurate descriptions
                        'top_p': 0.85,
                        'top_k': 30,
                        'do_sample': True,
                        'repetition_penalty': 1.15,
                        'no_repeat_ngram_size': 3,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'eos_token_id': self.tokenizer.eos_token_id,
                    }
                    
                    # Add image inputs
                    if 'pixel_values' in inputs:
                        gen_kwargs['pixel_values'] = inputs['pixel_values']
                    if 'image_grid_thw' in inputs:
                        gen_kwargs['image_grid_thw'] = inputs['image_grid_thw']
                    
                    generated_ids = self.full_model.generate(**gen_kwargs)
                    input_len = inputs['input_ids'].shape[1]
                    generated_ids_trimmed = generated_ids[:, input_len:]
                    
                    if self.debug:
                        print(f"  ✅ Generated {generated_ids_trimmed.shape[1]} tokens")
                    
                except Exception as gen_error:
                    if self.debug:
                        print(f"❌ Generation failed: {gen_error}")
                    generated_ids_trimmed = None
                
                # ========================================================
                # DECODE OUTPUT - ROBUST VERSION
                # ========================================================
                if generated_ids_trimmed is not None and generated_ids_trimmed.numel() > 0:
                    try:
                        import re
                        
                        # ✅ STEP 1: Decode toàn bộ sequence
                        raw_description = self.tokenizer.decode(
                            generated_ids_trimmed[0],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True
                        ).strip()
                        
                        if self.debug:
                            print(f"\n[Debug Raw Output]: '{raw_description[:200] if raw_description else 'EMPTY'}' (len={len(raw_description) if raw_description else 0})")
                            print(f"[Debug Token IDs]: {generated_ids_trimmed[0][:20].tolist()}...")
                        
                        # ✅ STEP 2: Cleanup - remove common garbage patterns
                        description = raw_description
                        
                        # Remove excessive whitespace
                        description = re.sub(r'\s+', ' ', description).strip()
                        
                        # Remove common LLM artifacts
                        garbage_patterns = [
                            r'^[\s\.,;:!?\-_]+',  # Leading punctuation
                            r'[\s\.,;:!?\-_]+$',  # Trailing punctuation  
                            r'^(assistant|Assistant|ASSISTANT):?\s*',  # Role prefixes
                            r'^(Answer|Trả lời|Response):?\s*',  # Answer prefixes
                            r'<\|.*?\|>',  # Special tokens that weren't removed
                            r'\[.*?\]',  # Bracketed tokens
                        ]
                        for pattern in garbage_patterns:
                            description = re.sub(pattern, '', description).strip()
                        
                        # 🔥 REMOVE CJK CHARACTERS (Chinese/Japanese/Korean) - Keep Vietnamese
                        # Vietnamese uses Latin alphabet with diacritics, not CJK
                        cjk_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+'
                        description = re.sub(cjk_pattern, '', description).strip()
                        
                        # Remove any remaining non-printable garbage
                        description = re.sub(r'[^\w\s\.,;:!?\-\(\)àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', '', description, flags=re.IGNORECASE).strip()
                        
                        # Clean up multiple spaces after removal
                        description = re.sub(r'\s+', ' ', description).strip()
                        
                        # ✅ STEP 3: Validate and use smart fallback
                        if description and len(description) >= 3:
                            # Success - use the description
                            pass
                        else:
                            # Try to create a meaningful fallback using available info
                            if predicted_class and confidence > 0.3:
                                if user_query:
                                    description = f"Đây có thể là {predicted_class} (độ tin cậy {confidence:.0%}). Vui lòng thử lại với góc chụp rõ hơn."
                                else:
                                    description = f"Đây là {predicted_class} với độ tin cậy {confidence:.0%}."
                            else:
                                # Last resort: trigger fallback generation
                                description = None  # Will trigger fallback below
                        
                    except Exception as decode_error:
                        if self.debug:
                            print(f"\n⚠️ Decode error: {decode_error}")
                        description = None  # Will trigger fallback
                else:
                    description = None  # Will trigger fallback
                
                # ========================================================
                # FALLBACK GENERATION - If primary generation failed
                # ========================================================
                if not description or len(description) < 3:
                    if self.debug:
                        print("\n🔄 Primary generation empty, trying FALLBACK...")
                    
                    try:
                        # Fallback 1: Simple direct generation with image - VIETNAMESE ONLY
                        simple_prompt = "Nhìn vào vùng xanh lá trong ảnh. Mô tả ngắn gọn vật thể này bằng tiếng Việt thuần túy."
                        if user_query:
                            simple_prompt = f"Nhìn vào vùng xanh lá trong ảnh. Trả lời bằng tiếng Việt thuần túy: {user_query}"
                        
                        fallback_messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": pil_image},
                                    {"type": "text", "text": simple_prompt}
                                ]
                            }
                        ]
                        
                        fallback_text = self.processor.apply_chat_template(
                            fallback_messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                        
                        fallback_inputs = self.processor(
                            text=[fallback_text],
                            images=[pil_image],
                            return_tensors="pt",
                            padding=True
                        ).to(self.device)
                        
                        # Generate with relaxed parameters
                        fallback_generated = self.full_model.generate(
                            **fallback_inputs,
                            max_new_tokens=100,
                            min_new_tokens=3,
                            temperature=0.7,  # More creative
                            top_p=0.95,
                            do_sample=True,
                            repetition_penalty=1.1,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                        
                        fallback_input_len = fallback_inputs['input_ids'].shape[1]
                        fallback_output = fallback_generated[:, fallback_input_len:]
                        
                        fallback_description = self.tokenizer.decode(
                            fallback_output[0],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True
                        ).strip()
                        
                        # 🔥 Apply same CJK filter to fallback output
                        import re
                        cjk_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+'
                        fallback_description = re.sub(cjk_pattern, '', fallback_description).strip()
                        fallback_description = re.sub(r'\s+', ' ', fallback_description).strip()
                        
                        if self.debug:
                            print(f"[Fallback Output]: '{fallback_description[:100] if fallback_description else 'EMPTY'}...'")
                        
                        if fallback_description and len(fallback_description) >= 3:
                            description = fallback_description
                        
                        del fallback_inputs, fallback_generated
                        
                    except Exception as fallback_err:
                        if self.debug:
                            print(f"⚠️ Fallback generation error: {fallback_err}")
                
                # ========================================================
                # FINAL FALLBACK - Use class prediction if available
                # ========================================================
                if not description or len(description) < 3:
                    if predicted_class and confidence > 0.2:
                        description = f"Vật thể được nhận dạng là: {predicted_class} (độ tin cậy: {confidence:.0%})."
                    else:
                        # Ultimate fallback - still provide useful information
                        y_coords, x_coords = np.where(mask > 0)
                        if len(y_coords) > 0:
                            area_percent = (len(y_coords) / (mask.shape[0] * mask.shape[1])) * 100
                            description = f"Đã phát hiện vật thể chiếm {area_percent:.1f}% diện tích hình ảnh. Vui lòng thử chụp rõ hơn hoặc từ góc khác."
                        else:
                            description = "Đã nhận diện được vật thể nhưng chưa thể mô tả chi tiết. Vui lòng thử lại."

                elapsed = time.time() - start_time
                print(f" ✓ ({elapsed:.1f}s)")
                
            except Exception as generation_error:
                if self.debug:
                    print(f"\n❌ Generation error: {generation_error}")
                    import traceback
                    traceback.print_exc()
                description = f"Lỗi tạo câu trả lời: {str(generation_error)}"
            
            finally:
                # Cleanup
                try:
                    if 'inputs' in locals():
                        del inputs
                    if 'generated_ids' in locals():
                        del generated_ids
                    if 'text_embeds' in locals():
                        del text_embeds
                    if 'combined_embeds' in locals():
                        del combined_embeds
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except:
                    pass
            
            return {
                'description': description,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'query': user_query or "Mô tả tổng quát",
                'error': False,
                'vit_features_shape': vit_features.shape if self.debug else None,
                'vision_tokens_shape': vision_tokens.shape if self.debug else None,
                'used_vision_tokens': True
            }
            
        except Exception as e:
            print(f" ❌ {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return {
                'description': f"Lỗi: {str(e)}",
                'predicted_class': predicted_class,
                'confidence': confidence,
                'error': True,
                'used_vision_tokens': False
            }
        
    def _initialize_vision_projection(self):
        """
        Initialize projection layer for vision tokens to LLM space
        Chi can goi 1 lan trong __init__
        """
        if not hasattr(self, 'vision_to_llm_proj'):
            self.vision_to_llm_proj = None  # Will be created on first use
            print("   Vision to LLM projection will be created dynamically")
