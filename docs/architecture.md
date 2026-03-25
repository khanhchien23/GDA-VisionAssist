# GDA-VisionAssist Architecture

## Overview

GDA (Global Description Acquisition) là hệ thống Vision-Language hỗ trợ người khiếm thị nhận diện và mô tả vật thể. Hệ thống kết hợp nhiều mô hình AI để tạo pipeline hoàn chỉnh từ đầu vào hình ảnh đến đầu ra mô tả bằng giọng nói.

## System Architecture

```
Input Image → ViT Encoder → [Seg Decoder + Adaptor] → Vision Tokens → LLM → Answer
                  ↓
              SAM 2 Mask
```

## Core Components

### 1. Shared ViT Encoder (Qwen2-VL)
- **Module**: `src/core/gda.py` → `self.vit_encoder`
- **Source**: Trích xuất từ Qwen2-VL pretrained model
- **Đầu ra**: Visual features `(B, N, 1536)`
- **Vai trò**: Encoder chia sẻ giữa nhánh VLM và nhánh Segmentation

### 2. DINOv2 Encoder
- **Module**: `src/models/dinov2_encoder.py`
- **Variant**: ViT-B/14 (768-dim)
- **Vai trò**: Cung cấp features cho SETR Segmentation Decoder
- **Ưu điểm**: Nhẹ (0.2GB VRAM), features tốt cho segmentation

### 3. SETR Segmentation Decoder
- **Module**: `src/models/segmentation.py`
- **Input**: DINOv2 features `(B, N, 768)`
- **Output**: Segmentation map `(B, 172, H, W)` (COCO-Stuff 171 classes + background)
- **Variants**: Naive, PUP (Progressive UPsampling), MLA

### 4. Vision-Language Adaptor
- **Module**: `src/models/adaptor.py`
- **Cơ chế**: Cross-attention với learnable query tokens
- **Input**: Masked ViT features `(B, N, 1536)`
- **Output**: Vision tokens `(B, 64, 1536)`
- **Khởi tạo**: Từ Qwen merger weights

### 5. Masked Feature Extractor
- **Module**: `src/models/vit_encoder.py`
- **Vai trò**: Trích xuất features từ vùng được chọn (SAM mask)

### 6. Vision Text Decoder
- **Module**: `src/models/text_decoder.py`
- **Architecture**: Transformer decoder với vision projection
- **Vai trò**: Cung cấp context bổ sung cho LLM

### 7. SAM 2 Segmenter
- **Module**: `src/models/sam_segmenter.py`
- **Vai trò**: Segment vật thể từ click point của người dùng
- **Hỗ trợ**: Iterative refinement

### 8. Prompt Constructor
- **Module**: `src/core/prompt.py`
- **Hỗ trợ**: Tiếng Việt + Tiếng Anh
- **Phân loại**: 8 loại câu hỏi (what_is, color, describe, material, ...)

## Application Layer

### GDA Application (`src/app/`)
- `gda_application.py`: Main loop, event handling, webcam integration
- `inference_manager.py`: Background inference với thread-safe queue
- `ui_renderer.py`: OpenCV-based UI rendering
- `config.py`: Dataclass-based configuration

### I/O Layer (`src/io/`)
- `voice.py`: STT (Google Speech) + TTS (Edge-TTS + pygame)
- `keyboard.py`: Pynput keyboard monitoring, click handling
- `camera.py`: OpenCV webcam initialization

## Data Flow

```
1. User nhấn SPACE → Bật chế độ chọn vùng
2. User CLICK vào vật thể → SAM 2 tạo mask
3. User nhấn ENTER hoặc GIỮ C + nói câu hỏi
4. Pipeline chạy:
   a. ViT Encoder → extract visual features
   b. DINOv2 + SETR → predict class
   c. MaskedFeatureExtractor → focused features from mask region
   d. Adaptor → vision tokens
   e. PromptConstructor → text prompt với class info
   f. Qwen2-VL LLM → generate answer
5. TTS → đọc kết quả
```

## Training Strategy

| Component | Status |
|-----------|--------|
| Qwen2-VL ViT | ❄️ Frozen |
| SAM 2 | ❄️ Frozen |
| Qwen2-VL LLM | ❄️ Frozen |
| SETR Decoder | 🔥 Trainable |
| Adaptor | 🔥 Trainable |
| Text Decoder | 🔥 Trainable |

## Directory Structure

```
GDA-VisionAssist/
├── app.py                  # Entry point
├── setup.py                # Package configuration
├── config/
│   ├── app_config.yaml     # Application config
│   └── model_config.yaml   # Model config
├── src/
│   ├── __init__.py
│   ├── constants.py        # COCO-Stuff class names (Vietnamese)
│   ├── app/                # Application layer
│   ├── core/               # Core GDA logic
│   ├── io/                 # Input/Output (voice, keyboard, camera)
│   ├── models/             # Neural network models
│   └── utils/              # Utility functions
├── scripts/                # Training & evaluation scripts
├── tests/                  # Unit tests
├── docs/                   # Documentation
└── examples/               # Usage examples
```
