# GDA-VisionAssist - Deployment Guide

## System Requirements

### Hardware
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA 6GB VRAM | NVIDIA RTX 3090 (24GB) |
| RAM | 16 GB | 32 GB |
| Storage | 10 GB | 20 GB |
| Webcam | 720p USB | 1080p |
| Microphone | Built-in | External USB |

### Software
- Python 3.8+
- CUDA 11.8+ 
- cuDNN 8.6+
- Windows 10/11 hoặc Ubuntu 20.04+

## Installation

### 1. Clone & Setup Environment

```bash
git clone https://github.com/ChinhocIT/GDA-VisionAssist.git
cd GDA-VisionAssist

python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
pip install -e .
```

### 2. Download Model Checkpoints

```bash
python scripts/download_models.py
```

Các model sẽ được tải về thư mục `checkpoints/`:
- `checkpoints/setr_dino_best.pth` — SETR segmentation decoder
- `checkpoints/adaptor_vizwiz/adaptor.pth` — Vision-Language adaptor
- `checkpoints/adaptor_vizwiz/masked_extractor.pth` — Masked feature extractor
- `checkpoints/adaptor_vizwiz/text_decoder.pth` — Text decoder

> **Note**: Qwen2-VL và SAM 2 sẽ được tải tự động từ HuggingFace khi chạy lần đầu.

### 3. Configuration

Copy và chỉnh sửa file config:

```bash
cp config/app_config.yaml config/app_config.local.yaml
```

Tham số quan trọng:
- `webcam.device_id`: Index webcam (mặc định `0`)
- `voice.enable_stt`: Bật/tắt nhận dạng giọng nói
- `voice.enable_tts`: Bật/tắt đọc kết quả

## Running

### Basic Usage

```bash
python app.py
```

### Advanced Options

```bash
# Custom checkpoints
python app.py --seg-checkpoint path/to/seg.pth --adaptor-checkpoint path/to/adaptor.pth

# Debug mode
python app.py --debug

# CPU mode (chậm hơn)
python app.py --device cpu
```

## Troubleshooting

### Common Issues

| Vấn đề | Giải pháp |
|---------|-----------|
| CUDA out of memory | Giảm `webcam.width/height`, dùng quantization 8-bit |
| Webcam không mở | Kiểm tra `webcam.device_id` trong config |
| Microphone không nhận | Cài PyAudio: `pip install pyaudio` |
| TTS không hoạt động | Cài edge-tts: `pip install edge-tts pygame` |
| Model download chậm | Dùng mirror HuggingFace hoặc tải thủ công |

### GPU Memory Usage

| Component | VRAM |
|-----------|------|
| Qwen2-VL (8-bit) | ~3 GB |
| SAM 2 | ~1 GB |
| DINOv2 (ViT-B) | ~0.2 GB |
| SETR Decoder | ~0.1 GB |
| Adaptor + TextDecoder | ~0.2 GB |
| **Total** | **~4.5 GB** |

## Production Notes

- Sử dụng quantization 8-bit (mặc định) để giảm VRAM
- `torch.backends.cudnn.benchmark = True` đã được bật để tối ưu tốc độ
- Frame cleanup mỗi 30 frames để tránh memory leak
- Inference chạy trên background thread, không block UI
