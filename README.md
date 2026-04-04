# GDA-VisionAssist

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**VISION-LANGUAGE SYSTEM FOR ASSISTING VISUALLY IMPAIRED INDIVIDUALS IN OBJECT RECOGNITION AND DESCRIPTION QUERY.**

## ✨ Features

- 🎯 **Precise object segmentation** with SAM 2 (Segment Anything Model)
- 🧠 **Vision-Language understanding** with Qwen2-VL
- 🗣️ **Voice interaction** (Vietnamese + English)
- ⚡ **Real-time inference** on webcam
- 🎨 **Semantic segmentation** with SETR decoder (COCO-Stuff 171 classes)
- 🔧 **Modular architecture** for easy extension

## 🏗️ Architecture

```
Input Image → ViT Encoder → [Seg Decoder + Adaptor] → Vision Tokens → LLM → Answer
                  ↓
              SAM 2 Mask
```

### Main Components:

1. **Shared ViT Encoder**: Extracts visual features from Qwen2-VL
2. **SETR Segmentation Decoder**: Predicts class for each region
3. **Vision-Language Adaptor**: Transforms visual features → language embeddings
4. **SAM 2 Segmenter**: Segments objects from user click
5. **LLM Generator**: Qwen2-VL language model generates answers

## 📋 System Requirements

- **Python**: 3.8+
- **GPU**: NVIDIA GPU with CUDA 11.8+ (recommended ≥8GB VRAM)
- **RAM**: 16GB+
- **OS**: Windows/Linux/macOS

## 🚀 Installation

### 1. Clone repository

```bash
git clone https://github.com/ChinhocIT/GDA-VisionAssist.git
cd GDA-VisionAssist
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download models

```bash
python scripts/download_models.py
```

### 5. Configuration

Create `.env` file from template:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Model paths
QWEN_MODEL_NAME=Qwen/Qwen2-VL-2B-Instruct
SAM_MODEL_NAME=facebook/sam-vit-huge
SEG_CHECKPOINT_PATH=checkpoints/seg_decoder_best.pth
ADAPTOR_CHECKPOINT_PATH=checkpoints/adaptor_best.pth

# Device
DEVICE=cuda
DEBUG=False

# Voice
ENABLE_STT=True
ENABLE_TTS=True
```

## 💡 Usage

### Basic Usage

```bash
python app.py
```

### Advanced Options

```bash
# Specify checkpoint
python app.py --seg-checkpoint path/to/seg.pth --adaptor-checkpoint path/to/adaptor.pth

# Enable debug mode
python app.py --debug

# Use CPU
python app.py --device cpu
```

### Keyboard Controls

| Key | Function |
|-----|----------|
| `Space` | Activate region selection mode |
| `C` (hold) + Voice | Ask a question by voice |
| `Enter` | Auto-describe selected region |
| `S` | Save current image |
| `D` | Toggle debug mode |
| `Q` | Quit |

### Python API

```python
from src.core.gda import GlobalDescriptionAcquisition
import cv2

# Initialize
gda = GlobalDescriptionAcquisition(device="cuda")

# Load image
image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Segment object (click point)
mask = gda.sam_segmenter.segment_from_point(image_rgb, point=(320, 240))

# Ask question
result = gda.process_region(image_rgb, mask, user_query="What is this?")

print(result['description'])
# Output: "This is a gray laptop with a black keyboard and the screen is on."
```

## 📚 Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api.md)
- [Training Guide](docs/training.md)
- [Deployment Guide](docs/deployment.md)

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Test with coverage
pytest --cov=src tests/

# Test specific module
pytest tests/test_models.py
```

## 🎓 Training

### Train Segmentation Decoder

```bash
python scripts/train_decoder.py \
  --dataset coco_stuff \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4
```

### Train Vision-Language Adaptor

```bash
python scripts/train_adaptor.py \
  --dataset vqa_v2 \
  --epochs 20 \
  --batch-size 4
```

## 📊 Performance

| Model | GPU | FPS | Accuracy |
|-------|-----|-----|----------|
| Full System | RTX 3090 | ~2-3 | 85%+ |
| Seg Decoder only | RTX 3090 | ~10 | 78% mIoU |
| SAM 2 only | RTX 3090 | ~8 | 92% IoU |

## 🤝 Contributing

We welcome all contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Run linting
black src/
flake8 src/
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- **Qwen2-VL**: Alibaba Cloud
- **SAM 2**: Meta AI
- **SETR**: Fudan University
- **COCO-Stuff**: Stanford University

## 📞 Contact

- **Author**: Khanh Chien Ngo
- **Email**: khanhchien6@gmail.com
- **GitHub**: [@KhanhChien](https://github.com/khanhchien23)

## ⭐ Citation

```bibtex
@software{gda_visionassist,
  author = {Khanh Chien},
  title = {GDA-VisionAssist: Vision-Language System for Assisting Visually Impaired Individuals},
  year = {2025},
  url = {https://github.com/ChinhocIT/GDA-VisionAssist}
}
```
