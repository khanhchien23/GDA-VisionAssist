# GDA-VisionAssist

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Vision-Language system for assisting visually impaired individuals in object recognition and description.**

---

## ✨ Key Features

- 🎯 **SAM 2** — click-based object segmentation
- 🦖 **DINOv2 + SETR** — semantic segmentation (COCO-Stuff 171 classes)
- 🧠 **Qwen2-VL-2B** — vision-language understanding (8-bit quantized)
- 🌉 **Vision-Language Adaptor** — bridges visual features to language space (64 query tokens)
- 📝 **Vision Text Decoder** — generates contextual descriptions from masked features
- 🗣️ **Voice interaction** — hold `M` to speak, auto-answer on release (Vietnamese + English)
- 🔊 **TTS** — reads results aloud via edge-tts (`vi-VN-HoaiMyNeural`)
- 📖 **OCR** — reads text within segmented regions
- ⚡ **Real-time** — Desktop app (OpenCV) & Web UI (FastAPI + WebRTC)

---

## 🏗️ Architecture

```
                    ┌─────────────┐
                    │ Input Image │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                  │
         ▼                 ▼                  ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ DINOv2 ViT-B │  │ Qwen2-VL ViT │  │    SAM 2     │
  │  (frozen)    │  │  (frozen)    │  │ (click→mask) │
  └──────┬───────┘  └──────┬───────┘  └──────────────┘
         │                 │
         ▼                 ▼
  ┌──────────────┐  ┌──────────────┐
  │ SETR Decoder │  │   Adaptor    │
  │ (172 classes)│  │ (64 queries) │
  └──────┬───────┘  └──────┬───────┘
         │                 │
         └────────┬────────┘
                  ▼
       ┌────────────────────┐
       │ Prompt Constructor │
       │ (spatial+semantic) │
       └─────────┬──────────┘
                 ▼
       ┌────────────────────┐
       │   Qwen2-VL LLM    │
       │  (8-bit quantized) │
       └─────────┬──────────┘
                 ▼
          ┌────────────┐
          │  TTS / OCR │
          └────────────┘
```

| #   | Component                   | Role                                                      |
| --- | --------------------------- | --------------------------------------------------------- |
| 1   | **DINOv2 ViT-B/14**         | Feature extraction for SETR segmentation (768-dim)        |
| 2   | **Qwen2-VL ViT**            | Visual feature extraction for VL understanding (1536-dim) |
| 3   | **SETR Decoder (PUP)**      | Semantic class prediction (172 classes, COCO-Stuff)       |
| 4   | **MaskedFeatureExtractor**  | Focuses ViT features on SAM-segmented region              |
| 5   | **Vision-Language Adaptor** | 64 query tokens bridging vision → language space          |
| 6   | **Vision Text Decoder**     | Generates context description from vision tokens          |
| 7   | **SAM 2 Segmenter**         | Segments object from user click point                     |
| 8   | **Prompt Constructor**      | Builds prompts with spatial + semantic context            |
| 9   | **Qwen2-VL LLM**            | Generates natural-language answers                        |

**Training strategy:**
❄️ Frozen: Qwen2-VL ViT + SAM 2 + LLM + DINOv2 · 🔥 Trainable: SETR Decoder + Adaptor + TextDecoder

---

## 📋 Requirements

| Component  | Requirement                                     |
| ---------- | ----------------------------------------------- |
| **Python** | 3.8+                                            |
| **GPU**    | NVIDIA with CUDA 11.8+ (≥ 8GB VRAM recommended) |
| **RAM**    | 16GB+                                           |
| **OS**     | Windows / Linux / macOS                         |

---

## 🚀 Installation

```bash
git clone https://github.com/ChinhocIT/GDA-VisionAssist.git
cd GDA-VisionAssist

python -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows

pip install -r requirements.txt

python scripts/download_models.py     # download pretrained models
```

---

## 💡 Usage

GDA-VisionAssist supports **3 modes** of operation:

### Mode 1: Desktop App (all-in-one)

Webcam + model inference in a single process.

```bash
python app.py

# Advanced options
python app.py --seg-checkpoint checkpoints/setr_dino_best.pth \
              --adaptor-checkpoint checkpoints/adaptor_vizwiz/adaptor.pth
python app.py --debug
```

### Mode 2: API Server + Camera Client

Splits workload — server handles GPU inference, client handles webcam.

```bash
# Step 1: Start API server
python -m src.app.api_server --host 127.0.0.1 --port 8765

# Step 2: Run camera client
python camera_client.py
```

### Mode 3: Web UI (Browser)

Open `app_mockup.html` in a browser while the API server is running.

- 📷 Live camera via WebRTC
- 🖱️ Click to segment with SAM
- 🎙️ Hold `M` to ask by voice (Web Speech API)
- 🔊 TTS reads results aloud
- 📖 OCR button to read text
- 🎨 Dark theme with real-time pipeline visualization

### Keyboard Controls

| Key         | Desktop App        | Camera Client      | Web UI             |
| ----------- | ------------------ | ------------------ | ------------------ |
| `Click`     | Select point → SAM | Select point → SAM | Select point → SAM |
| `D`         | Describe region    | Describe region    | 🔍 Button          |
| `O`         | OCR                | OCR                | 📖 Button          |
| `M` (hold)  | —                  | —                  | Voice input        |
| `R`         | Reset mask         | Reset mask         | ↺ Button           |
| `T`         | —                  | —                  | Toggle TTS         |
| `Q` / `ESC` | Quit               | Quit               | —                  |

### Python API

```python
from src.core.gda import GlobalDescriptionAcquisition
import cv2

gda = GlobalDescriptionAcquisition(device="cuda")

image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = gda.sam_segmenter.segment_from_point(image_rgb, point=(320, 240))
result = gda.process_region(image_rgb, mask, user_query="What is this?")

print(result['description'])      # "This is a gray laptop..."
print(result['predicted_class'])   # "laptop"
print(result['confidence'])        # 0.85
```

---

## 🌐 REST API

| Endpoint        | Method | Input                      | Output                                                      |
| --------------- | ------ | -------------------------- | ----------------------------------------------------------- |
| `/health`       | GET    | —                          | `{ ok, model_loaded }`                                      |
| `/api/segment`  | POST   | `image` + `x, y`           | `{ mask_png_base64, sam_sec, mask_area_ratio }`             |
| `/api/describe` | POST   | `image` + `mask` + `query` | `{ description, predicted_class, confidence, latency_sec }` |
| `/api/ocr`      | POST   | `image` + `mask`           | `{ description, latency_sec }`                              |

---

## 📂 Project Structure

```
GDA-VisionAssist/
├── app.py                  # Desktop entry point
├── camera_client.py        # Camera client → API calls
├── app_mockup.html         # Web UI (browser)
├── src/
│   ├── core/               # GDA pipeline, prompt constructor
│   ├── models/             # DINOv2, SETR, Adaptor, SAM2, TextDecoder
│   ├── app/                # FastAPI server, desktop controller, config
│   ├── io/                 # Camera, keyboard, voice (STT+TTS)
│   └── utils/              # Logger, visualization, data helpers
├── config/                 # app_config.yaml, model_config.yaml
├── checkpoints/            # Trained SETR + Adaptor weights
├── scripts/                # Training, evaluation, benchmarking
├── tests/                  # Unit tests
└── examples/               # Usage examples
```

---

## 🎓 Training

```bash
# SETR Decoder (DINOv2 → COCO-Stuff 171 classes)
python scripts/train_setr.py --epochs 15 --batch-size 8 --lr 1e-4

# Vision-Language Adaptor (VizWiz dataset)
python scripts/train_adaptor_vizwiz_v2.py --epochs 20 --batch-size 4

# End-to-end evaluation
python scripts/evaluate_e2e.py
python scripts/benchmark.py
```

---

## 🙏 Acknowledgments

This project builds upon several outstanding open-source works:

| Model / Dataset          | Source                                                                   | Usage in GDA                               |
| ------------------------ | ------------------------------------------------------------------------ | ------------------------------------------ |
| **Qwen2-VL-2B-Instruct** | [Alibaba Cloud](https://github.com/QwenLM/Qwen2-VL)                      | Vision encoder + LLM backbone              |
| **SAM 2**                | [Meta AI (FAIR)](https://github.com/facebookresearch/segment-anything-2) | Click-based object segmentation            |
| **DINOv2 ViT-B/14**      | [Meta AI (FAIR)](https://github.com/facebookresearch/dinov2)             | Feature extraction for SETR decoder        |
| **SETR**                 | [Fudan University](https://github.com/fudan-zvg/SETR)                    | Semantic segmentation decoder architecture |
| **COCO-Stuff**           | [Stanford University](https://github.com/nightrome/cocostuff)            | 171-class segmentation training data       |
| **VizWiz**               | [VizWiz.org](https://vizwiz.org/)                                        | Visual question answering training data    |
| **edge-tts**             | [rany2/edge-tts](https://github.com/rany2/edge-tts)                      | Text-to-Speech engine                      |
| **SpeechRecognition**    | [Uberi](https://github.com/Uberi/speech_recognition)                     | Speech-to-Text via Google API              |

---

## 📞 Contact

**Khanh Chien Ngo** — [khanhchien6@gmail.com](mailto:khanhchien6@gmail.com) — [@KhanhChien](https://github.com/khanhchien23)

```bibtex
@software{gda_visionassist,
  author = {Khanh Chien Ngo},
  title = {GDA-VisionAssist: Vision-Language System for Assisting Visually Impaired Individuals},
  year = {2025},
  url = {https://github.com/ChinhocIT/GDA-VisionAssist}
}
```
