# GDA-VisionAssist - API Reference

## Core API

### `GlobalDescriptionAcquisition`

Lớp chính điều phối toàn bộ hệ thống GDA.

```python
from src.core.gda import GlobalDescriptionAcquisition

gda = GlobalDescriptionAcquisition(
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    seg_checkpoint="checkpoints/setr_dino_best.pth",
    adaptor_checkpoint="checkpoints/adaptor_vizwiz/adaptor.pth",
    device="cuda",
    debug=False
)
```

#### Parameters

| Parameter            | Type | Default                     | Description                       |
| -------------------- | ---- | --------------------------- | --------------------------------- |
| `model_name`         | str  | `Qwen/Qwen2-VL-2B-Instruct` | Tên model Qwen2-VL                |
| `seg_checkpoint`     | str  | None                        | Đường dẫn checkpoint SETR decoder |
| `adaptor_checkpoint` | str  | None                        | Đường dẫn checkpoint adaptor      |
| `device`             | str  | `cuda`                      | Device (`cuda` hoặc `cpu`)        |
| `debug`              | bool | False                       | Bật debug mode                    |

#### Methods

##### `process_region(image_rgb, mask, user_query=None)`

Xử lý vùng đã chọn và trả về mô tả.

```python
result = gda.process_region(
    image_rgb=image_rgb,      # np.ndarray (H, W, 3) RGB
    mask=mask,                 # np.ndarray (H, W) binary mask
    user_query="Đây là gì?"   # Optional[str]
)
```

**Returns**: `Dict` với các key:

- `description` (str): Mô tả vật thể
- `predicted_class` (str | None): Class dự đoán từ SETR
- `confidence` (float): Độ tin cậy (0-1)
- `error` (bool): Có lỗi không
- `query` (str): Câu hỏi đầu vào

##### `predict_class_from_region(image_rgb, mask, image_shape)`

Dự đoán class cho vùng đã chọn dùng DINOv2 + SETR.

```python
predicted_class, confidence = gda.predict_class_from_region(
    image_rgb, mask, image_shape=(480, 640)
)
```

---

## Models API

### `SETRSegDecoder`

```python
from src.models.segmentation import SETRSegDecoder

decoder = SETRSegDecoder(
    vit_features_dim=768,   # DINOv2-B dim
    num_classes=172,          # COCO-Stuff + background
    decoder_type='pup',       # 'naive', 'pup', 'mla'
    device="cuda"
)

# Forward
seg_map = decoder(vit_features, target_size=(480, 640))
# seg_map: (B, 172, H, W)
```

### `ImprovedVisionLanguageAdaptor`

```python
from src.models.adaptor import ImprovedVisionLanguageAdaptor

adaptor = ImprovedVisionLanguageAdaptor(
    vision_dim=1536,
    llm_dim=1536,
    num_query_tokens=64
)

# Forward
vision_tokens = adaptor(vit_features)  # (B, 64, 1536)
```

### `SAM2Segmenter`

```python
from src.models.sam_segmenter import SAM2Segmenter

sam = SAM2Segmenter(device="cuda")

# Segment from point click
mask = sam.segment_from_point(
    image_rgb,                    # np.ndarray (H, W, 3)
    point=(320, 240),             # (x, y) tuple
    use_iterative=False
)
# mask: np.ndarray (H, W) binary
```

### `DINOv2Encoder`

```python
from src.models.dinov2_encoder import DINOv2Encoder

encoder = DINOv2Encoder(variant='vitb14', device='cuda')
features = encoder.extract_features(image_rgb)
# features: (1, N, 768)
```

### `VisionTextDecoder`

```python
from src.models.text_decoder import VisionTextDecoder

decoder = VisionTextDecoder(
    vision_dim=1536,
    hidden_dim=512,
    vocab_size=151936,
    num_decoder_layers=4,
    max_length=32
)
decoder.set_tokenizer(tokenizer)
```

---

## Application API

### `GDAApplication`

```python
from src.app import GDAApplication

app = GDAApplication(
    seg_checkpoint="checkpoints/setr_dino_best.pth",
    adaptor_checkpoint="checkpoints/adaptor_vizwiz/adaptor.pth",
    debug=False
)
app.run()
```

### `InferenceManager`

```python
from src.app import InferenceManager

manager = InferenceManager(gda_system, maxsize=1)
manager.start()
manager.submit(frame_rgb, mask, user_query)

is_processing, progress, result = manager.get_status()
result = manager.consume_result()

manager.stop()
```

---

## I/O API

### Voice Module

```python
from src.io import voice

# Initialize STT & TTS
stt_ready = voice.init_stt()
tts_ready = voice.init_tts()

# Start TTS worker
import threading
tts_thread = threading.Thread(target=voice.tts_worker, daemon=True)
tts_thread.start()

# Speak text
voice.speech_queue.put("Xin chào!")

# Stop TTS
voice.speech_queue.put(None)
```

### Prompt Constructor

```python
from src.core.prompt import PromptConstructor

pc = PromptConstructor()
prompt = pc.construct_prompt(
    mask=mask,
    predicted_class="laptop",
    user_query="Đây là gì?"
)
```
