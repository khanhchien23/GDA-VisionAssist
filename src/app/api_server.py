# src/app/api_server.py
"""HTTP API phục vụ SAM + describe + OCR (cùng pipeline với GDAApplication)."""
import argparse
import base64
import os
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

# UTF-8 trên Windows console
if sys.platform == "win32":
    import codecs
    try:
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    except Exception:
        pass

_gda = None
_gda_lock = threading.Lock()
_load_error: Optional[str] = None


def _decode_upload_image(contents: bytes) -> np.ndarray:
    arr = np.frombuffer(contents, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Không đọc được ảnh (JPEG/PNG).")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _decode_mask_png(contents: bytes, shape_hw: Tuple[int, int]) -> np.ndarray:
    arr = np.frombuffer(contents, dtype=np.uint8)
    m = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError("Không đọc được mask PNG.")
    h, w = shape_hw
    if m.shape[0] != h or m.shape[1] != w:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def _mask_to_png_b64(mask: np.ndarray) -> str:
    m = (mask.astype(np.uint8) * 255)
    ok, buf = cv2.imencode(".png", m)
    if not ok:
        raise RuntimeError("encode mask PNG thất bại")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def get_gda(seg_checkpoint: str, adaptor_checkpoint: str, debug: bool):
    global _gda, _load_error
    with _gda_lock:
        if _gda is not None:
            return _gda
        if _load_error is not None:
            raise RuntimeError(_load_error)
        try:
            import torch
            from ..core.gda import GlobalDescriptionAcquisition
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🖥️  API device: {device}")
            _gda = GlobalDescriptionAcquisition(
                seg_checkpoint=seg_checkpoint or None,
                adaptor_checkpoint=adaptor_checkpoint or None,
                device=device,
                debug=debug,
            )
            return _gda
        except Exception as e:
            _load_error = str(e)
            raise


def create_app(seg_checkpoint: str, adaptor_checkpoint: str, debug: bool = False):
    from fastapi import FastAPI, HTTPException, UploadFile, File, Form
    from fastapi.middleware.cors import CORSMiddleware
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(title="GDA VisionAssist API", version="1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"ok": True, "model_loaded": _gda is not None}

    # ── /api/segment ──────────────────────────────────────────────────────────
    def _sync_segment(raw: bytes, x: int, y: int):
        frame_rgb = _decode_upload_image(raw)
        h, w = frame_rgb.shape[0], frame_rgb.shape[1]
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError(f"Tọa độ (x,y)=({x},{y}) ngoài ảnh {w}x{h}")
        gda = get_gda(seg_checkpoint, adaptor_checkpoint, debug)
        t0 = time.time()
        with _gda_lock:
            mask = gda.sam_segmenter.segment_from_point(
                frame_rgb, (x, y), use_iterative=False
            )
        elapsed = time.time() - t0
        if mask is None or mask.sum() == 0:
            return {
                "ok": False,
                "message": "SAM không tạo được vùng — thử điểm khác.",
                "sam_sec": elapsed,
                "width": w,
                "height": h,
            }
        return {
            "ok": True,
            "mask_png_base64": _mask_to_png_b64(mask),
            "sam_sec": elapsed,
            "width": w,
            "height": h,
            "mask_area_ratio": float(mask.sum()) / float(h * w),
        }

    @app.post("/api/segment", response_model=None)
    async def api_segment(
        image: UploadFile = File(...),
        x: int = Form(0),
        y: int = Form(0),
    ):
        try:
            raw = await image.read()
            return await run_in_threadpool(_sync_segment, raw, x, y)
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ── /api/describe ─────────────────────────────────────────────────────────
    def _sync_describe(raw_img: bytes, raw_mask: bytes, query: str):
        frame_rgb = _decode_upload_image(raw_img)
        h, w = frame_rgb.shape[0], frame_rgb.shape[1]
        m = _decode_mask_png(raw_mask, (h, w))
        gda = get_gda(seg_checkpoint, adaptor_checkpoint, debug)
        t0 = time.time()
        with _gda_lock:
            result = gda.process_region(frame_rgb, m, query or None)
        elapsed = time.time() - t0
        return _normalize_result(result, "describe", query or "Mô tả", elapsed)

    @app.post("/api/describe", response_model=None)
    async def api_describe(
        image: UploadFile = File(...),
        mask: UploadFile = File(...),
        query: str = Form(""),
    ):
        try:
            raw_img = await image.read()
            raw_mask = await mask.read()
            return await run_in_threadpool(_sync_describe, raw_img, raw_mask, query)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ── /api/ocr ──────────────────────────────────────────────────────────────
    def _sync_ocr(raw_img: bytes, raw_mask: bytes):
        frame_rgb = _decode_upload_image(raw_img)
        h, w = frame_rgb.shape[0], frame_rgb.shape[1]
        m = _decode_mask_png(raw_mask, (h, w))
        gda = get_gda(seg_checkpoint, adaptor_checkpoint, debug)
        t0 = time.time()
        with _gda_lock:
            result = gda.ocr_region(frame_rgb, m)
        elapsed = time.time() - t0
        return _normalize_result(result, "ocr", "OCR", elapsed)

    @app.post("/api/ocr", response_model=None)
    async def api_ocr(
        image: UploadFile = File(...),
        mask: UploadFile = File(...),
    ):
        try:
            raw_img = await image.read()
            raw_mask = await mask.read()
            return await run_in_threadpool(_sync_ocr, raw_img, raw_mask)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


def _normalize_result(result: Any, task_type: str, query: str, elapsed: float) -> Dict[str, Any]:
    if isinstance(result, dict):
        d = dict(result)
        d.setdefault("task_type", task_type)
        d.setdefault("query", query)
        d.setdefault("latency_sec", elapsed)
        d.setdefault("error", False)
        return d
    return {
        "description": str(result),
        "error": False,
        "predicted_class": None,
        "confidence": 0.0,
        "query": query,
        "latency_sec": elapsed,
        "task_type": task_type,
    }


def main():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_seg = os.path.join(base, "checkpoints", "setr_dino_best.pth")
    default_adaptor = os.path.join(base, "checkpoints", "adaptor_vizwiz", "adaptor.pth")

    parser = argparse.ArgumentParser(description="GDA API server (FastAPI)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seg-checkpoint", default=default_seg)
    parser.add_argument("--adaptor-checkpoint", default=default_adaptor)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("Cài thêm: pip install -r requirements-api.txt")
        sys.exit(1)

    app = create_app(
        seg_checkpoint=args.seg_checkpoint,
        adaptor_checkpoint=args.adaptor_checkpoint,
        debug=args.debug,
    )
    print(f"🚀 API: http://{args.host}:{args.port}")
    print("   POST /api/segment  (image + x,y)")
    print("   POST /api/describe (image + mask PNG + query)")
    print("   POST /api/ocr      (image + mask PNG)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()