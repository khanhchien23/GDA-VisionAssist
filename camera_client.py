# camera_client.py
"""
GDA VisionAssist — Camera client thời gian thực (OpenCV).

Luồng:
  1. Mở webcam → hiển thị live feed
  2. Click chuột trái vào vật → gửi frame + tọa độ lên /api/segment → hiện mask xanh
  3. Nhấn [D] → gửi frame + mask lên /api/describe (hỏi mô tả)
  4. Nhấn [O] → gửi frame + mask lên /api/ocr (đọc chữ)
  5. Nhấn [R] → xóa mask, chọn vật khác
  6. Nhấn [Q] hoặc ESC → thoát

Phím tắt:
  Click trái  — chọn điểm, gọi SAM
  D           — Describe (mô tả vật)
  O           — OCR (đọc chữ)
  R           — Reset mask
  Q / ESC     — Thoát

Cài thêm nếu chưa có:
  pip install requests opencv-python numpy
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np
import requests

# ──────────────────────────────────────────────
# CẤU HÌNH
# ──────────────────────────────────────────────
API_BASE = "http://127.0.0.1:8765"   # Địa chỉ server FastAPI
CAMERA_INDEX = 0                      # 0 = webcam mặc định
WINDOW_NAME = "GDA VisionAssist  [Click=SAM | D=Mô tả | O=OCR | R=Reset | Q=Thoát]"

# Màu overlay mask (BGR)
MASK_COLOR = (0, 220, 80)   # xanh lá
MASK_ALPHA = 0.35           # độ trong suốt

# Màu UI text
COLOR_INFO   = (220, 220, 220)
COLOR_OK     = (80, 220, 80)
COLOR_ERROR  = (60, 80, 255)
COLOR_BUSY   = (220, 180, 40)
COLOR_POINT  = (60, 200, 255)


# ──────────────────────────────────────────────
# TRẠNG THÁI TOÀN CỤC (thread-safe qua lock)
# ──────────────────────────────────────────────
class State:
    def __init__(self):
        self._lock = threading.Lock()
        self.mask: Optional[np.ndarray] = None   # uint8 0/1, HxW
        self.click_point: Optional[tuple] = None  # (x, y) pixel trên frame
        self.status_msg: str = "Click vào vật để phân vùng (SAM)"
        self.status_color = COLOR_INFO
        self.busy: bool = False
        self.last_result: str = ""

    # --- getters / setters an toàn ---
    def set_mask(self, mask):
        with self._lock:
            self.mask = mask

    def get_mask(self):
        with self._lock:
            return self.mask

    def set_status(self, msg: str, color=COLOR_INFO):
        with self._lock:
            self.status_msg = msg
            self.status_color = color

    def get_status(self):
        with self._lock:
            return self.status_msg, self.status_color

    def set_busy(self, v: bool):
        with self._lock:
            self.busy = v

    def is_busy(self):
        with self._lock:
            return self.busy

    def set_click(self, pt):
        with self._lock:
            self.click_point = pt

    def get_click(self):
        with self._lock:
            return self.click_point

    def set_result(self, text: str):
        with self._lock:
            self.last_result = text

    def get_result(self):
        with self._lock:
            return self.last_result

    def reset(self):
        with self._lock:
            self.mask = None
            self.click_point = None
            self.last_result = ""
            self.status_msg = "Click vào vật để phân vùng (SAM)"
            self.status_color = COLOR_INFO
            self.busy = False


state = State()


# ──────────────────────────────────────────────
# HELPER: encode frame thành bytes JPEG/PNG
# ──────────────────────────────────────────────
def frame_to_bytes(frame_bgr: np.ndarray, ext: str = ".jpg") -> bytes:
    ok, buf = cv2.imencode(ext, frame_bgr)
    if not ok:
        raise RuntimeError("Không encode được frame")
    return buf.tobytes()


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    m = (mask.astype(np.uint8) * 255)
    ok, buf = cv2.imencode(".png", m)
    if not ok:
        raise RuntimeError("Không encode được mask")
    return buf.tobytes()


# ──────────────────────────────────────────────
# API CALLS (chạy trên thread riêng)
# ──────────────────────────────────────────────
def call_segment(frame_bgr: np.ndarray, x: int, y: int):
    """Gọi /api/segment, cập nhật state.mask."""
    state.set_busy(True)
    state.set_status(f"⏳ SAM đang phân vùng tại ({x},{y})...", COLOR_BUSY)
    try:
        img_bytes = frame_to_bytes(frame_bgr, ".jpg")
        resp = requests.post(
            f"{API_BASE}/api/segment",
            files={"image": ("frame.jpg", img_bytes, "image/jpeg")},
            data={"x": str(x), "y": str(y)},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("ok"):
            state.set_mask(None)
            state.set_status("⚠️ " + data.get("message", "SAM không tạo được mask"), COLOR_ERROR)
            return

        # Giải mã mask PNG base64
        import base64
        raw = base64.b64decode(data["mask_png_base64"])
        arr = np.frombuffer(raw, dtype=np.uint8)
        m = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        mask_bin = (m > 127).astype(np.uint8)
        state.set_mask(mask_bin)
        ratio = data.get("mask_area_ratio", 0) * 100
        elapsed = data.get("sam_sec", 0)
        state.set_status(
            f"✅ SAM xong ({elapsed:.2f}s) | Vùng: {ratio:.1f}%  →  [D] Mô tả  [O] Đọc chữ",
            COLOR_OK,
        )

    except requests.exceptions.ConnectionError:
        state.set_status("❌ Không kết nối được server — chạy python -m src.app.api_server", COLOR_ERROR)
        state.set_mask(None)
    except Exception as e:
        state.set_status(f"❌ SAM lỗi: {e}", COLOR_ERROR)
        state.set_mask(None)
    finally:
        state.set_busy(False)


def call_describe(frame_bgr: np.ndarray, mask: np.ndarray, query: str = ""):
    """Gọi /api/describe."""
    state.set_busy(True)
    state.set_status("⏳ Đang mô tả vật...", COLOR_BUSY)
    try:
        img_bytes = frame_to_bytes(frame_bgr, ".jpg")
        msk_bytes = mask_to_png_bytes(mask)
        resp = requests.post(
            f"{API_BASE}/api/describe",
            files={
                "image": ("frame.jpg", img_bytes, "image/jpeg"),
                "mask":  ("mask.png",  msk_bytes, "image/png"),
            },
            data={"query": query},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("description", "(Không có mô tả)")
        elapsed = data.get("latency_sec", 0)
        state.set_result(text)
        state.set_status(f"✅ Mô tả ({elapsed:.1f}s): {_truncate(text, 80)}", COLOR_OK)
        _print_result("MÔ TẢ", text)

    except requests.exceptions.ConnectionError:
        state.set_status("❌ Không kết nối được server", COLOR_ERROR)
    except Exception as e:
        state.set_status(f"❌ Describe lỗi: {e}", COLOR_ERROR)
    finally:
        state.set_busy(False)


def call_ocr(frame_bgr: np.ndarray, mask: np.ndarray):
    """Gọi /api/ocr."""
    state.set_busy(True)
    state.set_status("⏳ Đang đọc chữ (OCR)...", COLOR_BUSY)
    try:
        img_bytes = frame_to_bytes(frame_bgr, ".jpg")
        msk_bytes = mask_to_png_bytes(mask)
        resp = requests.post(
            f"{API_BASE}/api/ocr",
            files={
                "image": ("frame.jpg", img_bytes, "image/jpeg"),
                "mask":  ("mask.png",  msk_bytes, "image/png"),
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("description", "(Không có chữ)")
        elapsed = data.get("latency_sec", 0)
        state.set_result(text)
        state.set_status(f"✅ OCR ({elapsed:.1f}s): {_truncate(text, 80)}", COLOR_OK)
        _print_result("OCR", text)

    except requests.exceptions.ConnectionError:
        state.set_status("❌ Không kết nối được server", COLOR_ERROR)
    except Exception as e:
        state.set_status(f"❌ OCR lỗi: {e}", COLOR_ERROR)
    finally:
        state.set_busy(False)


# ──────────────────────────────────────────────
# HELPERS HIỂN THỊ
# ──────────────────────────────────────────────
def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "..."


def _print_result(label: str, text: str):
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(text)
    print('='*60)


def draw_overlay(frame_bgr: np.ndarray) -> np.ndarray:
    """Vẽ mask + UI lên frame, trả về frame đã vẽ."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    # 1) Mask overlay
    mask = state.get_mask()
    if mask is not None:
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        color_layer = np.zeros_like(out)
        color_layer[mask == 1] = MASK_COLOR
        out = cv2.addWeighted(out, 1.0, color_layer, MASK_ALPHA, 0)
        # Vẽ contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, MASK_COLOR, 2)

    # 2) Điểm click
    pt = state.get_click()
    if pt is not None:
        cv2.circle(out, pt, 8, COLOR_POINT, -1)
        cv2.circle(out, pt, 12, COLOR_POINT, 2)

    # 3) Status bar (nền mờ phía dưới)
    msg, color = state.get_status()
    bar_h = 36
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (15, 15, 15), -1)
    out = cv2.addWeighted(out, 0.35, overlay, 0.65, 0)
    cv2.putText(out, msg, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # 4) Busy spinner (góc trên phải)
    if state.is_busy():
        t = int(time.time() * 4) % 4
        dots = "." * (t + 1)
        cv2.putText(out, f"Đang xử lý{dots}", (w - 180, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_BUSY, 1, cv2.LINE_AA)

    # 5) Kết quả nhiều dòng (góc trên trái, tối đa 5 dòng)
    result = state.get_result()
    if result:
        lines = []
        for raw_line in result.split("\n"):
            # wrap dòng dài
            while len(raw_line) > 60:
                lines.append(raw_line[:60])
                raw_line = raw_line[60:]
            lines.append(raw_line)
        lines = lines[:6]
        bg = out.copy()
        cv2.rectangle(bg, (0, 0), (w, 14 + 22 * len(lines)), (10, 10, 10), -1)
        out = cv2.addWeighted(out, 0.25, bg, 0.75, 0)
        for i, line in enumerate(lines):
            cv2.putText(out, line, (8, 18 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_OK, 1, cv2.LINE_AA)

    return out


# ──────────────────────────────────────────────
# MOUSE CALLBACK
# ──────────────────────────────────────────────
_current_frame: Optional[np.ndarray] = None   # frame mới nhất để SAM


def on_mouse(event, x, y, flags, param):
    global _current_frame
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if state.is_busy():
        return
    frame = _current_frame
    if frame is None:
        return
    state.set_click((x, y))
    state.set_mask(None)
    state.set_result("")
    t = threading.Thread(target=call_segment, args=(frame.copy(), x, y), daemon=True)
    t.start()


# ──────────────────────────────────────────────
# VÒNG LẶP CHÍNH
# ──────────────────────────────────────────────
def main():
    global _current_frame

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[LỖI] Không mở được camera index={CAMERA_INDEX}")
        sys.exit(1)

    # Tăng buffer nhỏ để giảm lag
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 900, 600)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print("="*60)
    print("GDA VisionAssist — Camera Client")
    print(f"  API: {API_BASE}")
    print("  Click trái  → SAM phân vùng")
    print("  [D]         → Mô tả vật")
    print("  [O]         → Đọc chữ (OCR)")
    print("  [R]         → Reset mask")
    print("  [Q] / ESC   → Thoát")
    print("="*60)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[CẢNH BÁO] Không đọc được frame, thử lại...")
            time.sleep(0.05)
            continue

        _current_frame = frame  # lưu frame mới nhất cho mouse callback

        display = draw_overlay(frame)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):   # Q hoặc ESC
            break

        elif key in (ord('r'), ord('R')):      # Reset
            state.reset()

        elif key in (ord('d'), ord('D')):      # Describe
            mask = state.get_mask()
            if mask is None:
                state.set_status("⚠️ Chưa có mask — hãy click vào vật trước", COLOR_ERROR)
            elif state.is_busy():
                state.set_status("⏳ Đang bận, thử lại sau...", COLOR_BUSY)
            else:
                t = threading.Thread(
                    target=call_describe,
                    args=(frame.copy(), mask.copy(), ""),
                    daemon=True,
                )
                t.start()

        elif key in (ord('o'), ord('O')):      # OCR
            mask = state.get_mask()
            if mask is None:
                state.set_status("⚠️ Chưa có mask — hãy click vào vật trước", COLOR_ERROR)
            elif state.is_busy():
                state.set_status("⏳ Đang bận, thử lại sau...", COLOR_BUSY)
            else:
                t = threading.Thread(
                    target=call_ocr,
                    args=(frame.copy(), mask.copy()),
                    daemon=True,
                )
                t.start()

    cap.release()
    cv2.destroyAllWindows()
    print("Đã thoát.")


if __name__ == "__main__":
    main()