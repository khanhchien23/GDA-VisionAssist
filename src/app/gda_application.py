# src/app/gda_application.py
"""Main GDA Application class"""

import torch
import cv2
import numpy as np
import threading
import queue
import time
import sys
from datetime import datetime
from typing import Optional

# UTF-8 encoding for Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from .config import AppConfig
from .inference_manager import InferenceManager
from .ui_renderer import UIRenderer

from ..core.gda import GlobalDescriptionAcquisition
from ..io import voice
from ..io.keyboard import ClickHandler, KeyboardMonitor


class GDAApplication:
    """
    Main GDA Application class.
    Handles initialization, main loop, and event handling.
    """
    
    def __init__(self, 
                 seg_checkpoint: Optional[str] = None,
                 adaptor_checkpoint: Optional[str] = None,
                 debug: bool = False,
                 config: Optional[AppConfig] = None):
        """
        Initialize the GDA Application.
        
        Args:
            seg_checkpoint: Path to segmentation decoder checkpoint
            adaptor_checkpoint: Path to adaptor checkpoint
            debug: Enable debug mode
            config: Optional custom configuration
        """
        self.config = config or AppConfig()
        self.debug = debug
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Thiết bị: {self.device}\n")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        
        # Initialize GDA core system
        self.gda = GlobalDescriptionAcquisition(
            seg_checkpoint=seg_checkpoint or self.config.default_seg_checkpoint,
            adaptor_checkpoint=adaptor_checkpoint or self.config.default_adaptor_checkpoint,
            device=self.device,
            debug=debug
        )
        
        # Initialize components
        self.inference_manager = InferenceManager(self.gda, maxsize=1)
        self.ui_renderer = UIRenderer(self.config)
        
        # Initialize voice
        self.stt_available = voice.init_stt()
        self.tts_available = voice.init_tts()
        self.tts_thread: Optional[threading.Thread] = None
        
        if self.tts_available:
            self.tts_thread = threading.Thread(target=voice.tts_worker, daemon=True)
            self.tts_thread.start()
        
        # Keyboard monitor
        self.kb_monitor = KeyboardMonitor()
        
        # Click handler
        self.click_handler = ClickHandler()
        
        # State
        self.current_mask: Optional[np.ndarray] = None
        self.last_result: Optional[dict] = None
        self.frame_count = 0
        self._running = False
        self.cap: Optional[cv2.VideoCapture] = None
    
    def run(self):
        """Main application loop"""
        self._print_welcome()
        
        # Open webcam
        self.cap = cv2.VideoCapture(self.config.webcam.device_id)
        if not self.cap.isOpened():
            print("❌ Không mở được webcam")
            return
        
        # Configure webcam
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.webcam.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.webcam.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.webcam.fps)
        
        # Start components
        self.inference_manager.start()
        self.kb_monitor.start()
        
        # Setup window
        window_title = self.config.strings.WINDOW_TITLE
        cv2.namedWindow(window_title)
        cv2.setMouseCallback(window_title, self.click_handler.mouse_callback)
        
        self._running = True
        
        try:
            self._main_loop(window_title)
        except KeyboardInterrupt:
            print("\n⚠️ Dừng...")
        finally:
            self._cleanup()
    
    def _main_loop(self, window_title: str):
        """Main event loop"""
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_frame = frame.copy()
            
            # Draw mask overlay
            if self.current_mask is not None:
                display_frame = self.ui_renderer.draw_mask_overlay(
                    display_frame, self.current_mask
                )
            
            # Get inference status
            is_processing, progress, _ = self.inference_manager.get_status()
            
            # Draw status bar
            display_frame = self.ui_renderer.draw_status_bar(
                display_frame,
                is_processing=is_processing,
                progress=progress,
                waiting_for_click=self.click_handler.waiting_for_click,
                has_mask=self.current_mask is not None,
                debug=self.debug
            )
            
            # Draw result if available
            if self.last_result and not self.last_result.get('error'):
                display_frame = self.ui_renderer.draw_result(
                    display_frame, self.last_result, self.debug
                )
            
            # Check for new results
            self._check_inference_result()
            
            # Display
            cv2.imshow(window_title, display_frame)
            
            # Memory cleanup periodically
            self._periodic_cleanup()
            
            time.sleep(0.033)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            self._handle_key(key, frame_rgb)
            
            # Handle voice input
            if self.current_mask is not None and self.stt_available:
                if self.kb_monitor.c_pressed:
                    self._handle_voice_recording(frame_rgb)
            
            # Handle click segmentation
            self._handle_click_segmentation(frame_rgb)
    
    def _handle_key(self, key: int, frame_rgb: np.ndarray):
        """Handle keyboard input"""
        keys = self.config.keys
        
        if key == keys.QUIT:
            print("\n👋 Đang thoát...")
            self._running = False
        
        elif key == keys.SAVE:
            filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            print(f"\n💾 Đã lưu: {filename}\n")
        
        elif key == keys.DEBUG:
            self.debug = not self.debug
            self.gda.debug = self.debug
            status = "BẬT" if self.debug else "TẮT"
            print(f"\n🐛 Debug mode: {status}\n")
        
        elif key == keys.SPACE:
            self.click_handler.waiting_for_click = True
            self.click_handler.clicked_point = None
            self.current_mask = None
            self.last_result = None
            print("\n🎯 Chế độ chọn vùng - Click vào vật thể...")
        
        elif key == keys.ENTER and self.current_mask is not None:
            self._submit_inference(frame_rgb, None)
    
    def _submit_inference(self, frame_rgb: np.ndarray, 
                          user_query: Optional[str]):
        """Submit inference task"""
        if self.inference_manager.is_processing():
            print("⚠️ Đang xử lý, vui lòng đợi...")
            return False
        
        print("\n🚀 Đang mô tả (background)...")
        
        if self.inference_manager.submit(frame_rgb, self.current_mask, user_query):
            self.current_mask = None
            return True
        else:
            print("⚠️ Queue đầy")
            return False
    
    def _handle_voice_recording(self, frame_rgb: np.ndarray):
        """Handle voice recording when C is pressed"""
        import speech_recognition as sr
        
        print("\n🎤 GHI ÂM - THẢ C ĐỂ DỪNG...")
        
        saved_mask = self.current_mask.copy()
        saved_frame_rgb = frame_rgb.copy()
        
        audio_data = None
        recording_done = threading.Event()
        recording_error = [None]
        c_released = threading.Event()
        
        def capture_audio():
            nonlocal audio_data
            
            if not voice.microphone_lock.acquire(blocking=False):
                print("[🔒] Mic busy")
                recording_error[0] = "Mic in use"
                recording_done.set()
                return
            
            try:
                with voice.microphone as source:
                    try:
                        print("[🎤] Listening...")
                        start = time.time()
                        
                        while time.time() - start < self.config.inference.voice_timeout:
                            if c_released.is_set():
                                print("[👆] Stopped by user")
                                audio_data = None
                                return
                            
                            try:
                                audio_data = voice.recognizer.listen(
                                    source, timeout=0.3, phrase_time_limit=3
                                )
                                
                                if audio_data and hasattr(audio_data, 'frame_data'):
                                    size = len(audio_data.frame_data)
                                    if size > self.config.inference.min_audio_bytes:
                                        print(f"[✓] Got {size}B")
                                        return
                                    else:
                                        audio_data = None
                                        continue
                                        
                            except sr.WaitTimeoutError:
                                continue
                        
                        if audio_data is None:
                            print("[⏰] No speech")
                            
                    except Exception as listen_err:
                        print(f"[❌] {listen_err}")
                        recording_error[0] = str(listen_err)
                        
            finally:
                voice.microphone_lock.release()
                recording_done.set()
        
        # Start recording thread
        rec_thread = threading.Thread(target=capture_audio, daemon=True)
        rec_thread.start()
        
        rec_start = time.time()
        
        # Recording UI loop
        while True:
            elapsed = time.time() - rec_start
            
            if not self.kb_monitor.c_pressed:
                print(f"[👆] C released at {elapsed:.1f}s")
                c_released.set()
                time.sleep(0.15)
                break
            
            if recording_done.is_set():
                print("[✓] Recording complete")
                break
            
            if elapsed > self.config.inference.max_recording_time:
                print("[⏰] Max time reached")
                c_released.set()
                break
            
            key_check = cv2.waitKey(20) & 0xFF
            if key_check == self.config.keys.QUIT:
                c_released.set()
                break
            
            # Update display with recording indicator
            ret2, frame2 = self.cap.read()
            if ret2:
                display = frame2.copy()
                if saved_mask is not None:
                    display = self.ui_renderer.draw_mask_overlay(display, saved_mask)
                display = self.ui_renderer.draw_recording_indicator(
                    display, elapsed, self.kb_monitor.c_pressed
                )
                cv2.imshow(self.config.strings.WINDOW_TITLE, display)
        
        # Cleanup
        print("⏹️ Cleaning up...")
        c_released.set()
        recording_done.wait(timeout=0.5)
        rec_thread.join(timeout=0.3)
        
        # Process audio
        user_query = self._process_audio(audio_data, recording_error[0])
        
        # Submit if we got a query
        if user_query:
            print(f"📤 Đang gửi câu hỏi...")
            if not self.inference_manager.is_processing():
                if self.inference_manager.submit(saved_frame_rgb, saved_mask, user_query):
                    self.current_mask = None
                    print("✅ Đã gửi - đang xử lý...")
                else:
                    print("⚠️ Queue đầy")
            else:
                print("⚠️ Hệ thống đang bận")
        else:
            print("💡 Không có câu hỏi - mask vẫn còn")
            self.current_mask = saved_mask
    
    def _process_audio(self, audio_data, error) -> Optional[str]:
        """Process recorded audio and return transcribed text"""
        import speech_recognition as sr
        
        if error:
            print(f"⚠️ Lỗi: {error}")
            return None
        
        if audio_data is None:
            print("⚠️ Không ghi được âm thanh")
            print("   💡 Tips:")
            print("   - NÓI NGAY SAU KHI NHẤN C")
            print("   - GIỮ C trong khi nói (2-3s)")
            print("   - THẢ C SAU KHI NÓI XONG")
            print("   - Nói TO HƠN (90%+ volume)")
            return None
        
        try:
            print("🔊 Đang nhận dạng...")
            
            text = None
            try:
                text = voice.recognizer.recognize_google(audio_data, language="vi-VN")
            except:
                try:
                    text = voice.recognizer.recognize_google(audio_data, language="en-US")
                except:
                    pass
            
            if text and text.strip():
                user_query = text.strip()
                print(f"✅ Nhận được: \"{user_query}\"")
                return user_query
            else:
                print("⚠️ Không nhận dạng được")
                return None
                
        except sr.UnknownValueError:
            print("⚠️ Không hiểu được - nói rõ hơn")
            return None
        except sr.RequestError as e:
            print(f"❌ Lỗi API: {e}")
            return None
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            return None
    
    def _handle_click_segmentation(self, frame_rgb: np.ndarray):
        """Handle click-based segmentation"""
        if self.click_handler.clicked_point is None or self.current_mask is not None:
            return
        
        print("🔍 Đang phân đoạn với SAM 2...")
        
        point = self.click_handler.clicked_point
        mask = self.gda.sam_segmenter.segment_from_point(
            frame_rgb, point, use_iterative=False
        )
        
        print(" ✓")
        
        if mask is not None and mask.sum() > 0:
            self.current_mask = mask
            area_pixels = mask.sum()
            area_percent = (area_pixels / (mask.shape[0] * mask.shape[1])) * 100
            print(f"✅ Vùng: {area_pixels} px ({area_percent:.1f}%)")
            print("💬 Nhấn Enter hoặc giữ C để hỏi\n")
        else:
            print("⚠️ Thử lại")
        
        self.click_handler.waiting_for_click = False
        self.click_handler.clicked_point = None
    
    def _check_inference_result(self):
        """Check and process inference result"""
        result = self.inference_manager.consume_result()
        
        if result is not None:
            self.last_result = result
            
            print("\n" + "─"*70)
            print("📊 KẾT QUẢ:")
            
            if result.get('predicted_class'):
                conf = result.get('confidence', 0)
                print(f"  🏷️  Class: {result['predicted_class']} ({conf:.0%})")
            
            print(f"  🎯 Query: {result.get('query', 'N/A')}")
            print(f"  💬 Answer: {result.get('description', 'N/A')}")
            
            latency = result.get('latency_sec')
            if latency is not None:
                print(f"  ⏱️ Thời gian phản hồi: {latency:.2f} giây")
            
            if self.debug:
                if result.get('vit_features_shape'):
                    print(f"  🔍 ViT: {result['vit_features_shape']}")
                if result.get('vision_tokens_shape'):
                    print(f"  🔍 Tokens: {result['vision_tokens_shape']}")
            
            print("─"*70 + "\n")
            
            # Speak result via TTS
            if self.tts_available and not result.get('error'):
                desc = result.get('description')
                if desc:
                    voice.speech_queue.put(desc)
    
    def _periodic_cleanup(self):
        """Periodic memory cleanup"""
        self.frame_count += 1
        
        if self.frame_count % self.config.frame_cleanup_interval == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _print_welcome(self):
        """Print welcome message"""
        print("\n" + "="*70)
        print("🎬 HỆ THỐNG GDA - REFACTORED VERSION")
        print("="*70)
        print("📋 Cách dùng:")
        print("  1. Nhấn SPACE → Kích hoạt chế độ chọn vùng")
        print("  2. CLICK vào vật thể muốn hỏi")
        print("  3. SAM 2 sẽ phân đoạn vật thể")
        print("  4. GIỮ 'C' và nói câu hỏi (hoặc Enter để mô tả tự động)")
        print("  5. THẢ 'C' để xử lý")
        print("\n💡 Ví dụ câu hỏi:")
        print("  ✓ 'Đây là gì?'")
        print("  ✓ 'Vật này màu gì?'")
        print("  ✓ 'Mô tả vật này'")
        print("\n🔑 Phím tắt:")
        print("  - Space: Chọn vùng")
        print("  - C + Voice: Hỏi câu hỏi")
        print("  - Enter: Mô tả tự động")
        print("  - S: Lưu ảnh")
        print("  - D: Toggle debug mode")
        print("  - Q: Thoát")
        print("="*70 + "\n")
    
    def _cleanup(self):
        """Cleanup resources"""
        print("\n🛑 Shutting down...")
        
        self.kb_monitor.stop()
        self.inference_manager.stop()
        
        # Clear TTS queue
        if self.tts_available:
            while not voice.speech_queue.empty():
                try:
                    voice.speech_queue.get_nowait()
                except queue.Empty:
                    break
            
            voice.speech_queue.put(None)
            if self.tts_thread:
                self.tts_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ Thoát hoàn tất!")
