# src/app/inference_manager.py
"""Inference queue manager for GDA Application"""

import torch
import threading
import queue
import time
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass


@dataclass
class InferenceResult:
    """Container for inference results"""
    processing: bool = False
    result: Optional[Dict] = None
    progress: int = 0


class InferenceManager:
    """
    Manages background inference worker and result queue.
    Thread-safe design with Lock for state access.
    """
    
    def __init__(self, gda_system, maxsize: int = 1):
        """
        Args:
            gda_system: GlobalDescriptionAcquisition instance
            maxsize: Maximum queue size
        """
        self.gda = gda_system
        self.queue = queue.Queue(maxsize=maxsize)
        self.result = InferenceResult()
        self.lock = threading.Lock()
        self.worker_thread: Optional[threading.Thread] = None
        self._running = False
    
    def start(self):
        """Start the inference worker thread"""
        self._running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop, 
            daemon=True,
            name="InferenceWorker"
        )
        self.worker_thread.start()
    
    def stop(self):
        """Stop the worker thread gracefully"""
        self._running = False
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
    
    def submit(self, frame_rgb, mask, user_query: Optional[str] = None,
               task_type: str = 'describe') -> bool:
        """
        Submit a task to the inference queue.
        
        Args:
            frame_rgb: Camera frame
            mask: Segmentation mask (can be None for 'scene' mode)
            user_query: User's question
            task_type: 'describe' | 'ocr' | 'scene'
            
        Returns:
            True if submitted successfully, False if queue is full or busy
        """
        with self.lock:
            if self.result.processing:
                return False
        
        try:
            task_data = {
                'frame_rgb': frame_rgb.copy(),
                'mask': mask.copy() if mask is not None else None,
                'user_query': user_query,
                'task_type': task_type
            }
            self.queue.put_nowait(task_data)
            return True
        except queue.Full:
            return False
    
    def get_status(self) -> tuple:
        """
        Get current processing status.
        
        Returns:
            (is_processing, progress, result)
        """
        with self.lock:
            return (
                self.result.processing,
                self.result.progress,
                self.result.result
            )
    
    def consume_result(self) -> Optional[Dict]:
        """
        Get and clear the result if available.
        
        Returns:
            Result dict or None
        """
        with self.lock:
            if not self.result.processing and self.result.result is not None:
                result = self.result.result
                self.result.result = None
                return result
        return None
    
    def is_processing(self) -> bool:
        """Check if currently processing"""
        with self.lock:
            return self.result.processing
    
    def _worker_loop(self):
        """Main worker loop - runs in background thread"""
        while self._running:
            try:
                task = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if task is None:
                break
            
            # Support both old tuple format and new dict format
            if isinstance(task, dict):
                frame_rgb = task['frame_rgb']
                mask = task['mask']
                user_query = task['user_query']
                task_type = task.get('task_type', 'describe')
            else:
                frame_rgb, mask, user_query = task
                task_type = 'describe'
            
            # Initialize processing state
            with self.lock:
                self.result.processing = True
                self.result.result = None
                self.result.progress = 0
            
            try:
                start_time = time.time()
                
                type_labels = {
                    'describe': '🔍 Mô tả',
                    'ocr': '📖 OCR',
                    'scene': '🌍 Toàn cảnh'
                }
                type_label = type_labels.get(task_type, task_type)
                query_text = user_query if user_query else type_label
                print(f"\n🔄 Worker [{type_label}]: {query_text}")
                
                # Update progress
                with self.lock:
                    self.result.progress = 10
                
                # Route to correct GDA method based on task_type
                if task_type == 'ocr':
                    result = self.gda.ocr_region(frame_rgb, mask)
                elif task_type == 'scene':
                    result = self.gda.describe_scene(frame_rgb)
                else:
                    result = self.gda.process_region(frame_rgb, mask, user_query)
                
                elapsed = time.time() - start_time
                
                # Normalize result
                if isinstance(result, dict):
                    result.setdefault('latency_sec', elapsed)
                    result.setdefault('task_type', task_type)
                    result.setdefault('query', query_text)
                else:
                    result = {
                        'description': str(result),
                        'error': False,
                        'predicted_class': None,
                        'confidence': 0.0,
                        'query': query_text,
                        'latency_sec': elapsed,
                        'task_type': task_type,
                    }
                
                with self.lock:
                    self.result.progress = 95
                
                print("✅ Worker hoàn thành")
                
            except Exception as e:
                print(f"❌ Worker lỗi: {e}")
                if self.gda.debug:
                    import traceback
                    traceback.print_exc()
                
                elapsed = time.time() - start_time
                result = {
                    'description': f"Lỗi: {str(e)}",
                    'error': True,
                    'predicted_class': None,
                    'confidence': 0.0,
                    'query': user_query if user_query else "Lỗi",
                    'latency_sec': elapsed,
                    'task_type': task_type if 'task_type' in dir() else 'describe',
                }
            
            finally:
                with self.lock:
                    self.result.processing = False
                    self.result.result = result
                    self.result.progress = 0
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.queue.task_done()
                time.sleep(0.05)
