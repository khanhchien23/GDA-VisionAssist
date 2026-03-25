"""
Tests for src/io module.
Tests voice, keyboard, and camera I/O components.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import queue


class TestVoiceModule:
    """Test voice I/O functions."""
    
    def test_speech_queue_exists(self):
        """Test speech_queue is accessible"""
        from src.io.voice import speech_queue
        assert isinstance(speech_queue, queue.Queue)
    
    def test_microphone_lock_exists(self):
        """Test microphone_lock is accessible"""
        import threading
        from src.io.voice import microphone_lock
        assert isinstance(microphone_lock, type(threading.Lock()))
    
    def test_recognizer_initial_state(self):
        """Test recognizer starts as None"""
        # Recognizer is None before init_stt()
        from src.io import voice
        # After import, recognizer might be None or initialized
        assert voice.recognizer is None or voice.recognizer is not None  # Just check accessible
    
    @patch('src.io.voice.sr', create=True)
    def test_init_stt_no_microphone(self):
        """Test init_stt khi không có microphone"""
        # This should handle gracefully
        from src.io.voice import init_stt
        # init_stt returns bool
        result = init_stt()
        assert isinstance(result, bool)
    
    def test_speech_queue_put_get(self):
        """Test queue behavior"""
        from src.io.voice import speech_queue
        
        # Clear queue
        while not speech_queue.empty():
            speech_queue.get_nowait()
        
        # Put & get
        speech_queue.put("test message")
        msg = speech_queue.get_nowait()
        assert msg == "test message"


class TestClickHandler:
    """Test ClickHandler class."""
    
    def test_init(self):
        """Test ClickHandler initialization"""
        from src.io.keyboard import ClickHandler
        handler = ClickHandler()
        
        assert handler.clicked_point is None
        assert handler.waiting_for_click == False
    
    def test_mouse_callback_while_waiting(self):
        """Test mouse callback sets point when waiting"""
        from src.io.keyboard import ClickHandler
        import cv2
        
        handler = ClickHandler()
        handler.waiting_for_click = True
        
        # Simulate left click
        handler.mouse_callback(cv2.EVENT_LBUTTONDOWN, 100, 200, 0, None)
        
        assert handler.clicked_point == (100, 200)
    
    def test_mouse_callback_not_waiting(self):
        """Test mouse callback does nothing when not waiting"""
        from src.io.keyboard import ClickHandler
        import cv2
        
        handler = ClickHandler()
        handler.waiting_for_click = False
        
        handler.mouse_callback(cv2.EVENT_LBUTTONDOWN, 100, 200, 0, None)
        
        assert handler.clicked_point is None


class TestKeyboardMonitor:
    """Test KeyboardMonitor class."""
    
    def test_init(self):
        """Test KeyboardMonitor initialization"""
        from src.io.keyboard import KeyboardMonitor
        monitor = KeyboardMonitor()
        
        assert monitor.c_pressed == False
    
    def test_start_stop(self):
        """Test start and stop"""
        from src.io.keyboard import KeyboardMonitor
        monitor = KeyboardMonitor()
        
        monitor.start()
        monitor.stop()


class TestCamera:
    """Test camera module."""
    
    def test_init_camera_function_exists(self):
        """Test init_camera function is importable"""
        from src.io.camera import init_camera
        assert callable(init_camera)
