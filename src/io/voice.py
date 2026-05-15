import queue
import threading
import asyncio
import tempfile
import os
import uuid

recognizer = None
microphone = None
speech_queue = queue.Queue()
microphone_lock = threading.Lock()

def init_stt():
    global recognizer, microphone
    print("🎤 Đang khởi tạo Voice Recognition...")
    try:
        import speech_recognition as sr
        
        # List microphones
        print("\n📋 Danh sách microphones:")
        mic_list = sr.Microphone.list_microphone_names()
        
        if len(mic_list) == 0:
            print("❌ KHÔNG TÌM THẤY MICROPHONE!")
            return False
        
        for index, name in enumerate(mic_list):
            print(f"  [{index}] {name}")
        
        # Find best mic
        best_mic_index = None
        
        for index, name in enumerate(mic_list):
            name_lower = name.lower()
            if 'loopback' not in name_lower and 'stereo mix' not in name_lower:
                if any(kw in name_lower for kw in ['microphone', 'mic', 'headset', 'webcam', 'usb']):
                    best_mic_index = index
                    print(f"\n✅ Chọn: [{index}] {name}")
                    break
        
        if best_mic_index is None:
            best_mic_index = 0
            print(f"\n⚠️ Dùng mic mặc định: [{best_mic_index}] {mic_list[0]}")
        
        # Initialize
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 200
        recognizer.dynamic_energy_threshold = True
        recognizer.dynamic_energy_adjustment_damping = 0.10
        recognizer.dynamic_energy_ratio = 1.5
        recognizer.pause_threshold = 0.6
        recognizer.phrase_threshold = 0.3
        recognizer.non_speaking_duration = 0.4
        
        microphone = sr.Microphone(device_index=best_mic_index)
        
        # Calibrate
        print("\n🔧 Đang hiệu chỉnh...")
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print(f"   Energy threshold: {recognizer.energy_threshold:.0f}")

        print("\n🧪 Testing mic (say 'HELLO')...")
        try:
            with microphone as source:
                test_audio = recognizer.listen(source, timeout=3, phrase_time_limit=2)
                
                if test_audio:
                    size = len(test_audio.frame_data)
                    print(f"   ✅ Working! ({size}B)")
                    
                    if size < 1000:
                        print(f"   ⚠️ Weak signal - increase volume!")
                else:
                    print("   ⚠️ No audio captured")
                    
        except Exception as test_err:
            print(f"   ⚠️ Test failed: {test_err}")
            print("   → Mic might not work properly")
        
        print("✅ Voice Recognition ready!\n")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi init Voice: {e}")
        return False

def init_tts():
    """Initialize TTS"""
    print("🔊 Đang khởi tạo TTS...")
    try:
        import edge_tts
        import pygame
        
        # Test edge-tts version
        print(f"   📦 edge-tts version: {edge_tts.__version__ if hasattr(edge_tts, '__version__') else 'unknown'}")
        
        # Initialize pygame mixer
        pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=2048)
        
        print("✅ TTS ready!\n")
        return True
        
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("   💡 Install: pip install edge-tts pygame")
        return False
    except Exception as e:
        print(f"⚠️ TTS init error: {e}")
        return False

def tts_worker():
    """TTS worker — streaming từ memory, không ghi file"""
    import edge_tts
    import pygame
    import io
    import asyncio
    import time as _time
    
    # Create event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def speak_streaming(text, max_retries=2):
        """Stream TTS audio trực tiếp vào memory → play ngay"""
        
        voices = [
            "vi-VN-HoaiMyNeural",
            "vi-VN-NamMinhNeural",
        ]
        
        for attempt in range(max_retries):
            for voice in voices:
                try:
                    if attempt > 0:
                        print(f"   🔄 Retry {attempt+1}/{max_retries} with {voice}...")
                    
                    t_tts_total = _time.time()
                    
                    communicate = edge_tts.Communicate(
                        text,
                        voice,
                        rate="+5%",
                    )
                    
                    # Stream audio chunks trực tiếp vào BytesIO (không ghi file)
                    audio_buffer = io.BytesIO()
                    
                    t_gen = _time.time()
                    await asyncio.wait_for(
                        _collect_stream(communicate, audio_buffer),
                        timeout=10.0
                    )
                    t_gen_elapsed = _time.time() - t_gen
                    
                    # Kiểm tra buffer có dữ liệu
                    if audio_buffer.tell() < 1000:
                        raise ValueError("Audio buffer too small")
                    
                    # Play trực tiếp từ memory
                    audio_buffer.seek(0)
                    t_play = _time.time()
                    pygame.mixer.music.load(audio_buffer, "mp3")
                    pygame.mixer.music.play()
                    
                    while pygame.mixer.music.get_busy():
                        await asyncio.sleep(0.05)
                    t_play_elapsed = _time.time() - t_play
                    
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                    
                    # Giải phóng buffer
                    audio_buffer.close()
                    
                    t_tts_total_elapsed = _time.time() - t_tts_total
                    print(f"   ⏱️ TTS: tổng={t_tts_total_elapsed:.3f}s (sinh audio={t_gen_elapsed:.3f}s, phát={t_play_elapsed:.3f}s)")
                    
                    return  # Thành công
                    
                except asyncio.TimeoutError:
                    print(f"   ⏰ TTS timeout: {voice}")
                    continue
                except Exception as e:
                    # Nếu pygame không hỗ trợ load từ BytesIO → fallback file
                    if "load" in str(e).lower() or "format" in str(e).lower():
                        await _speak_file_fallback(text, voice)
                        return
                    print(f"   ⚠️ TTS error ({voice}): {e}")
                    continue
            
            # Hết voices, retry
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5)
        
        print(f"   ❌ TTS failed after {max_retries} attempts")
    
    async def _collect_stream(communicate, buffer):
        """Thu thập tất cả audio chunks từ edge-tts stream vào buffer"""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])
    
    async def _speak_file_fallback(text, voice):
        """Fallback: ghi file nếu streaming không hoạt động"""
        import tempfile, os, uuid
        
        temp_file = os.path.join(
            tempfile.gettempdir(),
            f"tts_{uuid.uuid4().hex[:8]}.mp3"
        )
        
        try:
            communicate = edge_tts.Communicate(text, voice, rate="+5%")
            await asyncio.wait_for(communicate.save(temp_file), timeout=10.0)
            
            if os.path.exists(temp_file) and os.path.getsize(temp_file) > 1000:
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.05)
                
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
        except Exception as e:
            print(f"   ❌ TTS file fallback error: {e}")
        finally:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    # Main worker loop
    while True:
        text = speech_queue.get()
        
        if text is None:
            break
        
        try:
            loop.run_until_complete(speak_streaming(text))
        except Exception as e:
            print(f"   ❌ TTS worker error: {e}")
        finally:
            speech_queue.task_done()
    
    loop.close()