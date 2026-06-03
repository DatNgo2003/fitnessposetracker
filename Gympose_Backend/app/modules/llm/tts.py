import asyncio
import logging
import os
import time
from typing import Optional
import uuid

import aiohttp
import aiofiles

from app.core.config import get_settings

_logger = logging.getLogger(__name__)


class FPTTTSManager:
    """Manager for FPT.AI Text-to-Speech integration"""

    def __init__(self):
        self.settings = get_settings()
        # Use absolute path to ensure it's in the mounted volume
        self.audio_dir = os.path.join(os.getcwd(), "data", "tts_audio")
        _logger.info(f"TTS Manager initialized with audio_dir: {os.path.abspath(self.audio_dir)}")
        self._ensure_audio_dir()

    def _ensure_audio_dir(self) -> None:
        """Ensure audio directory exists"""
        abs_path = os.path.abspath(self.audio_dir)
        _logger.info(f"Creating TTS audio directory: {abs_path}")
        os.makedirs(self.audio_dir, exist_ok=True)
        _logger.info(f"TTS audio directory ready: {abs_path}")

    async def text_to_speech(self, text: str, voice: str = "banmai", speed: str = "1") -> Optional[str]:
        """
        Convert text to speech using FPT.AI TTS API
        
        Args:
            text: Text to convert to speech
            voice: Voice type (banmai, leminh, etc.)
            speed: Speech speed (0.8, 1, 1.2)
            
        Returns:
            Path to generated audio file or None if error
        """
        if not self.settings.fpt_tts_api_key:
            _logger.error("FPT TTS API key not configured")
            return None

        if not text or len(text.strip()) < 1:
            _logger.warning("Empty text provided for TTS")
            return None

        try:
            start_time = time.time()
            _logger.info(f"🔊 Converting text to speech: {text[:50]}...")

            # Generate unique filename
            audio_filename = f"tts_{uuid.uuid4().hex[:8]}_{int(time.time())}.mp3"
            audio_path = os.path.join(self.audio_dir, audio_filename)
            _logger.info(f"Saving audio to: {os.path.abspath(audio_path)}")

            # FPT.AI TTS API request
            url = "https://api.fpt.ai/hmi/tts/v5"
            headers = {
                "api-key": self.settings.fpt_tts_api_key,
                "speed": speed,
                "voice": voice
            }

            async with aiohttp.ClientSession() as session:
                # Send TTS request
                async with session.post(
                    url, 
                    data=text.encode("utf-8"), 
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        _logger.error(f"FPT TTS API error: {response.status}")
                        return None
                    
                    data = await response.json()
                    audio_url = data.get("async")

                    if not audio_url:
                        _logger.error("No audio URL returned from FPT TTS API")
                        return None

                # Download audio file with retry logic
                _logger.info(f"📥 Downloading audio from: {audio_url}")
                
                # Retry download up to 3 times with shorter delays to avoid blocking
                max_retries = 3
                retry_delays = [2, 4, 6]  # seconds - shorter delays to avoid blocking pipeline
                
                for attempt in range(max_retries):
                    try:
                        if attempt > 0:
                            _logger.info(f"Retry attempt {attempt + 1}/{max_retries} after {retry_delays[attempt - 1]}s delay")
                            await asyncio.sleep(retry_delays[attempt - 1])
                        
                        async with session.get(audio_url, timeout=aiohttp.ClientTimeout(total=10)) as audio_response:
                            if audio_response.status == 200:
                                # Save audio file
                                async with aiofiles.open(audio_path, "wb") as f:
                                    async for chunk in audio_response.content.iter_chunked(8192):
                                        await f.write(chunk)
                                break  # Success, exit retry loop
                            elif audio_response.status == 404 and attempt < max_retries - 1:
                                _logger.warning(f"Audio not ready yet (404), retrying in {retry_delays[attempt]}s...")
                                continue
                            else:
                                _logger.error(f"Failed to download audio: {audio_response.status}")
                                return None
                                
                    except asyncio.TimeoutError:
                        if attempt < max_retries - 1:
                            _logger.warning(f"Download timeout, retrying in {retry_delays[attempt]}s...")
                            continue
                        else:
                            _logger.error("Download timeout after all retries")
                            return None
                else:
                    # All retries failed - log but don't break the conversation
                    _logger.error(f"Failed to download audio after {max_retries} retries. FPT.AI may be experiencing issues.")
                    _logger.warning("Skipping TTS for this response to maintain conversation flow")
                    return None

            processing_time = time.time() - start_time
            _logger.info(f"✅ TTS completed in {processing_time:.2f}s: {audio_path}")
            
            return audio_filename  # Return relative filename

        except asyncio.TimeoutError:
            _logger.error("TTS request timeout")
            return None
        except Exception as e:
            _logger.error(f"TTS error: {e}")
            return None

    async def cleanup_old_files(self, max_age_hours: int = 24) -> None:
        """
        Clean up old TTS audio files
        
        Args:
            max_age_hours: Maximum age of files to keep in hours
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for filename in os.listdir(self.audio_dir):
                if filename.startswith("tts_") and filename.endswith(".mp3"):
                    file_path = os.path.join(self.audio_dir, filename)
                    file_age = current_time - os.path.getctime(file_path)
                    
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        _logger.info(f"Cleaned up old TTS file: {filename}")
                        
        except Exception as e:
            _logger.error(f"Error cleaning up TTS files: {e}")

    def get_audio_path(self, filename: str) -> str:
        """Get full path to audio file"""
        return os.path.join(self.audio_dir, filename)


# Global instance
tts_manager = FPTTTSManager()
