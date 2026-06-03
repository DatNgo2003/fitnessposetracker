from typing import Optional
import os
import time
import uuid
import logging

from google.cloud import texttospeech
from google.oauth2 import service_account

from app.core.config import get_settings

_logger = logging.getLogger(__name__)


class GoogleTTSManager:
    """Manager for Google Cloud Text-to-Speech integration"""

    def __init__(self):
        self.settings = get_settings()
        self.client = None
        # Use absolute path to ensure it's in the mounted volume
        self.audio_dir = os.path.join(os.getcwd(), "data", "tts_audio")
        _logger.info(
            f"Google TTS Manager initialized with audio_dir: {os.path.abspath(self.audio_dir)}"
        )
        self._ensure_audio_dir()
        self._initialize_client()

    def _ensure_audio_dir(self) -> None:
        """Ensure audio directory exists"""
        abs_path = os.path.abspath(self.audio_dir)
        _logger.info(f"Creating TTS audio directory: {abs_path}")
        os.makedirs(self.audio_dir, exist_ok=True)
        _logger.info(f"TTS audio directory ready: {abs_path}")

    def _initialize_client(self) -> None:
        """Initialize Google Cloud TTS client with credentials"""
        try:
            credentials_path = self.settings.tts_google_credentials

            # If relative path, resolve it relative to project root
            if credentials_path and not os.path.isabs(credentials_path):
                # Get project root (where Dockerfile is)
                project_root = os.getcwd()
                credentials_path = os.path.join(project_root, credentials_path)

            _logger.info(f"Google TTS credentials path: {credentials_path}")
            _logger.info(f"Current working directory: {os.getcwd()}")

            if not credentials_path or not os.path.exists(credentials_path):
                _logger.error(
                    f"Google TTS credentials file not found: {credentials_path}"
                )
                # List files in config directory for debugging
                config_dir = os.path.join(os.getcwd(), "config")
                if os.path.exists(config_dir):
                    _logger.info(f"Files in config directory: {os.listdir(config_dir)}")
                return

            # Load credentials from JSON file
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )

            # Initialize the Text-to-Speech client
            self.client = texttospeech.TextToSpeechClient(credentials=credentials)
            _logger.info("Google Cloud TTS client initialized successfully")

        except Exception as e:
            _logger.error(f"Failed to initialize Google TTS client: {e}")
            self.client = None

    async def text_to_speech(
        self,
        text: str,
        language_code: str = "vi-VN",
        voice_name: str = "vi-VN-Neural2-D",
        speaking_rate: float = 1.0,
    ) -> Optional[str]:
        """
        Convert text to speech using Google Cloud TTS API

        Args:
            text: Text to convert to speech
            language_code: Language code (vi-VN for Vietnamese)
            voice_name: Voice name (vi-VN-Neural2-A, vi-VN-Neural2-D, etc.)
            speaking_rate: Speaking rate (0.25 to 4.0, default 1.0)

        Returns:
            Path to generated audio file or None if error
        """
        if not self.client:
            _logger.error("Google TTS client not initialized")
            return None

        if not text or len(text.strip()) < 1:
            _logger.warning("Empty text provided for TTS")
            return None

        try:
            start_time = time.time()
            _logger.info(f"🔊 Converting text to speech (Google Cloud): {text[:50]}...")

            # Generate unique filename
            audio_filename = f"tts_{uuid.uuid4().hex[:8]}_{int(time.time())}.mp3"
            audio_path = os.path.join(self.audio_dir, audio_filename)
            _logger.info(f"Saving audio to: {os.path.abspath(audio_path)}")

            # Set the text input to be synthesized
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Build the voice request
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code, name=voice_name
            )

            # Select the type of audio file you want returned
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=speaking_rate,
            )

            # Perform the text-to-speech request
            response = self.client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            # Write the response to the output file
            with open(audio_path, "wb") as out:
                out.write(response.audio_content)

            processing_time = time.time() - start_time
            _logger.info(
                f"✅ Google TTS completed in {processing_time:.2f}s: {audio_path}"
            )

            return audio_filename  # Return relative filename

        except Exception as e:
            _logger.error(f"Google TTS error: {e}")
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
                    file_age = current_time - os.path.getmtime(file_path)

                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        _logger.info(f"Cleaned up old TTS file: {filename}")

        except Exception as e:
            _logger.error(f"Error cleaning up TTS files: {e}")

    def get_audio_path(self, filename: str) -> str:
        """Get full path to audio file"""
        return os.path.join(self.audio_dir, filename)


# Global instance
google_tts_manager = GoogleTTSManager()
