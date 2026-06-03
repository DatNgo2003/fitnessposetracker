import re
import logging
import os
from typing import Optional

import google.generativeai as genai
from google.oauth2 import service_account

from app.core.config import get_settings

_logger = logging.getLogger(__name__)


class GeminiLLMManager:
    """Manager for Gemini LLM integration"""

    def __init__(self):
        self.settings = get_settings()
        self.model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize Gemini model with credentials from JSON file"""
        try:
            credentials_path = self.settings.llm_gemini_credentials

            # If relative path, resolve it relative to project root
            if credentials_path and not os.path.isabs(credentials_path):
                project_root = os.getcwd()
                credentials_path = os.path.join(project_root, credentials_path)

            _logger.info(f"Gemini credentials path: {credentials_path}")
            _logger.info(f"Current working directory: {os.getcwd()}")

            # Check if using service account credentials
            if credentials_path and os.path.exists(credentials_path):
                _logger.info(f"Using Gemini credentials from file: {credentials_path}")
                # For Vertex AI, we'll use service account
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path
                )
                # Set environment variable for Google Cloud
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            else:
                # Fallback to API key if credentials file not found
                api_key = self.settings.gemini_api_key
                _logger.info(
                    f"Gemini API key loaded: {'Yes' if api_key else 'No'} (length: {len(api_key) if api_key else 0})"
                )

                if not api_key:
                    _logger.error("Gemini API key not configured")
                    return

                genai.configure(api_key=api_key)

            # Use the newer flash model
            self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
            _logger.info(
                "Gemini model initialized successfully with gemini-2.0-flash-exp"
            )
        except Exception as e:
            _logger.error(f"Failed to initialize Gemini model: {e}")
            self.model = None

    async def generate_response(self, user_text: str) -> Optional[str]:
        """
        Generate response from user text using Gemini API

        Args:
            user_text: Text from STT to process

        Returns:
            Processed text ready for TTS or None if error
        """
        if not self.model:
            _logger.error("Gemini model not initialized")
            return None

        try:
            # Tạo prompt phù hợp với fitness context - nhấn mạnh độ ngắn gọn
            prompt = f"""
            Bạn là huấn luyện viên thể hình. Người dùng nói: "{user_text}"
            
            Trả lời NGẮN GỌN (1-2 câu, tối đa 30 từ):
            - Động viên, thân thiện
            - Không markdown, không ký tự đặc biệt
            - Phù hợp cho text-to-speech
            - Ví dụ: "Tuyệt vời! Hãy tập trung vào hiệp này và đẩy hết sức nhé!"
            """

            # Configure generation parameters for short, real-time responses
            generation_config = genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=40,
                max_output_tokens=150,  # Short responses for real-time conversation
                candidate_count=1,
            )

            _logger.info(f"Sending request to Gemini for text: {user_text}")
            response = self.model.generate_content(
                prompt, generation_config=generation_config
            )

            if not response or not response.text:
                _logger.error("Empty response from Gemini")
                return None

            # Xử lý text để phù hợp với TTS
            processed_text = self._process_for_tts(response.text)

            # Kiểm tra độ dài response
            word_count = len(processed_text.split())
            if word_count > 30:  # Nếu quá dài, cắt ngắn
                words = processed_text.split()[:25]
                processed_text = " ".join(words) + "..."
                _logger.warning(
                    f"Response too long ({word_count} words), truncated to {len(processed_text.split())} words"
                )

            _logger.info(
                f"Generated response ({len(processed_text.split())} words): {processed_text}"
            )

            return processed_text

        except Exception as e:
            _logger.error(f"Error generating Gemini response: {e}")
            return None

    def _process_for_tts(self, text: str) -> str:
        """
        Process Gemini response text to be TTS-ready

        Args:
            text: Raw text from Gemini

        Returns:
            Cleaned text ready for TTS
        """
        # Xóa Markdown và ký tự đặc biệt
        text = re.sub(r"(\*\*|##|---|\*+|`)", "", text)

        # Xóa các ký tự xuống dòng và khoảng trắng thừa
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        # Nối thành đoạn liền mạch
        tts_ready_text = " ".join(lines)

        # Xóa khoảng trắng thừa
        tts_ready_text = re.sub(r"\s+", " ", tts_ready_text).strip()

        return tts_ready_text


# Global instance
gemini_manager = GeminiLLMManager()
