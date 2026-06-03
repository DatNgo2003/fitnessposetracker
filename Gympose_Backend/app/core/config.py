from functools import lru_cache
import os
from typing import List

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from .const import DEFAULT_CORS_ORIGINS


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GYMPOSE_",
        extra="ignore",
        protected_namespaces=("settings_",),  # Fix Pydantic warning
    )

    app_name: str = "Gympose Backend"
    log_level: str = "INFO"
    cors_origins: List[str] = Field(
        default=DEFAULT_CORS_ORIGINS,
        validation_alias=AliasChoices("GYMPOSE_CORS_ORIGINS", "CORS_ORIGINS"),
    )

    # STUN/TURN servers for WebRTC
    stun_servers: List[str] = [
        "stun:stun.l.google.com:19302",
        "stun:stun1.l.google.com:19302",
        "stun:stun2.l.google.com:19302",
    ]

    # Inference configuration
    detection_score_threshold: float = 0.7
    process_fps: float = 10.0  # Max FPS for processing frames

    # === PHẦN CỦA BẠN (GIỮ LẠI) ===
    # Model paths
    model_dir: str = Field(
        default=os.path.join(
            os.path.dirname(__file__), "../../architectures/mmpose/models"
        ),
        description="Directory containing model files",
    )

    # File storage
    upload_dir: str = Field(
        default="/tmp/gympose/uploads",
        description="Directory for temporary file uploads",
    )
    output_dir: str = Field(
        default="/tmp/gympose/outputs", description="Directory for processed outputs"
    )

    # Processing limits
    max_image_size: int = Field(default=1920, description="Maximum image dimension")
    max_video_duration: int = Field(
        default=300, description="Maximum video duration in seconds"
    )
    max_batch_size: int = Field(
        default=50, description="Maximum number of images in a batch"
    )

    # File cleanup
    file_retention_hours: int = Field(
        default=24, description="Number of hours to keep processed files"
    )

    # Soniox STT configuration
    soniox_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GYMPOSE_SONIOX_API_KEY", "SONIOX_API_KEY"),
    )
    stt_sample_rate: int = 16000
    stt_channels: int = 1
    stt_block_size: int = 1024
    stt_silence_threshold: float = 10.0
    stt_long_silence: float = 10.0

    # Gemini LLM configuration
    gemini_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GYMPOSE_GEMINI_API_KEY", "GEMINI_API_KEY"),
    )

    # FPT.AI TTS configuration
    fpt_tts_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GYMPOSE_FPT_TTS_API_KEY", "FPT_TTS_API_KEY"),
    )

    # Google Cloud credentials
    llm_gemini_credentials: str = Field(
        default="",
        validation_alias=AliasChoices(
            "GYMPOSE_LLM_GEMINI_CREDENTIALS", "LLM_GEMINI_CREDENTIALS"
        ),
    )
    tts_google_credentials: str = Field(
        default="",
        validation_alias=AliasChoices(
            "GYMPOSE_TTS_GOOGLE_CREDENTIALS", "TTS_GOOGLE_CREDENTIALS"
        ),
    )
    # === KẾT THÚC PHẦN CỦA NGƯỜI KHÁC ===


@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
