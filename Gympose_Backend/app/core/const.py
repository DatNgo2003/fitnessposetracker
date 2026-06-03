from typing import List


# Default CORS configuration for the API
DEFAULT_CORS_ORIGINS: List[str] = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://10.1.56.210:8080",
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",
    "ws://localhost:8000",
    "ws://127.0.0.1:8000",
    "ws://10.1.56.210:8000",
    # Allow all origins for development (remove in production)
    "*",
]

# Soniox STT constants
SONIOX_WEBSOCKET_URL: str = "wss://stt-rt.soniox.com/transcribe-websocket"
SONIOX_MODEL: str = "stt-rt-preview-v2"
SONIOX_AUDIO_FORMAT: str = "pcm_s16le"
SONIOX_LANGUAGE_HINTS: List[str] = ["vi"]
