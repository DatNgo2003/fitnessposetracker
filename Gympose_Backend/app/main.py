# app/main.py
import sys
import os
import logging

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.logger import configure_logging
from app.api import router as api_router

# Initialize settings and logging FIRST
settings = get_settings()
configure_logging(settings.log_level)

_logger = logging.getLogger(__name__)

# ✨ IMPORTANT: Import managers AFTER logging is configured
# This ensures we can see initialization logs
_logger.info("🚀 Starting AI services initialization...")

# Force immediate logging output
logging.getLogger("app.modules.llm.gemini").setLevel(logging.INFO)
logging.getLogger("app.modules.llm.tts_google").setLevel(logging.INFO)
logging.getLogger("app.modules.llm.stt").setLevel(logging.INFO)

from app.modules.llm.gemini import gemini_manager
from app.modules.llm.tts_google import google_tts_manager
from app.modules.llm.stt import stt_manager

_logger.info("✅ AI services loaded successfully")
_logger.info(f"Gemini model ready: {gemini_manager.model is not None}")
_logger.info(f"Google TTS client ready: {google_tts_manager.client is not None}")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure for large file handling
app.router.route_class.max_size = 1024 * 1024 * 1000  # 1000MB


# Health check endpoint (without prefix for Docker healthcheck)
@app.get("/health")
def health_check():
    """Health check endpoint for Docker"""
    return {"status": "ok"}


# STT WebSocket endpoint (without prefix for frontend compatibility)
@app.websocket("/ws/stt")
async def ws_stt_direct(websocket: WebSocket):
    """STT WebSocket endpoint for frontend"""
    await stt_manager.handle_websocket(websocket)


# Audio serving endpoint (without prefix for frontend compatibility)
@app.get("/audio/{filename}")
async def get_audio_file_direct(filename: str):
    """Serve TTS audio files directly"""
    from fastapi.responses import FileResponse

    audio_path = google_tts_manager.get_audio_path(filename)
    abs_path = os.path.abspath(audio_path)

    _logger.info(f"Serving audio file: {filename}")
    _logger.info(f"Audio path: {abs_path}")
    _logger.info(f"File exists: {os.path.exists(audio_path)}")

    if not os.path.exists(audio_path):
        _logger.error(f"Audio file not found: {abs_path}")
        return {"error": "Audio file not found"}, 404

    return FileResponse(
        audio_path,
        filename=filename,
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "public, max-age=3600",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "*",
        },
    )


# Debug endpoint (without prefix)
@app.get("/debug/tts")
async def debug_tts_direct():
    """Debug TTS configuration and files"""
    audio_dir = google_tts_manager.audio_dir
    abs_path = os.path.abspath(audio_dir)

    files = []
    if os.path.exists(audio_dir):
        files = os.listdir(audio_dir)

    return {
        "audio_dir": audio_dir,
        "abs_path": abs_path,
        "exists": os.path.exists(audio_dir),
        "files": files,
        "cwd": os.getcwd(),
    }


# Include all API routes with /api prefix
app.include_router(api_router, prefix="/api")
