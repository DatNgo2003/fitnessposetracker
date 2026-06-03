"""
Video Processing Progress Manager
Manages WebSocket connections for real-time progress updates
"""
import logging
from typing import Dict, Optional
from fastapi import WebSocket

_logger = logging.getLogger(__name__)


class ProgressManager:
    """Manages progress tracking for video processing sessions"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
    
    async def connect(self, session_id: str, websocket: WebSocket):
        """Register a new WebSocket connection for progress updates"""
        await websocket.accept()
        self.connections[session_id] = websocket
        _logger.info(f"Progress tracking connected for session: {session_id}")
    
    def disconnect(self, session_id: str):
        """Remove WebSocket connection"""
        if session_id in self.connections:
            del self.connections[session_id]
            _logger.info(f"Progress tracking disconnected for session: {session_id}")
    
    async def update_progress(self, session_id: str, progress: int, message: str):
        """Send progress update to connected client"""
        if session_id in self.connections:
            try:
                await self.connections[session_id].send_json({
                    "progress": progress,
                    "message": message
                })
                _logger.debug(f"Progress update sent for {session_id}: {progress}% - {message}")
            except Exception as e:
                _logger.error(f"Error sending progress update for {session_id}: {e}")
                self.disconnect(session_id)


# Global singleton instance
progress_manager = ProgressManager()

