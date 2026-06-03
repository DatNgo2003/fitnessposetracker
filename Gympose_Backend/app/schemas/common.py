from pydantic import BaseModel, Field
from typing import Optional, Any, Dict


class BaseResponse(BaseModel):
    """Base response schema"""
    success: bool = Field(default=True, description="Request success status")
    message: Optional[str] = Field(default=None, description="Response message")
    data: Optional[Any] = Field(default=None, description="Response data")


class ErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = Field(default=False, description="Request success status")
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")


class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(default=1, ge=1, description="Page number")
    limit: int = Field(default=10, ge=1, le=100, description="Items per page")


class PaginatedResponse(BaseModel):
    """Paginated response schema"""
    items: list = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page")
    limit: int = Field(..., description="Items per page")
    pages: int = Field(..., description="Total number of pages")


class STTToken(BaseModel):
    """Speech-to-text token"""
    text: str = Field(..., description="Token text")
    is_final: bool = Field(..., description="Whether token is final")


class STTResponse(BaseModel):
    """Speech-to-text response"""
    type: str = Field(..., description="Response type: transcript, partial, llm_response, tts_response, error, ping, finished")
    transcript: Optional[str] = Field(default=None, description="Full transcript")
    tokens: Optional[list[STTToken]] = Field(default=None, description="Token list")
    error_code: Optional[str] = Field(default=None, description="Error code if any")
    error_message: Optional[str] = Field(default=None, description="Error message if any")
    finished: bool = Field(default=False, description="Whether session finished")


class LLMResponse(BaseModel):
    """LLM response schema"""
    type: str = Field(default="llm_response", description="Response type")
    user_text: str = Field(..., description="Original user text from STT")
    ai_response: str = Field(..., description="AI generated response")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")


class TTSResponse(BaseModel):
    """TTS response schema"""
    type: str = Field(default="tts_response", description="Response type")
    text: str = Field(..., description="Text that was converted to speech")
    audio_url: str = Field(..., description="URL to download the audio file")
    audio_filename: str = Field(..., description="Audio filename")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")