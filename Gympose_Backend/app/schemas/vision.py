from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any  # Add Dict and Any for flexible feedback


class VideoProcessingRequest(BaseModel):
    """Request parameters for video processing"""

    draw_bbox: bool = Field(default=True, description="Whether to draw bounding boxes")
    draw_keypoints: bool = Field(default=True, description="Whether to draw keypoints")
    save_keypoints: bool = Field(
        default=True, description="Whether to save keypoint data"
    )
    min_confidence: float = Field(
        default=0.3, description="Minimum confidence threshold"
    )
    output_fps: int = Field(default=30, description="Output video FPS")


class KeypointData(BaseModel):
    """Structure for a single keypoint"""

    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    confidence: float = Field(..., description="Keypoint confidence score")


class PersonDetection(BaseModel):
    """Structure for a detected person in a frame"""

    frame_id: int = Field(..., description="Frame number in the video")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    keypoints: List[KeypointData] = Field(..., description="List of detected keypoints")
    detection_confidence: float = Field(
        ..., description="Person detection confidence score"
    )


class RepFeedback(BaseModel):
    """Feedback for a single squat rep"""

    rep_id: int = Field(..., description="Rep number")
    depth_feedback: str = Field(..., description="Depth feedback message")
    back_feedback: str = Field(..., description="Back posture feedback message")
    lowest_knee_angle: float = Field(..., description="Lowest knee angle achieved")
    is_valid: bool = Field(..., description="Whether the rep is valid")


class VideoProcessingResponse(BaseModel):
    """Response for video processing"""

    processed_video_url: str = Field(
        ..., description="URL to download the processed video"
    )
    keypoints_data_url: Optional[str] = Field(
        None, description="URL to download the keypoints JSON data"
    )
    frame_count: int = Field(..., description="Total number of frames processed")
    processing_fps: float = Field(
        ..., description="Processing speed in frames per second"
    )
    processing_time: float = Field(..., description="Total processing time in seconds")
    detected_persons_count: int = Field(
        ..., description="Total number of persons detected across all frames"
    )
    # Squat analysis fields
    total_squats: Optional[int] = Field(
        None, description="Total number of squats detected"
    )
    valid_squats: Optional[int] = Field(None, description="Number of valid squats")
    invalid_squats: Optional[int] = Field(None, description="Number of invalid squats")
    rep_feedbacks: Optional[List[RepFeedback]] = Field(
        None, description="Per-rep feedback"
    )
    # Pushup analysis fields
    total_pushups: Optional[int] = Field(
        None, description="Total number of pushups detected"
    )
    valid_pushups: Optional[int] = Field(
        None, description="Number of valid pushup reps"
    )
    invalid_pushups: Optional[int] = Field(
        None, description="Number of invalid pushup reps"
    )
    pushup_rep_feedbacks: Optional[List[Dict[str, Any]]] = Field(
        None, description="Detailed feedback for each pushup rep (flexible dict format)"
    )
    # Barbell dead row analysis fields
    total_barbells: Optional[int] = Field(
        None, description="Total number of barbell dead rows detected"
    )
    valid_barbells: Optional[int] = Field(
        None, description="Number of valid barbell reps"
    )
    invalid_barbells: Optional[int] = Field(
        None, description="Number of invalid barbell reps"
    )
    barbell_rep_feedbacks: Optional[List[Dict[str, Any]]] = Field(
        None, description="Detailed feedback for each barbell rep (flexible dict format)"
    )
    # Dumbbell reverse lunge analysis fields
    total_lunges: Optional[int] = Field(
        None, description="Total number of dumbbell reverse lunges detected"
    )
    valid_lunges: Optional[int] = Field(
        None, description="Number of valid lunge reps"
    )
    invalid_lunges: Optional[int] = Field(
        None, description="Number of invalid lunge reps"
    )
    lunge_rep_feedbacks: Optional[List[Dict[str, Any]]] = Field(
        None, description="Detailed feedback for each lunge rep (flexible dict format)"
    )



class Offer(BaseModel):
    """WebRTC offer schema"""

    sdp: str = Field(..., description="Session Description Protocol")
    type: str = Field(..., description="Offer type")
    session_id: str = Field(..., description="Unique session identifier")


class Answer(BaseModel):
    """WebRTC answer schema"""

    sdp: str = Field(..., description="Session Description Protocol")
    type: str = Field(..., description="Answer type")


class PoseKeypoint(BaseModel):
    """Individual pose keypoint"""

    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    score: float = Field(..., description="Confidence score")


class PoseResult(BaseModel):
    """Pose detection result"""

    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    keypoints: List[List[float]] = Field(
        ..., description="Keypoints [[x, y, score], ...]"
    )


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service status")


class StunServersResponse(BaseModel):
    """STUN servers response"""

    iceServers: List[dict] = Field(..., description="List of ICE servers")


