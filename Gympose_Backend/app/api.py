# FILE: Gympose_Backend/app/api.py

from __future__ import annotations

import logging
import os
from pathlib import Path
import asyncio
import uuid

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from app.core.config import get_settings

# === KẾT HỢP TẤT CẢ IMPORT TỪ CẢ HAI BÊN ===
from app.schemas.vision import (
    Answer,
    Offer,
    HealthResponse,
    StunServersResponse,
    VideoProcessingResponse,
)
from app.modules.vision.pose import PoseEstimator
from app.modules.llm.stt import stt_manager
from app.modules.llm.tts import tts_manager

# ===============================================

router = APIRouter()
_logger = logging.getLogger(__name__)

# === GIỮ LẠI CÁC KHỞI TẠO TỪ CẢ HAI BÊN ===
# Khởi tạo pose estimator của BẠN
pose_estimator = PoseEstimator()

# Tạo thư mục output/upload của BẠN
settings = get_settings()
UPLOAD_DIR = Path(settings.upload_dir)
OUTPUT_DIR = Path(settings.output_dir)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ===============================================


# === GIỮ LẠI ENDPOINT HEALTH CHECK CỦA NGƯỜI KHÁC (VÌ NÓ ĐẦY ĐỦ HƠN) ===
@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint"""
    return HealthResponse(status="ok")


# ======================================================================


# === GIỮ LẠI TOÀN BỘ TÍNH NĂNG WEBRTC (REAL-TIME) CỦA NGƯỜI KHÁC ===
@router.get("/vision/stun", response_model=StunServersResponse)
def get_stun_servers() -> StunServersResponse:
    """Get STUN servers for WebRTC connection"""
    return StunServersResponse(iceServers=[{"urls": get_settings().stun_servers}])


@router.post("/vision/offer", response_model=Answer)
async def create_answer(offer: Offer) -> Answer:
    """Create WebRTC answer from offer"""
    from app.modules.vision.webrtc import webrtc_manager

    data = await webrtc_manager.create_answer(
        session_id=offer.session_id, sdp=offer.sdp, type_=offer.type
    )
    return Answer(**data)


@router.websocket("/vision/ws/{session_id}")
async def ws_pose(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for real-time pose data"""
    await websocket.accept()
    from app.modules.vision.webrtc import webrtc_manager

    queue = await webrtc_manager.subscribe(session_id)
    try:
        while True:
            try:
                data = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(data)
            except asyncio.TimeoutError:
                # keep the connection alive
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        _logger.info("WebSocket disconnected: session=%s", session_id)
    finally:
        await webrtc_manager.unsubscribe(session_id, queue)


# === PROGRESS TRACKING WEBSOCKET ===
@router.websocket("/ws/progress/{session_id}")
async def ws_progress(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for video processing progress"""
    from app.modules.vision.progress_manager import progress_manager
    
    await progress_manager.connect(session_id, websocket)
    try:
        # Keep connection alive and listen for client messages
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _logger.info("Progress WebSocket disconnected: session=%s", session_id)
    finally:
        progress_manager.disconnect(session_id)


# ======================================================================


# === GIỮ LẠI TOÀN BỘ TÍNH NĂNG XỬ LÝ VIDEO (UPLOAD) CỦA BẠN ===
@router.post("/video/process", response_model=VideoProcessingResponse)
async def process_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    draw_bbox: bool = Form(default=True),
    draw_keypoints: bool = Form(default=True),
    save_keypoints: bool = Form(default=True),
    min_confidence: float = Form(default=0.3),
    output_fps: int = Form(default=30),
    analyze_squat: bool = Form(default=False),
    analyze_pushup: bool = Form(default=False),
    analyze_barbell: bool = Form(default=False),
    analyze_lunge: bool = Form(default=False),
    camera_angle: str = Form(default='front_view'),
    session_id: str = Form(default=None),  # Optional session ID for progress tracking
) -> VideoProcessingResponse:
    """
    Process video file for pose estimation, squat, pushup, barbell dead row, and dumbbell reverse lunge analysis
    """
    from app.modules.vision.progress_manager import progress_manager
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    # Save uploaded video
    if session_id:
        await progress_manager.update_progress(session_id, 5, "Đang tải video lên...")
    
    video_path = UPLOAD_DIR / video.filename
    with open(video_path, "wb") as buffer:
        content = await video.read()
        buffer.write(content)

    if session_id:
        await progress_manager.update_progress(session_id, 10, "Đang khởi tạo AI...")
        await asyncio.sleep(0.5)  # Small delay to ensure message is sent
        
        await progress_manager.update_progress(session_id, 12, "Đang chuẩn bị xử lý video...")
        await asyncio.sleep(0.3)

    # Process video - always output as MP4 for compatibility
    # Replace input extension with .mp4 and ensure filename uniqueness to avoid caching
    base_name = os.path.splitext(video.filename)[0]
    safe_base_name = base_name.replace(" ", "_")
    unique_suffix = uuid.uuid4().hex[:8]
    output_filename = f"processed_{safe_base_name}_{unique_suffix}.mp4"
    output_path = OUTPUT_DIR / output_filename

    if session_id:
        await progress_manager.update_progress(session_id, 15, "Bắt đầu phân tích...")
        await asyncio.sleep(0.2)

    # Run video processing in thread pool to allow async progress updates
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        output_path, stats = await loop.run_in_executor(
            pool,
            lambda: pose_estimator.process_video(
                str(video_path),
                str(output_path),
                draw_bbox=draw_bbox,
                draw_keypoints=draw_keypoints,
                min_confidence=min_confidence,
                save_keypoints=save_keypoints,
                output_fps=output_fps,
                analyze_squat=analyze_squat,
                analyze_pushup=analyze_pushup,
                analyze_barbell=analyze_barbell,
                analyze_lunge=analyze_lunge,
                camera_angle=camera_angle,
                session_id=session_id,  # Pass session_id for progress tracking
                event_loop=loop,  # Pass event loop for progress updates
            )
        )

    # Clean up input file in background
    background_tasks.add_task(os.unlink, video_path)

    # Prepare response
    response = VideoProcessingResponse(
        processed_video_url=f"/downloads/{Path(output_path).name}",
        frame_count=stats["frame_count"],
        processing_fps=stats["fps"],
        processing_time=stats["processing_time"],
        detected_persons_count=stats.get("total_detections", 0),
    )

    if "keypoints_file" in stats:
        response.keypoints_data_url = f"/downloads/{Path(stats['keypoints_file']).name}"

    # Add squat analysis data
    if analyze_squat and "total_squats" in stats:
        response.total_squats = stats["total_squats"]
        response.valid_squats = stats["valid_squats"]
        response.invalid_squats = stats["invalid_squats"]
        response.rep_feedbacks = stats["rep_feedbacks"]
    
    # Add pushup analysis data
    if analyze_pushup and "total_pushups" in stats:
        response.total_pushups = stats["total_pushups"]
        response.valid_pushups = stats.get("valid_pushups", 0)
        response.invalid_pushups = stats.get("invalid_pushups", 0)
        response.pushup_rep_feedbacks = stats.get("pushup_rep_feedbacks", [])
    
    # Add barbell analysis data
    if analyze_barbell and "total_barbells" in stats:
        response.total_barbells = stats["total_barbells"]
        response.valid_barbells = stats["valid_barbells"]
        response.invalid_barbells = stats["invalid_barbells"]
        response.barbell_rep_feedbacks = stats.get("barbell_rep_feedbacks", [])
    
    # Add lunge analysis data
    if analyze_lunge and "total_lunges" in stats:
        response.total_lunges = stats["total_lunges"]
        response.valid_lunges = stats["valid_lunges"]
        response.invalid_lunges = stats["invalid_lunges"]
        response.lunge_rep_feedbacks = stats.get("lunge_rep_feedbacks", [])

    return response


@router.get("/downloads/{filename}")
async def get_processed_file(filename: str) -> FileResponse:
    """
    Download processed files (video or keypoint data)
    """
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        # Using a proper exception would be better, but for now this works.
        return {"error": "File not found"}, 404

    return FileResponse(
        str(file_path), 
        filename=filename, 
        media_type="video/mp4",
        headers={
            "Content-Disposition": "inline",  # Encourage browser playback
            "Accept-Ranges": "bytes",  # Enable range requests for seeking
        }
    )


# ======================================================================


# === GIỮ LẠI TÍNH NĂNG SPEECH-TO-TEXT (STT) CỦA NGƯỜI KHÁC ===
@router.websocket("/ws/stt")
async def ws_stt(websocket: WebSocket) -> None:
    await stt_manager.handle_websocket(websocket)


# === DEBUG ENDPOINT FOR TTS ===
@router.get("/debug/tts")
async def debug_tts():
    """Debug TTS configuration and files"""
    import os
    from app.modules.llm.tts import tts_manager

    audio_dir = tts_manager.audio_dir
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


# === TTS AUDIO SERVING ENDPOINT ===
@router.get("/audio/{filename}")
async def get_audio_file(filename: str) -> FileResponse:
    """
    Serve TTS audio files
    """
    audio_path = tts_manager.get_audio_path(filename)
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
            "Access-Control-Allow-Origin": "*",  # Allow CORS for audio
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "*",
        },
    )


# === REP AUDIO SERVING ENDPOINT ===
@router.get("/rep-audio/{filename}")
async def get_rep_audio(filename: str) -> FileResponse:
    """
    Serve rep audio files (rep1.mp3, rep2.mp3, etc.)
    """
    import os
    from fastapi.responses import FileResponse
    
    # Path to rep audio directory
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    gympose_backend_dir = os.path.dirname(current_file_dir)
    audio_path = os.path.join(gympose_backend_dir, "data", "rep", filename)
    
    if not os.path.exists(audio_path):
        _logger.error(f"Rep audio file not found: {audio_path}")
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Audio file not found: {filename}")
    
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


# === VOICE FEEDBACK AUDIO SERVING ENDPOINT ===
@router.get("/voice-feedback/{exercise_type}/{filename}")
async def get_voice_feedback_audio(exercise_type: str, filename: str) -> FileResponse:
    """
    Serve voice feedback audio files for exercises
    exercise_type: squat, pushup, barbell, lunge
    """
    import os
    from fastapi.responses import FileResponse
    
    # Path to voice feedback directory
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    gympose_backend_dir = os.path.dirname(current_file_dir)
    audio_path = os.path.join(gympose_backend_dir, "data", "voice_feedback", exercise_type, filename)
    
    if not os.path.exists(audio_path):
        _logger.error(f"Voice feedback audio file not found: {audio_path}")
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Audio file not found: {filename}")
    
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


# === SAMPLE VIDEO SERVING ENDPOINT ===
@router.get("/sample-videos/squat")
async def get_sample_squat_video() -> FileResponse:
    """
    Serve sample squat demonstration video
    """
    # Path to sample video in dataset directory
    # api.py is in Gympose_Backend/app/, dataset is in Gympose_Backend/dataset/
    # So we go up one level from app/ to Gympose_Backend/, then into dataset/
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    gympose_backend_dir = os.path.dirname(
        current_file_dir
    )  # Go up from app/ to Gympose_Backend/
    video_path = os.path.join(gympose_backend_dir, "dataset", "IMG_8503.mp4")

    _logger.info(f"Current file dir: {current_file_dir}")
    _logger.info(f"Gympose Backend dir: {gympose_backend_dir}")
    _logger.info(f"Serving sample squat video: {video_path}")

    if not os.path.exists(video_path):
        _logger.error(f"Sample video not found: {video_path}")
        # List what's actually in the dataset directory for debugging
        dataset_dir = os.path.join(gympose_backend_dir, "dataset")
        if os.path.exists(dataset_dir):
            files = os.listdir(dataset_dir)
            _logger.error(f"Files in dataset dir: {files}")
        return {"error": "Sample video not found"}, 404

    return FileResponse(
        video_path,
        filename="squat_demo.mp4",
        media_type="video/mp4",
        headers={
            "Cache-Control": "public, max-age=86400",  # Cache for 1 day
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "*",
        },
    )


@router.get("/sample-videos/pushup")
async def get_sample_pushup_video() -> FileResponse:
    """
    Serve sample pushup demonstration video
    """
    # Path to sample video in dataset directory
    # In Docker: Dataset_gympose is mounted at /app/dataset
    # Locally: api.py is in Gympose_Backend/app/, dataset is ../dataset/ relative to Gympose_Backend
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    gympose_backend_dir = os.path.dirname(current_file_dir)  # Go up to Gympose_Backend/ or /app
    
    # Try Docker path first (/app/dataset), then local path
    docker_video_path = os.path.join(gympose_backend_dir, "dataset", "IMG_8519-pushup.mp4")
    
    if os.path.exists(docker_video_path):
        video_path = docker_video_path
    else:
        # Local development: go up to parent and into Dataset_gympose
        parent_dir = os.path.dirname(gympose_backend_dir)
        video_path = os.path.join(parent_dir, "Dataset_gympose", "IMG_8519-pushup.mp4")

    _logger.info(f"Current file dir: {current_file_dir}")
    _logger.info(f"Gympose Backend dir: {gympose_backend_dir}")
    _logger.info(f"Serving sample pushup video: {video_path}")

    if not os.path.exists(video_path):
        _logger.error(f"Sample pushup video not found: {video_path}")
        # List what's in dataset directories for debugging
        docker_dataset_dir = os.path.join(gympose_backend_dir, "dataset")
        if os.path.exists(docker_dataset_dir):
            files = [f for f in os.listdir(docker_dataset_dir) if f.endswith('.mp4')]
            _logger.error(f"MP4 files in {docker_dataset_dir}: {files}")
        return {"error": "Sample pushup video not found"}, 404

    return FileResponse(
        video_path,
        filename="pushup_demo.mp4",
        media_type="video/mp4",
        headers={
            "Cache-Control": "public, max-age=86400",  # Cache for 1 day
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "*",
        },
    )


@router.get("/sample-videos/barbell")
async def get_sample_barbell_video() -> FileResponse:
    """
    Serve sample barbell dead row demonstration video
    """
    # Path to sample video in dataset directory
    # In Docker: Dataset_gympose is mounted at /app/dataset
    # Locally: api.py is in Gympose_Backend/app/, dataset is ../dataset/ relative to Gympose_Backend
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    gympose_backend_dir = os.path.dirname(current_file_dir)  # Go up to Gympose_Backend/ or /app
    
    # Try Docker path first (/app/dataset), then local path
    docker_video_path = os.path.join(gympose_backend_dir, "dataset", "baber_deadle_row.mp4")
    
    if os.path.exists(docker_video_path):
        video_path = docker_video_path
    else:
        # Local development: go up to parent and into Dataset_gympose
        parent_dir = os.path.dirname(gympose_backend_dir)
        video_path = os.path.join(parent_dir, "Dataset_gympose", "baber_deadle_row.mp4")

    _logger.info(f"Serving sample barbell video: {video_path}")

    if not os.path.exists(video_path):
        _logger.error(f"Sample barbell video not found: {video_path}")
        # List what's in dataset directories for debugging
        docker_dataset_dir = os.path.join(gympose_backend_dir, "dataset")
        if os.path.exists(docker_dataset_dir):
            files = [f for f in os.listdir(docker_dataset_dir) if f.endswith('.mp4')]
            _logger.error(f"MP4 files in {docker_dataset_dir}: {files}")
        return {"error": "Sample barbell video not found"}, 404

    return FileResponse(
        video_path,
        filename="barbell_demo.mp4",
        media_type="video/mp4",
        headers={
            "Cache-Control": "public, max-age=86400",  # Cache for 1 day
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "*",
        },
    )


@router.get("/sample-videos/lunge")
async def get_sample_lunge_video() -> FileResponse:
    """
    Serve sample dumbbell reverse lunge demonstration video
    """
    # Path to sample video in dataset directory
    # In Docker: Dataset_gympose is mounted at /app/dataset
    # Locally: api.py is in Gympose_Backend/app/, dataset is ../dataset/ relative to Gympose_Backend
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    gympose_backend_dir = os.path.dirname(current_file_dir)  # Go up to Gympose_Backend/ or /app
    
    # Try Docker path first (/app/dataset), then local path
    docker_video_path = os.path.join(gympose_backend_dir, "dataset", "dumbel_reverse.mp4")
    
    if os.path.exists(docker_video_path):
        video_path = docker_video_path
    else:
        # Local development: go up to parent and into Dataset_gympose
        parent_dir = os.path.dirname(gympose_backend_dir)
        video_path = os.path.join(parent_dir, "Dataset_gympose", "dumbel_reverse.mp4")

    _logger.info(f"Serving sample lunge video: {video_path}")

    if not os.path.exists(video_path):
        _logger.error(f"Sample lunge video not found: {video_path}")
        # List what's in dataset directories for debugging
        docker_dataset_dir = os.path.join(gympose_backend_dir, "dataset")
        if os.path.exists(docker_dataset_dir):
            files = [f for f in os.listdir(docker_dataset_dir) if f.endswith('.mp4')]
            _logger.error(f"MP4 files in {docker_dataset_dir}: {files}")
        return {"error": "Sample lunge video not found"}, 404

    return FileResponse(
        video_path,
        filename="lunge_demo.mp4",
        media_type="video/mp4",
        headers={
            "Cache-Control": "public, max-age=86400",  # Cache for 1 day
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "*",
        },
    )


@router.get("/sample-videos/squat-side")
async def get_sample_squat_side_video() -> FileResponse:
    """
    Serve sample squat demonstration video (side view)
    """
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    gympose_backend_dir = os.path.dirname(current_file_dir)
    
    # Try Docker path first (/app/dataset), then local path
    docker_video_path = os.path.join(gympose_backend_dir, "dataset", "squat-ngang.mp4")
    
    if os.path.exists(docker_video_path):
        video_path = docker_video_path
    else:
        # Local development: go up to parent and into Dataset_gympose
        parent_dir = os.path.dirname(gympose_backend_dir)
        video_path = os.path.join(parent_dir, "Dataset_gympose", "squat-ngang.mp4")

    _logger.info(f"Serving sample squat side view video: {video_path}")

    if not os.path.exists(video_path):
        _logger.error(f"Sample squat side view video not found: {video_path}")
        return {"error": "Sample squat side view video not found"}, 404

    return FileResponse(
        video_path,
        filename="squat_side_demo.mp4",
        media_type="video/mp4",
        headers={
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "*",
        },
    )


@router.get("/sample-videos/pushup-side")
async def get_sample_pushup_side_video() -> FileResponse:
    """
    Serve sample pushup demonstration video (side view)
    """
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    gympose_backend_dir = os.path.dirname(current_file_dir)
    
    docker_video_path = os.path.join(gympose_backend_dir, "dataset", "pushup-ngang.mp4")
    
    if os.path.exists(docker_video_path):
        video_path = docker_video_path
    else:
        parent_dir = os.path.dirname(gympose_backend_dir)
        video_path = os.path.join(parent_dir, "Dataset_gympose", "pushup-ngang.mp4")

    _logger.info(f"Serving sample pushup side view video: {video_path}")

    if not os.path.exists(video_path):
        _logger.error(f"Sample pushup side view video not found: {video_path}")
        return {"error": "Sample pushup side view video not found"}, 404

    return FileResponse(
        video_path,
        filename="pushup_side_demo.mp4",
        media_type="video/mp4",
        headers={
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "*",
        },
    )


@router.get("/sample-videos/barbell-side")
async def get_sample_barbell_side_video() -> FileResponse:
    """
    Serve sample barbell dead row demonstration video (side view)
    """
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    gympose_backend_dir = os.path.dirname(current_file_dir)
    
    docker_video_path = os.path.join(gympose_backend_dir, "dataset", "baber_deadle_row-ngang.mp4")
    
    if os.path.exists(docker_video_path):
        video_path = docker_video_path
    else:
        parent_dir = os.path.dirname(gympose_backend_dir)
        video_path = os.path.join(parent_dir, "Dataset_gympose", "baber_deadle_row-ngang.mp4")

    _logger.info(f"Serving sample barbell side view video: {video_path}")

    if not os.path.exists(video_path):
        _logger.error(f"Sample barbell side view video not found: {video_path}")
        return {"error": "Sample barbell side view video not found"}, 404

    return FileResponse(
        video_path,
        filename="barbell_side_demo.mp4",
        media_type="video/mp4",
        headers={
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "*",
        },
    )


@router.get("/sample-videos/lunge-side")
async def get_sample_lunge_side_video() -> FileResponse:
    """
    Serve sample dumbbell reverse lunge demonstration video (side view)
    """
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    gympose_backend_dir = os.path.dirname(current_file_dir)
    
    docker_video_path = os.path.join(gympose_backend_dir, "dataset", "dumbel_reverse-ngang.mp4")
    
    if os.path.exists(docker_video_path):
        video_path = docker_video_path
    else:
        parent_dir = os.path.dirname(gympose_backend_dir)
        video_path = os.path.join(parent_dir, "Dataset_gympose", "dumbel_reverse-ngang.mp4")

    _logger.info(f"Serving sample lunge side view video: {video_path}")

    if not os.path.exists(video_path):
        _logger.error(f"Sample lunge side view video not found: {video_path}")
        return {"error": "Sample lunge side view video not found"}, 404

    return FileResponse(
        video_path,
        filename="lunge_side_demo.mp4",
        media_type="video/mp4",
        headers={
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "*",
        },
    )


# ======================================================================
