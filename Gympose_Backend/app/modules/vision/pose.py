from __future__ import annotations
import logging
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import json
from mmpose.apis import inference_topdown


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert (dict, list, numpy types, etc.)
        
    Returns:
        Converted object with Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

from app.core.config import get_settings
from app.modules.vision.detection import PersonDetector
from app.modules.vision.squat_analyzer import SquatAnalyzer
from app.modules.vision.pushup_analyzer import PushupAnalyzer
from app.modules.vision.barbell_analyzer import BarbellDeadRowAnalyzer
from app.modules.vision.lunge_analyzer import DumbbellReverseLungeAnalyzer

_logger = logging.getLogger(__name__)


def remove_vietnamese_accents(text: str) -> str:
    """
    Remove Vietnamese accents from text for OpenCV display.
    OpenCV cannot display Vietnamese Unicode characters correctly.
    
    Args:
        text: Vietnamese text with accents
        
    Returns:
        Text without accents (ASCII compatible)
    """
    # Vietnamese character mapping
    replacements = {
        'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
        'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
        'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
        'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
        'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
        'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
        'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
        'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
        'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
        'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
        'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
        'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
        'đ': 'd',
        'À': 'A', 'Á': 'A', 'Ả': 'A', 'Ã': 'A', 'Ạ': 'A',
        'Ă': 'A', 'Ằ': 'A', 'Ắ': 'A', 'Ẳ': 'A', 'Ẵ': 'A', 'Ặ': 'A',
        'Â': 'A', 'Ầ': 'A', 'Ấ': 'A', 'Ẩ': 'A', 'Ẫ': 'A', 'Ậ': 'A',
        'È': 'E', 'É': 'E', 'Ẻ': 'E', 'Ẽ': 'E', 'Ẹ': 'E',
        'Ê': 'E', 'Ề': 'E', 'Ế': 'E', 'Ể': 'E', 'Ễ': 'E', 'Ệ': 'E',
        'Ì': 'I', 'Í': 'I', 'Ỉ': 'I', 'Ĩ': 'I', 'Ị': 'I',
        'Ò': 'O', 'Ó': 'O', 'Ỏ': 'O', 'Õ': 'O', 'Ọ': 'O',
        'Ô': 'O', 'Ồ': 'O', 'Ố': 'O', 'Ổ': 'O', 'Ỗ': 'O', 'Ộ': 'O',
        'Ơ': 'O', 'Ờ': 'O', 'Ớ': 'O', 'Ở': 'O', 'Ỡ': 'O', 'Ợ': 'O',
        'Ù': 'U', 'Ú': 'U', 'Ủ': 'U', 'Ũ': 'U', 'Ụ': 'U',
        'Ư': 'U', 'Ừ': 'U', 'Ứ': 'U', 'Ử': 'U', 'Ữ': 'U', 'Ự': 'U',
        'Ỳ': 'Y', 'Ý': 'Y', 'Ỷ': 'Y', 'Ỹ': 'Y', 'Ỵ': 'Y',
        'Đ': 'D',
    }
    
    result = text
    for vn_char, ascii_char in replacements.items():
        result = result.replace(vn_char, ascii_char)
    
    return result


class PoseEstimator:
    _initialized: bool = False

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None
        self._detector = PersonDetector()
        self._visualizer = None
        self._initialized = False
        self._squat_analyzer = SquatAnalyzer(smoothing_window=5)
        self._pushup_analyzer = PushupAnalyzer(side='right', smoothing_window=5)
        self._barbell_analyzer = BarbellDeadRowAnalyzer(side='right', smoothing_window=5)
        self._lunge_analyzer = DumbbellReverseLungeAnalyzer(smoothing_window=5)

    def _lazy_init(self) -> None:
        if self._initialized:
            return

        device = "cuda" if self._cuda_available() else "cpu"
        _logger.info(
            "Initializing Pose Estimator (may download weights on first run)..."
        )

        # Initialize pose estimator with custom RTMPose model
        # pose.py is at: /app/app/modules/vision/pose.py
        # config is at: /app/config/rtmpose_s_fullfinetune.py
        # models is at: /app/models/best_coco_AP_epoch_80.pth
        # From pose.py: go up 3 levels (vision -> modules -> app -> /app)
        pose_config = os.path.join(
            os.path.dirname(__file__),
            "../../../config/rtmpose-l_fullfinetune_gym.py",
        )
        pose_checkpoint = os.path.join(
            os.path.dirname(__file__),
            "../../../models/rtmpose_l_fullfintune_60e_phase2.pth",
        )

        self._model = init_pose_estimator(
            pose_config,
            pose_checkpoint,
            device=device,
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))),
        )

        # Initialize visualizer
        self._visualizer = VISUALIZERS.build(self._model.cfg.visualizer)
        self._visualizer.set_dataset_meta(
            self._model.dataset_meta, skeleton_style="mmpose"
        )

        _logger.info(f"Visualizer initialized: {type(self._visualizer).__name__}")
        _logger.info(
            f"Dataset meta: {self._model.dataset_meta.get('dataset_name', 'unknown')}"
        )

        self._initialized = True

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def _draw_squat_feedback(
        self, frame: np.ndarray, squat_analysis: Dict[str, Any]
    ) -> np.ndarray:
        """
        Draw squat feedback overlay on frame.
        Shows HUD with rep counter, state, angles on left side
        and feedback messages on right side.

        Args:
            frame: Input frame
            squat_analysis: Analysis results from SquatAnalyzer

        Returns:
            Frame with HUD overlay
        """
        if not squat_analysis:
            return frame
        
        # Left side HUD - Black box with info
        cv2.rectangle(frame, (10, 10), (450, 200), (0, 0, 0), -1)
        
        # Line 1: Rep counter (large)
        cv2.putText(frame, f"REPS: {squat_analysis.get('rep_counter', 0)}", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        # Line 2: State - Remove accents
        state_text = remove_vietnamese_accents(squat_analysis.get('state', 'IDLE'))
        cv2.putText(frame, f"STATE: {state_text}", 
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Line 3: Knee angle
        angle_knee = squat_analysis.get('angle_knee')
        if angle_knee is not None:
            knee_color = (0, 255, 0) if angle_knee < 90 else (0, 255, 255)  # Green if deep enough
            cv2.putText(frame, f"KNEE: {int(angle_knee)} deg", 
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, knee_color, 2)
        
        # Line 4: Back angle
        angle_back = squat_analysis.get('angle_back', 0)
        if angle_back > 0:
            back_color = (0, 255, 0) if angle_back < 40 else (0, 0, 255)  # Green if straight
            cv2.putText(frame, f"BACK: {int(angle_back)} deg", 
                        (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.8, back_color, 2)
        
        # Right side - Feedback messages (remove accents)
        y_offset = 50
        
        # Depth feedback
        depth_fb = squat_analysis.get('depth_feedback', '')
        if depth_fb:
            fb_color = (0, 255, 0) if 'tot' in depth_fb.lower() or 'du' in depth_fb.lower() else (0, 0, 255)
            depth_ascii = remove_vietnamese_accents(depth_fb)
            cv2.putText(frame, depth_ascii, 
                        (470, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, fb_color, 2)
            y_offset += 50
        
        # Back feedback
        back_fb = squat_analysis.get('back_feedback', '')
        if back_fb:
            fb_color = (0, 255, 0) if 'tot' in back_fb.lower() or 'thang' in back_fb.lower() else (0, 0, 255)
            back_ascii = remove_vietnamese_accents(back_fb)
            cv2.putText(frame, back_ascii, 
                        (470, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, fb_color, 2)

        # DEPTH ANGLE FEEDBACK:
        # SIDE
        if squat_analysis.get("s_hip"):
            pt = tuple(map(int, squat_analysis["s_hip"]))
            cv2.circle(frame, pt, 5, (0, 255, 255), -1)  # Yellow for selected hip
        if squat_analysis.get("s_knee"):
            pt = tuple(map(int, squat_analysis["s_knee"]))
            cv2.circle(frame, pt, 5, (255, 0, 255), -1)  # Magenta for selected knee
        if squat_analysis.get("s_ankle"):
            pt = tuple(map(int, squat_analysis["s_ankle"]))
            cv2.circle(frame, pt, 5, (255, 255, 0), -1)  # Cyan for selected ankle

        # Draw lines for selected side
        if squat_analysis.get("s_hip") and squat_analysis.get("s_knee"):
            pt1 = tuple(map(int, squat_analysis["s_hip"]))
            pt2 = tuple(map(int, squat_analysis["s_knee"]))
            cv2.line(frame, pt1, pt2, (100, 100, 255), 2)  # Light blue line
        if squat_analysis.get("s_knee") and squat_analysis.get("s_ankle"):
            pt2 = tuple(map(int, squat_analysis["s_knee"]))
            pt3 = tuple(map(int, squat_analysis["s_ankle"]))
            cv2.line(frame, pt2, pt3, (100, 100, 255), 2)  # Light blue line

        # FRONT
        if squat_analysis.get("hip_midpoint"):
            pt = tuple(map(int, squat_analysis["hip_midpoint"]))
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)  # Green for hip midpoint
        if squat_analysis.get("knee_midpoint"):
            pt = tuple(map(int, squat_analysis["knee_midpoint"]))
            cv2.circle(frame, pt, 5, (255, 0, 0), -1)  # Blue for knee midpoint
        if squat_analysis.get("ankle_midpoint"):
            pt = tuple(map(int, squat_analysis["ankle_midpoint"]))
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)  # Red for ankle midpoint

        # Draw lines for front
        if squat_analysis.get("hip_midpoint") and squat_analysis.get("knee_midpoint"):  
            pt1 = tuple(map(int, squat_analysis["hip_midpoint"]))
            pt2 = tuple(map(int, squat_analysis["knee_midpoint"]))
            cv2.line(frame, pt1, pt2, (200, 200, 0), 2)  # Cyan line
        if squat_analysis.get("knee_midpoint") and squat_analysis.get("ankle_midpoint"):
            pt2 = tuple(map(int, squat_analysis["knee_midpoint"]))
            pt3 = tuple(map(int, squat_analysis["ankle_midpoint"]))
            cv2.line(frame, pt2, pt3, (200, 200, 0), 2)  # Cyan line

        # BACK ANGLE FEEDBACK:
        # Draw visualization points (shoulder_midpoint, hip_midpoint, vertical_point)
        if squat_analysis.get("shoulder_midpoint"):
            pt = tuple(map(int, squat_analysis["shoulder_midpoint"]))
            cv2.circle(frame, pt, 5, (255, 0, 0), -1)  # Blue for shoulder midpoint

        if squat_analysis.get("hip_midpoint"):
            pt = tuple(map(int, squat_analysis["hip_midpoint"]))
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)  # Green for hip midpoint

        if squat_analysis.get("vertical_point"):
            pt = tuple(map(int, squat_analysis["vertical_point"]))
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)  # Red for vertical point
        
        # Draw line connecting shoulder to hip to vertical point
        if squat_analysis.get("shoulder_midpoint") and squat_analysis.get("hip_midpoint"):
            pt1 = tuple(map(int, squat_analysis["shoulder_midpoint"]))
            pt2 = tuple(map(int, squat_analysis["hip_midpoint"]))
            cv2.line(frame, pt1, pt2, (200, 200, 0), 2)  # Cyan line
        
        if squat_analysis.get("hip_midpoint") and squat_analysis.get("vertical_point"):
            pt2 = tuple(map(int, squat_analysis["hip_midpoint"]))
            pt3 = tuple(map(int, squat_analysis["vertical_point"]))
            cv2.line(frame, pt2, pt3, (200, 200, 0), 2)  # Cyan line
        
        return frame

    def _draw_pushup_feedback(
        self, frame: np.ndarray, pushup_analysis: Dict[str, Any]
    ) -> np.ndarray:
        """
        Draw pushup feedback overlay on frame.
        Shows HUD with rep counter, state, angles on left side
        and feedback messages on right side.

        Args:
            frame: Input frame
            pushup_analysis: Analysis results from PushupAnalyzer

        Returns:
            Frame with HUD overlay
        """
        if not pushup_analysis:
            return frame
        
        # Left side HUD - Black box with info
        cv2.rectangle(frame, (10, 10), (450, 215), (0, 0, 0), -1)
        
        # Line 1: Rep counter (large)
        cv2.putText(frame, f"REPS: {pushup_analysis.get('rep_counter', 0)}", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        # Line 2: State - Remove accents
        state_text = remove_vietnamese_accents(pushup_analysis.get('state', 'IDLE'))
        cv2.putText(frame, f"STATE: {state_text}", 
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Line 3: Elbow angle
        elbow_angle = pushup_analysis.get('angle_elbow', 0)
        if elbow_angle > 0:
            cv2.putText(frame, f"ELBOW: {int(elbow_angle)} deg", 
                        (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Line 4: Body angle
        body_angle = pushup_analysis.get('angle_body', 0)
        if body_angle > 0:
            cv2.putText(frame, f"BODY: {int(body_angle)} deg", 
                        (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Right side - Feedback messages (remove accents)
        y_offset = 50
        
        # Main feedback
        feedback = pushup_analysis.get('feedback', '')
        if feedback:
            fb_color = (255, 255, 0)  # Yellow default
            if 'TOT' in feedback or 'GOOD' in feedback:
                fb_color = (0, 255, 0)
            elif 'SAU' in feedback or 'DEPTH' in feedback:
                fb_color = (0, 0, 255)
            feedback_ascii = remove_vietnamese_accents(feedback)
            cv2.putText(frame, feedback_ascii, 
                        (470, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, fb_color, 2)
            y_offset += 50
        
        # Body feedback
        body_fb = pushup_analysis.get('body_feedback', '')
        if body_fb:
            fb_color = (0, 255, 0) if 'THANG' in body_fb or 'STRAIGHT' in body_fb else (0, 0, 255)
            body_ascii = remove_vietnamese_accents(body_fb)
            cv2.putText(frame, body_ascii, 
                        (470, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, fb_color, 2)
        
        return frame

    def _draw_barbell_feedback(
        self, frame: np.ndarray, barbell_analysis: Dict[str, Any]
    ) -> np.ndarray:
        """
        Draw barbell dead row feedback overlay on frame.
        Shows HUD with rep counter, state, angles on left side
        and feedback messages on right side.

        Args:
            frame: Input frame
            barbell_analysis: Analysis results from BarbellDeadRowAnalyzer

        Returns:
            Frame with HUD overlay
        """
        if not barbell_analysis:
            return frame
        
        # Left side HUD - Black box with info
        cv2.rectangle(frame, (10, 10), (450, 180), (0, 0, 0), -1)
        
        # Line 1: Rep counter (large)
        cv2.putText(frame, f"REPS: {barbell_analysis.get('rep_counter', 0)}", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        # Line 2: State - Remove accents
        state_text = remove_vietnamese_accents(barbell_analysis.get('state', 'IDLE'))
        cv2.putText(frame, f"STATE: {state_text}", 
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Line 3: Knee angle (depth feedback) - Remove accents
        depth_fb = barbell_analysis.get('depth_feedback', '')
        if depth_fb:
            depth_color = (0, 255, 0) if 'DU' in depth_fb or 'GOOD' in depth_fb else (0, 0, 255)
            depth_ascii = remove_vietnamese_accents(depth_fb)
            cv2.putText(frame, depth_ascii, 
                        (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.8, depth_color, 2)
        
        # Right side - Feedback messages (remove accents)
        y_offset = 50
        
        # Back feedback
        back_fb = barbell_analysis.get('back_feedback', '')
        if back_fb:
            fb_color = (0, 255, 0) if 'TOT' in back_fb or 'GOOD' in back_fb or 'THANG' in back_fb else (0, 0, 255)
            back_ascii = remove_vietnamese_accents(back_fb)
            cv2.putText(frame, back_ascii, 
                        (470, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, fb_color, 2)
        
        return frame

    def _draw_lunge_feedback(
        self, frame: np.ndarray, lunge_analysis: Dict[str, Any]
    ) -> np.ndarray:
        """
        Draw dumbbell reverse lunge feedback overlay on frame.
        Shows HUD with rep counter, state, angles on left side
        and feedback messages on right side.

        Args:
            frame: Input frame
            lunge_analysis: Analysis results from DumbbellReverseLungeAnalyzer

        Returns:
            Frame with HUD overlay
        """
        if not lunge_analysis:
            return frame
        
        # Left side HUD - Black box with info
        cv2.rectangle(frame, (10, 10), (550, 220), (0, 0, 0), -1)
        
        # Line 1: Rep counter (large)
        cv2.putText(frame, f"REPS: {lunge_analysis.get('rep_count', 0)}", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        # Line 2: State - Remove accents
        state_text = remove_vietnamese_accents(lunge_analysis.get('state', 'IDLE'))
        cv2.putText(frame, f"STATE: {state_text}", 
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Line 3: Back angle
        back_angle = lunge_analysis.get('back_angle', 0)
        if back_angle > 0:
            back_color = (0, 255, 0) if back_angle >= 160 else (0, 0, 255)  # Green if straight
            back_text = f"BACK: {int(back_angle)} deg"
            if back_angle < 160:
                back_text += " (BENT)"
            else:
                back_text += " (STRAIGHT)"
            cv2.putText(frame, back_text, 
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, back_color, 2)
        
        # Line 4: Depth feedback - Remove accents
        depth_fb = lunge_analysis.get('depth_feedback', '')
        if depth_fb:
            depth_color = (0, 255, 0) if 'TOT' in depth_fb or 'GOOD' in depth_fb else (0, 0, 255)
            depth_ascii = remove_vietnamese_accents(depth_fb)
            cv2.putText(frame, depth_ascii, 
                        (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.8, depth_color, 2)
        
        # Line 5: Rep feedback - Remove accents
        rep_fb = lunge_analysis.get('rep_feedback', '')
        if rep_fb:
            rep_color = (0, 255, 0) if 'TOT' in rep_fb or 'GOOD' in rep_fb else (0, 0, 255)
            rep_ascii = remove_vietnamese_accents(rep_fb)
            cv2.putText(frame, rep_ascii, 
                        (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, rep_color, 2)
        
        # Right side - Feedback messages (remove accents)
        y_offset = 50
        
        # Back feedback
        back_fb = lunge_analysis.get('back_feedback', '')
        if back_fb:
            fb_color = (0, 255, 0) if 'THANG' in back_fb or 'STRAIGHT' in back_fb else (0, 0, 255)
            back_ascii = remove_vietnamese_accents(back_fb)
            cv2.putText(frame, back_ascii, 
                        (470, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, fb_color, 2)
        
        return frame

    def _extract_keypoints_for_pushup(
        self, keypoints: np.ndarray, scores: np.ndarray, min_confidence: float = 0.3
    ) -> Dict[str, Tuple[int, int]]:
        """
        Extract keypoints in format expected by PushupAnalyzer.
        
        Args:
            keypoints: Keypoints array (17, 2) with [x, y]
            scores: Confidence scores (17,)
            min_confidence: Minimum confidence threshold (lowered to 0.3 for pushup)
            
        Returns:
            Dictionary mapping joint name to (x, y) coordinates, or None if critical points missing
        """
        from app.modules.vision.pushup_analyzer import COCO_KEYPOINT_INDEXES
        
        # Handle batch dimension if present
        if len(keypoints.shape) == 3:
            keypoints = keypoints[0]
        if len(scores.shape) == 2:
            scores = scores[0]
        
        kp_dict = {}
        for joint_name, index in COCO_KEYPOINT_INDEXES.items():
            if index < len(scores) and scores[index] > min_confidence:
                x, y = keypoints[index]
                kp_dict[joint_name] = (int(x), int(y))
        
        # For pushup, we need at least shoulder, elbow, hip from one side
        # Allow partial detection instead of requiring all keypoints
        critical_points = ['right_shoulder', 'right_elbow', 'right_hip']
        has_critical = all(point in kp_dict for point in critical_points)
        
        # Return dict even with partial keypoints if critical points exist
        return kp_dict if has_critical else None

    def _extract_keypoints_for_barbell(
        self, keypoints: np.ndarray, scores: np.ndarray, min_confidence: float = 0.5
    ) -> Dict[str, Tuple[int, int]]:
        """
        Extract keypoints in format expected by BarbellDeadRowAnalyzer.
        
        Args:
            keypoints: Keypoints array (17, 2) with [x, y]
            scores: Confidence scores (17,)
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary mapping joint name to (x, y) coordinates
        """
        from app.modules.vision.barbell_analyzer import COCO_KEYPOINT_INDEXES
        
        # Handle batch dimension if present
        if len(keypoints.shape) == 3:
            keypoints = keypoints[0]
        if len(scores.shape) == 2:
            scores = scores[0]
        
        kp_dict = {}
        for joint_name, index in COCO_KEYPOINT_INDEXES.items():
            if index < len(scores) and scores[index] > min_confidence:
                x, y = keypoints[index]
                kp_dict[joint_name] = (int(x), int(y))
        
        return kp_dict if kp_dict else None

    def _extract_keypoints_for_lunge(
        self, keypoints: np.ndarray, scores: np.ndarray, min_confidence: float = 0.5
    ) -> Dict[str, Tuple[int, int]]:
        """
        Extract keypoints in format expected by DumbbellReverseLungeAnalyzer.
        
        Args:
            keypoints: Keypoints array (17, 2) with [x, y]
            scores: Confidence scores (17,)
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary mapping joint name to (x, y) coordinates
        """
        from app.modules.vision.lunge_analyzer import COCO_KEYPOINT_INDEXES
        
        # Handle batch dimension if present
        if len(keypoints.shape) == 3:
            keypoints = keypoints[0]
        if len(scores.shape) == 2:
            scores = scores[0]
        
        kp_dict = {}
        for joint_name, index in COCO_KEYPOINT_INDEXES.items():
            if index < len(scores) and scores[index] > min_confidence:
                x, y = keypoints[index]
                kp_dict[joint_name] = (int(x), int(y))
        
        return kp_dict if kp_dict else None

    def process_frame(
        self,
        frame: np.ndarray,
        draw_bbox: bool = True,
        draw_keypoints: bool = True,
        min_confidence: float = 0.3,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process a single frame from video.

        Args:
            frame: Input BGR frame
            draw_bbox: Whether to draw bounding boxes
            draw_keypoints: Whether to draw keypoints
            min_confidence: Minimum confidence threshold

        Returns:
            Tuple of (visualized frame, keypoint data)
        """
        # Detect persons
        detections = self._detector.detect_persons(frame)
        if not detections:
            return frame, []

        # Filter detections by confidence
        bboxes = np.array([det[:4] for det in detections if det[4] >= min_confidence])
        if len(bboxes) == 0:
            return frame, []

        # Run pose estimation
        pose_results = inference_topdown(self._model, frame, bboxes)
        data_samples = merge_data_samples(pose_results)

        # Prepare visualization
        if draw_bbox or draw_keypoints:
            self._visualizer.add_datasample(
                "result",
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                data_sample=data_samples,
                draw_gt=False,
                draw_heatmap=False,
                draw_bbox=draw_bbox,
                show_kpt_idx=False,
                show=False,
                wait_time=0,
                kpt_thr=min_confidence,
            )
            vis_frame = cv2.cvtColor(self._visualizer.get_image(), cv2.COLOR_RGB2BGR)
        else:
            vis_frame = frame.copy()

        # Extract keypoint data
        frame_data = []
        if hasattr(data_samples, "pred_instances"):
            pred_instances = data_samples.pred_instances
            for bbox, keypoints in zip(pred_instances.bboxes, pred_instances.keypoints):
                detection = {"bbox": bbox.tolist(), "keypoints": keypoints.tolist()}
                frame_data.append(detection)

        return vis_frame, frame_data

    def process_video(
        self,
        video_path: str,
        output_path: str,
        draw_bbox: bool = True,
        draw_keypoints: bool = True,
        min_confidence: float = 0.3,
        save_keypoints: bool = True,
        output_fps: int = 30,
        analyze_squat: bool = False,
        analyze_pushup: bool = False,
        analyze_barbell: bool = False,
        analyze_lunge: bool = False,
        camera_angle: str = 'front_view',
        session_id: str = None,  # Add session_id for progress tracking
        event_loop = None,  # Pass event loop for progress updates
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Process a video file for pose estimation.

        Args:
            video_path: Path to input video
            output_path: Path to save output video
            draw_bbox: Whether to draw bounding boxes
            draw_keypoints: Whether to draw keypoints
            min_confidence: Minimum confidence threshold
            save_keypoints: Whether to save keypoint data
            output_fps: FPS for output video
            analyze_squat: Whether to perform squat analysis
            analyze_pushup: Whether to perform pushup analysis
            analyze_barbell: Whether to perform barbell dead row analysis
            analyze_lunge: Whether to perform dumbbell reverse lunge analysis
            camera_angle: Camera angle ('front_view' or 'side_view')

        Returns:
            Tuple of (output video path, processing stats)
        """
        self._lazy_init()

        # Determine if using side view
        is_side_view = (camera_angle == 'side_view')

        # Reset analyzers for new video with appropriate camera angle settings
        if analyze_squat:
            # Reinitialize with camera angle
            self._squat_analyzer = SquatAnalyzer(smoothing_window=5, is_side_view=is_side_view)
            _logger.info(f"Squat analysis enabled - camera angle: {camera_angle}")
        
        if analyze_pushup:
            # Reinitialize with camera angle
            # Front view: track 'front' side → không chấm body straightness
            # Side view: track 'right' side → chấm cả depth + body straightness
            side = 'right' if is_side_view else 'front'
            self._pushup_analyzer = PushupAnalyzer(side=side, smoothing_window=5)
            _logger.info(f"Pushup analysis enabled - camera angle: {camera_angle}, side={side}")
        
        if analyze_barbell:
            # Reinitialize with camera angle
            # Front view: track 'front' side → không chấm lưng
            # Side view: track 'right' side → chấm cả depth + lưng
            side = 'right' if is_side_view else 'front'
            self._barbell_analyzer = BarbellDeadRowAnalyzer(
                side=side,
                smoothing_window=5,
            )
            _logger.info(f"Barbell dead row analysis enabled - camera angle: {camera_angle}, side={side}")
        
        if analyze_lunge:
            # Reinitialize lunge analyzer with camera angle
            self._lunge_analyzer = DumbbellReverseLungeAnalyzer(smoothing_window=5, is_side_view=is_side_view)
            if is_side_view:
                _logger.info("Dumbbell reverse lunge analysis enabled (side view) - will analyze form and count reps")
            else:
                _logger.info("Dumbbell reverse lunge front view - analysis disabled (no depth/back scoring)")

        # Convert input video to MP4 if it's WebM (OpenCV has issues with WebM)
        original_video_path = video_path
        if video_path.lower().endswith('.webm'):
            _logger.info("Converting WebM input to MP4 for better compatibility...")
            import subprocess
            temp_mp4_path = video_path.replace('.webm', '_converted.mp4')
            
            try:
                # Convert WebM to MP4 with FIXED 30 FPS to avoid FPS issues
                result = subprocess.run([
                    'ffmpeg', '-i', video_path,
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    '-r', '30',  # FORCE output to 30 FPS
                    '-y', temp_mp4_path
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0 and os.path.exists(temp_mp4_path):
                    video_path = temp_mp4_path
                    _logger.info("Successfully converted WebM to MP4 @ 30 FPS")
                else:
                    _logger.warning(f"WebM conversion failed, will try to use original: {result.stderr[:200]}")
            except Exception as e:
                _logger.warning(f"Failed to convert WebM: {e}, will try to use original")

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Fix invalid FPS (some webcam recordings have extremely high FPS values)
        if input_fps <= 0 or input_fps > 120:
            _logger.warning(f"Invalid input FPS: {input_fps}, using default 30 FPS")
            input_fps = 30

        # Use reasonable output FPS
        if output_fps <= 0 or output_fps > 60:
            output_fps = min(input_fps, 30)

        _logger.info(f"Processing video: {video_path}")
        _logger.info(
            f"Resolution: {frame_width}x{frame_height}, FPS: {input_fps} -> {output_fps}"
        )

        # Calculate output dimensions (limit to 720p max height for faster loading)
        MAX_HEIGHT = 720
        output_width = frame_width
        output_height = frame_height
        should_resize = False

        if frame_height > MAX_HEIGHT:
            # Calculate new dimensions maintaining aspect ratio
            scale_factor = MAX_HEIGHT / frame_height
            output_width = int(frame_width * scale_factor)
            output_height = MAX_HEIGHT
            should_resize = True
            _logger.info(
                f"Video will be resized: {frame_width}x{frame_height} -> {output_width}x{output_height} "
                f"(scale: {scale_factor:.2f}x) for faster loading"
            )

        # Prepare output video writer
        # Strategy: Write to AVI first (more reliable), then convert to MP4
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use AVI container with MJPEG codec (most reliable for OpenCV)
        temp_output_path = output_path.replace('.mp4', '_temp.avi')
        
        video_writer = cv2.VideoWriter(
            temp_output_path,
            cv2.VideoWriter_fourcc(*'MJPG'),
            output_fps,
            (output_width, output_height),  # Use calculated output dimensions
        )

        if not video_writer.isOpened():
            raise ValueError(f"Failed to create video writer for {temp_output_path}")

        # Process video frames
        keypoints_data = []
        frame_count = 0
        total_detections = 0
        start_time = time.time()
        
        # Import progress manager if session_id provided
        progress_manager = None
        async_loop = event_loop  # Use the passed event loop
        if session_id and event_loop:
            from app.modules.vision.progress_manager import progress_manager as pm
            progress_manager = pm
            _logger.info(f"Progress tracking enabled for session: {session_id}")

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    _logger.info(
                        f"End of video or read failure at frame {frame_count}/{total_frames}"
                    )
                    break
                
                # Update progress (15% to 85% during frame processing)
                if progress_manager and session_id and total_frames > 0 and async_loop:
                    progress_value = int(15 + (frame_count / total_frames) * 70)
                    # Update more frequently (every 5 frames) and always on first frame
                    if frame_count % 5 == 0 or frame_count == 0:
                        try:
                            import asyncio
                            future = asyncio.run_coroutine_threadsafe(
                                progress_manager.update_progress(
                                    session_id, 
                                    progress_value, 
                                    f"Đang phân tích frame {frame_count + 1}/{total_frames}..."
                                ),
                                async_loop
                            )
                            # Wait a bit to ensure update is sent
                            future.result(timeout=0.2)
                            _logger.info(f"Progress update sent: {progress_value}% (frame {frame_count + 1}/{total_frames})")
                        except Exception as e:
                            _logger.error(f"Progress update error: {e}")

                # Detect persons
                detections = self._detector.detect_persons(frame)
                if not detections:
                    # Resize frame if needed before writing
                    output_frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA) if should_resize else frame
                    video_writer.write(output_frame)
                    frame_count += 1
                    continue

                # Filter detections by confidence
                bboxes = np.array(
                    [det[:4] for det in detections if det[4] >= min_confidence]
                )
                if len(bboxes) == 0:
                    # Resize frame if needed before writing
                    output_frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA) if should_resize else frame
                    video_writer.write(output_frame)
                    frame_count += 1
                    continue

                # Run pose estimation
                pose_results = inference_topdown(self._model, frame, bboxes)
                data_samples = merge_data_samples(pose_results)

                # Prepare visualization
                if draw_bbox or draw_keypoints:
                    self._visualizer.add_datasample(
                        "result",
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        data_sample=data_samples,
                        draw_gt=False,
                        draw_heatmap=False,
                        draw_bbox=draw_bbox,
                        show_kpt_idx=False,
                        show=False,
                        wait_time=0,
                        kpt_thr=min_confidence,
                    )
                    vis_frame = cv2.cvtColor(
                        self._visualizer.get_image(), cv2.COLOR_RGB2BGR
                    )
                    if frame_count == 0:
                        _logger.info(
                            f"Drawing skeleton: bbox={draw_bbox}, keypoints={draw_keypoints}"
                        )
                else:
                    vis_frame = frame.copy()

                # Extract keypoint data
                frame_data = []
                squat_analysis = None
                pushup_analysis = None
                barbell_analysis = None
                lunge_analysis = None

                if hasattr(data_samples, "pred_instances"):
                    pred_instances = data_samples.pred_instances
                    total_detections += len(pred_instances.bboxes)

                    for bbox, keypoints, scores in zip(
                        pred_instances.bboxes,
                        pred_instances.keypoints,
                        pred_instances.keypoint_scores,
                    ):
                        detection = {
                            "bbox": bbox.tolist(),
                            "keypoints": keypoints.tolist(),
                            "scores": scores.tolist(),
                        }
                        frame_data.append(detection)

                        # Perform squat analysis on first detected person
                        if analyze_squat and squat_analysis is None:
                            squat_analysis = self._squat_analyzer.process_frame(
                                keypoints, scores
                            )
                        
                        # Perform pushup analysis on first detected person
                        if analyze_pushup and pushup_analysis is None:
                            # Convert keypoints to format expected by pushup analyzer
                            kp_dict = self._extract_keypoints_for_pushup(keypoints, scores)
                            if kp_dict:
                                self._pushup_analyzer.process_frame(kp_dict)
                                pushup_analysis = self._pushup_analyzer.get_analysis_result()
                        
                        # Perform barbell analysis on first detected person
                        if analyze_barbell and barbell_analysis is None:
                            barbell_analysis = self._barbell_analyzer.process_frame(
                                keypoints, scores
                            )
                        
                        # Perform lunge analysis on first detected person
                        if analyze_lunge and lunge_analysis is None:
                            kp_dict = self._extract_keypoints_for_lunge(keypoints, scores, min_confidence)
                            if kp_dict:
                                lunge_result = self._lunge_analyzer.process_frame(kp_dict)

                                if isinstance(lunge_result, tuple):
                                    if len(lunge_result) == 5:
                                        state, rep_count, angles, feedback, vertical_point = lunge_result
                                    else:
                                        state, rep_count, angles, feedback = lunge_result
                                        vertical_point = None
                                else:
                                    continue

                                lunge_analysis = {
                                    'state': state,
                                    'rep_count': rep_count,
                                    'depth_angle': angles.get('depth') if isinstance(angles, dict) else None,
                                    'back_angle': angles.get('back') if isinstance(angles, dict) else None,
                                    'depth_feedback': feedback.get('depth', '') if isinstance(feedback, dict) else '',
                                    'back_feedback': feedback.get('back', '') if isinstance(feedback, dict) else '',
                                    'rep_feedback': feedback.get('rep', '') if isinstance(feedback, dict) else '',
                                    'vertical_point': vertical_point
                                }

                # Draw squat feedback if analyzing
                if analyze_squat and squat_analysis is not None:
                    vis_frame = self._draw_squat_feedback(vis_frame, squat_analysis)
                
                # Draw pushup feedback if analyzing
                if analyze_pushup and pushup_analysis is not None:
                    vis_frame = self._draw_pushup_feedback(vis_frame, pushup_analysis)
                
                # Draw barbell feedback if analyzing
                if analyze_barbell and barbell_analysis is not None:
                    vis_frame = self._draw_barbell_feedback(vis_frame, barbell_analysis)
                
                # Draw lunge feedback if analyzing
                if analyze_lunge and lunge_analysis is not None:
                    vis_frame = self._draw_lunge_feedback(vis_frame, lunge_analysis)

                # Save frame data if requested
                if save_keypoints:
                    keypoints_data.append(
                        {
                            "frame_id": frame_count,
                            "detections": frame_data,
                            "squat_analysis": squat_analysis,
                            "pushup_analysis": pushup_analysis,
                        }
                    )

                # Write output frame (resize if needed)
                output_frame = cv2.resize(vis_frame, (output_width, output_height), interpolation=cv2.INTER_AREA) if should_resize else vis_frame
                video_writer.write(output_frame)
                frame_count += 1

                # Log progress periodically
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    _logger.info(
                        f"Processed {frame_count}/{total_frames} frames ({current_fps:.2f} FPS)"
                    )

        finally:
            cap.release()
            video_writer.release()
            
            # Cleanup converted WebM file if exists
            if video_path != original_video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    _logger.info("Cleaned up temporary converted video")
                except Exception as e:
                    _logger.warning(f"Failed to cleanup temp file: {e}")

        # Convert AVI to MP4 with H264 for browser compatibility
        if os.path.exists(temp_output_path):
            _logger.info("Converting AVI to MP4 with H264 for browser compatibility...")
            import subprocess

            try:
                # Use ffmpeg to convert to H264 MP4 with good browser compatibility
                result = subprocess.run(
                    [
                        "ffmpeg",
                        "-i", temp_output_path,
                        "-c:v", "libx264",
                        "-preset", "fast",
                        "-crf", "23",
                        "-pix_fmt", "yuv420p",  # Ensures compatibility
                        "-movflags", "+faststart",  # Enable streaming
                        "-y",  # Overwrite
                        output_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                if result.returncode == 0 and os.path.exists(output_path):
                    # Success, remove temp AVI
                    os.remove(temp_output_path)
                    _logger.info("Successfully converted to H264 MP4")
                else:
                    _logger.error(f"FFmpeg conversion failed: {result.stderr}")
                    # Fallback: rename AVI to MP4 (won't play in browser but at least something)
                    if os.path.exists(temp_output_path):
                        os.rename(temp_output_path, output_path)
                        _logger.warning("Using AVI file as fallback (may not play in browser)")
            except Exception as e:
                _logger.error(f"Failed to convert video to H264: {e}")
                # Fallback: rename AVI to MP4
                if os.path.exists(temp_output_path):
                    os.rename(temp_output_path, output_path)
                    _logger.warning("Using AVI file as fallback (may not play in browser)")
        else:
            raise ValueError(f"Temporary output video not found: {temp_output_path}")

        # Calculate statistics
        processing_time = time.time() - start_time
        fps = frame_count / processing_time if processing_time > 0 else 0

        _logger.info(
            f"Video processing completed: {frame_count}/{total_frames} frames "
            f"in {processing_time:.2f}s ({fps:.2f} FPS)"
        )

        stats = {
            "frame_count": frame_count,
            "total_detections": total_detections,
            "processing_time": processing_time,
            "fps": fps,
            "input_fps": input_fps,
            "output_fps": output_fps,
            "resolution": f"{frame_width}x{frame_height}",
        }

        # Add squat analysis summary
        if analyze_squat:
            squat_summary = self._squat_analyzer.get_summary()
            stats.update(squat_summary)
            _logger.info(
                f"Squat Analysis: {squat_summary['total_squats']} total "
                f"({squat_summary['valid_squats']} valid, "
                f"{squat_summary['invalid_squats']} invalid)"
            )
        
        # Add pushup analysis summary
        if analyze_pushup:
            pushup_summary = self._pushup_analyzer.get_summary()
            stats['total_pushups'] = pushup_summary['total_reps']
            stats['valid_pushups'] = pushup_summary['valid_reps']
            stats['invalid_pushups'] = pushup_summary['invalid_reps']
            stats['pushup_rep_feedbacks'] = pushup_summary['rep_feedbacks']
            _logger.info(
                f"Pushup Analysis: {pushup_summary['total_reps']} total reps "
                f"({pushup_summary['valid_reps']} valid, "
                f"{pushup_summary['invalid_reps']} invalid)"
            )
        
        # Add barbell analysis summary
        if analyze_barbell:
            barbell_summary = self._barbell_analyzer.get_summary()
            stats['total_barbells'] = barbell_summary['total_reps']
            stats['valid_barbells'] = barbell_summary['valid_reps']
            stats['invalid_barbells'] = barbell_summary['invalid_reps']
            stats['barbell_rep_feedbacks'] = convert_numpy_types(barbell_summary.get('rep_feedbacks', []))
            _logger.info(
                f"Barbell Dead Row Analysis: {barbell_summary['total_reps']} total reps "
                f"({barbell_summary['valid_reps']} valid, "
                f"{barbell_summary['invalid_reps']} invalid)"
            )
        
        # Add lunge analysis summary
        if analyze_lunge:
            lunge_summary = self._lunge_analyzer.get_summary()
            stats['total_lunges'] = lunge_summary['total_reps']
            stats['valid_lunges'] = lunge_summary['valid_reps']
            stats['invalid_lunges'] = lunge_summary['invalid_reps']
            stats['lunge_rep_feedbacks'] = convert_numpy_types(lunge_summary.get('rep_feedbacks', []))
            _logger.info(
                f"Dumbbell Reverse Lunge Analysis: {lunge_summary['total_reps']} total reps "
                f"({lunge_summary['valid_reps']} valid, "
                f"{lunge_summary['invalid_reps']} invalid)"
            )

        # Save keypoints if requested
        if save_keypoints and keypoints_data:
            keypoints_path = os.path.splitext(output_path)[0] + "_keypoints.json"
            
            # Prepare summaries for JSON - convert numpy types to Python native types
            squat_summary_json = convert_numpy_types(self._squat_analyzer.get_summary()) if analyze_squat else None
            pushup_summary_json = convert_numpy_types(self._pushup_analyzer.get_summary()) if analyze_pushup else None
            barbell_summary_json = convert_numpy_types(self._barbell_analyzer.get_summary()) if analyze_barbell else None
            lunge_summary_json = convert_numpy_types(self._lunge_analyzer.get_summary()) if analyze_lunge else None
            
            # Convert entire data structure to avoid any numpy type issues
            data_to_save = convert_numpy_types({
                "video_file": os.path.basename(video_path),
                "total_frames": frame_count,
                "total_detections": total_detections,
                "fps": input_fps,
                "resolution": f"{frame_width}x{frame_height}",
                "squat_summary": squat_summary_json,
                "pushup_summary": pushup_summary_json,
                "barbell_summary": barbell_summary_json,
                "lunge_summary": lunge_summary_json,
                "keypoints_data": keypoints_data,
            })
            
            with open(keypoints_path, "w") as f:
                json.dump(data_to_save, f, indent=2)
                
            stats["keypoints_file"] = keypoints_path
        
        # Final progress update
        if progress_manager and session_id and async_loop:
            try:
                import asyncio
                future = asyncio.run_coroutine_threadsafe(
                    progress_manager.update_progress(session_id, 100, "Hoàn thành!"),
                    async_loop
                )
                future.result(timeout=0.5)
                _logger.info(f"Final progress update sent: 100%")
            except Exception as e:
                _logger.warning(f"Final progress update error: {e}")

        return output_path, stats
