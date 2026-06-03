"""
Dumbbell Reverse Lunge Analysis Module
Analyzes dumbbell reverse lunge form using rule-based logic on pose keypoints.
Based on squat_analyzer.py structure with side-specific knee tracking.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


# COCO Keypoint indexes for lunge analysis
COCO_KEYPOINT_INDEXES = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


def calculate_angle(a, b, c) -> float:
    """
    Calculate angle between three points.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    angle_radians = np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0])
    angle_degrees = np.degrees(angle_radians)
    angle_degrees = np.abs(angle_degrees)
    if angle_degrees > 180.0:
        angle_degrees = 360 - angle_degrees
    return angle_degrees


class DumbbellReverseLungeAnalyzer:
    """
    Analyzes dumbbell reverse lunge form in real-time using state machine.
    Tracks reps and provides feedback on depth and back angle.
    
    Key features:
    - Depth check: Based on knee angle of the tracked side (left/right)
    - Back check: Same as squat (shoulder-hip angle vs vertical)
    - Deep is GOOD (like squat) - knee angle should be below threshold
    """

    def __init__(self, side: str = 'right', smoothing_window: int = 5, is_side_view: bool = False):
        self.state = "CHƯA VÀO TƯ THẾ"
        self.rep_counter = 0
        self.down_frames_counter = 0
        
        # Store the side parameter for tracking (which leg to track)
        self.tracking_side = side  # 'left', 'right', or 'front'
        
        # Determine is_side_view based on side parameter or explicit is_side_view
        if side in ['left', 'right']:
            self.is_side_view = True
        else:
            self.is_side_view = is_side_view
        
        self.hip_history: List[float] = []

        # Thresholds and configurations
        self.SMOOTHING_WINDOW = smoothing_window
        self.DEPTH_THRESHOLD = 90  # Knee angle threshold for good depth
        self.KNEE_START_BEND_THRESHOLD = 160
        self.KNEE_STRAIGHT_THRESHOLD = 150
        self.MOTION_TRIGGER_FRAMES = 3  
        self.BACK_THRESHOLD = 25

        self.lowest_knee_angle = 180.0
        self.current_knee_angle = 180.0
        self.current_back_angle = 0.0

        self.depth_feedback = "" 
        self.back_feedback = ""
        self.rep_feedbacks: List[Dict[str, Any]] = []

        self.check_depth = True
        self.check_back = self.is_side_view
        self.has_back_error_in_rep = False

        # Visualization points
        self.shoulder_midpoint = None
        self.hip_midpoint = None
        self.vertical_point = None

    def reset(self):
        """Reset analyzer state for new video/session."""
        self.state = "CHƯA VÀO TƯ THẾ"
        self.rep_counter = 0
        self.down_frames_counter = 0
        self.hip_history = []
        self.lowest_knee_angle = 180.0
        self.current_knee_angle = 180.0
        self.current_back_angle = 0.0
        self.depth_feedback = ""
        self.back_feedback = ""
        self.rep_feedbacks = []
        self.has_back_error_in_rep = False

    def _calculate_midpoint(self, p1, p2):
        """Calculate midpoint between two points."""
        return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

    def process_frame(self, keypoints: Dict[str, tuple]) -> tuple:
        """
        Process a single frame and update state machine.
        
        Args:
            keypoints: Dict of keypoint names to (x, y) coordinates
            
        Returns:
            Tuple of (state, rep_count, angles_dict, feedback_dict)
        """
        self.back_feedback = ""
        
        if not keypoints:
            return self._get_result_tuple()

        # Check required keypoints
        s_hip_key = f"{self.tracking_side}_hip"
        s_knee_key = f"{self.tracking_side}_knee"
        if s_knee_key not in keypoints:
            return self._get_result_tuple()

        shoulder_midpoint, hip_midpoint = None, None
        
        # --- 1. CALCULATE KNEE ANGLE (DEPTH) ---
        try:
            if self.is_side_view and self.tracking_side in ['left', 'right']:
                s_hip = f"{self.tracking_side}_hip"
                s_knee = f"{self.tracking_side}_knee"
                s_ankle = f"{self.tracking_side}_ankle"

                if s_hip in keypoints and s_knee in keypoints and s_ankle in keypoints:
                    self.current_knee_angle = calculate_angle(
                        keypoints[s_hip], keypoints[s_knee], keypoints[s_ankle]
                    )
            else:
                # Front view: use average of both legs
                if all(k in keypoints for k in ["left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"]):
                    angle_l = calculate_angle(
                        keypoints["left_hip"], keypoints["left_knee"], keypoints["left_ankle"]
                    )
                    angle_r = calculate_angle(
                        keypoints["right_hip"], keypoints["right_knee"], keypoints["right_ankle"]
                    )
                    self.current_knee_angle = (angle_l + angle_r) / 2.0
        except (KeyError, TypeError):
            pass 

        # --- 2. CALCULATE BACK ANGLE (shoulder-hip vs vertical) ---
        try:
            if "left_shoulder" in keypoints and "right_shoulder" in keypoints:
                shoulder_midpoint = self._calculate_midpoint(
                    keypoints["left_shoulder"], keypoints["right_shoulder"]
                )
                self.shoulder_midpoint = shoulder_midpoint
                
            if "left_hip" in keypoints and "right_hip" in keypoints:
                hip_midpoint = self._calculate_midpoint(
                    keypoints["left_hip"], keypoints["right_hip"]
                )
                self.hip_midpoint = hip_midpoint
                
            if shoulder_midpoint is not None and hip_midpoint is not None:
                vertical_point = (hip_midpoint[0], hip_midpoint[1] - 100)
                self.vertical_point = vertical_point
                self.current_back_angle = calculate_angle(
                    shoulder_midpoint, hip_midpoint, vertical_point
                )
        except (KeyError, TypeError):
            pass

        # --- 3. HIP MOVEMENT TRACKING ---
        hip_to_track_y = None
        try:
            if self.is_side_view and self.tracking_side in ['left', 'right']:
                s_hip = f"{self.tracking_side}_hip"
                if s_hip in keypoints:
                    hip_to_track_y = keypoints[s_hip][1]
            elif hip_midpoint is not None:
                hip_to_track_y = hip_midpoint[1]
        except (KeyError, TypeError):
            hip_to_track_y = None

        # --- 4. STATE MACHINE ---
        if hip_to_track_y is not None:
            self.hip_history.append(hip_to_track_y)
            if len(self.hip_history) > self.SMOOTHING_WINDOW * 2:
                self.hip_history.pop(0)
            if len(self.hip_history) < self.SMOOTHING_WINDOW * 2:
                return self._get_result_tuple()

            recent_hip_avg = np.mean(self.hip_history[-self.SMOOTHING_WINDOW:])
            previous_hip_avg = np.mean(
                self.hip_history[-self.SMOOTHING_WINDOW * 2 : -self.SMOOTHING_WINDOW]
            )

            is_hip_moving_down = recent_hip_avg > previous_hip_avg + 2
            is_hip_moving_up = recent_hip_avg < previous_hip_avg - 2
            is_knee_bending = self.current_knee_angle < self.KNEE_START_BEND_THRESHOLD

            if self.state == "CHƯA VÀO TƯ THẾ":
                if is_hip_moving_down and is_knee_bending:
                    self.down_frames_counter += 1
                else:
                    self.down_frames_counter = 0

                if self.down_frames_counter >= self.MOTION_TRIGGER_FRAMES:
                    self.state = "XUỐNG"
                    self.lowest_knee_angle = self.current_knee_angle
                    self.down_frames_counter = 0
                    self.depth_feedback = "BẮT ĐẦU"
                    self.has_back_error_in_rep = False

            elif self.state == "XUỐNG":
                self.lowest_knee_angle = min(
                    self.lowest_knee_angle, self.current_knee_angle
                )

                if is_hip_moving_up:
                    self.rep_counter += 1 
                    if self.check_depth:
                        if self.lowest_knee_angle <= self.DEPTH_THRESHOLD:
                            self.depth_feedback = f"LẦN {self.rep_counter}: Độ sâu rất chuẩn! Đầu gối sau hạ thấp gần sàn, giúp kích hoạt tối đa cơ đùi và mông."
                        else:
                            self.depth_feedback = f"LẦN {self.rep_counter}: Bạn chưa hạ đầu gối sau đủ thấp. Hãy hạ đến khi đầu gối gần chạm sàn để đạt biên độ chuẩn."
                    else:
                        self.depth_feedback = f"LẦN {self.rep_counter}"
                    
                    self._save_rep_feedback()
                    self.state = "LÊN"
                    self.has_back_error_in_rep = False

            elif self.state == "LÊN":
                if self.current_knee_angle > self.KNEE_STRAIGHT_THRESHOLD:
                    self.state = "CHƯA VÀO TƯ THẾ"
                    self.depth_feedback = "HOÀN THÀNH"

        # --- 5. BACK CHECK (same as squat) ---
        if self.check_back and self.state in ["XUỐNG", "LÊN"]:
            if self.current_back_angle > self.BACK_THRESHOLD:
                self.back_feedback = "Bạn đang nghiêng người về trước. Hãy giữ ngực mở và lưng thẳng để phân bổ trọng lượng đều hơn."
                self.has_back_error_in_rep = True
            else:
                self.back_feedback = "Rất tốt! Lưng bạn thẳng và ổn định, giúp tối ưu sức mạnh và giảm nguy cơ chấn thương."

        return self._get_result_tuple()

    def _get_result_tuple(self) -> tuple:
        """Return result tuple compatible with pose.py expectations."""
        angles = {
            'depth': self.current_knee_angle,
            'back': self.current_back_angle
        }
        feedback = {
            'depth': self.depth_feedback,
            'back': self.back_feedback,
        }
        return (self.state, self.rep_counter, angles, feedback)

    def _save_rep_feedback(self):
        """Save feedback for the completed rep."""
        if self.has_back_error_in_rep:
            final_back_feedback = "Bạn đang nghiêng người về trước. Hãy giữ ngực mở và lưng thẳng để phân bổ trọng lượng đều hơn."
        else:
            final_back_feedback = "Rất tốt! Lưng bạn thẳng và ổn định, giúp tối ưu sức mạnh và giảm nguy cơ chấn thương."
        
        rep_data = {
            "rep_id": self.rep_counter,
            "depth_feedback": self.depth_feedback,
            "back_feedback": final_back_feedback,
            "lowest_knee_angle": float(self.lowest_knee_angle),
            "is_valid": "Bạn chưa hạ đầu gối sau đủ thấp" not in self.depth_feedback
            and "Bạn đang nghiêng người về trước" not in final_back_feedback,
        }
        self.rep_feedbacks.append(rep_data)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all reps analyzed."""
        valid_reps = sum(1 for rep in self.rep_feedbacks if rep["is_valid"])
        invalid_reps = len(self.rep_feedbacks) - valid_reps

        return {
            "total_reps": self.rep_counter,
            "valid_reps": valid_reps,
            "invalid_reps": invalid_reps,
            "rep_feedbacks": self.rep_feedbacks,
        }
