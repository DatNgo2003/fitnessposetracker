"""
Squat Analysis Module
Analyzes squat form using rule-based logic on pose keypoints.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


# COCO Keypoint indexes for squat analysis
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


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculate angle between three points.

    Args:
        a: First point (x, y)
        b: Vertex point (x, y)
        c: Third point (x, y)

    Returns:
        Angle in degrees
    """
    ba = a - b
    bc = c - b
    angle_radians = np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0])
    angle_degrees = np.degrees(angle_radians)
    angle_degrees = np.abs(angle_degrees)
    if angle_degrees > 180.0:
        angle_degrees = 360 - angle_degrees
    return angle_degrees


class SquatAnalyzer:
    """
    Analyzes squat form in real-time using state machine.
    Tracks reps and provides feedback on depth and back angle.
    Supports both front view and side view camera angles.
    """

    # --- INIT FUNCTION MODIFIED FOR COMPATIBILITY ---
    def __init__(self, smoothing_window: int = 5, is_side_view: bool = False):
        self.state = "CHƯA VÀO TƯ THẾ"
        self.rep_counter = 0
        self.down_frames_counter = 0
        
        # --- Automatic conversion logic (NEW) ---
        # Convert 'is_side_view' (True/False) to 'side' logic
        self.is_side_view = is_side_view
        if is_side_view:
            # If side view, default to tracking right side (can change to 'left' if needed)
            self.tracking_side = 'right'
        else:
            # If front view
            self.tracking_side = 'front'
        # ----------------------------------------
        
        self.hip_history: List[float] = []

        # (Remaining thresholds and configurations)
        self.SMOOTHING_WINDOW = smoothing_window
        self.DEPTH_THRESHOLD = 75
        self.KNEE_START_BEND_THRESHOLD = 170 
        self.KNEE_STRAIGHT_THRESHOLD = 160 
        self.MOTION_TRIGGER_FRAMES = 3  
        self.BACK_THRESHOLD = 25

        self.lowest_knee_angle = 180.0
        self.current_knee_angle = 180.0
        self.current_back_angle = 0.0

        self.depth_feedback = "" 
        self.back_feedback = ""
        self.rep_feedbacks: List[Dict[str, Any]] = []

        self.check_depth = True
        # Chỉ kiểm tra lưng khi quay ngang (side view); góc chính diện chỉ xét độ sâu
        self.check_back = self.is_side_view
        
        # Track if any back error occurred during current rep
        self.has_back_error_in_rep = False

        # Depth angle for visualization
        # SIDE KEYPOINTS
        self.s_hip: Optional[np.ndarray] = None
        self.s_knee: Optional[np.ndarray] = None
        self.s_ankle: Optional[np.ndarray] = None

        # FRONT KEYPOINTS
        self.hip_midpoint: Optional[np.ndarray] = None
        self.knee_midpoint: Optional[np.ndarray] = None
        self.ankle_midpoint: Optional[np.ndarray] = None
        
        # Back angle for visualization
        self.shoulder_midpoint: Optional[np.ndarray] = None
        self.hip_midpoint: Optional[np.ndarray] = None
        self.vertical_point: Optional[np.ndarray] = None

    def reset(self):

        self.state = "CHƯA VÀO TƯ THẾ"
        self.rep_counter = 0
        # ... (remaining parts unchanged)

    def _extract_keypoints(
        self, keypoints_array: np.ndarray, scores: np.ndarray, min_score: float = 0.5
    ) -> Dict[str, np.ndarray]:

        keypoints = {}
        for joint_name, index in COCO_KEYPOINT_INDEXES.items():
            if scores[index] > min_score:
                keypoints[joint_name] = keypoints_array[index]
        return keypoints

    def _calculate_midpoint(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:

        return (p1 + p2) / 2.0

    def process_frame(
        self, keypoints_array: np.ndarray, scores: np.ndarray
    ) -> Dict[str, Any]:
        
        self.back_feedback = ""
        keypoints = self._extract_keypoints(keypoints_array, scores)
        if not keypoints:
            return self._get_current_state()

        shoulder_midpoint, hip_midpoint = None, None
        
        # --- 1. ANGLE CALCULATION (FLEXIBLE - STILL WORKING WELL) ---
        try:
            if self.is_side_view: # For side view
                s_hip = f"{self.tracking_side}_hip"
                s_knee = f"{self.tracking_side}_knee"
                s_ankle = f"{self.tracking_side}_ankle"

                self.s_hip = keypoints[s_hip]
                self.s_knee = keypoints[s_knee] 
                self.s_ankle = keypoints[s_ankle]

                self.current_knee_angle = calculate_angle(
                    keypoints[s_hip], keypoints[s_knee], keypoints[s_ankle]
                )
            else: # For front view
                angle_l = calculate_angle(
                    keypoints["left_hip"], keypoints["left_knee"], keypoints["left_ankle"]
                )
                angle_r = calculate_angle(
                    keypoints["right_hip"], keypoints["right_knee"], keypoints["right_ankle"]
                )
                self.current_knee_angle = (angle_l + angle_r) / 2.0
                hip_midpoint = self._calculate_midpoint(
                    keypoints["left_hip"], keypoints["right_hip"]  )
                knee_midpoint = self._calculate_midpoint(
                    keypoints["left_knee"], keypoints["right_knee"]  )
                ankle_midppoint = self._calculate_midpoint(
                    keypoints["left_ankle"], keypoints["right_ankle"]  )

        except KeyError:
             pass 

        # (Calculate back angle - unchanged)
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
            vertical_point = np.array([hip_midpoint[0], hip_midpoint[1] - 100])
            self.vertical_point = vertical_point
            self.current_back_angle = calculate_angle(
                shoulder_midpoint, hip_midpoint, vertical_point
            )

        # --- 2. HIP MOVEMENT TRACKING (FLEXIBLE - STILL WORKING WELL) ---
        hip_to_track_y = None
        try:
            if self.is_side_view:
                s_hip = f"{self.tracking_side}_hip"
                hip_to_track_y = keypoints[s_hip][1]
            elif hip_midpoint is not None: # For front view
                hip_to_track_y = hip_midpoint[1]
        except KeyError:
            hip_to_track_y = None


        # --- 3. STATE MACHINE (Upgraded logic maintained) ---
        if hip_to_track_y is not None:
            self.hip_history.append(hip_to_track_y)
            if len(self.hip_history) > self.SMOOTHING_WINDOW * 2:
                self.hip_history.pop(0)
            if len(self.hip_history) < self.SMOOTHING_WINDOW * 2:
                return self._get_current_state()

            recent_hip_avg = np.mean(self.hip_history[-self.SMOOTHING_WINDOW :])
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
                    self.feedback = "BẮT ĐẦU"
                    # Reset back error flag for new rep
                    self.has_back_error_in_rep = False

            elif self.state == "XUỐNG":
                self.lowest_knee_angle = min(
                    self.lowest_knee_angle, self.current_knee_angle
                )

                if is_hip_moving_up:
                    self.rep_counter += 1 
                    if self.check_depth:
                        if self.lowest_knee_angle <= self.DEPTH_THRESHOLD:
                            self.depth_feedback = f"LẦN {self.rep_counter}: Độ sâu rất chuẩn! Hông đạt đến vị trí ngang gối, đảm bảo an toàn và hiệu quả."
                        else:
                            self.depth_feedback = f"LẦN {self.rep_counter}: Bạn chưa hạ người đủ sâu. Hãy hạ hông đến ngang hoặc thấp hơn gối để kích hoạt tối đa nhóm cơ đùi và mông."
                    else:
                        self.depth_feedback = f"LẦN {self.rep_counter}"
                    
                    self._save_rep_feedback()
                    self.state = "LÊN"
                    # Reset back error flag for next rep
                    self.has_back_error_in_rep = False

            elif self.state == "LÊN":
                if self.current_knee_angle > self.KNEE_STRAIGHT_THRESHOLD:
                    self.state = "CHƯA VÀO TƯ THẾ"
                    self.depth_feedback = "HOÀN THÀNH"

        # (Back checking logic maintained)
        if self.check_back and self.state in ["XUỐNG", "LÊN"]:
            if self.current_back_angle > self.BACK_THRESHOLD:
                self.back_feedback = "Lưng bạn đang bị cong hoặc nghiêng quá mức. Hãy mở ngực, siết cơ bụng và giữ cột sống trung lập trong suốt chuyển động."
                # Mark that this rep has a back error
                self.has_back_error_in_rep = True
            else:
                self.back_feedback = "Rất tốt! Lưng bạn thẳng và ổn định, giúp tối ưu sức mạnh và giảm nguy cơ chấn thương."

        return self._get_current_state()
    def _save_rep_feedback(self):
        """Save feedback for the completed rep."""
        # Determine back_feedback based on whether any back error occurred during this rep
        # Logic: use my code, but messages from their version
        if self.has_back_error_in_rep:
            final_back_feedback = "Lưng bạn đang bị cong hoặc nghiêng quá mức. Hãy mở ngực, siết cơ bụng và giữ cột sống trung lập trong suốt chuyển động."
        else:
            final_back_feedback = "Rất tốt! Lưng bạn thẳng và ổn định, giúp tối ưu sức mạnh và giảm nguy cơ chấn thương."
        
        rep_data = {
            "rep_id": self.rep_counter ,
            "depth_feedback": self.depth_feedback,
            "back_feedback": final_back_feedback,
            "lowest_knee_angle": float(self.lowest_knee_angle),
            "is_valid": bool(
                "Bạn chưa hạ người đủ sâu" not in self.depth_feedback
                and "Lưng bạn đang bị cong hoặc nghiêng quá mức" not in final_back_feedback
            ),
        }
        self.rep_feedbacks.append(rep_data)

    def _get_current_state(self) -> Dict[str, Any]:
        """Get current analysis state."""
        return {
            "state": self.state,
            "rep_counter": self.rep_counter,
            "angle_knee": self.lowest_knee_angle if self.state != "CHƯA VÀO TƯ THẾ" else None,
            "angle_back": self.current_back_angle,
            "depth_feedback": self.depth_feedback,
            "back_feedback": self.back_feedback,
            "shoulder_midpoint": self.shoulder_midpoint.tolist() if self.shoulder_midpoint is not None else None,
            "hip_midpoint": self.hip_midpoint.tolist() if self.hip_midpoint is not None else None,
            "vertical_point": self.vertical_point.tolist() if self.vertical_point is not None else None,
            "s_hip": self.s_hip.tolist() if self.s_hip is not None else None,
            "s_knee": self.s_knee.tolist() if self.s_knee is not None else None,
            "s_ankle": self.s_ankle.tolist() if self.s_ankle is not None else None,
            "hip_midpoint": self.hip_midpoint.tolist() if self.hip_midpoint is not None else None,
            "knee_midpoint": self.knee_midpoint.tolist() if self.knee_midpoint is not None else None,
            "ankle_midpoint": self.ankle_midpoint.tolist() if self.ankle_midpoint is not None else None,
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all reps analyzed.

        Returns:
            Summary dictionary with total_squats and per-rep feedback
        """
        valid_reps = sum(1 for rep in self.rep_feedbacks if rep["is_valid"])
        invalid_reps = len(self.rep_feedbacks) - valid_reps

        return {
            "total_squats": self.rep_counter,
            "valid_squats": valid_reps,
            "invalid_squats": invalid_reps,
            "rep_feedbacks": self.rep_feedbacks,
        }
