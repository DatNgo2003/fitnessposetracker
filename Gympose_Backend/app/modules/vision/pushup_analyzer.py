"""
Pushup Analysis Module

This module provides real-time pushup form analysis using pose estimation keypoints.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple

_logger = logging.getLogger(__name__)


# COCO Keypoint indices
COCO_KEYPOINT_INDEXES = {
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16
}


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculate angle at point b formed by points a, b, c.
    
    Args:
        a, b, c: Points as numpy arrays [x, y] or tuples (x, y)
        
    Returns:
        Angle in degrees (0-180)
    """
    # Convert to numpy arrays if they are tuples
    a = np.array(a) if not isinstance(a, np.ndarray) else a
    b = np.array(b) if not isinstance(b, np.ndarray) else b
    c = np.array(c) if not isinstance(c, np.ndarray) else c
    
    ba = a - b
    bc = c - b
    angle_radians = np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0])
    angle_degrees = np.degrees(angle_radians)
    angle_degrees = np.abs(angle_degrees)
    if angle_degrees > 180.0:
        angle_degrees = 360 - angle_degrees
    return angle_degrees


class PushupAnalyzer:
    def __init__(self, side: str = 'right', smoothing_window: int = 5, check_depth: bool = True, check_body_straight: bool = True):
        """
        Initialize Pushup Analyzer with improved motion detection.
        
        Args:
            side: 'left', 'right', or 'front' - camera view angle
            smoothing_window: Number of frames for smoothing
            check_depth: Whether to check pushup depth
            check_body_straight: Whether to check body alignment
        """
        # State tracking
        self.state = 'CHƯA VÀO TƯ THẾ'
        self.rep_counter = 0
        self.valid_reps = 0
        self.invalid_reps = 0
        self.rep_feedbacks = []
        self.current_rep_errors = []
        self.worst_body_angle_in_rep = 180
        
        # History tracking
        self.shoulder_history = []
        self.elbow_angle_history = []
        
        # Store the side/view parameter for tracking
        # Giá trị hợp lệ: 'left', 'right', 'front'
        # - 'left': Camera quay từ bên trái người tập (side view)
        # - 'right': Camera quay từ bên phải người tập (side view)
        # - 'front': Camera quay từ phía trước người tập (front view)
        self.tracking_side = side
        
        # Xác định loại góc nhìn dựa trên tham số side
        # Side view: có thể đo được góc body (thân thẳng) chính xác
        # Front view: chỉ đo được góc khuỷu tay, không đo được body angle chính xác
        if side in ['left', 'right']:
            self.side = side  # Dùng cho tracking keypoints
            self.is_side_view = True
            self.is_front_view = False
        else:  # side == 'front' hoặc các giá trị khác
            self.side = 'right'  # Mặc định dùng right cho tracking
            self.is_side_view = False
            self.is_front_view = True
        
        # Noise filtering counters
        self.stable_ready_frames = 0
        self.READY_FRAME_THRESHOLD = 5
        self.down_frames_counter = 0
        self.DOWN_FRAME_THRESHOLD = 3
        self.up_frames_counter = 0
        self.UP_FRAME_THRESHOLD = 3

        # Feature toggles
        self.check_depth = check_depth
        # Body straight chỉ kiểm tra khi side view (vì front view không đo được chính xác)
        self.check_body_straight = check_body_straight and self.is_side_view

        # Thresholds
        self.SMOOTHING_WINDOW = smoothing_window
        self.ELBOW_ANGLE_THRESHOLD_MIN = 90
        self.BODY_ANGLE_THRESHOLD_MIN = 160
        self.HIP_STRAIGHT_ANGLE_THRESHOLD = 160
        self.MOVEMENT_THRESHOLD = 2  # Ngưỡng chuyển động vai (pixels)
        self.ELBOW_CHANGE_THRESHOLD = 3  # Ngưỡng thay đổi góc khuỷu tay (degrees)
        
        # Front view plank detection thresholds
        self.PLANK_VERTICAL_RATIO = 2.0  # vertical_span < shoulder_width * ratio = plank
        self.PLANK_ELBOW_THRESHOLD = 140  # Góc khuỷu tay tối thiểu khi ở tư thế plank (tay gần duỗi thẳng)

        # Current state variables
        self.lowest_elbow_angle = 180
        self.current_elbow_angle = 0
        self.current_body_angle = 0
        self.current_hip_angle = 0
        self.feedback = "HÃY VÀO TƯ THẾ PLANK"
        self.body_feedback = ""
    
    def reset(self) -> None:
        """Reset analyzer state for new video."""
        self.state = 'CHƯA VÀO TƯ THẾ'
        self.rep_counter = 0
        self.valid_reps = 0
        self.invalid_reps = 0
        self.shoulder_history = []
        self.elbow_angle_history = []
        self.stable_ready_frames = 0
        self.down_frames_counter = 0
        self.up_frames_counter = 0
        self.lowest_elbow_angle = 180
        self.current_elbow_angle = 0
        self.current_body_angle = 0
        self.current_hip_angle = 0
        self.feedback = "HÃY VÀO TƯ THẾ PLANK"
        self.body_feedback = ""
        self.rep_feedbacks = []
        self.current_rep_errors = []
        self.worst_body_angle_in_rep = 180
    
    def process_frame(self, keypoints):
        """
        Process one frame of keypoints with improved motion detection.
        Uses BOTH shoulder movement AND elbow angle change rate.
        """
        self.body_feedback = ""
        if not keypoints:
            return None, None, None

        is_horizontal = False
        is_moving_down = False
        is_moving_up = False
        is_hip_straight = False
        is_plank_front_view = False  # Flag cho front view plank detection
        
        # Generate joint names based on selected side
        s_shoulder = f"{self.side}_shoulder"
        s_hip = f"{self.side}_hip"
        s_knee = f"{self.side}_knee"
        s_ankle = f"{self.side}_ankle"
        s_elbow = f"{self.side}_elbow"
        s_wrist = f"{self.side}_wrist"
        
        # Get opposite side joint names
        o_side = 'left' if self.side == 'right' else 'right'
        o_shoulder = f"{o_side}_shoulder"
        o_hip = f"{o_side}_hip"
        o_ankle = f"{o_side}_ankle"
        
        try:
            # Get joints of the selected side
            p_shoulder_s = keypoints[s_shoulder]
            p_hip_s = keypoints[s_hip]
            p_ankle_s = keypoints[s_ankle]
            p_knee_s = keypoints[s_knee]
            p_elbow_s = keypoints[s_elbow]
            p_wrist_s = keypoints[s_wrist]
            
            # Calculate elbow angle from OPPOSITE side
            p_shoulder_o = keypoints[o_shoulder]
            p_elbow_o = keypoints[f"{o_side}_elbow"]
            p_wrist_o = keypoints[f"{o_side}_wrist"]
            self.current_elbow_angle = calculate_angle(p_shoulder_o, p_elbow_o, p_wrist_o)
            
            # Calculate hip angle
            self.current_hip_angle = calculate_angle(p_shoulder_s, p_hip_s, p_knee_s)
            
            # Calculate body angle using midpoints if possible
            if o_shoulder in keypoints and o_hip in keypoints and o_ankle in keypoints:
                shoulder_mid = (np.array(p_shoulder_s) + np.array(keypoints[o_shoulder])) / 2
                hip_mid = (np.array(p_hip_s) + np.array(keypoints[o_hip])) / 2
                ankle_mid = (np.array(p_ankle_s) + np.array(keypoints[o_ankle])) / 2
                self.current_body_angle = calculate_angle(shoulder_mid, hip_mid, ankle_mid)

            # === SIDE VIEW: Check if body is horizontal ===
            all_x = [p_shoulder_s[0], p_hip_s[0], p_ankle_s[0]]
            all_y = [p_shoulder_s[1], p_hip_s[1], p_ankle_s[1]]
            body_width = max(all_x) - min(all_x)
            body_height = max(all_y) - min(all_y)
            is_horizontal = body_width > body_height

            # Check hip straight (side view)
            is_hip_straight = self.current_hip_angle > self.HIP_STRAIGHT_ANGLE_THRESHOLD
            
            # === FRONT VIEW: Check plank position ===
            # Khi ĐỨNG: vertical_span lớn (vai ở trên, mắt cá ở dưới)
            # Khi PLANK: vertical_span nhỏ (các keypoints cùng độ cao Y)
            if o_shoulder in keypoints and o_ankle in keypoints:
                # Tính shoulder midpoint và ankle midpoint
                left_shoulder_y = keypoints['left_shoulder'][1]
                right_shoulder_y = keypoints['right_shoulder'][1]
                left_ankle_y = keypoints['left_ankle'][1]
                right_ankle_y = keypoints['right_ankle'][1]
                
                shoulder_mid_y = (left_shoulder_y + right_shoulder_y) / 2
                ankle_mid_y = (left_ankle_y + right_ankle_y) / 2
                
                # Khoảng cách vertical giữa vai và mắt cá
                vertical_span = abs(ankle_mid_y - shoulder_mid_y)
                
                # Chiều rộng vai (để làm tham chiếu)
                shoulder_width = abs(keypoints['left_shoulder'][0] - keypoints['right_shoulder'][0])
                
                # Nếu vertical_span nhỏ hơn shoulder_width * ratio → đang ở tư thế PLANK
                # Vì khi nằm ngang nhìn từ front, các điểm gần như cùng độ cao Y
                if shoulder_width > 0:  # Tránh chia cho 0
                    is_plank_front_view = vertical_span < shoulder_width * self.PLANK_VERTICAL_RATIO
                    
                    # Thêm điều kiện: góc khuỷu tay phải gần thẳng (đang chống tay)
                    if is_plank_front_view:
                        is_plank_front_view = self.current_elbow_angle > self.PLANK_ELBOW_THRESHOLD

            # Track shoulder position history
            self.shoulder_history.append(p_shoulder_s[1])
            if len(self.shoulder_history) > self.SMOOTHING_WINDOW * 2:
                self.shoulder_history.pop(0)
            
            # Track elbow angle history
            self.elbow_angle_history.append(self.current_elbow_angle)
            if len(self.elbow_angle_history) > self.SMOOTHING_WINDOW * 2:
                self.elbow_angle_history.pop(0)
            
            # === CONDITION 1: Shoulder Movement (Original Logic) ===
            is_shoulder_moving_down = False
            is_shoulder_moving_up = False
            
            if len(self.shoulder_history) >= self.SMOOTHING_WINDOW * 2:
                recent_avg = np.mean(self.shoulder_history[-self.SMOOTHING_WINDOW:])
                previous_avg = np.mean(self.shoulder_history[-self.SMOOTHING_WINDOW*2:-self.SMOOTHING_WINDOW])
                
                if recent_avg > previous_avg + self.MOVEMENT_THRESHOLD:
                    is_shoulder_moving_down = True
                if recent_avg < previous_avg - self.MOVEMENT_THRESHOLD:
                    is_shoulder_moving_up = True
            
            # === CONDITION 2: Elbow Angle Change Rate (New Logic) ===
            is_elbow_decreasing = False  # Angle decreasing = going down
            is_elbow_increasing = False  # Angle increasing = pushing up
            
            if len(self.elbow_angle_history) >= self.SMOOTHING_WINDOW * 2:
                recent_elbow_avg = np.mean(self.elbow_angle_history[-self.SMOOTHING_WINDOW:])
                previous_elbow_avg = np.mean(self.elbow_angle_history[-self.SMOOTHING_WINDOW*2:-self.SMOOTHING_WINDOW])
                elbow_change_rate = recent_elbow_avg - previous_elbow_avg
                
                # If angle decreases > threshold = going down
                if elbow_change_rate < -self.ELBOW_CHANGE_THRESHOLD:
                    is_elbow_decreasing = True
                # If angle increases > threshold = pushing up
                if elbow_change_rate > self.ELBOW_CHANGE_THRESHOLD:
                    is_elbow_increasing = True
            
            # === COMBINE BOTH CONDITIONS ===
            # DOWN: Shoulder moving down AND elbow angle decreasing
            is_moving_down = is_shoulder_moving_down and is_elbow_decreasing
            
            # UP: Shoulder moving up AND elbow angle increasing
            is_moving_up = is_shoulder_moving_up and is_elbow_increasing
            
        except KeyError:
            # Skip frame if any important joint is missing
            return self.current_elbow_angle, self.current_body_angle, self.current_hip_angle

        # === STATE MACHINE ===
        if self.state == 'CHƯA VÀO TƯ THẾ':
            # Điều kiện vào tư thế READY phụ thuộc vào góc nhìn
            # Side view: is_horizontal (nằm ngang) AND is_hip_straight (hông thẳng)
            # Front view: is_plank_front_view (vertical_span nhỏ + khuỷu tay thẳng)
            if self.is_side_view:
                is_ready = is_horizontal and is_hip_straight
            else:
                is_ready = is_plank_front_view
            
            if is_ready:
                self.stable_ready_frames += 1
                if self.stable_ready_frames >= self.READY_FRAME_THRESHOLD:
                    self.state = 'SẴN SÀNG'
                    self.feedback = "SẴN SÀNG BẮT ĐẦU"
                    self.stable_ready_frames = 0
                    _logger.info(f"Pushup: Đã vào trạng thái READY")
            else:
                self.stable_ready_frames = 0
                self.feedback = "HÃY VÀO TƯ THẾ PLANK"
        
        elif self.state == 'SẴN SÀNG' or self.state == 'LÊN':
            # Must satisfy BOTH: shoulder down AND elbow angle decreasing
            if is_moving_down:
                self.down_frames_counter += 1
                if self.down_frames_counter >= self.DOWN_FRAME_THRESHOLD:
                    self.state = 'XUỐNG'
                    self.feedback = "HẠ XUỐNG"
                    self.lowest_elbow_angle = self.current_elbow_angle
                    self.down_frames_counter = 0
                    _logger.info(f"Pushup: Chuyển sang XUỐNG (góc: {self.current_elbow_angle:.1f}°)")
            else:
                self.down_frames_counter = 0
                if self.state == 'SẴN SÀNG':
                    self.feedback = "SẴN SÀNG TẬP"

        elif self.state == 'XUỐNG':
            # Track lowest elbow angle
            self.lowest_elbow_angle = min(self.lowest_elbow_angle, self.current_elbow_angle)
            
            # Must satisfy BOTH: shoulder up AND elbow angle increasing
            if is_moving_up:
                self.up_frames_counter += 1
                if self.up_frames_counter >= self.UP_FRAME_THRESHOLD:
                    self.rep_counter += 1
                    
                    # Determine if rep is valid based on depth and form
                    is_valid = True
                    errors = []
                    
                    if self.check_depth:
                        if self.lowest_elbow_angle <= self.ELBOW_ANGLE_THRESHOLD_MIN:
                            depth_feedback = "Tốt lắm! Độ sâu vừa phải, đảm bảo kích hoạt đầy đủ cơ mà không gây áp lực thừa."
                        else:
                            depth_feedback = "Bạn chưa hạ người đủ thấp. Hãy hạ ngực gần chạm sàn để cơ ngực và tay sau hoạt động hiệu quả hơn."
                            is_valid = False
                            errors.append("Độ sâu không đủ")
                    else:
                        depth_feedback = ""
                    
                    # Check form (body straightness) and record body_feedback
                    # Chỉ kiểm tra khi side view (check_body_straight đã được set = False khi front view)
                    if self.check_body_straight:
                        if self.current_body_angle < self.BODY_ANGLE_THRESHOLD_MIN:
                            is_valid = False
                            errors.append("Lưng không thẳng")
                            final_body_feedback = "Thân người chưa tạo thành đường thẳng. Hãy siết cơ bụng và mông để giữ lưng, vai, và gót chân thẳng hàng."
                        else:
                            final_body_feedback = "Thân người tạo thành đường thẳng đẹp từ vai đến gót chân. Rất tốt!"
                    else:
                        final_body_feedback = ""
                    
                    # Record this rep's feedback (use rep_id for consistency with other exercises)
                    rep_feedback = {
                        'rep_id': self.rep_counter,
                        'is_valid': is_valid,
                        'lowest_elbow_angle': self.lowest_elbow_angle,
                        'worst_body_angle': self.current_body_angle if self.is_side_view else None,
                        'depth_feedback': f"LẦN {self.rep_counter}: {depth_feedback}",
                        'back_feedback': final_body_feedback,
                        'errors': errors
                    }
                    self.rep_feedbacks.append(rep_feedback)
                    
                    self.feedback = f"LẦN {self.rep_counter}: {depth_feedback}"
                    _logger.info(f"Pushup: Rep #{self.rep_counter}, góc thấp nhất: {self.lowest_elbow_angle:.1f}°, valid: {is_valid}")
                    
                    self.state = 'LÊN'
                    self.up_frames_counter = 0
            else:
                self.up_frames_counter = 0

        # === Check body straightness (realtime feedback) ===
        if self.check_body_straight:
            if self.state in ['XUỐNG', 'LÊN', 'SẴN SÀNG']:
                if self.current_body_angle < self.BODY_ANGLE_THRESHOLD_MIN and self.current_body_angle > 0:
                    self.body_feedback = "Thân người chưa tạo thành đường thẳng. Hãy siết cơ bụng và mông để giữ lưng, vai, và gót chân thẳng hàng."
                else:
                    self.body_feedback = "Thân người tạo thành đường thẳng đẹp từ vai đến gót chân. Rất tốt!"

        return self.current_elbow_angle, self.current_body_angle, self.current_hip_angle

    
    def get_analysis_result(self) -> Dict[str, Any]:
        """
        Get current analysis state.
        
        Returns:
            Dictionary with analysis results
        """
        return {
            'state': self.state,
            'rep_counter': self.rep_counter,
            'angle_elbow': self.current_elbow_angle,
            'angle_body': self.current_body_angle,
            'angle_hip': self.current_hip_angle,
            'feedback': self.feedback,
            'body_feedback': self.body_feedback,
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the session.
        
        Returns:
            Dictionary with summary statistics
        """
        # Calculate valid/invalid reps based on rep_feedbacks
        valid_reps = sum(1 for rep in self.rep_feedbacks if rep.get("is_valid", True))
        invalid_reps = len(self.rep_feedbacks) - valid_reps
        
        return {
            'total_reps': self.rep_counter,
            'valid_reps': valid_reps,
            'invalid_reps': invalid_reps,
            'rep_feedbacks': self.rep_feedbacks
        }