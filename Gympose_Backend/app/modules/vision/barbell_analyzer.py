"""
Barbell Dead Row Analysis Module
Analyzes barbell dead row form using rule-based logic on pose keypoints.

Luồng trạng thái:
1. IDLE: Đứng thẳng, đi lại bình thường (chưa chuẩn bị tập)
2. READY: Đã khuỵu gối + cúi lưng về trước (tư thế chuẩn bị tập, cầm tạ) - CHỈ KIỂM TRA LẦN ĐẦU
3. ASCENDING: Kéo tạ lên (từ READY hoặc DOWN đi lên)
4. DESCENDING: Hạ tạ xuống (từ trên đi xuống)

Luồng rep:
- Lần đầu: IDLE -> (check ready conditions) -> READY -> UP -> DOWN = 1 rep
- Lần sau: DOWN -> UP -> DOWN = 1 rep (KHÔNG CẦN check ready nữa)
- Reset về IDLE chỉ khi đứng thẳng người (thoát tư thế tập)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


# COCO Keypoint indexes for barbell dead row analysis
COCO_KEYPOINT_INDEXES = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
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


class BarbellDeadRowAnalyzer:
    """
    Phân tích động tác Barbell Dead Row (Bent-Over Barbell Row).
    
    Luồng trạng thái:
    1. IDLE: Đứng thẳng, đi lại bình thường (chưa chuẩn bị tập)
    2. READY: Đã khuỵu gối + cúi lưng về trước (tư thế chuẩn bị tập, cầm tạ) - CHỈ KIỂM TRA LẦN ĐẦU
    3. ASCENDING: Kéo tạ lên (từ READY hoặc DOWN đi lên)
    4. DESCENDING: Hạ tạ xuống (từ trên đi xuống)
    
    Luồng rep:
    - Lần đầu: IDLE -> (check ready conditions) -> READY -> UP -> DOWN = 1 rep
    - Lần sau: DOWN -> UP -> DOWN = 1 rep (KHÔNG CẦN check ready nữa)
    - Reset về IDLE chỉ khi đứng thẳng người (thoát tư thế tập)
    """

    def __init__(self, side: str = 'right', smoothing_window: int = 5):
        self.state = 'IDLE'
        self.rep_counter = 0
        self.ready_frames_counter = 0
        self.motion_frames_counter = 0
        self.hip_history: List[float] = []

        self.check_depth = True  # Kiểm tra góc gối trong lúc tập
        self.check_back = True   # Kiểm tra góc lưng trong lúc tập
        
        # Store the side/view parameter for tracking
        # Giá trị hợp lệ: 'left', 'right', 'front'
        # - 'left': Camera quay từ bên trái người tập (side view)
        # - 'right': Camera quay từ bên phải người tập (side view)
        # - 'front': Camera quay từ phía trước người tập (front view)
        self.tracking_side = side
        
        # Xác định loại góc nhìn dựa trên tham số side
        # Side view: có thể đo được góc lưng chính xác
        # Front view: chỉ đo được góc gối, không đo được góc lưng chính xác
        if side in ['left', 'right']:
            self.is_side_view = True
            self.is_front_view = False
        else:  # side == 'front' hoặc các giá trị khác
            self.is_side_view = False
            self.is_front_view = True
        
        self.SMOOTHING_WINDOW = smoothing_window
        
        # === CÁC NGƯỠNG CHO BARBELL DEAD ROW ===
        
        # --- Ngưỡng để vào tư thế READY (chỉ kiểm tra lần đầu) ---
        self.READY_KNEE_THRESHOLD = 170      # Gối hơi khuỵu (góc < 170 độ) để vào tư thế
        self.READY_BACK_THRESHOLD = 20       # Lưng cúi nhẹ (góc > 20 độ) để vào tư thế
        
        # --- Ngưỡng kiểm tra DEPTH trong lúc tập (feedback cho mỗi rep) ---
        self.DEPTH_THRESHOLD = 110           # Nếu gối khuỵu quá sâu (góc < 110 độ) = lỗi
        
        # --- Ngưỡng kiểm tra BACK trong lúc tập (feedback cho mỗi rep) ---
        self.BACK_MAX_THRESHOLD = 45         # Lưng không được cúi quá 45 độ (lỗi tư thế)
        
        # --- Ngưỡng để reset về IDLE (thoát khỏi tư thế tập) ---
        self.IDLE_KNEE_THRESHOLD = 175       # Gối gần thẳng (góc > 175 độ)
        self.IDLE_BACK_THRESHOLD = 12        # Lưng gần thẳng (góc < 12 độ)
        
        # --- Ngưỡng chuyển động elbow ---
        self.ELBOW_MOVE_THRESHOLD = 15       # Khuỷu tay di chuyển ít nhất 15 pixel
        
        self.MOTION_TRIGGER_FRAMES = 3       # Số frame liên tiếp để xác nhận chuyển động
        # =======================================
        
        self.current_knee_angle = 180.0  # Khởi tạo là đang đứng thẳng
        self.current_back_angle = 0.0
        self.lowest_elbow_position: Optional[float] = None
        self.highest_elbow_position: Optional[float] = None
        
        # Tracking cho mỗi rep
        self.lowest_knee_in_rep = 180.0    # Góc gối thấp nhất trong rep (khuỵu sâu nhất)
        self.highest_back_in_rep = 0.0     # Góc lưng cao nhất trong rep (cúi sâu nhất)
        
        self.depth_feedback = ""
        self.back_feedback = ""
        self.form_feedback = ""
        self.rep_feedbacks: List[Dict[str, Any]] = []

        # Keypoints for visualization
        # SIDE KEYPOINTS
        self.s_hip: Optional[np.ndarray] = None
        self.s_knee: Optional[np.ndarray] = None
        self.s_ankle: Optional[np.ndarray] = None
        self.s_elbow: Optional[np.ndarray] = None

        # FRONT KEYPOINTS / Midpoints
        self.shoulder_midpoint: Optional[np.ndarray] = None
        self.hip_midpoint: Optional[np.ndarray] = None
        self.knee_midpoint: Optional[np.ndarray] = None
        self.ankle_midpoint: Optional[np.ndarray] = None
        self.elbow_midpoint: Optional[np.ndarray] = None
        self.vertical_point: Optional[np.ndarray] = None

    def reset(self):
        """Reset analyzer state."""
        self.state = 'IDLE'
        self.rep_counter = 0
        self.ready_frames_counter = 0
        self.motion_frames_counter = 0
        self.hip_history = []
        self.current_knee_angle = 180.0
        self.current_back_angle = 0.0
        self.lowest_elbow_position = None
        self.highest_elbow_position = None
        self.lowest_knee_in_rep = 180.0
        self.highest_back_in_rep = 0.0
        self.depth_feedback = ""
        self.back_feedback = ""
        self.form_feedback = ""
        self.rep_feedbacks = []

    def _extract_keypoints(
        self, keypoints_array: np.ndarray, scores: np.ndarray, min_score: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """Extract keypoints from array format to dictionary."""
        keypoints = {}
        for joint_name, index in COCO_KEYPOINT_INDEXES.items():
            if scores[index] > min_score:
                keypoints[joint_name] = keypoints_array[index]
        return keypoints

    def _calculate_midpoint(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Calculate midpoint between two points."""
        return (p1 + p2) / 2.0

    def process_frame(
        self, keypoints_array: np.ndarray, scores: np.ndarray
    ) -> Dict[str, Any]:
        """
        Process a single frame and update state.
        
        Args:
            keypoints_array: Array of keypoint coordinates
            scores: Confidence scores for each keypoint
            
        Returns:
            Current state dictionary
        """
        self.back_feedback = ""
        self.form_feedback = ""
        keypoints = self._extract_keypoints(keypoints_array, scores)
        if not keypoints:
            return self._get_current_state()

        shoulder_midpoint, hip_midpoint, elbow_midpoint = None, None, None
        
        # === TÍNH GÓC GỐI ===
        knee_angle_left, knee_angle_right = None, None
        try:
            if self.is_side_view:
                s_hip = f"{self.tracking_side}_hip"
                s_knee = f"{self.tracking_side}_knee"
                s_ankle = f"{self.tracking_side}_ankle"

                self.s_hip = keypoints.get(s_hip)
                self.s_knee = keypoints.get(s_knee)
                self.s_ankle = keypoints.get(s_ankle)

                if self.s_hip is not None and self.s_knee is not None and self.s_ankle is not None:
                    self.current_knee_angle = calculate_angle(
                        self.s_hip, self.s_knee, self.s_ankle
                    )
            else:
                # Front view - tính trung bình góc 2 bên
                if "left_hip" in keypoints and "left_knee" in keypoints and "left_ankle" in keypoints:
                    knee_angle_left = calculate_angle(
                        keypoints["left_hip"], 
                        keypoints["left_knee"], 
                        keypoints["left_ankle"]
                    )
                if "right_hip" in keypoints and "right_knee" in keypoints and "right_ankle" in keypoints:
                    knee_angle_right = calculate_angle(
                        keypoints["right_hip"], 
                        keypoints["right_knee"], 
                        keypoints["right_ankle"]
                    )
                
                if knee_angle_left is not None and knee_angle_right is not None:
                    self.current_knee_angle = (knee_angle_left + knee_angle_right) / 2
                elif knee_angle_left is not None:
                    self.current_knee_angle = knee_angle_left
                elif knee_angle_right is not None:
                    self.current_knee_angle = knee_angle_right
        except KeyError:
            pass

        # === TÍNH GÓC LƯNG ===
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
        except KeyError:
            pass

        if shoulder_midpoint is not None and hip_midpoint is not None:
            vertical_point = np.array([hip_midpoint[0], hip_midpoint[1] - 100])
            self.vertical_point = vertical_point
            self.current_back_angle = calculate_angle(shoulder_midpoint, hip_midpoint, vertical_point)
        
        # === TÍNH VỊ TRÍ KHUỶU TAY (để theo dõi chuyển động lên/xuống) ===
        try:
            if self.is_side_view:
                s_elbow = f"{self.tracking_side}_elbow"
                if s_elbow in keypoints:
                    elbow_midpoint = keypoints[s_elbow]
                    self.s_elbow = elbow_midpoint
            else:
                if "left_elbow" in keypoints and "right_elbow" in keypoints:
                    elbow_midpoint = self._calculate_midpoint(
                        keypoints["left_elbow"], keypoints["right_elbow"]
                    )
                elif "left_elbow" in keypoints:
                    elbow_midpoint = keypoints["left_elbow"]
                elif "right_elbow" in keypoints:
                    elbow_midpoint = keypoints["right_elbow"]
            self.elbow_midpoint = elbow_midpoint
        except KeyError:
            pass

        # Sử dụng vị trí hip để theo dõi chuyển động
        hip_to_track_y = None
        try:
            if self.is_side_view:
                s_hip = f"{self.tracking_side}_hip"
                if s_hip in keypoints:
                    hip_to_track_y = keypoints[s_hip][1]
            elif hip_midpoint is not None:
                hip_to_track_y = hip_midpoint[1]
        except KeyError:
            pass

        if hip_to_track_y is not None:
            self.hip_history.append(hip_to_track_y)

            if len(self.hip_history) > self.SMOOTHING_WINDOW * 2: 
                self.hip_history.pop(0)
            if len(self.hip_history) < self.SMOOTHING_WINDOW * 2: 
                return self._get_current_state()

            recent_hip_avg = np.mean(self.hip_history[-self.SMOOTHING_WINDOW:])
            previous_hip_avg = np.mean(self.hip_history[-self.SMOOTHING_WINDOW*2:-self.SMOOTHING_WINDOW])

            # === ĐIỀU KIỆN VÀO TƯ THẾ READY (CHỈ KIỂM TRA LẦN ĐẦU) ===
            is_ready_knee = self.current_knee_angle < self.READY_KNEE_THRESHOLD
            is_ready_back = self.current_back_angle > self.READY_BACK_THRESHOLD
            
            # === ĐIỀU KIỆN THOÁT VỀ IDLE (ĐỨNG THẲNG NGƯỜI) ===
            # Side view: cần cả gối thẳng VÀ lưng thẳng
            # Front view: chỉ cần gối thẳng (không đo được góc lưng chính xác)
            is_standing_knee = self.current_knee_angle > self.IDLE_KNEE_THRESHOLD
            is_standing_back = self.current_back_angle < self.IDLE_BACK_THRESHOLD
            
            if self.is_side_view:
                is_standing_straight = is_standing_knee and is_standing_back
            else:
                is_standing_straight = is_standing_knee  # Front view chỉ kiểm tra gối

            # === STATE MACHINE CHO BARBELL DEAD ROW ===
            if self.state == 'IDLE':
                # Kiểm tra điều kiện READY phụ thuộc vào góc nhìn
                # Side view: cần cả gối khuỵu VÀ lưng cúi (đo được cả 2)
                # Front view: chỉ cần gối khuỵu (không đo được góc lưng chính xác)
                if self.is_side_view:
                    is_ready = is_ready_knee and is_ready_back
                else:
                    is_ready = is_ready_knee  # Front view chỉ kiểm tra gối
                
                if is_ready:
                    self.ready_frames_counter += 1
                else:
                    self.ready_frames_counter = 0

                if self.ready_frames_counter >= self.MOTION_TRIGGER_FRAMES:
                    self.state = 'READY'
                    self.ready_frames_counter = 0
                    # Reset tracking cho rep mới
                    self.lowest_knee_in_rep = self.current_knee_angle
                    self.highest_back_in_rep = self.current_back_angle
                    self.depth_feedback = "Tư thế sẵn sàng"
                    
            elif self.state == 'READY':
                # Kiểm tra nếu người tập đứng thẳng dậy (thoát khỏi tư thế)
                if is_standing_straight:
                    self.state = 'IDLE'
                    self.depth_feedback = ""
                    self.lowest_elbow_position = None
                    self.highest_elbow_position = None
                    return self._get_current_state()
                
                # Tracking góc trong rep
                self.lowest_knee_in_rep = min(self.lowest_knee_in_rep, self.current_knee_angle)
                self.highest_back_in_rep = max(self.highest_back_in_rep, self.current_back_angle)
                
                # Phát hiện chuyển động đi LÊN (bắt đầu kéo tạ)
                if elbow_midpoint is not None:
                    elbow_y = elbow_midpoint[1] if isinstance(elbow_midpoint, np.ndarray) else elbow_midpoint
                    
                    if self.lowest_elbow_position is None:
                        self.lowest_elbow_position = elbow_y
                    
                    # Nếu khuỷu tay đi lên (giá trị Y giảm)
                    elbow_moving_up = elbow_y < self.lowest_elbow_position - self.ELBOW_MOVE_THRESHOLD
                    
                    if elbow_moving_up:
                        self.motion_frames_counter += 1
                    else:
                        self.motion_frames_counter = 0
                        self.lowest_elbow_position = elbow_y
                    
                    if self.motion_frames_counter >= self.MOTION_TRIGGER_FRAMES:
                        self.state = 'ASCENDING'
                        self.highest_elbow_position = elbow_y
                        self.motion_frames_counter = 0
                        self.depth_feedback = "Kéo lên"

            elif self.state == 'ASCENDING':
                # Kiểm tra nếu người tập đứng thẳng dậy (thoát khỏi tư thế)
                if is_standing_straight:
                    self.state = 'IDLE'
                    self.depth_feedback = ""
                    self.lowest_elbow_position = None
                    self.highest_elbow_position = None
                    return self._get_current_state()
                
                # Tracking góc trong rep
                self.lowest_knee_in_rep = min(self.lowest_knee_in_rep, self.current_knee_angle)
                self.highest_back_in_rep = max(self.highest_back_in_rep, self.current_back_angle)
                
                # Cập nhật vị trí cao nhất của khuỷu tay
                if elbow_midpoint is not None:
                    elbow_y = elbow_midpoint[1] if isinstance(elbow_midpoint, np.ndarray) else elbow_midpoint
                    
                    self.highest_elbow_position = min(self.highest_elbow_position or elbow_y, elbow_y)
                    
                    # Phát hiện chuyển động đi XUỐNG (hạ tạ)
                    elbow_moving_down = elbow_y > self.highest_elbow_position + self.ELBOW_MOVE_THRESHOLD
                    
                    if elbow_moving_down:
                        self.motion_frames_counter += 1
                    else:
                        self.motion_frames_counter = 0
                    
                    if self.motion_frames_counter >= self.MOTION_TRIGGER_FRAMES:
                        # HOÀN THÀNH 1 REP: đã kéo lên và bắt đầu hạ xuống
                        self.rep_counter += 1
                        self.state = 'DESCENDING'
                        self.motion_frames_counter = 0
                        
                        # === ĐÁNH GIÁ REP DỰA TRÊN DEPTH VÀ BACK ===
                        depth_ok = self.lowest_knee_in_rep >= self.DEPTH_THRESHOLD
                        
                        # Side view: kiểm tra cả depth và back
                        # Front view: chỉ kiểm tra depth (không đo được góc lưng)
                        if self.is_side_view:
                            back_ok = self.highest_back_in_rep <= self.BACK_MAX_THRESHOLD
                            
                            if depth_ok and back_ok:
                                self.depth_feedback = f"LẦN {self.rep_counter}: Tư thế rất chuẩn! Bạn đã giữ hông ở vị trí cao, đảm bảo an toàn cho lưng dưới và tập trung vào cơ lưng giữa."
                            elif not depth_ok:
                                self.depth_feedback = f"LẦN {self.rep_counter}: Bạn đang hạ người quá sâu. Hãy giữ hông cao hơn, tránh gập gối quá nhiều."
                            elif not back_ok:
                                self.depth_feedback = f"LẦN {self.rep_counter}: Lưng bạn đang bị cong hoặc nghiêng quá mức. Hãy mở ngực, siết cơ bụng và giữ cột sống trung lập trong suốt chuyển động."
                        else:
                            # Front view: chỉ đánh giá depth
                            if depth_ok:
                                self.depth_feedback = f"LẦN {self.rep_counter}: Tư thế rất chuẩn! Bạn đã giữ hông ở vị trí cao, đảm bảo an toàn cho lưng dưới và tập trung vào cơ lưng giữa."
                            else:
                                self.depth_feedback = f"LẦN {self.rep_counter}: Bạn đang hạ người quá sâu. Hãy giữ hông cao hơn, tránh gập gối quá nhiều."
                        
                        self._save_rep_feedback()

            elif self.state == 'DESCENDING':
                # Kiểm tra nếu người tập đứng thẳng dậy (thoát khỏi tư thế)
                if is_standing_straight:
                    self.state = 'IDLE'
                    self.depth_feedback = "Hoàn thành"
                    self.lowest_elbow_position = None
                    self.highest_elbow_position = None
                    return self._get_current_state()
                
                # Chờ cho đến khi về lại tư thế sẵn sàng cho rep tiếp theo
                # KHÔNG CẦN kiểm tra điều kiện ready nữa, chỉ cần khuỷu tay về vị trí thấp
                if elbow_midpoint is not None:
                    elbow_y = elbow_midpoint[1] if isinstance(elbow_midpoint, np.ndarray) else elbow_midpoint
                    
                    # Kiểm tra nếu đã về gần vị trí thấp nhất
                    back_to_ready = elbow_y > self.highest_elbow_position + (self.ELBOW_MOVE_THRESHOLD * 2)
                    
                    if back_to_ready:
                        # Chuyển về trạng thái READY để bắt đầu rep tiếp theo
                        # KHÔNG CẦN kiểm tra điều kiện ready (gối khuỵu, lưng cúi) nữa
                        self.state = 'READY'
                        self.lowest_elbow_position = elbow_y
                        self.highest_elbow_position = None
                        # Reset tracking cho rep mới
                        self.lowest_knee_in_rep = self.current_knee_angle
                        self.highest_back_in_rep = self.current_back_angle
                        self.depth_feedback = "Sẵn sàng tiếp"
        
        # === KIỂM TRA TƯ THẾ LƯNG REALTIME (CHỈ CHO SIDE VIEW) ===
        if self.check_back and self.is_side_view:
            if self.state in ['READY', 'ASCENDING', 'DESCENDING']:
                if self.current_back_angle > self.BACK_MAX_THRESHOLD:
                    self.back_feedback = "Lưng bạn đang bị cong hoặc nghiêng quá mức. Hãy mở ngực, siết cơ bụng và giữ cột sống trung lập."
                else:
                    self.back_feedback = "Rất tốt! Lưng bạn thẳng và ổn định, giúp tối ưu sức mạnh và giảm nguy cơ chấn thương."
        
        # === KIỂM TRA GÓC GỐI REALTIME (DEPTH) ===
        if self.check_depth:
            if self.state in ['READY', 'ASCENDING', 'DESCENDING']:
                if self.current_knee_angle < self.DEPTH_THRESHOLD:
                    self.form_feedback = f"Cảnh báo: Gối quá thấp ({int(self.current_knee_angle)}°)"
                else:
                    self.form_feedback = "Độ sâu tốt"

        return self._get_current_state()

    def _save_rep_feedback(self):
        """Save feedback for the completed rep."""
        # Determine back_feedback based on whether back was within threshold
        # Front view: không đánh giá back
        if self.is_side_view:
            if self.highest_back_in_rep > self.BACK_MAX_THRESHOLD:
                final_back_feedback = "Lưng bạn đang bị cong hoặc nghiêng quá mức. Hãy mở ngực, siết cơ bụng và giữ cột sống trung lập trong suốt chuyển động."
            else:
                final_back_feedback = "Rất tốt! Lưng bạn thẳng và ổn định, giúp tối ưu sức mạnh và giảm nguy cơ chấn thương."
            
            # Side view: kiểm tra cả depth và back
            is_valid = bool(self.lowest_knee_in_rep >= self.DEPTH_THRESHOLD 
                and self.highest_back_in_rep <= self.BACK_MAX_THRESHOLD)
        else:
            final_back_feedback = ""  # Front view không đánh giá back
            # Front view: chỉ kiểm tra depth
            is_valid = bool(self.lowest_knee_in_rep >= self.DEPTH_THRESHOLD)
        
        rep_data = {
            "rep_id": self.rep_counter,
            "depth_feedback": self.depth_feedback,
            "back_feedback": final_back_feedback,
            "lowest_knee_angle": float(self.lowest_knee_in_rep),
            "highest_back_angle": float(self.highest_back_in_rep) if self.is_side_view else None,
            "is_valid": is_valid,
        }
        self.rep_feedbacks.append(rep_data)

    def _get_current_state(self) -> Dict[str, Any]:
        """Get current analysis state."""
        return {
            "state": self.state,
            "rep_counter": self.rep_counter,
            "angle_knee": self.current_knee_angle,
            "angle_back": self.current_back_angle,
            "depth_feedback": self.depth_feedback,
            "back_feedback": self.back_feedback,
            "form_feedback": self.form_feedback,
            "shoulder_midpoint": self.shoulder_midpoint.tolist() if self.shoulder_midpoint is not None else None,
            "hip_midpoint": self.hip_midpoint.tolist() if self.hip_midpoint is not None else None,
            "vertical_point": self.vertical_point.tolist() if self.vertical_point is not None else None,
            "elbow_midpoint": self.elbow_midpoint.tolist() if self.elbow_midpoint is not None else None,
            "s_hip": self.s_hip.tolist() if self.s_hip is not None else None,
            "s_knee": self.s_knee.tolist() if self.s_knee is not None else None,
            "s_ankle": self.s_ankle.tolist() if self.s_ankle is not None else None,
            "s_elbow": self.s_elbow.tolist() if self.s_elbow is not None else None,
            "knee_midpoint": self.knee_midpoint.tolist() if self.knee_midpoint is not None else None,
            "ankle_midpoint": self.ankle_midpoint.tolist() if self.ankle_midpoint is not None else None,
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all reps analyzed.

        Returns:
            Summary dictionary with total_reps and per-rep feedback
        """
        valid_reps = sum(1 for rep in self.rep_feedbacks if rep["is_valid"])
        invalid_reps = len(self.rep_feedbacks) - valid_reps

        return {
            "total_reps": self.rep_counter,
            "valid_reps": valid_reps,
            "invalid_reps": invalid_reps,
            "rep_feedbacks": self.rep_feedbacks,
        }
