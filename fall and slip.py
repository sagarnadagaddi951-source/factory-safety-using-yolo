"""
Real-time Fall and Slip Detection System using YOLOv8-Pose
Author: EC Engineering Student Project
Description: Detects human falls and slips in factory environments
"""

import cv2
from ultralytics import YOLO
import numpy as np
import math
import time
import threading
import os
from collections import deque

try:
    import winsound
    WINDOWS = True
except:
    WINDOWS = False

# Global variables for alert control
alert_active = False
last_alert_time = 0

# Fall detection state tracking
person_states = {}

class PersonState:
    """Track individual person's state for fall detection"""
    def __init__(self, person_id):
        self.person_id = person_id
        self.position_history = deque(maxlen=30)  # 1 second at 30fps
        self.body_angle_history = deque(maxlen=30)
        self.vertical_pos_history = deque(maxlen=30)
        self.fall_detected = False
        self.fall_start_time = None
        self.last_seen = time.time()
        self.stationary_frames = 0
        
    def update(self, center_y, body_angle, current_time):
        """Update person's state with new frame data"""
        self.position_history.append(center_y)
        self.body_angle_history.append(body_angle)
        self.last_seen = current_time
        
    def detect_sudden_drop(self, threshold=50):
        """Detect if person's vertical position dropped suddenly"""
        if len(self.position_history) < 10:
            return False
        
        recent_positions = list(self.position_history)[-10:]
        initial_pos = np.mean(recent_positions[:3])
        current_pos = np.mean(recent_positions[-3:])
        
        # Positive drop means falling down (y increases downward in images)
        drop = current_pos - initial_pos
        
        return drop > threshold
    
    def is_horizontal(self):
        """Check if person is in horizontal position"""
        if len(self.body_angle_history) < 3:
            return False
        
        recent_angles = list(self.body_angle_history)[-5:]
        avg_angle = np.mean(recent_angles)
        
        # Body angle > 60 degrees indicates horizontal position
        return avg_angle > 60
    
    def is_stationary(self, frames_threshold=15):
        """Check if person has been stationary while horizontal"""
        if len(self.position_history) < frames_threshold:
            return False
        
        recent_positions = list(self.position_history)[-frames_threshold:]
        position_variance = np.var(recent_positions)
        
        # Low variance means person is not moving much
        return position_variance < 20

def play_alert_sound():
    """Play alert sound in a separate thread"""
    global alert_active
    try:
        if WINDOWS:
            for _ in range(5):
                winsound.Beep(1500, 300)
                time.sleep(0.1)
        else:
            for _ in range(5):
                os.system('printf "\a"')
                time.sleep(0.1)
    except:
        print("\a" * 5)
    alert_active = False

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points"""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    
    ba = a - b
    bc = c - b
    
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    
    return np.degrees(angle)

def detect_fall(keypoints, person_id, current_time):
    """
    Detect falls based on:
    1. Sudden vertical drop in center of mass
    2. Body orientation change (vertical to horizontal)
    3. Stationary horizontal position (person not getting up)
    """
    
    if keypoints is None or len(keypoints) == 0:
        return "Unknown", (128, 128, 128), False, None
    
    kp = keypoints[0]
    
    # Extract keypoints with confidence check
    nose = kp[0][:2] if kp[0][2] > 0.3 else None
    l_shoulder = kp[5][:2] if kp[5][2] > 0.3 else None
    r_shoulder = kp[6][:2] if kp[6][2] > 0.3 else None
    l_hip = kp[11][:2] if kp[11][2] > 0.3 else None
    r_hip = kp[12][:2] if kp[12][2] > 0.3 else None
    l_knee = kp[13][:2] if kp[13][2] > 0.3 else None
    r_knee = kp[14][:2] if kp[14][2] > 0.3 else None
    
    # Check critical keypoints
    critical_points = [l_shoulder, r_shoulder, l_hip, r_hip]
    if any(point is None for point in critical_points):
        return "Unknown", (128, 128, 128), False, None
    
    # Calculate body center (center of mass approximation)
    shoulder_mid = [(l_shoulder[0] + r_shoulder[0])/2, (l_shoulder[1] + r_shoulder[1])/2]
    hip_mid = [(l_hip[0] + r_hip[0])/2, (l_hip[1] + r_hip[1])/2]
    body_center_y = (shoulder_mid[1] + hip_mid[1]) / 2
    
    # Calculate body angle from vertical
    body_vertical = abs(shoulder_mid[1] - hip_mid[1])
    body_horizontal = abs(shoulder_mid[0] - hip_mid[0])
    body_angle_deg = math.degrees(math.atan2(body_horizontal, body_vertical + 1e-6))
    
    # Initialize or update person state
    if person_id not in person_states:
        person_states[person_id] = PersonState(person_id)
    
    state = person_states[person_id]
    state.update(body_center_y, body_angle_deg, current_time)
    
    # FALL DETECTION LOGIC
    is_fallen = False
    fall_type = None
    
    # Condition 1: Sudden drop detected
    sudden_drop = state.detect_sudden_drop(threshold=40)
    
    # Condition 2: Body is horizontal
    is_horizontal = state.is_horizontal()
    
    # Condition 3: Person is stationary while horizontal
    is_stationary = state.is_stationary(frames_threshold=15)
    
    # FALL DETECTED if:
    # - Sudden drop AND body is now horizontal (active fall)
    # - Body horizontal AND stationary for extended time (person down)
    
    if sudden_drop and is_horizontal:
        is_fallen = True
        fall_type = "FALL DETECTED!"
        if not state.fall_detected:
            state.fall_detected = True
            state.fall_start_time = current_time
            
    elif is_horizontal and is_stationary:
        is_fallen = True
        fall_type = "PERSON DOWN!"
        if not state.fall_detected:
            state.fall_detected = True
            state.fall_start_time = current_time
    
    # Recovery detection: person is vertical again
    elif body_angle_deg < 35 and state.fall_detected:
        state.fall_detected = False
        state.fall_start_time = None
        state.stationary_frames = 0
    
    # Continue alert if fall was detected and person still down
    if state.fall_detected:
        is_fallen = True
        duration = current_time - state.fall_start_time if state.fall_start_time else 0
        fall_type = f"PERSON DOWN! ({duration:.1f}s)"
    
    # Determine status color and label
    if is_fallen:
        return fall_type, (0, 0, 255), True, state
    elif body_angle_deg > 60:
        return "Lying/Sitting", (255, 165, 0), False, state
    elif body_angle_deg < 25:
        return "Standing", (0, 255, 0), False, state
    else:
        return "Moving", (255, 255, 0), False, state

def cleanup_old_states(max_age=2.0):
    """Remove tracking data for people no longer in frame"""
    current_time = time.time()
    to_remove = []
    
    for person_id, state in person_states.items():
        if current_time - state.last_seen > max_age:
            to_remove.append(person_id)
    
    for person_id in to_remove:
        del person_states[person_id]

def main():
    global alert_active, last_alert_time
    
    print("\n" + "="*60)
    print("FALL & SLIP DETECTION SYSTEM - YOLOv8 Pose")
    print("="*60)
    
    # User input for video source
    print("\nSelect video source:")
    print("1. Webcam (default)")
    print("2. Upload video file")
    choice = input("Enter choice (1 or 2): ").strip()
    
    video_source = 0  # Default to webcam
    
    if choice == '2':
        video_path = input("Enter video file path: ").strip()
        if os.path.exists(video_path):
            video_source = video_path
            print(f"Loading video: {video_path}")
        else:
            print(f"Error: File not found - {video_path}")
            print("Defaulting to webcam...")
            video_source = 0
    
    # Load YOLOv8-Pose model
    print("\nLoading YOLO-Pose model...")
    try:
        model = YOLO('yolov8n-pose.pt')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure 'yolov8n-pose.pt' is downloaded")
        return
    
    # Open video source
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # Default FPS
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video opened: {width}x{height} @ {fps} FPS")
    print("\nControls:")
    print("  'q' - Quit")
    print("  'p' - Pause/Resume")
    print("  'r' - Reset fall detection states")
    print("\n" + "="*60)
    print("SYSTEM ACTIVE - Monitoring for falls and slips...")
    print("="*60 + "\n")
    
    ALERT_COOLDOWN = 2  # seconds between audio alerts
    paused = False
    frame_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("\nEnd of video or camera error")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Run YOLO pose estimation
            results = model(frame, conf=0.4, verbose=False)
            
            # Get keypoints
            keypoints_data = results[0].keypoints.data.cpu().numpy() if results[0].keypoints is not None else None
            
            # Annotate frame
            annotated_frame = results[0].plot()
            
            # Track falls
            any_fall_detected = False
            fall_count = 0
            
            if keypoints_data is not None and len(keypoints_data) > 0:
                for idx, kp in enumerate(keypoints_data):
                    person_id = f"person_{idx}"
                    
                    status, color, is_fallen, state = detect_fall([kp], person_id, current_time)
                    
                    if is_fallen:
                        any_fall_detected = True
                        fall_count += 1
                    
                    # Get bounding box
                    if results[0].boxes is not None and len(results[0].boxes) > idx:
                        box = results[0].boxes.xyxy[idx].cpu().numpy()
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Display status label
                        label = f"ID:{idx+1} - {status}"
                        
                        # Background for text
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(annotated_frame,
                                    (x1, y1 - text_size[1] - 10),
                                    (x1 + text_size[0] + 10, y1),
                                    color, -1)
                        
                        cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Draw bounding box with color based on status
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # Cleanup old person states
            cleanup_old_states()
            
            # Trigger audio alert
            if any_fall_detected and not alert_active and (current_time - last_alert_time) > ALERT_COOLDOWN:
                alert_active = True
                last_alert_time = current_time
                threading.Thread(target=play_alert_sound, daemon=True).start()
            
            # Display alert banner
            if any_fall_detected:
                if int(time.time() * 3) % 2 == 0:  # Blinking effect
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                    
                    cv2.putText(annotated_frame, "!!! FALL DETECTED - EMERGENCY !!!", 
                              (int(width/2) - 300, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # Info overlay
            info_bg = annotated_frame.copy()
            cv2.rectangle(info_bg, (10, 10), (400, 180), (0, 0, 0), -1)
            cv2.addWeighted(info_bg, 0.6, annotated_frame, 0.4, 0, annotated_frame)
            
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"People Detected: {len(keypoints_data) if keypoints_data is not None else 0}",
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Falls Detected: {fall_count}",
                       (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if fall_count > 0 else (255, 255, 255), 2)
            
            # Legend
            legend_y = 110
            cv2.putText(annotated_frame, "Standing", (20, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(annotated_frame, "Moving", (20, legend_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(annotated_frame, "Lying/Sitting", (20, legend_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
            cv2.putText(annotated_frame, "FALL/DOWN", (20, legend_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow('Fall Detection System - YOLOv8 Pose', annotated_frame)
        
        else:
            # Paused - just show last frame with PAUSED text
            paused_frame = annotated_frame.copy()
            cv2.putText(paused_frame, "PAUSED - Press 'p' to resume", 
                       (int(width/2) - 250, int(height/2)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            cv2.imshow('Fall Detection System - YOLOv8 Pose', paused_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("PAUSED" if paused else "RESUMED")
        elif key == ord('r'):
            person_states.clear()
            print("Fall detection states reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n" + "="*60)
    print("Program terminated successfully")
    print("="*60)

if __name__ == "__main__":
    main()
