"""
Real-time Factory Safety Monitoring System using YOLOv8-Pose
Author: EC Engineering Student Project
Features: 
1. Fall & Slip Detection
2. Proximity to Dangerous Machines Detection
3. Incorrect Lifting Posture Detection (REBA/RULA Assessment)
"""

import cv2
from ultralytics import YOLO
import numpy as np
import math
import time
import threading
import os
from collections import deque
import json

try:
    import winsound
    WINDOWS = True
except:
    WINDOWS = False

# Global variables
alert_active = False
last_alert_time = 0
person_states = {}
danger_zones = []

# REBA Score Thresholds
REBA_NEGLIGIBLE = 1      # Negligible risk
REBA_LOW = 2             # Low risk
REBA_MEDIUM = 4          # Medium risk - investigate
REBA_HIGH = 8            # High risk - investigate and change soon
REBA_VERY_HIGH = 11      # Very high risk - implement change

class SafetyZone:
    """Represents a danger zone around machinery"""
    def __init__(self, zone_id, zone_type, coordinates, machine_name="Machine", 
                 approach_speed=1.6, stop_time=0.5, intrusion_distance=0.2):
        self.zone_id = zone_id
        self.zone_type = zone_type
        self.coordinates = coordinates
        self.machine_name = machine_name
        self.approach_speed = approach_speed
        self.stop_time = stop_time
        self.intrusion_distance = intrusion_distance
        
        # ISO 13855: S = (K × T) + C
        self.min_safe_distance_m = (approach_speed * stop_time) + intrusion_distance
        self.min_safe_distance_px = self.min_safe_distance_m * 100
        self.warning_distance_px = self.min_safe_distance_px * 1.5
        
    def is_point_in_danger(self, point):
        x, y = point
        if self.zone_type == 'rectangle':
            x1, y1, x2, y2 = self.coordinates
            return x1 <= x <= x2 and y1 <= y <= y2
        elif self.zone_type == 'circle':
            cx, cy, radius = self.coordinates
            distance = math.sqrt((x - cx)**2 + (y - cy)**2)
            return distance <= radius
        return False
    
    def get_distance_to_zone(self, point):
        x, y = point
        if self.zone_type == 'rectangle':
            x1, y1, x2, y2 = self.coordinates
            dx = max(x1 - x, 0, x - x2)
            dy = max(y1 - y, 0, y - y2)
            return math.sqrt(dx**2 + dy**2)
        elif self.zone_type == 'circle':
            cx, cy, radius = self.coordinates
            distance_to_center = math.sqrt((x - cx)**2 + (y - cy)**2)
            return max(0, distance_to_center - radius)
        return float('inf')
    
    def draw(self, frame):
        if self.zone_type == 'rectangle':
            x1, y1, x2, y2 = self.coordinates
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            warning_margin = int(self.warning_distance_px)
            cv2.rectangle(frame, (x1-warning_margin, y1-warning_margin), 
                         (x2+warning_margin, y2+warning_margin), (0, 255, 255), 2)
        elif self.zone_type == 'circle':
            cx, cy, radius = self.coordinates
            cv2.circle(frame, (cx, cy), radius, (0, 0, 255), 3)
            warning_radius = radius + int(self.warning_distance_px)
            cv2.circle(frame, (cx, cy), warning_radius, (0, 255, 255), 2)
        
        label_pos = self.get_label_position()
        cv2.putText(frame, f"DANGER: {self.machine_name}", label_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Safe Dist: {self.min_safe_distance_m:.2f}m", 
                   (label_pos[0], label_pos[1] + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def get_label_position(self):
        if self.zone_type == 'rectangle':
            x1, y1, x2, y2 = self.coordinates
            return (x1, y1 - 10)
        elif self.zone_type == 'circle':
            cx, cy, radius = self.coordinates
            return (cx - 80, cy - radius - 30)
        return (10, 10)

class PersonState:
    """Track individual person's state"""
    def __init__(self, person_id):
        self.person_id = person_id
        self.position_history = deque(maxlen=30)
        self.body_angle_history = deque(maxlen=30)
        self.fall_detected = False
        self.fall_start_time = None
        self.last_seen = time.time()
        self.proximity_alerts = {}
        
        # Lifting posture tracking
        self.lifting_detected = False
        self.reba_scores = deque(maxlen=30)
        self.bad_posture_frames = 0
        self.lifting_start_time = None
        
    def update(self, center_y, body_angle, current_time):
        self.position_history.append(center_y)
        self.body_angle_history.append(body_angle)
        self.last_seen = current_time
        
    def detect_sudden_drop(self, threshold=50):
        if len(self.position_history) < 10:
            return False
        recent_positions = list(self.position_history)[-10:]
        initial_pos = np.mean(recent_positions[:3])
        current_pos = np.mean(recent_positions[-3:])
        drop = current_pos - initial_pos
        return drop > threshold
    
    def is_horizontal(self):
        if len(self.body_angle_history) < 3:
            return False
        recent_angles = list(self.body_angle_history)[-5:]
        avg_angle = np.mean(recent_angles)
        return avg_angle > 60
    
    def is_stationary(self, frames_threshold=15):
        if len(self.position_history) < frames_threshold:
            return False
        recent_positions = list(self.position_history)[-frames_threshold:]
        position_variance = np.var(recent_positions)
        return position_variance < 20

def play_alert_sound(alert_type="fall"):
    """Play alert sound"""
    global alert_active
    try:
        if WINDOWS:
            if alert_type == "proximity":
                for _ in range(3):
                    winsound.Beep(2000, 200)
                    time.sleep(0.05)
            elif alert_type == "lifting":
                for _ in range(4):
                    winsound.Beep(1800, 250)
                    time.sleep(0.08)
            else:
                for _ in range(5):
                    winsound.Beep(1500, 300)
                    time.sleep(0.1)
        else:
            print("\a" * 5)
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

def calculate_reba_score(trunk_angle, neck_angle, leg_angle, knee_angle, is_bending_forward):
    """
    Simplified REBA (Rapid Entire Body Assessment) score calculation
    REBA Score ranges from 1-15
    1 = Negligible risk
    2-3 = Low risk
    4-7 = Medium risk (investigate and change)
    8-10 = High risk (investigate and implement change soon)
    11+ = Very high risk (implement change immediately)
    
    Focus on lifting posture:
    - Trunk (spine) angle from vertical
    - Leg position (should bend knees when lifting)
    - Whether lifting with back vs legs
    """
    
    score = 0
    
    # TRUNK SCORE (Most critical for lifting)
    # Upright (0-20°) = 1 point
    # Bent forward 20-60° = 2-3 points
    # Bent forward >60° = 4 points (DANGER)
    if trunk_angle < 20:
        trunk_score = 1
    elif trunk_angle < 60:
        trunk_score = 2 if trunk_angle < 40 else 3
    else:
        trunk_score = 4  # Severe bending
    
    # Add +1 if twisting or side bending (simplified: we check if bending forward)
    if is_bending_forward and trunk_angle > 30:
        trunk_score += 1
    
    score += trunk_score
    
    # NECK SCORE (Secondary factor)
    if neck_angle > 20:
        score += 1
    
    # LEG SCORE (Critical for proper lifting)
    # Legs bent (knees flexed) = Good (0 points)
    # Legs straight while bending = Bad (+3 points)
    if leg_angle > 160 and trunk_angle > 30:  # Straight legs + bent trunk = VERY BAD
        score += 3  # Lifting with back, not legs!
    elif knee_angle < 120:  # Knees are bent (good form)
        score += 0
    else:
        score += 1
    
    # LOAD COUPLING (assume moderate load)
    score += 1
    
    # ACTIVITY SCORE (assume static posture during lift)
    score += 1
    
    return min(score, 15)  # Cap at 15

def detect_lifting_posture(keypoints, person_id, current_time):
    """
    Detect incorrect lifting posture
    Returns: (is_lifting, posture_status, reba_score, risk_level, details)
    """
    if keypoints is None or len(keypoints) == 0:
        return False, "Unknown", 0, "none", {}
    
    kp = keypoints[0]
    
    # Extract keypoints
    nose = kp[0][:2] if kp[0][2] > 0.3 else None
    l_shoulder = kp[5][:2] if kp[5][2] > 0.3 else None
    r_shoulder = kp[6][:2] if kp[6][2] > 0.3 else None
    l_hip = kp[11][:2] if kp[11][2] > 0.3 else None
    r_hip = kp[12][:2] if kp[12][2] > 0.3 else None
    l_knee = kp[13][:2] if kp[13][2] > 0.3 else None
    r_knee = kp[14][:2] if kp[14][2] > 0.3 else None
    l_ankle = kp[15][:2] if kp[15][2] > 0.3 else None
    r_ankle = kp[16][:2] if kp[16][2] > 0.3 else None
    
    # Check if critical points available
    if any(p is None for p in [l_shoulder, r_shoulder, l_hip, r_hip]):
        return False, "Unknown", 0, "none", {}
    
    # Calculate body segment positions
    shoulder_mid = [(l_shoulder[0] + r_shoulder[0])/2, (l_shoulder[1] + r_shoulder[1])/2]
    hip_mid = [(l_hip[0] + r_hip[0])/2, (l_hip[1] + r_hip[1])/2]
    
    # TRUNK ANGLE (spine angle from vertical)
    trunk_vertical = abs(shoulder_mid[1] - hip_mid[1])
    trunk_horizontal = abs(shoulder_mid[0] - hip_mid[0])
    trunk_angle = math.degrees(math.atan2(trunk_horizontal, trunk_vertical + 1e-6))
    
    # NECK ANGLE (head to shoulders)
    neck_angle = 0
    if nose is not None:
        neck_angle = calculate_angle(nose, shoulder_mid, hip_mid)
        # Convert to deviation from neutral
        neck_angle = abs(180 - neck_angle)
    
    # LEG ANGLES (hip-knee-ankle)
    left_leg_angle = 180
    right_leg_angle = 180
    left_knee_angle = 180
    right_knee_angle = 180
    
    if l_knee is not None and l_ankle is not None:
        left_leg_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
    
    if r_knee is not None and r_ankle is not None:
        right_leg_angle = calculate_angle(r_shoulder, r_hip, r_knee)
        right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
    
    avg_leg_angle = (left_leg_angle + right_leg_angle) / 2
    avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
    
    # Detect if person is bending forward (potential lifting)
    is_bending_forward = trunk_angle > 25 and trunk_angle < 75
    
    # Check if legs are straight (bad lifting form)
    legs_straight = avg_leg_angle > 160
    
    # Detect lifting posture
    is_lifting = is_bending_forward and trunk_angle > 30
    
    # Calculate REBA score
    reba_score = calculate_reba_score(
        trunk_angle, 
        neck_angle, 
        avg_leg_angle, 
        avg_knee_angle,
        is_bending_forward
    )
    
    # Determine risk level
    if reba_score >= REBA_VERY_HIGH:
        risk_level = "very_high"
        posture_status = "CRITICAL - STOP!"
    elif reba_score >= REBA_HIGH:
        risk_level = "high"
        posture_status = "DANGEROUS POSTURE"
    elif reba_score >= REBA_MEDIUM:
        risk_level = "medium"
        posture_status = "Poor Posture"
    elif reba_score >= REBA_LOW:
        risk_level = "low"
        posture_status = "Acceptable"
    else:
        risk_level = "negligible"
        posture_status = "Good Posture"
    
    # Update person state
    if person_id not in person_states:
        person_states[person_id] = PersonState(person_id)
    
    state = person_states[person_id]
    state.reba_scores.append(reba_score)
    
    # Check if consistently bad posture
    if reba_score >= REBA_HIGH:
        state.bad_posture_frames += 1
        if not state.lifting_detected:
            state.lifting_detected = True
            state.lifting_start_time = current_time
    else:
        state.bad_posture_frames = max(0, state.bad_posture_frames - 1)
        if state.bad_posture_frames == 0:
            state.lifting_detected = False
    
    # Check for specific dangerous pattern: bent back + straight legs
    back_injury_risk = trunk_angle > 45 and legs_straight
    
    details = {
        'trunk_angle': trunk_angle,
        'neck_angle': neck_angle,
        'leg_angle': avg_leg_angle,
        'knee_angle': avg_knee_angle,
        'back_injury_risk': back_injury_risk,
        'lifting_with_back': trunk_angle > 40 and legs_straight,
        'duration': current_time - state.lifting_start_time if state.lifting_start_time else 0
    }
    
    return is_lifting, posture_status, reba_score, risk_level, details

def check_proximity_danger(keypoints, person_id, danger_zones):
    """Check proximity to danger zones"""
    if keypoints is None or len(keypoints) == 0:
        return False, 'safe', []
    
    kp = keypoints[0]
    
    critical_parts = {
        'Head': 0,
        'Left Wrist': 9,
        'Right Wrist': 10,
        'Left Elbow': 7,
        'Right Elbow': 8,
        'Left Shoulder': 5,
        'Right Shoulder': 6,
    }
    
    proximity_details = []
    max_danger_level = 'safe'
    
    for zone in danger_zones:
        for part_name, part_idx in critical_parts.items():
            if kp[part_idx][2] > 0.3:
                point = (int(kp[part_idx][0]), int(kp[part_idx][1]))
                
                if zone.is_point_in_danger(point):
                    proximity_details.append({
                        'zone': zone.machine_name,
                        'body_part': part_name,
                        'level': 'critical',
                        'distance': 0,
                        'point': point
                    })
                    max_danger_level = 'critical'
                else:
                    distance = zone.get_distance_to_zone(point)
                    
                    if distance < zone.min_safe_distance_px:
                        proximity_details.append({
                            'zone': zone.machine_name,
                            'body_part': part_name,
                            'level': 'critical',
                            'distance': distance,
                            'distance_m': distance / 100,
                            'point': point
                        })
                        max_danger_level = 'critical'
                    
                    elif distance < zone.warning_distance_px:
                        if max_danger_level != 'critical':
                            max_danger_level = 'warning'
                        proximity_details.append({
                            'zone': zone.machine_name,
                            'body_part': part_name,
                            'level': 'warning',
                            'distance': distance,
                            'distance_m': distance / 100,
                            'point': point
                        })
    
    is_dangerous = max_danger_level in ['critical', 'warning']
    return is_dangerous, max_danger_level, proximity_details

def detect_fall(keypoints, person_id, current_time):
    """Detect falls"""
    if keypoints is None or len(keypoints) == 0:
        return "Unknown", (128, 128, 128), False, None
    
    kp = keypoints[0]
    
    l_shoulder = kp[5][:2] if kp[5][2] > 0.3 else None
    r_shoulder = kp[6][:2] if kp[6][2] > 0.3 else None
    l_hip = kp[11][:2] if kp[11][2] > 0.3 else None
    r_hip = kp[12][:2] if kp[12][2] > 0.3 else None
    
    critical_points = [l_shoulder, r_shoulder, l_hip, r_hip]
    if any(point is None for point in critical_points):
        return "Unknown", (128, 128, 128), False, None
    
    shoulder_mid = [(l_shoulder[0] + r_shoulder[0])/2, (l_shoulder[1] + r_shoulder[1])/2]
    hip_mid = [(l_hip[0] + r_hip[0])/2, (l_hip[1] + r_hip[1])/2]
    body_center_y = (shoulder_mid[1] + hip_mid[1]) / 2
    
    body_vertical = abs(shoulder_mid[1] - hip_mid[1])
    body_horizontal = abs(shoulder_mid[0] - hip_mid[0])
    body_angle_deg = math.degrees(math.atan2(body_horizontal, body_vertical + 1e-6))
    
    if person_id not in person_states:
        person_states[person_id] = PersonState(person_id)
    
    state = person_states[person_id]
    state.update(body_center_y, body_angle_deg, current_time)
    
    is_fallen = False
    fall_type = None
    
    sudden_drop = state.detect_sudden_drop(threshold=40)
    is_horizontal = state.is_horizontal()
    is_stationary = state.is_stationary(frames_threshold=15)
    
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
    
    elif body_angle_deg < 35 and state.fall_detected:
        state.fall_detected = False
        state.fall_start_time = None
    
    if state.fall_detected:
        is_fallen = True
        duration = current_time - state.fall_start_time if state.fall_start_time else 0
        fall_type = f"PERSON DOWN! ({duration:.1f}s)"
    
    if is_fallen:
        return fall_type, (0, 0, 255), True, state
    elif body_angle_deg > 60:
        return "Lying/Sitting", (255, 165, 0), False, state
    elif body_angle_deg < 25:
        return "Standing", (0, 255, 0), False, state
    else:
        return "Moving", (255, 255, 0), False, state

def setup_danger_zones(frame_width, frame_height):
    """Setup danger zones"""
    zones = []
    
    print("\n" + "="*60)
    print("DANGER ZONE SETUP")
    print("="*60)
    print("\nDo you want to define custom danger zones?")
    print("1. Yes - Define custom zones")
    print("2. No - Use default example zones")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        print("\nHow many danger zones do you want to create?")
        num_zones = int(input("Number of zones: ").strip())
        
        for i in range(num_zones):
            print(f"\n--- ZONE {i+1} ---")
            machine_name = input("Machine name (e.g., 'Lathe Machine'): ").strip()
            
            print("Zone type:")
            print("1. Rectangle")
            print("2. Circle")
            zone_type_choice = input("Enter choice (1 or 2): ").strip()
            
            if zone_type_choice == '1':
                print(f"Frame size: {frame_width}x{frame_height}")
                print("Enter rectangle coordinates:")
                x1 = int(input("  Top-left X: "))
                y1 = int(input("  Top-left Y: "))
                x2 = int(input("  Bottom-right X: "))
                y2 = int(input("  Bottom-right Y: "))
                
                zone = SafetyZone(
                    zone_id=f"zone_{i+1}",
                    zone_type='rectangle',
                    coordinates=[x1, y1, x2, y2],
                    machine_name=machine_name,
                    approach_speed=1.6,
                    stop_time=0.5
                )
                zones.append(zone)
            
            elif zone_type_choice == '2':
                print(f"Frame size: {frame_width}x{frame_height}")
                print("Enter circle parameters:")
                cx = int(input("  Center X: "))
                cy = int(input("  Center Y: "))
                radius = int(input("  Radius: "))
                
                zone = SafetyZone(
                    zone_id=f"zone_{i+1}",
                    zone_type='circle',
                    coordinates=[cx, cy, radius],
                    machine_name=machine_name,
                    approach_speed=1.6,
                    stop_time=0.5
                )
                zones.append(zone)
    
    else:
        print("\nUsing default example zones...")
        
        zone1 = SafetyZone(
            zone_id="zone_1",
            zone_type='rectangle',
            coordinates=[int(frame_width*0.65), int(frame_height*0.3), 
                        int(frame_width*0.85), int(frame_height*0.7)],
            machine_name="Rotating Machine",
            approach_speed=1.6,
            stop_time=0.5,
            intrusion_distance=0.2
        )
        zones.append(zone1)
        
        zone2 = SafetyZone(
            zone_id="zone_2",
            zone_type='circle',
            coordinates=[int(frame_width*0.2), int(frame_height*0.25), 120],
            machine_name="Press Machine",
            approach_speed=2.0,
            stop_time=0.8,
            intrusion_distance=0.3
        )
        zones.append(zone2)
    
    print(f"\n✓ {len(zones)} danger zone(s) configured")
    return zones

def cleanup_old_states(max_age=2.0):
    """Remove old person states"""
    current_time = time.time()
    to_remove = [pid for pid, state in person_states.items() 
                 if current_time - state.last_seen > max_age]
    for pid in to_remove:
        del person_states[pid]

def main():
    global alert_active, last_alert_time, danger_zones
    
    print("\n" + "="*60)
    print("FACTORY SAFETY MONITORING SYSTEM - YOLOv8 Pose")
    print("Features: Fall + Proximity + Lifting Posture Detection")
    print("="*60)
    
    # Video source
    print("\nSelect video source:")
    print("1. Webcam")
    print("2. Video file")
    choice = input("Enter choice (1 or 2): ").strip()
    
    video_source = 0
    
    if choice == '2':
        video_path = input("Enter video file path: ").strip()
        if os.path.exists(video_path):
            video_source = video_path
            print(f"✓ Video loaded: {video_path}")
        else:
            print(f"✗ File not found: {video_path}")
            print("Defaulting to webcam...")
    
    # Load YOLO
    print("\nLoading YOLOv8-Pose model...")
    try:
        model = YOLO('yolov8n-pose.pt')
        print("✓ Model loaded!")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("✗ Could not open video")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✓ Video: {width}x{height} @ {fps} FPS")
    
    # Setup zones
    danger_zones = setup_danger_zones(width, height)
    
    print("\n" + "="*60)
    print("SYSTEM ACTIVE")
    print("="*60)
    print("\nControls: 'q'=Quit | 'p'=Pause | 'r'=Reset | 'd'=Toggle Zones")
    print("="*60 + "\n")
    
    ALERT_COOLDOWN = 1.5
    paused = False
    show_zones = True
    frame_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n✓ End of video")
                break
            
            frame_count += 1
            current_time = time.time()
            
            results = model(frame, conf=0.4, verbose=False)
            keypoints_data = results[0].keypoints.data.cpu().numpy() if results[0].keypoints is not None else None
            annotated_frame = results[0].plot()
            
            if show_zones:
                for zone in danger_zones:
                    zone.draw(annotated_frame)
            
            # Counters
            fall_count = 0
            proximity_count = 0
            lifting_alert_count = 0
            any_fall = False
            any_proximity = False
            any_lifting_danger = False
            
            if keypoints_data is not None and len(keypoints_data) > 0:
                for idx, kp in enumerate(keypoints_data):
                    person_id = f"person_{idx}"
                    
                    # Fall detection
                    status, color, is_fallen, state = detect_fall([kp], person_id, current_time)
                    if is_fallen:
                        any_fall = True
                        fall_count += 1
                    
                    # Proximity detection
                    is_dangerous, danger_level, prox_details = check_proximity_danger([kp], person_id, danger_zones)
                    if is_dangerous:
                        any_proximity = True
                        proximity_count += 1
                        if danger_level == 'critical':
                            color = (0, 0, 255)
                            status = "DANGER - TOO CLOSE!"
                        elif danger_level == 'warning':
                            color = (0, 255, 255)
                            status = "WARNING - Approaching"
                    
                    # Lifting posture detection
                    is_lifting, posture_status, reba_score, risk_level, lift_details = detect_lifting_posture([kp], person_id, current_time)
                    
                    if risk_level in ['high', 'very_high']:
                        any_lifting_danger = True
                        lifting_alert_count += 1
                        
                        # Override status for critical lifting posture
                        if risk_level == 'very_high':
                            color = (128, 0, 128)  # Purple for lifting danger
                            status = f"STOP! REBA:{reba_score}"
                        elif risk_level == 'high':
                            color = (0, 140, 255)  # Orange
                            status = f"Bad Posture REBA:{reba_score}"
                    
                    # Draw on frame
                    if results[0].boxes is not None and len(results[0].boxes) > idx:
                        box = results[0].boxes.xyxy[idx].cpu().numpy()
                        x1, y1, x2, y2 = map(int, box)
                        
                        label = f"ID:{idx+1} - {status}"
                        
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(annotated_frame,
                                    (x1, y1 - text_size[1] - 10),
                                    (x1 + text_size[0] + 10, y1),
                                    color, -1)
                        
                        cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                        
                        # Show lifting details if dangerous
                        if risk_level in ['high', 'very_high']:
                            detail_y = y2 + 25
                            cv2.putText(annotated_frame, f"Trunk: {lift_details['trunk_angle']:.0f}deg", 
                                      (x1, detail_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            if lift_details['lifting_with_back']:
                                cv2.putText(annotated_frame, "LIFTING WITH BACK!", 
                                          (x1, detail_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        # Draw proximity warning lines
                        if prox_details:
                            for detail in prox_details:
                                if detail['level'] == 'critical':
                                    cv2.circle(annotated_frame, detail['point'], 8, (0, 0, 255), -1)
                                    cv2.line(annotated_frame, detail['point'], 
                                           (x1 + (x2-x1)//2, y1 + (y2-y1)//2),
                                           (0, 0, 255), 2)
            
            cleanup_old_states()
            
            # Trigger alerts
            if (any_fall or any_proximity or any_lifting_danger) and not alert_active and \
               (current_time - last_alert_time) > ALERT_COOLDOWN:
                alert_active = True
                last_alert_time = current_time
                
                # Priority: fall > proximity > lifting
                if any_fall:
                    alert_type = "fall"
                elif any_proximity:
                    alert_type = "proximity"
                else:
                    alert_type = "lifting"
                
                threading.Thread(target=play_alert_sound, args=(alert_type,), daemon=True).start()
            
            # Alert banners
            banner_y = 0
            
            if any_fall:
                if int(time.time() * 3) % 2 == 0:
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (0, banner_y), (width, banner_y + 80), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                    cv2.putText(annotated_frame, "!!! FALL DETECTED - EMERGENCY !!!", 
                              (width//2 - 300, banner_y + 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                banner_y += 90
            
            if any_proximity:
                if int(time.time() * 2) % 2 == 0:
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (0, banner_y), (width, banner_y + 80), (0, 165, 255), -1)
                    cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                    cv2.putText(annotated_frame, "⚠ PROXIMITY ALERT - MOVE BACK! ⚠", 
                              (width//2 - 320, banner_y + 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
                banner_y += 90
            
            if any_lifting_danger:
                if int(time.time() * 2.5) % 2 == 0:
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (0, banner_y), (width, banner_y + 80), (128, 0, 128), -1)
                    cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                    cv2.putText(annotated_frame, "⚠ INCORRECT LIFTING - BEND KNEES! ⚠", 
                              (width//2 - 340, banner_y + 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            
            # Info panel
            info_bg = annotated_frame.copy()
            cv2.rectangle(info_bg, (10, height-210), (420, height-10), (0, 0, 0), -1)
            cv2.addWeighted(info_bg, 0.7, annotated_frame, 0.3, 0, annotated_frame)
            
            info_y = height - 180
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (20, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Workers: {len(keypoints_data) if keypoints_data is not None else 0}",
                       (20, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Falls: {fall_count}",
                       (20, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 0, 255) if fall_count > 0 else (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Proximity Alerts: {proximity_count}",
                       (20, info_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (0, 165, 255) if proximity_count > 0 else (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Lifting Alerts: {lifting_alert_count}",
                       (20, info_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (128, 0, 128) if lifting_alert_count > 0 else (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Danger Zones: {len(danger_zones)}",
                       (20, info_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Factory Safety Monitoring System', annotated_frame)
        
        else:
            paused_frame = annotated_frame.copy()
            cv2.putText(paused_frame, "PAUSED - Press 'p' to resume", 
                       (width//2 - 250, height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            cv2.imshow('Factory Safety Monitoring System', paused_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("⏸ PAUSED" if paused else "▶ RESUMED")
        elif key == ord('r'):
            person_states.clear()
            print("🔄 States reset")
        elif key == ord('d'):
            show_zones = not show_zones
            print(f"👁 Zones: {'VISIBLE' if show_zones else 'HIDDEN'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n" + "="*60)
    print("✓ Program terminated")
    print("="*60)

if __name__ == "__main__":
    main()
