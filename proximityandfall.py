"""
Real-time Safety Monitoring System using YOLOv8-Pose
Author: EC Engineering Student Project
Features: 
1. Fall & Slip Detection
2. Proximity to Dangerous Machines Detection
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

# Safety zone configuration based on ISO 13855
class SafetyZone:
    """Represents a danger zone around machinery"""
    def __init__(self, zone_id, zone_type, coordinates, machine_name="Machine", 
                 approach_speed=1.6, stop_time=0.5, intrusion_distance=0.2):
        """
        zone_type: 'rectangle' or 'circle'
        coordinates: 
            - rectangle: [x1, y1, x2, y2]
            - circle: [center_x, center_y, radius]
        approach_speed (K): m/s (default 1.6 m/s for hand speed)
        stop_time (T): seconds (default 0.5s)
        intrusion_distance (C): meters (default 0.2m)
        """
        self.zone_id = zone_id
        self.zone_type = zone_type
        self.coordinates = coordinates
        self.machine_name = machine_name
        self.approach_speed = approach_speed
        self.stop_time = stop_time
        self.intrusion_distance = intrusion_distance
        
        # Calculate minimum safe distance using ISO 13855
        # S = (K × T) + C
        self.min_safe_distance_m = (approach_speed * stop_time) + intrusion_distance
        
        # Convert to pixels (approximate: 1 meter ≈ 100 pixels at standard distance)
        self.min_safe_distance_px = self.min_safe_distance_m * 100
        
        self.warning_distance_px = self.min_safe_distance_px * 1.5  # 150% for warning
        
    def is_point_in_danger(self, point):
        """Check if a point is in danger zone"""
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
        """Calculate minimum distance from point to zone boundary"""
        x, y = point
        
        if self.zone_type == 'rectangle':
            x1, y1, x2, y2 = self.coordinates
            # Distance to nearest edge
            dx = max(x1 - x, 0, x - x2)
            dy = max(y1 - y, 0, y - y2)
            return math.sqrt(dx**2 + dy**2)
        
        elif self.zone_type == 'circle':
            cx, cy, radius = self.coordinates
            distance_to_center = math.sqrt((x - cx)**2 + (y - cy)**2)
            return max(0, distance_to_center - radius)
        
        return float('inf')
    
    def draw(self, frame):
        """Draw the danger zone on frame"""
        if self.zone_type == 'rectangle':
            x1, y1, x2, y2 = self.coordinates
            # Danger zone (red)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            # Warning zone (yellow)
            warning_margin = int(self.warning_distance_px)
            cv2.rectangle(frame, (x1-warning_margin, y1-warning_margin), 
                         (x2+warning_margin, y2+warning_margin), (0, 255, 255), 2)
            
        elif self.zone_type == 'circle':
            cx, cy, radius = self.coordinates
            # Danger zone (red)
            cv2.circle(frame, (cx, cy), radius, (0, 0, 255), 3)
            # Warning zone (yellow)
            warning_radius = radius + int(self.warning_distance_px)
            cv2.circle(frame, (cx, cy), warning_radius, (0, 255, 255), 2)
        
        # Label
        label_pos = self.get_label_position()
        cv2.putText(frame, f"DANGER: {self.machine_name}", label_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Safe Dist: {self.min_safe_distance_m:.2f}m", 
                   (label_pos[0], label_pos[1] + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def get_label_position(self):
        """Get position for zone label"""
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
        self.proximity_alerts = {}  # zone_id: alert_info
        
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
                # Higher pitch for proximity
                for _ in range(3):
                    winsound.Beep(2000, 200)
                    time.sleep(0.05)
            else:
                # Lower pitch for fall
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

def check_proximity_danger(keypoints, person_id, danger_zones):
    """
    Check if worker body parts are too close to danger zones
    Returns: (is_dangerous, alert_level, details)
    alert_level: 'critical', 'warning', or 'safe'
    """
    if keypoints is None or len(keypoints) == 0:
        return False, 'safe', []
    
    kp = keypoints[0]
    
    # Critical body parts to monitor (hands, arms, head)
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
            if kp[part_idx][2] > 0.3:  # Confidence check
                point = (int(kp[part_idx][0]), int(kp[part_idx][1]))
                
                # Check if in danger zone
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
                    # Check distance to zone
                    distance = zone.get_distance_to_zone(point)
                    
                    if distance < zone.min_safe_distance_px:
                        # Too close - critical
                        proximity_details.append({
                            'zone': zone.machine_name,
                            'body_part': part_name,
                            'level': 'critical',
                            'distance': distance,
                            'distance_m': distance / 100,  # Convert to meters
                            'point': point
                        })
                        max_danger_level = 'critical'
                    
                    elif distance < zone.warning_distance_px:
                        # Warning zone
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
    """Setup danger zones - Can be customized by user"""
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
        # Default example zones
        print("\nUsing default example zones...")
        
        # Zone 1: Rectangle in center-right (simulating a machine)
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
        
        # Zone 2: Circle in top-left (simulating another hazard)
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
    print("Features: Fall Detection + Proximity Alert")
    print("="*60)
    
    # Video source selection
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
            video_source = 0
    
    # Load YOLO model
    print("\nLoading YOLOv8-Pose model...")
    try:
        model = YOLO('yolov8n-pose.pt')
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("✗ Could not open video source")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✓ Video opened: {width}x{height} @ {fps} FPS")
    
    # Setup danger zones
    danger_zones = setup_danger_zones(width, height)
    
    print("\n" + "="*60)
    print("SYSTEM ACTIVE")
    print("="*60)
    print("\nControls:")
    print("  'q' - Quit")
    print("  'p' - Pause/Resume")
    print("  'r' - Reset detection states")
    print("  'd' - Toggle danger zones visibility")
    print("\n" + "="*60)
    
    ALERT_COOLDOWN = 2
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
            
            # YOLO inference
            results = model(frame, conf=0.4, verbose=False)
            keypoints_data = results[0].keypoints.data.cpu().numpy() if results[0].keypoints is not None else None
            annotated_frame = results[0].plot()
            
            # Draw danger zones
            if show_zones:
                for zone in danger_zones:
                    zone.draw(annotated_frame)
            
            # Detection flags
            any_fall_detected = False
            any_proximity_alert = False
            fall_count = 0
            proximity_count = 0
            
            if keypoints_data is not None and len(keypoints_data) > 0:
                for idx, kp in enumerate(keypoints_data):
                    person_id = f"person_{idx}"
                    
                    # Fall detection
                    status, color, is_fallen, state = detect_fall([kp], person_id, current_time)
                    
                    if is_fallen:
                        any_fall_detected = True
                        fall_count += 1
                    
                    # Proximity detection
                    is_dangerous, danger_level, proximity_details = check_proximity_danger(
                        [kp], person_id, danger_zones)
                    
                    if is_dangerous:
                        any_proximity_alert = True
                        proximity_count += 1
                        
                        # Override color for proximity alerts
                        if danger_level == 'critical':
                            color = (0, 0, 255)  # Red
                            status = "DANGER - TOO CLOSE!"
                        elif danger_level == 'warning':
                            color = (0, 255, 255)  # Yellow
                            status = "WARNING - Approaching"
                    
                    # Draw bounding box and label
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
                        
                        # Draw lines from body parts to danger zones
                        if proximity_details:
                            for detail in proximity_details:
                                if detail['level'] == 'critical':
                                    cv2.circle(annotated_frame, detail['point'], 8, (0, 0, 255), -1)
                                    # Draw warning line
                                    cv2.line(annotated_frame, detail['point'], 
                                           (x1 + (x2-x1)//2, y1 + (y2-y1)//2),
                                           (0, 0, 255), 2)
            
            cleanup_old_states()
            
            # Trigger alerts
            if (any_fall_detected or any_proximity_alert) and not alert_active and \
               (current_time - last_alert_time) > ALERT_COOLDOWN:
                alert_active = True
                last_alert_time = current_time
                alert_type = "proximity" if any_proximity_alert else "fall"
                threading.Thread(target=play_alert_sound, args=(alert_type,), daemon=True).start()
            
            # Alert banners
            if any_fall_detected:
                if int(time.time() * 3) % 2 == 0:
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                    cv2.putText(annotated_frame, "!!! FALL DETECTED - EMERGENCY !!!", 
                              (width//2 - 300, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            if any_proximity_alert:
                if int(time.time() * 2) % 2 == 0:
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (0, 90), (width, 170), (0, 165, 255), -1)
                    cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                    cv2.putText(annotated_frame, "⚠ PROXIMITY ALERT - MOVE BACK! ⚠", 
                              (width//2 - 320, 135),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
            
            # Info panel
            info_bg = annotated_frame.copy()
            cv2.rectangle(info_bg, (10, height-180), (380, height-10), (0, 0, 0), -1)
            cv2.addWeighted(info_bg, 0.7, annotated_frame, 0.3, 0, annotated_frame)
            
            info_y = height - 155
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
            cv2.putText(annotated_frame, f"Danger Zones: {len(danger_zones)}",
                       (20, info_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Factory Safety Monitoring System', annotated_frame)
        
        else:
            paused_frame = annotated_frame.copy()
            cv2.putText(paused_frame, "PAUSED - Press 'p' to resume", 
                       (width//2 - 250, height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            cv2.imshow('Factory Safety Monitoring System', paused_frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("⏸ PAUSED" if paused else "▶ RESUMED")
        elif key == ord('r'):
            person_states.clear()
            print("🔄 Detection states reset")
        elif key == ord('d'):
            show_zones = not show_zones
            print(f"👁 Danger zones: {'VISIBLE' if show_zones else 'HIDDEN'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n" + "="*60)
    print("✓ Program terminated successfully")
    print("="*60)

if __name__ == "__main__":
    main()
