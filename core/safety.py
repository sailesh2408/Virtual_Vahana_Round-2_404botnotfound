import cv2
import numpy as np
import time
import carla

class SafetyModule:
    def __init__(self):
        self.bumper_offset = 2.5   
        self.height_max = 1.0      
        self.height_min = -1.5     
        
        self.prev_min_dist = float('inf')
        self.prev_time = time.time()
        self.aeb_hold_frames = 0 
        
        self.stop_sign_hold_frames = 0
        self.ignore_stop_sign_frames = 0 

    def evaluate_risk(self, detections, frame, point_cloud, current_speed_kmh, ego_veh=None, is_parking=False):
        aeb_trigger = False
        overtake_requested = False
        warning_msg = ""
        front_dist = float('inf') 
        
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt <= 0: dt = 0.001 
        
        v_ms = current_speed_kmh / 3.6 
        dynamic_distance = 4.5 + (v_ms * 0.5) + ((v_ms ** 2) / (2 * 4.0))
        panic_bubble = np.clip(dynamic_distance, 4.5, 20.0)

        if self.stop_sign_hold_frames > 0:
            self.stop_sign_hold_frames -= 1
            aeb_trigger = True
            warning_msg = "STOP SIGN: MANDATORY HALT"
            front_dist = 0.0
            if self.stop_sign_hold_frames == 0:
                self.ignore_stop_sign_frames = 100 
                
        if self.ignore_stop_sign_frames > 0:
            self.ignore_stop_sign_frames -= 1

        if detections:
            if ego_veh and ego_veh.is_at_traffic_light():
                traffic_light_visible = any(det['class'] == 'traffic light' for det in detections)
                if traffic_light_visible and ego_veh.get_traffic_light_state() == carla.TrafficLightState.Red:
                    aeb_trigger = True
                    warning_msg = "RED LIGHT: STOPPING"
                    front_dist = 0.0

            for det in detections:
                if det['class'] == 'stop sign' and self.stop_sign_hold_frames == 0 and self.ignore_stop_sign_frames == 0:
                    x1, y1, x2, y2 = det['bbox']
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area > 3500: 
                        self.stop_sign_hold_frames = 60 
                        aeb_trigger = True
                        warning_msg = "STOP SIGN DETECTED"
                        front_dist = 0.0
                        break

                if 'obj_x' in det and 'obj_y' in det:
                    obs_x = det['obj_x']
                    obs_y = det['obj_y']
                    
                    if det['class'] == 'pedestrian':
                        lateral_threshold = 1.5 if obs_x < 12.0 else 1.0 
                        if abs(obs_y) < lateral_threshold and 0 < obs_x < (panic_bubble * 1.5):
                            aeb_trigger = True
                            self.aeb_hold_frames = 15 
                            warning_msg = f"PEDESTRIAN IN PATH: {obs_x:.1f}m"
                            front_dist = min(front_dist, obs_x)
                            break
                            
                    elif det['class'] in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                        if abs(obs_y) < 0.9: 
                            front_dist = min(front_dist, obs_x) 
                            if obs_x < 20.0: 
                                if obs_x < panic_bubble:
                                    aeb_trigger = True
                                    warning_msg = f"BLOCKED! AEB STOPPING: {obs_x:.1f}m"
                                else:
                                    overtake_requested = True
                                    warning_msg = f"OVERTAKE TRIGGERED: {obs_x:.1f}m"
                else:
                    if det['class'] == 'pedestrian':
                        x1, y1, x2, y2 = det['bbox']
                        box_height = y2 - y1
                        center_x = (x1 + x2) / 2
                        if box_height > 100 and 250 < center_x < 550:
                            aeb_trigger = True
                            self.aeb_hold_frames = 15
                            warning_msg = "PEDESTRIAN DETECTED (VISION AEB)"
                            front_dist = min(front_dist, 2.0)
                            break

        lidar_lookahead = min(18.0, panic_bubble + 2.0)
        
        # ---------------------------------------------------------
        # PARKING MODE LIDAR ADJUSTMENTS
        # ---------------------------------------------------------
        lat_bound = 0.85 if is_parking else 1.2
        close_lat_bound = 0.85 if is_parking else 1.5
        z_floor = -1.5 if is_parking else -2.2
        close_lookahead = 2.0 if is_parking else 5.0 # FIX: Don't look 5m ahead while parking
        
        if not aeb_trigger and point_cloud is not None:
            emergency_points = point_cloud[
                (point_cloud[:, 0] > self.bumper_offset) & (point_cloud[:, 0] < lidar_lookahead) &  
                (point_cloud[:, 1] > -lat_bound) & (point_cloud[:, 1] < lat_bound) & 
                (point_cloud[:, 2] > self.height_min) & (point_cloud[:, 2] < self.height_max)
            ]
            
            close_points = point_cloud[
                (point_cloud[:, 0] > self.bumper_offset) & (point_cloud[:, 0] < self.bumper_offset + close_lookahead) &  
                (point_cloud[:, 1] > -close_lat_bound) & (point_cloud[:, 1] < close_lat_bound) & 
                (point_cloud[:, 2] > z_floor) & (point_cloud[:, 2] < self.height_max)
            ]
            
            if len(emergency_points) > 15 or len(close_points) > 15:
                dist1 = np.min(emergency_points[:, 0]) if len(emergency_points) > 15 else float('inf')
                dist2 = np.min(close_points[:, 0]) if len(close_points) > 15 else float('inf')
                current_min_dist = min(dist1, dist2) - self.bumper_offset
                
                front_dist = min(front_dist, current_min_dist)
                
                if current_min_dist < panic_bubble:
                    aeb_trigger = True
                    if not warning_msg:
                        warning_msg = f"LIDAR AEB STOPPING: {current_min_dist:.1f}m"
                self.prev_min_dist = current_min_dist
            else:
                self.prev_min_dist = float('inf')

        if self.aeb_hold_frames > 0 and not aeb_trigger:
            aeb_trigger = True
            self.aeb_hold_frames -= 1
            if not warning_msg:
                warning_msg = "AEB HOLD ACTIVE"

        self.prev_time = current_time

        if warning_msg:
            color = (0, 0, 255) if aeb_trigger else (0, 165, 255) 
            cv2.putText(frame, warning_msg, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 4)
                        
        return aeb_trigger, overtake_requested, frame, front_dist