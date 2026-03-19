import numpy as np
import cv2
import math

class LocalPlanner:
    def __init__(self):
        self.lookahead_dist = 8.0 
        self.trajectory_cache = []
        self.route_index = 0
        self.current_route = []
        self.active_lateral_offset = 0.0 

    def set_route(self, global_route):
        self.current_route = global_route
        self.route_index = 0
        self.trajectory_cache = []
        self.active_lateral_offset = 0.0

    def generate_trajectory(self, ego_transform, current_speed_kmh, is_overtaking=False, lane_clear_dir=0):
        if not self.current_route:
            self.trajectory_cache = []
            return None

        ego_loc = ego_transform.location
        ego_yaw = math.radians(ego_transform.rotation.yaw)
        
        while self.route_index < len(self.current_route):
            wp = self.current_route[self.route_index][0]
            
            # Convert waypoint to vehicle coordinate frame to see if it's behind us
            dx = wp.transform.location.x - ego_loc.x
            dy = wp.transform.location.y - ego_loc.y
            wp_veh_x = dx * math.cos(-ego_yaw) - dy * math.sin(-ego_yaw)
            
            # CRITICAL FIX: If distance is close, OR if the waypoint is physically behind the car, pop it!
            if wp.transform.location.distance(ego_loc) < 6.5 or wp_veh_x < 0.0:
                self.route_index += 1
            else:
                break
                
        if self.route_index >= len(self.current_route):
            self.current_route = [] 
            return None

        target_wp = None
        search_idx = self.route_index
        while search_idx < len(self.current_route):
            wp = self.current_route[search_idx][0]
            if wp.transform.location.distance(ego_loc) >= self.lookahead_dist:
                target_wp = wp
                break
            search_idx += 1
            
        if not target_wp:
            target_wp = self.current_route[-1][0] 
            
        dx = target_wp.transform.location.x - ego_loc.x
        dy = target_wp.transform.location.y - ego_loc.y
        
        veh_x = dx * math.cos(-ego_yaw) - dy * math.sin(-ego_yaw)
        veh_y = dx * math.sin(-ego_yaw) + dy * math.cos(-ego_yaw)
        
        veh_x = max(veh_x, 5.0) 
        
        target_lateral_offset = 0.0
        if is_overtaking:
            target_lateral_offset = 3.5 if lane_clear_dir == 1 else -3.5
            
        if current_speed_kmh > 1.0:
            self.active_lateral_offset += (target_lateral_offset - self.active_lateral_offset) * 0.05
        
        self.trajectory_cache = []
        for i in range(1, 11):
            t = i / 10.0 
            pt_x = veh_x * t
            pt_y = veh_y * (t ** 2) * np.sign(veh_y) if abs(veh_y) > 0.5 else veh_y * t
            
            pt_y += self.active_lateral_offset 
            self.trajectory_cache.append((pt_x, pt_y))
            
        return (veh_x, veh_y + self.active_lateral_offset)

    def render_frenet_plot(self, current_speed_kmh, detections=None, aeb_active=False):
        plot = np.ones((400, 400, 3), dtype=np.uint8) * 255
        for i in range(0, 400, 40):
            cv2.line(plot, (i, 0), (i, 400), (220, 220, 220), 1)
            cv2.line(plot, (0, i), (400, i), (220, 220, 220), 1)
            
        def veh_to_plot(x, y):
            plot_x = int(50 + (x / 30.0) * 330) 
            plot_y = int(200 + (y / 15.0) * 180) 
            return (plot_x, plot_y)

        cv2.line(plot, (0, 200), (400, 200), (100, 100, 100), 2) 
        cv2.line(plot, (50, 0), (50, 400), (100, 100, 100), 2)   

        if detections:
            for det in detections:
                if 'obj_x' in det and 'obj_y' in det:
                    obs_x = det['obj_x']
                    obs_y = det['obj_y']
                    if 0 < obs_x < 35 and abs(obs_y) < 20:
                        obs_px, obs_py = veh_to_plot(obs_x, obs_y)
                        cv2.rectangle(plot, (obs_px - 8, obs_py - 8), (obs_px + 8, obs_py + 8), (130, 70, 20), -1)
                        cv2.putText(plot, det['class'], (obs_px - 10, obs_py - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

        ego_px, ego_py = veh_to_plot(0, 0)
        cv2.rectangle(plot, (ego_px - 10, ego_py - 10), (ego_px + 10, ego_py + 10), (180, 100, 50), -1)

        if not aeb_active and self.trajectory_cache:
            for pt_x, pt_y in self.trajectory_cache:
                dot_px, dot_py = veh_to_plot(pt_x, pt_y)
                cv2.circle(plot, (dot_px, dot_py), 6, (0, 0, 255), -1)
                
        cv2.putText(plot, f"v(m/s): {current_speed_kmh/3.6:.2f}", (140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return plot