import numpy as np
import cv2
import math

class SensorFusion:
    def __init__(self):
        self.image_w = 800
        self.image_h = 600
        self.fov = 90
        self.focal = self.image_w / (2.0 * np.tan(self.fov * np.pi / 360.0))
        self.cx = self.image_w / 2.0
        self.cy = self.image_h / 2.0

    def fuse_lidar_to_camera(self, image, point_cloud, detections=None, yaw_offset=0.0):
        if point_cloud is None or len(point_cloud) == 0:
            return image

        valid_points = point_cloud[(point_cloud[:, 2] > -3.0) & (point_cloud[:, 2] < 1.0)]
        if len(valid_points) == 0:
            return image

        x_raw, y_raw, z_lidar = valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]
        theta = np.radians(-yaw_offset)
        x_local = x_raw * np.cos(theta) - y_raw * np.sin(theta)
        y_local = x_raw * np.sin(theta) + y_raw * np.cos(theta)

        front_mask = x_local > 0.1
        x_local, y_local, z_lidar = x_local[front_mask], y_local[front_mask], z_lidar[front_mask]
        x_raw_f, y_raw_f = x_raw[front_mask], y_raw[front_mask]

        if len(x_local) == 0:
            return image

        u = (self.focal * y_local / x_local) + self.cx
        v = (self.focal * -z_lidar / x_local) + self.cy
        valid_pixels = np.where((u >= 0) & (u < self.image_w) & (v >= 0) & (v < self.image_h))[0]

        if detections is not None:
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                inside_box_idx = valid_pixels[
                    (u[valid_pixels] >= x1) & (u[valid_pixels] <= x2) & 
                    (v[valid_pixels] >= y1) & (v[valid_pixels] <= y2)
                ]
                
                if len(inside_box_idx) > 0:
                    box_x, box_y, box_z = x_raw_f[inside_box_idx], y_raw_f[inside_box_idx], z_lidar[inside_box_idx]
                    
                    # UPDATED: Lowered Z-threshold to catch feet
                    valid_obj_mask = box_z > -2.2
                    if not np.any(valid_obj_mask):
                        continue 
                        
                    box_x = box_x[valid_obj_mask]
                    box_y = box_y[valid_obj_mask]
                    box_z = box_z[valid_obj_mask]
                    
                    dist_array = np.sqrt(box_x**2 + box_y**2 + box_z**2)
                    closest_idx = np.argmin(dist_array)
                    
                    obj_x, obj_y, obj_z = box_x[closest_idx], box_y[closest_idx], box_z[closest_idx]
                    dist = dist_array[closest_idx]
                    
                    det['true_distance'] = dist
                    det['obj_x'] = obj_x
                    det['obj_y'] = obj_y
                    
                    ground_z = obj_z + 2.4  
                    text = f"{dist:.1f}m | X:{obj_x:.1f} Y:{abs(obj_y):.1f} Z:{ground_z:.1f}"
                    cv2.putText(image, text, (x1, max(20, y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return image

    def project_carla_lanes(self, image, ego_transform, carla_map):
        ego_loc = ego_transform.location
        ego_yaw = math.radians(-ego_transform.rotation.yaw)

        wp = carla_map.get_waypoint(ego_loc)
        if not wp: return image

        h, w = image.shape[:2]
        cam_fov = 90
        cam_focal = w / (2.0 * math.tan(cam_fov * math.pi / 360.0))
        cam_cx = w / 2.0
        cam_cy = h / 2.0
        cam_z_offset = 1.3 if h == 720 else 2.4

        left_lane_pts, right_lane_pts = [], []

        for dist in range(2, 40, 2):
            next_wps = wp.next(dist)
            # CRITICAL FIX: Explicit length check to prevent IndexError
            if not next_wps or len(next_wps) == 0: 
                break
            next_wp = next_wps[0]

            lw = next_wp.lane_width
            wp_yaw = math.radians(next_wp.transform.rotation.yaw)

            lx = next_wp.transform.location.x + math.cos(wp_yaw - math.pi/2) * (lw/2)
            ly = next_wp.transform.location.y + math.sin(wp_yaw - math.pi/2) * (lw/2)
            
            rx = next_wp.transform.location.x + math.cos(wp_yaw + math.pi/2) * (lw/2)
            ry = next_wp.transform.location.y + math.sin(wp_yaw + math.pi/2) * (lw/2)
            
            pz = next_wp.transform.location.z

            def world_to_pixel(px, py):
                dx = px - ego_loc.x
                dy = py - ego_loc.y
                veh_x = dx * math.cos(ego_yaw) - dy * math.sin(ego_yaw)
                veh_y = dx * math.sin(ego_yaw) + dy * math.cos(ego_yaw)

                cam_x = veh_x - 1.5
                cam_y = veh_y
                cam_z = pz - (ego_loc.z + cam_z_offset)

                if cam_x > 1.0: 
                    u = int((cam_focal * cam_y / cam_x) + cam_cx)
                    v = int((cam_focal * -cam_z / cam_x) + cam_cy)
                    if 0 <= u < w and 0 <= v < h:
                        return (u, v)
                return None

            pt_l = world_to_pixel(lx, ly)
            if pt_l: left_lane_pts.append(pt_l)

            pt_r = world_to_pixel(rx, ry)
            if pt_r: right_lane_pts.append(pt_r)

        if len(left_lane_pts) > 2:
            pts = np.array(left_lane_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], False, (255, 255, 0), 4) 
            for p in left_lane_pts: cv2.circle(image, p, 5, (255, 0, 0), -1)

        if len(right_lane_pts) > 2:
            pts = np.array(right_lane_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], False, (255, 255, 0), 4)
            for p in right_lane_pts: cv2.circle(image, p, 5, (255, 0, 0), -1)

        return image