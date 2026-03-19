import sys
import os
import glob
import carla
import cv2
import math
import numpy as np

target_destination = None
route_planned = False
just_arrived = False 
parking_search_active = False 

stuck_frames = 0
is_reversing = False
is_overtaking = False
overtake_timer = 0
active_lane_clear_dir = 0 
    
try:
    sys.path.append(glob.glob('../CARLA_0.9.16/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
sys.path.append(os.path.abspath('../CARLA_0.9.16/PythonAPI/carla'))

from utils.carla_utils import CarlaEnvironment
from core.perception import PerceptionModule
from core.fusion import SensorFusion
from core.global_planner import GlobalPlanner
from core.local_planner import LocalPlanner
from core.control import VehicleController
from core.safety import SafetyModule

def get_speed(vehicle):
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

def minimap_click(event, x, y, flags, param):
    global target_destination, route_planned, just_arrived, parking_search_active
    global is_reversing, is_overtaking, active_lane_clear_dir
    ego_veh = param['ego_veh']
    carla_map = param['carla_map']
    g_planner = param['g_planner']

    if event == cv2.EVENT_LBUTTONDOWN:
        if 1200 <= x <= 1600 and 0 <= y <= 400:
            click_x = x - 1200
            click_y = y
            world_x = ((click_y - 20) / g_planner.scale) + g_planner.min_x
            world_y = ((click_x - 20) / g_planner.scale) + g_planner.min_y
            target_loc = carla.Location(x=world_x, y=world_y, z=0.0)
            
            waypoint = carla_map.get_waypoint(target_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if waypoint:
                target_destination = waypoint.transform.location
                route_planned = False 
                just_arrived = False
                parking_search_active = False 
                is_reversing = False
                is_overtaking = False
                active_lane_clear_dir = 0
                print(f"\n[SYSTEM] Mapped New Destination: X:{target_loc.x:.1f}, Y:{target_loc.y:.1f}")

def main():
    global target_destination, route_planned, just_arrived, parking_search_active
    global stuck_frames, is_reversing, is_overtaking, overtake_timer, active_lane_clear_dir
    
    env = CarlaEnvironment()
    perception = PerceptionModule()
    fusion = SensorFusion()
    safety = SafetyModule()
    local_planner = LocalPlanner()
    controller = VehicleController(dt=0.05)
    
    try:
        ego_veh = env.spawn_ego_vehicle()
        if not ego_veh: return

        env.attach_camera()
        env.attach_lidar()
        ego_veh.set_autopilot(False)
        
        global_planner = GlobalPlanner(env.world)
        cruise_speed = 30.0 

        window_name = "Team 404botnotfound - Advanced ADAS"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, minimap_click, param={
            'ego_veh': ego_veh, 'carla_map': env.world.get_map(), 'g_planner': global_planner
        })
        
        print("\n[READY] System Online.")

        while True:
            if all(env.camera_data.get(pos) is not None for pos in ['center', 'left', 'right', 'ai_vision']):
                
                if target_destination is not None and not route_planned and not just_arrived:
                    global_planner.set_destination(ego_veh.get_transform(), target_destination)
                    local_planner.set_route(global_planner.route)
                    route_planned = True
                
                current_lidar = env.lidar_data
                
                ann_center, center_dets, _ = perception.process_frame(env.camera_data['center'], run_lanenet=False)
                ann_center = fusion.fuse_lidar_to_camera(ann_center, current_lidar, center_dets, yaw_offset=0)
                
                ann_ai, _, lanes = perception.process_frame(env.camera_data['ai_vision'], run_lanenet=True)
                ann_ai = fusion.project_carla_lanes(ann_ai, ego_veh.get_transform(), env.world.get_map())
                
                current_speed = get_speed(ego_veh)
                
                # UPDATED: Pass is_parking=parking_search_active
                aeb_active, overtake_req, ann_center, front_dist = safety.evaluate_risk(
                    center_dets, ann_center, current_lidar, current_speed, ego_veh=ego_veh, is_parking=parking_search_active
                )
                
                current_wp = env.world.get_map().get_waypoint(ego_veh.get_transform().location)
                
                left_blindspot_clear = True
                right_blindspot_clear = True
                
                if current_lidar is not None:
                    left_points = current_lidar[
                        (current_lidar[:, 0] > -8.0) & (current_lidar[:, 0] < 2.0) &
                        (current_lidar[:, 1] > -4.0) & (current_lidar[:, 1] < -1.5) &
                        (current_lidar[:, 2] > -1.5) & (current_lidar[:, 2] < 1.0)
                    ]
                    if len(left_points) > 15: left_blindspot_clear = False
                        
                    right_points = current_lidar[
                        (current_lidar[:, 0] > -8.0) & (current_lidar[:, 0] < 2.0) &
                        (current_lidar[:, 1] > 1.5) & (current_lidar[:, 1] < 4.0) &
                        (current_lidar[:, 2] > -1.5) & (current_lidar[:, 2] < 1.0)
                    ]
                    if len(right_points) > 15: right_blindspot_clear = False

                # ==========================================
                # CRITICAL FIX: SAME-DIRECTION OVERTAKING
                # Verifies that the lane you are swerving into is moving
                # in the same direction to prevent oncoming traffic crashes!
                # ==========================================
                def is_safe_lane(curr, adj, blindspot_clear):
                    if adj and adj.lane_type == carla.LaneType.Driving and blindspot_clear:
                        if curr.lane_id * adj.lane_id > 0:
                            return True
                    return False

                if overtake_req and not is_overtaking:
                    left_wp = current_wp.get_left_lane()
                    right_wp = current_wp.get_right_lane()
                    
                    if is_safe_lane(current_wp, left_wp, left_blindspot_clear):
                        active_lane_clear_dir = -1
                        is_overtaking = True
                        overtake_timer = 120 
                    elif is_safe_lane(current_wp, right_wp, right_blindspot_clear):
                        active_lane_clear_dir = 1
                        is_overtaking = True
                        overtake_timer = 120 
                    else:
                        overtake_req = False
                        is_overtaking = False

                if is_overtaking:
                    lane_clear_dir = active_lane_clear_dir
                    if aeb_active and overtake_timer < 90:
                        is_overtaking = False
                        overtake_timer = 0
                    elif current_speed > 2.0:
                        overtake_timer -= 1
                        
                    if overtake_timer <= 0:
                        is_overtaking = False
                else:
                    lane_clear_dir = 0
                    active_lane_clear_dir = 0
                
                rear_clear = True
                if current_lidar is not None:
                    rear_points = current_lidar[
                        (current_lidar[:, 0] < -2.0) & (current_lidar[:, 0] > -8.0) &
                        (abs(current_lidar[:, 1]) < 0.95) 
                    ]
                    if len(rear_points) > 5:
                        rear_clear = False
                        
                if route_planned and not just_arrived:
                    if aeb_active and current_speed < 0.5:
                        if not is_reversing:
                            stuck_frames += 1
                            if stuck_frames > 20: # Start reversing much faster (1 second)
                                is_reversing = True
                                stuck_frames = 0
                    else:
                        stuck_frames = 0
                        
                if is_reversing:
                    # Will now reverse until it creates a 15 meter gap for a safe overtake!
                    if not rear_clear or front_dist > 15.0:
                        is_reversing = False
                        stuck_frames = 0
                
                if just_arrived:
                    target_speed, local_target = 0.0, None
                    status_text, status_color = "PARKED SUCCESSFULLY!", (0, 255, 0)
                
                elif not route_planned:
                    target_speed, local_target = 0.0, None
                    status_text, status_color = "WAITING FOR MAP CLICK ->", (0, 165, 255) 
                
                elif is_reversing:
                    local_target = None
                    target_speed = 5.0 
                    status_text, status_color = f"BUILDING OVERTAKE GAP... ({front_dist:.1f}m)", (0, 165, 255)
                    
                elif aeb_active and not parking_search_active:
                    target_speed = 0.0
                    local_target = local_planner.generate_trajectory(ego_veh.get_transform(), current_speed, is_overtaking, lane_clear_dir)
                    status_text, status_color = "AEB ACTIVE - STOPPING", (0, 0, 255) 
                
                else:
                    local_target = local_planner.generate_trajectory(ego_veh.get_transform(), current_speed, is_overtaking, lane_clear_dir)
                    
                    if local_target is None and not parking_search_active:
                        parking_search_active = True
                        status_text, status_color = "SEARCHING FOR PARKING...", (255, 100, 0)
                        
                        outermost_wp = current_wp
                        next_right = current_wp.get_right_lane()
                        
                        while next_right and next_right.lane_type in [carla.LaneType.Driving, carla.LaneType.Parking, carla.LaneType.Shoulder]:
                            outermost_wp = next_right
                            next_right = next_right.get_right_lane()
                        
                        if outermost_wp.lane_type in [carla.LaneType.Parking, carla.LaneType.Shoulder] or outermost_wp != current_wp:
                            # SAFE FIX: Build the route safely, and stop if the lane ends early
                            parking_route = []
                            for d in range(2, 32, 2):
                                next_wps = outermost_wp.next(d)
                                if next_wps and len(next_wps) > 0:
                                    parking_route.append((next_wps[0], None))
                                else:
                                    break # The shoulder/parking lane ended, so we stop plotting here
                                    
                            local_planner.set_route(parking_route)
                            local_target = local_planner.generate_trajectory(ego_veh.get_transform(), current_speed)
                        else:
                            just_arrived = True
                            
                    elif parking_search_active:
                        status_text, status_color = "PULLING OVER...", (255, 150, 0)
                        if aeb_active:
                            target_speed = 0.0
                            status_text, status_color = "SPOT BLOCKED - WAITING", (0, 0, 255)
                        else:
                            waypoints_left = len(local_planner.current_route) - local_planner.route_index
                            target_speed = 5.0 if waypoints_left < 5 else 10.0 
                            
                            if local_target is None:
                                just_arrived = True
                                parking_search_active = False
                                target_speed = 0.0
                    else:
                        if is_overtaking:
                            target_speed = cruise_speed + 5.0
                            status_text, status_color = "EXECUTING OVERTAKE", (0, 165, 255)
                        else:
                            target_speed = cruise_speed
                            status_text, status_color = "AUTONOMOUS DRIVING", (0, 255, 0)

                if is_reversing and rear_clear:
                    control_cmd = carla.VehicleControl()
                    control_cmd.throttle = 0.40 
                    control_cmd.brake = 0.0
                    control_cmd.steer = 0.0
                    control_cmd.reverse = True
                else:
                    control_cmd = controller.run_step(target_speed, current_speed, local_target)
                    control_cmd.reverse = False
                    
                    if not route_planned or just_arrived:
                        control_cmd.throttle, control_cmd.brake, control_cmd.steer, control_cmd.hand_brake = 0.0, 1.0, 0.0, True
                    elif aeb_active and not is_reversing:
                        control_cmd.throttle = 0.0
                        control_cmd.brake = 1.0
                        control_cmd.hand_brake = False
                    
                ego_veh.apply_control(control_cmd)
                
                display_canvas = np.zeros((600, 1600, 3), dtype=np.uint8)
                display_canvas[0:600, 0:800] = ann_center
                
                frenet_plot = local_planner.render_frenet_plot(current_speed, center_dets, aeb_active)
                display_canvas[0:400, 800:1200] = frenet_plot
                
                cv2.putText(display_canvas, status_text, (820, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(display_canvas, f"Speed: {int(current_speed)} / {int(target_speed)} km/h", (820, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(display_canvas, f"Steer: {control_cmd.steer:.2f}", (820, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                minimap = global_planner.render_minimap(ego_veh.get_transform())
                display_canvas[0:400, 1200:1600] = minimap
                
                ai_resized = cv2.resize(ann_ai, (355, 200))
                
                if is_reversing and env.camera_data.get('rear') is not None:
                    rear_resized = cv2.resize(env.camera_data['rear'], (355, 200))
                    display_canvas[400:600, 1245:1600] = rear_resized
                    cv2.putText(display_canvas, "REAR CAMERA", (1250, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                else:
                    display_canvas[400:600, 1245:1600] = ai_resized
                    cv2.putText(display_canvas, "AI LANE VISION", (1250, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                cv2.imshow(window_name, display_canvas)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'):
                target_destination, route_planned, just_arrived, parking_search_active, is_reversing, is_overtaking = None, False, False, False, False, False
                global_planner.route, local_planner.current_route = [], []
                ego_veh.destroy()
                ego_veh = env.spawn_ego_vehicle()
                env.attach_camera()
                env.attach_lidar()
                cv2.setMouseCallback(window_name, minimap_click, param={'ego_veh': ego_veh, 'carla_map': env.world.get_map(), 'g_planner': global_planner})
                
    except KeyboardInterrupt: pass
    finally:
        env.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()