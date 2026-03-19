from simple_pid import PID
import carla
import math
import numpy as np

class VehicleController:
    def __init__(self, dt=0.05):
        # Longitudinal Controller (PID)
        self.lon_pid = PID(1.0, 0.1, 0.05, setpoint=0)
        self.lon_pid.output_limits = (-1.0, 1.0) 
        self.lon_pid.sample_time = dt
        
        # Lateral Controller (Stanley Gain)
        self.k_stanley = 0.55 # Tuned up slightly for tighter spline tracking
        self.k_soft = 1.0

    def run_step(self, target_speed_kmh, current_speed_kmh, local_target_coords):
        control = carla.VehicleControl()
        
        # ==========================================
        # 1. LONGITUDINAL CONTROL (Throttle/Brake)
        # ==========================================
        if target_speed_kmh == 0.0:
            control.throttle = 0.0
            control.brake = 1.0  
            self.lon_pid.reset() 
        else:
            self.lon_pid.setpoint = target_speed_kmh
            control_signal = self.lon_pid(current_speed_kmh)
            
            if control_signal > 0:
                control.throttle = min(control_signal, 1.0)
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = min(abs(control_signal), 1.0)

        # ==========================================
        # 2. LATERAL CONTROL (Stanley Kinematics)
        # ==========================================
        if local_target_coords is not None:
            # Target is in Vehicle Frame: Z is forward distance, X is lateral distance (cross-track error)
            target_z, target_x = local_target_coords
            v_meters_per_sec = max(current_speed_kmh / 3.6, 1.0)
            
            # A. Heading Error: The angle to the target point on our spline
            heading_error = math.atan2(target_x, target_z)

            # B. Cross-Track Error: The lateral distance to the path
            crosstrack_error = target_x

            # C. Stanley Control Law
            steering_angle = heading_error + math.atan2(self.k_stanley * crosstrack_error, v_meters_per_sec + self.k_soft)
            
            # CARLA steering input is strictly bounded [-1.0, 1.0]
            control.steer = float(np.clip(steering_angle, -1.0, 1.0))
        else:
            # If vision loses the lane temporarily, hold the wheel steady
            control.steer = 0.0

        return control