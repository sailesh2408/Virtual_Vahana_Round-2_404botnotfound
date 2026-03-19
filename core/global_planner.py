import carla
import cv2
import numpy as np

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ImportError:
    pass

class GlobalPlanner:
    def __init__(self, world, resolution=2.0):
        self.world = world
        self.carla_map = world.get_map()
        self.grp = GlobalRoutePlanner(self.carla_map, resolution)
        self.route = []
        
        self.minimap_size = 400
        
        # PRE-CALCULATE MAP BOUNDS using High-Density Waypoints (Smooth Curves)
        print("Caching High-Definition Map Topology...")
        self.waypoints = self.carla_map.generate_waypoints(2.0)
        
        xs = [wp.transform.location.x for wp in self.waypoints]
        ys = [wp.transform.location.y for wp in self.waypoints]
        self.min_x, self.max_x = min(xs), max(xs)
        self.min_y, self.max_y = min(ys), max(ys)
        
        width = self.max_x - self.min_x
        height = self.max_y - self.min_y
        self.scale = min(self.minimap_size / (width * 1.1), self.minimap_size / (height * 1.1))

    def world_to_pixel(self, location):
        px = int((location.y - self.min_y) * self.scale + (self.minimap_size * 0.05))
        py = int((location.x - self.min_x) * self.scale + (self.minimap_size * 0.05))
        return (px, py)

    def set_destination(self, start_transform, end_location):
        self.route = self.grp.trace_route(start_transform.location, end_location)
        return self.route

    def render_minimap(self, ego_transform):
        minimap = np.zeros((self.minimap_size, self.minimap_size, 3), dtype=np.uint8)
        
        # Draw Smooth Roads (Dots instead of jagged lines)
        for wp in self.waypoints:
            px, py = self.world_to_pixel(wp.transform.location)
            cv2.circle(minimap, (px, py), 1, (60, 60, 60), -1)

        # Draw Global Route (Green)
        if self.route:
            for i in range(len(self.route) - 1):
                loc1 = self.route[i][0].transform.location
                loc2 = self.route[i+1][0].transform.location
                cv2.line(minimap, self.world_to_pixel(loc1), self.world_to_pixel(loc2), (0, 255, 0), 2)

        # Draw Ego Vehicle
        ego_px, ego_py = self.world_to_pixel(ego_transform.location)
        cv2.circle(minimap, (ego_px, ego_py), 4, (255, 255, 0), -1)

        return minimap