# Distance calculation and coordinate mapping
import time
from typing import Tuple, Dict

class SpatialSynchronizer:
    def __init__(self, config: Dict):
        self.cfg = config
        self.px_to_mm_scale = self.cfg['camera']['fov_width_mm'] / self.cfg['camera']['resolution'][0]
        self.actuator_dist = self.cfg['camera']['dist_to_actuator_mm']
        self.belt_speed = self.cfg['conveyor']['speed_mm_per_sec']
        
        # Total latency compensation in seconds
        self.total_latency = (
            self.cfg['system_latency']['inference_ms'] + 
            self.cfg['system_latency']['actuator_response_ms'] + 
            self.cfg['system_latency']['communication_ms']
        ) / 1000.0

    def get_centroid(self, bbox: Tuple[float, float, float, float]) -> Tuple[int, int]:
        """Calculates the center (x, y) of a YOLO bounding box."""
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return (center_x, center_y)

    def calculate_trigger_delay(self, pixel_x: int) -> float:
        """
        Calculates how many seconds to wait before triggering the actuator.
        Logic:
        1. Find distance from object to camera center in mm.
        2. Find total distance to actuator.
        3. Divide by speed to get travel time.
        4. Subtract system latency.
        """
        # Distance from the left edge of the frame in mm
        dist_from_origin_mm = pixel_x * self.px_to_mm_scale
        
        # Distance from camera center (assuming camera is at fov_width / 2)
        dist_to_camera_center = (self.cfg['camera']['fov_width_mm'] / 2) - dist_from_origin_mm
        
        # Total physical distance the object must travel to reach the nozzle
        total_travel_dist = self.actuator_dist + dist_to_camera_center
        
        # Theoretical time until arrival
        time_to_arrival = total_travel_dist / self.belt_speed
        
        # Compensate for system thinking time
        final_delay = time_to_arrival - self.total_latency
        
        return max(0, final_delay)

# Example Usage:
# sync = SpatialSynchronizer(yaml_config)
# delay = sync.calculate_trigger_delay(960) # Object is exactly in the middle