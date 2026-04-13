import cv2
import math
import time
import os
import numpy as np

# Constants
SPEED_LIMIT = 80  # km/hr
DISTANCE_THRESHOLD = 70
SPEED_MULTIPLIER = 214.15
TIMER_START_RANGE = (410, 430)
TIMER_END_RANGE = (235, 255)
CAPTURE_THRESHOLD = 235
RECORD_FILE = "SpeedRecord.txt"
OUTPUT_DIR = "TrafficRecord"
EXCEEDED_DIR = os.path.join(OUTPUT_DIR, "exceeded")


class EuclideanDistTracker:
    def __init__(self):
        """Initialize the tracker with data structures for tracking objects."""
        self.center_points = {}
        self.id_count = 0
        self.speed_times_1 = {}  # Start times for speed measurement
        self.speed_times_2 = {}  # End times for speed measurement
        self.speeds = {}  # Calculated speeds
        self.capture_flags = {}  # Flag to capture once per object
        self.count = 0
        self.exceeded = 0
        
        # Initialize record file
        with open(RECORD_FILE, "w") as f:
            f.write("ID \t SPEED\n------\t-------\n")
    
    def _in_range(self, value: int, range_tuple: tuple) -> bool:
        """Check if value is within the given range."""
        return range_tuple[0] <= value <= range_tuple[1]


    def update(self, objects_rect: list) -> list:
        """Update object tracking with new detections."""
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Check if object was already detected
            same_object_detected = False

            for obj_id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < DISTANCE_THRESHOLD:
                    self.center_points[obj_id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, obj_id])
                    same_object_detected = True

                    # Start timer when object crosses start line
                    if self._in_range(y, TIMER_START_RANGE):
                        self.speed_times_1[obj_id] = time.time()

                    # Stop timer and calculate speed when object crosses end line
                    if self._in_range(y, TIMER_END_RANGE):
                        self.speed_times_2[obj_id] = time.time()
                        elapsed = self.speed_times_2[obj_id] - self.speed_times_1.get(obj_id, 0)
                        self.speeds[obj_id] = elapsed

                    # Set capture flag when object reaches top of ROI
                    if y < CAPTURE_THRESHOLD:
                        self.capture_flags[obj_id] = 1

                    break

            # Detect new object
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.speed_times_1[self.id_count] = 0
                self.speed_times_2[self.id_count] = 0
                self.speeds[self.id_count] = 0
                self.capture_flags[self.id_count] = 0
                self.id_count += 1

        # Update center points to only tracked objects
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            new_center_points[object_id] = self.center_points[object_id]

        self.center_points = new_center_points
        return objects_bbs_ids

    def getsp(self, obj_id: int) -> int:
        """Calculate speed in km/hr for the given object ID."""
        elapsed_time = self.speeds.get(obj_id, 0)
        if elapsed_time != 0:
            speed = SPEED_MULTIPLIER / elapsed_time
        else:
            speed = 0
        return int(speed)

    def capture(self, img, x: int, y: int, h: int, w: int, speed: int, obj_id: int) -> None:
        """Capture and save vehicle image with speed violation status."""
        if self.capture_flags.get(obj_id, 0) == 0:
            self.capture_flags[obj_id] = 1
            
            crop_img = img[y - 5:y + h + 5, x - 5:x + w + 5]
            filename = f"{obj_id}_speed_{speed}.jpg"
            
            # Create base directory
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            base_path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(base_path, crop_img)
            self.count += 1
            
            # Write to record file
            with open(RECORD_FILE, "a") as f:
                if speed > SPEED_LIMIT:
                    # Create exceeded directory and save
                    os.makedirs(EXCEEDED_DIR, exist_ok=True)
                    exceeded_path = os.path.join(EXCEEDED_DIR, filename)
                    cv2.imwrite(exceeded_path, crop_img)
                    f.write(f"{obj_id} \t {speed}<---exceeded\n")
                    self.exceeded += 1
                else:
                    f.write(f"{obj_id} \t {speed}\n")

    def limit(self) -> int:
        """Get the speed limit."""
        return SPEED_LIMIT

    def end(self) -> None:
        """Write summary statistics to record file."""
        with open(RECORD_FILE, "a") as f:
            f.write("\n-------------\n")
            f.write("-------------\n")
            f.write("SUMMARY\n")
            f.write("-------------\n")
            f.write(f"Total Vehicles :\t{self.count}\n")
            f.write(f"Exceeded speed limit :\t{self.exceeded}")

