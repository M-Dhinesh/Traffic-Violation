import cv2
import numpy as np
from tracker2 import EuclideanDistTracker

# Constants
VIDEO_SOURCE = 'Resources/traffic4.mp4'
FRAME_RATE = 25
FRAME_DELAY = int(1000 / (FRAME_RATE - 1))
FRAME_SCALE = 0.5
ROI_COORDS = (30, 540, 200, 970)  # (y_start, y_end, x_start, x_end)
MASK_THRESHOLD = 200
AREA_THRESHOLD = 1000
DISTANCE_THRESHOLD = 70
TIMER_START_RANGE = (410, 430)
TIMER_END_RANGE = (235, 255)
CAPTURE_THRESHOLD = 235

# Morphological kernels
KERNEL_OPEN = np.ones((3, 3), np.uint8)
KERNEL_CLOSE = np.ones((11, 11), np.uint8)
KERNEL_ERODE = np.ones((5, 5), np.uint8)

# Initialize tracker and video capture
tracker = EuclideanDistTracker()
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.resize(frame, None, fx=FRAME_SCALE, fy=FRAME_SCALE)
    height, width, _ = frame.shape
    
    # Extract ROI
    roi = frame[ROI_COORDS[0]:ROI_COORDS[1], ROI_COORDS[2]:ROI_COORDS[3]]

    # Apply background subtraction and morphological operations
    fgmask = bg_subtractor.apply(roi)
    _, binary_mask = cv2.threshold(fgmask, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask_opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, KERNEL_OPEN)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, KERNEL_CLOSE)
    eroded_mask = cv2.erode(mask_closed, KERNEL_ERODE)

    # Detect contours
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > AREA_THRESHOLD:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
            detections.append([x, y, w, h])

    # Track objects
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, obj_id = box_id
        obj_speed = tracker.getsp(obj_id)

        # Change color based on speed limit
        if obj_speed < tracker.limit():
            color = (0, 255, 0)
            text_color = (255, 255, 0)
        else:
            color = (0, 165, 255)
            text_color = (0, 0, 255)

        cv2.putText(roi, f"{obj_id} {obj_speed}", (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, text_color, 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), color, 3)

        # Capture violating vehicles
        if tracker.capture_flags.get(obj_id, 0) == 1 and obj_speed != 0:
            tracker.capture(roi, x, y, h, w, obj_speed, obj_id)

    # Draw speed check lines
    cv2.line(roi, (0, TIMER_START_RANGE[0]), (960, TIMER_START_RANGE[0]), (0, 0, 255), 2)
    cv2.line(roi, (0, TIMER_START_RANGE[1]), (960, TIMER_START_RANGE[1]), (0, 0, 255), 2)
    cv2.line(roi, (0, TIMER_END_RANGE[0]), (960, TIMER_END_RANGE[0]), (0, 0, 255), 2)
    cv2.line(roi, (0, TIMER_END_RANGE[1]), (960, TIMER_END_RANGE[1]), (0, 0, 255), 2)

    # Display
    cv2.imshow("ROI", roi)

    key = cv2.waitKey(FRAME_DELAY - 10)
    if key == 27:
        tracker.end()
        break

tracker.end()
cap.release()
cv2.destroyAllWindows()
