# yolo_pose_identify_suspicious_activities_better.py

import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import os
from collections import deque

# ==============================
# Setup
# ==============================
model = YOLO('models/yolov8s-pose.pt')
model.overrides['verbose'] = False

cap = cv2.VideoCapture('files/vid5.mp4')

# Create "detected" folder
if not os.path.exists("detected"):
    os.makedirs("detected")

labels = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
          "left_wrist", "right_wrist", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle"]

count = 0
angle_buffers = {}   # store recent angles for each person
saved_ids = set()    # IDs already saved

# ==============================
# Processing loop
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))
    count += 1

    # Optional: skip every 2nd frame for speed
    if count % 2 != 0:
        continue

    # Run YOLO pose estimation
    result = model.track(frame)

    if result[0].boxes is not None and result[0].boxes.id is not None:
        boxes = result[0].boxes.xyxy.int().cpu().tolist()
        keypoints = result[0].keypoints.xy.cpu().numpy()
        track_ids = result[0].boxes.id.int().cpu().tolist()
        confs = result[0].boxes.conf.cpu().tolist()

        for box, t_id, keypoint, conf in zip(boxes, track_ids, keypoints, confs):
            if t_id is None or conf * 100 < 50:
                continue

            x1, y1, x2, y2 = box
            cx, cy = None, None
            cx1, cy1 = None, None
            cx2, cy2 = None, None

            for j, point in enumerate(keypoint):
                if j == 6:  # Left shoulder
                    cx, cy = int(point[0]), int(point[1])

                if j == 8:  # Left hip
                    cx1, cy1 = int(point[0]), int(point[1])

                if j == 10:  # Left elbow
                    cx2, cy2 = int(point[0]), int(point[1])

                    if cx is not None and cy is not None and cx1 is not None and cy1 is not None:
                        # Compute angle at hip between shoulder-hip and elbow-hip
                        v1 = np.array([cx, cy]) - np.array([cx1, cy1])
                        v2 = np.array([cx2, cy2]) - np.array([cx1, cy1])

                        norm_v1 = np.linalg.norm(v1)
                        norm_v2 = np.linalg.norm(v2)

                        if norm_v1 != 0 and norm_v2 != 0:
                            cos_angle = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
                            angle_degree = np.degrees(np.arccos(cos_angle))

                            # Draw angle on screen
                            cvzone.putTextRect(frame, f'Angle: {angle_degree:.2f}', (50, 60), 1, 1)

                            # Store angle in buffer
                            if t_id not in angle_buffers:
                                angle_buffers[t_id] = deque(maxlen=5)
                            angle_buffers[t_id].append(angle_degree)

                            # Check if angle stayed suspicious for all recent frames
                            if len(angle_buffers[t_id]) == 5:  # only check when buffer full
                                avg_angle = np.mean(angle_buffers[t_id])

                                if 100 <= avg_angle <= 113:
                                    color = (0, 0, 255)  # red suspicious box
                                    label = f"SUSPICIOUS ID: {t_id}"

                                    # Save frame only once
                                    if t_id not in saved_ids:
                                        save_path = os.path.join("detected", f"frame{count}_id{t_id}.jpg")
                                        cv2.imwrite(save_path, frame)
                                        saved_ids.add(t_id)

                                else:
                                    color = (0, 255, 0)  # green normal
                                    label = f"ID: {t_id}"

                                # Draw rectangle and ID
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                cvzone.putTextRect(frame, label, (x1, y1), 1, 1)

    # Show video
    cv2.imshow("RGB", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
