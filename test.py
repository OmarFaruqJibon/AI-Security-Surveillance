import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import time
import smtplib
import ssl
import threading
from email.message import EmailMessage
from email.utils import formatdate
import os
from dotenv import load_dotenv


# Email config
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

def send_email_alert(image_path, subject="Security Alert", body="Person detected."):
    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Date'] = formatdate(localtime=True)
        msg.set_content(body)

        with open(image_path, 'rb') as f:
            img_data = f.read()
            msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename='detected.jpg')

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)

        print(" [Email] Alert with image sent.")

    except Exception as e:
        print(f" [Email] Alert send Failed: {e}")

def send_email_in_thread(image_path, subject, body):
    thread = threading.Thread(target=send_email_alert, args=(image_path, subject, body))
    thread.daemon = True
    thread.start()

# YOLO setup
model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture("demo2.mp4")   # Change to 0 for webcam

# Restricted polygon area
area = [(380, 78), (358, 236), (582, 302), (614, 90)]
area_np = np.array(area, np.int32)

# Function: compute overlap ratio
def overlap_ratio(bbox, polygon):
    # Convert bbox -> rectangle polygon
    x1, y1, x2, y2 = bbox
    rect = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], np.int32)

    rect = rect.reshape((-1,1,2)).astype(np.float32)
    poly = polygon.reshape((-1,1,2)).astype(np.float32)

    inter_area, _ = cv2.intersectConvexConvex(rect, poly)
    rect_area = cv2.contourArea(rect)

    if rect_area == 0: 
        return 0.0
    return inter_area / rect_area

# Main loop
last_sent_time = 0
SEND_INTERVAL = 5  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Cannot read video/webcam")
        break

    frame = cv2.resize(frame, (1020, 600))
    results = model.track(frame, persist=True, classes=[0])  # person only

    current_time = time.time()

    if results and results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        for track_id, box in zip(ids, boxes):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2)//2, (y1 + y2)//2

            ratio = overlap_ratio((x1, y1, x2, y2), area_np)

            if ratio > 0.3:   # at least 30% overlap → intrusion
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'ID {track_id} {ratio:.2f}', (x1, y1), 1, 1)

                if current_time - last_sent_time > SEND_INTERVAL:
                    filename = "detected.jpg"
                    cv2.imwrite(filename, frame)
                    subject = f"Intrusion detected (ID {track_id})"
                    body = "PERSON DETECTED IN RESTRICTED AREA."
                    send_email_in_thread(filename, subject, body)
                    last_sent_time = current_time

    # Draw polygon
    cv2.polylines(frame, [area_np], True, (0,0,255), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
