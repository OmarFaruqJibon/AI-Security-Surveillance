# YOLO + webcam + restricted area detection + alarm + email alerts

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
import winsound   # For Windows Beep
import os
from dotenv import load_dotenv

# ========== EMAIL CONFIG ==========
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


# ========== ALARM SYSTEM ==========
alarm_active = False
stop_alarm = False

def alarm_loop():
    global alarm_active, stop_alarm
    while not stop_alarm:
        if alarm_active:
            # Beep at 1000 Hz for 300 ms
            winsound.Beep(1000, 300)
            time.sleep(0.2)
        else:
            time.sleep(0.1)  # avoid CPU overuse


# Start alarm thread
threading.Thread(target=alarm_loop, daemon=True).start()


# ========== YOLO SETUP ==========
model = YOLO("models/yolov8s.pt")
model.overrides['verbose'] = False
cap = cv2.VideoCapture("files/demo2.mp4")

# Define restricted polygon area
area = [(380, 78), (358, 236), (582, 302), (614, 90)] # FOR DEMO2.MP4 VEDIO
# area = [(720, 207), (720, 584), (930, 584), (930, 207)]

# MAIN LOOP
last_sent_time = 0
SEND_INTERVAL = 5  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Cannot read frame (end of video or error)")
        break

    frame = cv2.resize(frame, (1020, 600))
    results = model(frame, classes=[0])  # detect persons only

    current_time = time.time()
    person_inside = False

    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs):
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            inside = (
                cv2.pointPolygonTest(np.array(area, np.int32), (int(cx), int(cy)), False) >= 0 or
                cv2.pointPolygonTest(np.array(area, np.int32), (int(x1), int(y1)), False) >= 0 or
                cv2.pointPolygonTest(np.array(area, np.int32), (int(x2), int(y2)), False) >= 0 or
                cv2.pointPolygonTest(np.array(area, np.int32), (int(x1), int(y2)), False) >= 0
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

            if inside:
                person_inside = True
                cvzone.putTextRect(frame, "ALERT", (x1, y1 - 20), 1, 2, colorT=(255,255,255), colorR=(0,0,255))

                if current_time - last_sent_time > SEND_INTERVAL:
                    filename = "detected.jpg"
                    cv2.imwrite(filename, frame)
                    subject = "ALERT Detected"
                    body = "PERSON DETECTED IN RESTRICTED AREA."
                    send_email_in_thread(filename, subject, body)
                    last_sent_time = current_time

    # 🔔 Control alarm
    alarm_active = person_inside

    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 0, 255), 2)
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

stop_alarm = True
cap.release()
cv2.destroyAllWindows()
