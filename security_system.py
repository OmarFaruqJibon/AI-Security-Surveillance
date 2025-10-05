# Combined Security System
# YOLO restricted-area detection + InsightFace face recognition
# Alarm + Email when UNKNOWN person enters restricted area

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
import winsound
import os
import insightface
from numpy.linalg import norm
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
            winsound.Beep(1000, 300)  # short beep
            time.sleep(0.2)           # gap between beeps
        else:
            time.sleep(0.1)

threading.Thread(target=alarm_loop, daemon=True).start()


# ========== FACE RECOGNITION (InsightFace) ==========
# app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
app = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

known_embeddings = []
known_names = []

print("🔄 Loading known faces...")
for file in os.listdir("known_faces"):
    if file.lower().endswith((".jpg", ".png")):
        img = cv2.imread(os.path.join("known_faces", file))
        faces = app.get(img)
        if faces:
            known_embeddings.append(faces[0].normed_embedding)
            known_names.append(os.path.splitext(file)[0])
            print(f" ✅ Loaded: {file}")
        else:
            print(f" ⚠️ No face found in {file}, skipping...")

print(f"✅ Total registered users: {len(known_names)}")

def recognize_face(face_embedding, threshold=0.30):
    if not known_embeddings:
        return "Unknown"
    # embeddings are already normalized
    sims = [np.dot(face_embedding, e) for e in known_embeddings]
    best_match = np.argmax(sims)
    best_score = sims[best_match]
    print(f"[DEBUG] Best match: {known_names[best_match]} (score={best_score:.3f})")
    if best_score >= threshold:
        return known_names[best_match]
    return "Unknown"


# ========== YOLO SETUP ==========
model = YOLO("models/yolov8s.pt")
model.overrides['verbose'] = False
cap = cv2.VideoCapture(0)   # use 0 for webcam

# Define restricted polygon area
area = [(720, 207), (720, 584), (930, 584), (930, 207)]  # WEBCAM

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
    authorized_inside = False

    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs):
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            inside = (
                cv2.pointPolygonTest(np.array(area, np.int32), (int(cx), int(cy)), False) >= 0
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

            if inside:
                person_inside = True

                # Crop face region from detected person
                face_crop = frame[y1:y2, x1:x2]
                faces = app.get(face_crop)

                if faces:
                    name = recognize_face(faces[0].normed_embedding)
                    if name != "Unknown":
                        authorized_inside = True
                        cvzone.putTextRect(frame, f"Authorized: {name}", (x1, y1 - 20), 1, 2,
                                           colorT=(255,255,255), colorR=(0,255,0))
                    else:
                        cvzone.putTextRect(frame, "Alert", (x1, y1 - 20), 1, 2,
                                           colorT=(255,255,255), colorR=(0,0,255))
                else:
                    cvzone.putTextRect(frame, "NO FACE DETECTED", (x1, y1 - 20), 1, 2,
                                       colorT=(255,255,255), colorR=(0,0,255))

    # 🔔 Trigger alarm only if unknown inside
    if person_inside and not authorized_inside:
        alarm_active = True
        if current_time - last_sent_time > SEND_INTERVAL:
            filename = "detected.jpg"
            cv2.imwrite(filename, frame)
            subject = "ALERT Detected"
            body = "UNKNOWN PERSON DETECTED IN RESTRICTED AREA."
            send_email_in_thread(filename, subject, body)
            last_sent_time = current_time
    else:
        alarm_active = False

    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 0, 255), 2)
    cv2.imshow("Security System", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

stop_alarm = True
cap.release()
cv2.destroyAllWindows()
