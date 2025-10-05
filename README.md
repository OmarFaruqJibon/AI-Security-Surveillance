# 🔒 AI Security Surveillance – YOLO + InsightFace

A real-time **AI-powered security system** that combines **YOLO (object detection)** with **InsightFace (face recognition)** to monitor restricted areas.

- Detects when someone enters a **restricted zone**.
- **Recognizes known faces** (authorized users).
- **Triggers alarms + sends email alerts** when an **unknown person** is detected.

---

## ✨ Features

- 🎯 **Restricted-area monitoring** using YOLO.
- 🧑‍🤝‍🧑 **Face recognition with InsightFace** (buffalo_l model).
- 🔔 **Alarm system** (short beep pulses).
- 📩 **Email alerts with snapshots** of intruders.
- 🖼️ **Real-time video display** with bounding boxes and labels.
- 🚀 Works fully on **CPU** (no GPU required).

---

## 🛠️ Tech Stack

- [YOLOv8](https://docs.ultralytics.com/models/yolov8/) – person detection.
- [InsightFace](https://github.com/deepinsight/insightface) – face recognition.
- OpenCV – video stream processing.
- NumPy – math operations.
- cvzone – easy OpenCV utilities.
- Python standard libs: threading, smtplib, winsound, etc.

---

## 🚀 Getting Started

### 1️⃣ Clone the repo

```bash
git clone https://github.com/OmarFaruqJibon/AI-Security-Surveillance
cd AI-Security-Surveillance
```

### 2️⃣ Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add your known faces

- Place clear **frontal images** of authorized users inside `known_faces/`.
- Use filenames as their names (e.g., `John.jpg` → "John").

### 5️⃣ Run the system

```bash
python main_security_system.py
```

---

## ⚙️ Configuration

- **Restricted Area**: Update polygon coordinates inside `main_security_system.py`.
- **Email Alerts**: Update `EMAIL_SENDER`, `EMAIL_PASSWORD`, and `EMAIL_RECEIVER`.
- **YOLO Model**: Replace `models/yolov8s.pt` with your trained/custom model if needed.
- **Thresholds**: Adjust face recognition similarity threshold inside code for stricter/looser matching.

---

## 📧 Email Alert Example

When an **unknown person** enters the restricted area:

- A **beeping alarm** is triggered.
- A snapshot (`detected.jpg`) is saved.
- An **email with the snapshot attached** is sent automatically.

---

## 🔮 Future Improvements

- Add database for **logging intruder history**.
- Cloud integration (send alerts to Telegram/Slack).
- Add GPU support for faster processing.
- Support multi-camera setups.

---

## 👨‍💻 Author

**MD Omar Faruq** 🚀
