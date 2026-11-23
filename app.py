import cv2
import time
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# ==========================================
# 1. Configuration & Model Paths
# ==========================================
WORKER_MODEL_PATH = "yolo26n.pt"  # Pre-trained for accurate persons
PPE_MODEL_PATH = r"U:\PPE Detection\ppe.pt"
VIDEO_PATH = r"U:\PPE Detection\chrome_IsDr11UOzr.mp4"

worker_model = YOLO(WORKER_MODEL_PATH)
ppe_model = YOLO(PPE_MODEL_PATH)

# Styling & Colors
COLORS = {
    "Worker": (255, 144, 30),
    "Hard_hat": (0, 255, 255),
    "Vest": (0, 165, 255),
    "Gloves": (0, 255, 0),
    "Safety_boots": (255, 0, 255),
    "Mask": (255, 255, 0)
}
TARGET_PPE = ["Hard_hat", "Vest", "Gloves", "Safety_boots", "Mask"]

# For Tracking Trails & Stable Counting
track_history = defaultdict(lambda: [])
unique_workers = set()
MAX_TRAIL_LENGTH = 30

# ==========================================
# 2. Main Processing Loop
# ==========================================
cap = cv2.VideoCapture(VIDEO_PATH)
prev_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video ended.")
        break

    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time

    # Current Frame Counters
    current_frame_counts = {"Hard_hat": 0, "Vest": 0, "Worker": 0}

    # Run Models
    worker_results = worker_model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.30, classes=[0], verbose=False)
    ppe_results = ppe_model.predict(frame, conf=0.10, iou=0.45, verbose=False)

    # ==========================================
    # 3. Draw Worker Boxes & Motion Trails
    # ==========================================
    if worker_results[0].boxes is not None and len(worker_results[0].boxes) > 0:
        boxes = worker_results[0].boxes.xyxy.int().cpu().tolist()
        if worker_results[0].boxes.id is not None:
            track_ids = worker_results[0].boxes.id.int().cpu().tolist()
        else:
            track_ids = [-1] * len(boxes)

        for box, track_id in zip(boxes, track_ids):
            if track_id != -1:
                unique_workers.add(track_id) # Stable Unique Count
            
            current_frame_counts["Worker"] += 1
            x1, y1, x2, y2 = box
            
            # Center of the worker for trail
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            track_history[track_id].append(center)
            
            # Keep trail to a maximum length
            if len(track_history[track_id]) > MAX_TRAIL_LENGTH:
                track_history[track_id].pop(0)

            # Draw Trail (Motion Path)
            points = np.hstack(track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=COLORS["Worker"], thickness=2)

            # Draw Worker Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["Worker"], 2)
            
            label = f"[ID: {track_id}] Worker"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 22), (x1 + w, y1), COLORS["Worker"], -1)
            cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # ==========================================
    # 4. Draw PPE Boxes
    # ==========================================
    if ppe_results[0].boxes is not None and len(ppe_results[0].boxes) > 0:
        boxes = ppe_results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = ppe_results[0].boxes.cls.int().cpu().tolist()
        confs = ppe_results[0].boxes.conf.cpu().tolist()

        for box, class_id, conf in zip(boxes, class_ids, confs):
            class_name = ppe_model.names[class_id]

            if class_name in TARGET_PPE:
                current_frame_counts[class_name] = current_frame_counts.get(class_name, 0) + 1
                x1, y1, x2, y2 = box
                color = COLORS.get(class_name, (200, 200, 200))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # ==========================================
    # 5. UI/UX: Translucent KPI Sidebar (LinkedIn Wow Factor)
    # ==========================================
    # Calculate dummy compliance rate based on visible helmets/vests vs workers
    workers_now = current_frame_counts["Worker"]
    helmets_now = current_frame_counts["Hard_hat"]
    compliance_rate = int((helmets_now / workers_now * 100)) if workers_now > 0 else 100

    # Create Transparent Overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (350, 1080), (20, 20, 20), -1) # Dark sidebar
    
    # Header background
    cv2.rectangle(overlay, (0, 0), (350, 80), (255, 144, 30), -1) 
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Sidebar Text & KPIs
    cv2.putText(frame, "AI SAFETY VISION", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2)
    
    # KPI 1: FPS
    cv2.putText(frame, f"System FPS: {int(fps)}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(frame, (20, 150), (330, 150), (100, 100, 100), 1)

    # KPI 2: Stable Total Unique Count
    cv2.putText(frame, "TOTAL UNIQUE WORKERS", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"{len(unique_workers)}", (20, 240), cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 144, 30), 2)
    
    # KPI 3: Live Compliance Rate
    cv2.putText(frame, "LIVE COMPLIANCE RATE", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    comp_color = (0, 255, 0) if compliance_rate >= 80 else (0, 0, 255)
    cv2.putText(frame, f"{compliance_rate}%", (20, 350), cv2.FONT_HERSHEY_DUPLEX, 1.8, comp_color, 2)

    # KPI 4: Active Frame Assets
    cv2.putText(frame, "CURRENT FRAME DETECTIONS", (20, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Active Workers: {workers_now}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["Worker"], 2)
    cv2.putText(frame, f"Hard Hats: {current_frame_counts['Hard_hat']}", (20, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["Hard_hat"], 2)
    cv2.putText(frame, f"Safety Vests: {current_frame_counts['Vest']}", (20, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["Vest"], 2)

    # Display Window
    frame_resized = cv2.resize(frame, (1280, 720))
    cv2.imshow("LinkedIn Ready PPE Tracking", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()