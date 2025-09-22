import cv2
import os
import csv
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO
import onnxruntime as ort
from scipy.spatial.distance import cosine
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
import threading

# Load YOLOv8-Face model
model = YOLO('yolov8n-face.pt')

# Load ArcFace ONNX model
onnx_session = ort.InferenceSession('arcface.onnx', providers=['CPUExecutionProvider'])

# Load known faces database
face_db = defaultdict(list)
db_path = 'faces_db'
for img_name in os.listdir(db_path):
    img_path = os.path.join(db_path, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    embedding = (cv2.resize(img, (112, 112)).astype(np.float32) - 127.5) / 128.0
    embedding = np.transpose(embedding, (2, 0, 1))[np.newaxis, :]
    emb = onnx_session.run(None, {'input.1': embedding})[0]
    emb = emb / np.linalg.norm(emb)
    face_db[img_name.split('_')[0]].append(emb.flatten())

print(f"[INFO] Loaded {sum(len(v) for v in face_db.values())} face images for {len(face_db)} identities.")

# CSV Log Setup
csv_file = open('logs.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'Room', 'TrackID', 'Name', 'Event', 'Confidence', 'Similarity', 'Xmin', 'Ymin', 'Xmax', 'Ymax'])
csv_lock = threading.Lock()

# Function to run face tracking on a camera
def camera_tracking(room_name, video_source):
    tracker = DeepSort(max_age=15)
    track_id_to_name = {}
    active_names = set()

    cap = cv2.VideoCapture(video_source)
    print(f"[INFO] Starting Face Tracking on {room_name}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = []

        boxes = results[0].boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf >= 0.7:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'face'))

        tracks = tracker.update_tracks(detections, frame=frame)
        current_frame_names = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue

            # Get embedding
            face_img = cv2.resize(face_crop, (112, 112))
            face_img = (cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 128.0
            face_img = np.transpose(face_img, (2, 0, 1))[np.newaxis, :]
            emb = onnx_session.run(None, {'input.1': face_img})[0]
            emb = emb / np.linalg.norm(emb)
            emb = emb.flatten()

            # Recognition
            best_match = "Unknown"
            best_similarity = 0
            for db_name, embeddings in face_db.items():
                for db_emb in embeddings:
                    similarity = 1 - cosine(emb, db_emb)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = db_name

            name = best_match if best_similarity >= 0.5 else "Unknown"
            track_id_to_name[track_id] = name

            # Draw Box & Info
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID:{track_id} {name} Sim:{best_similarity:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if name != "Unknown":
                current_frame_names.add(name)
                # ENTRY
                if name not in active_names:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with csv_lock:
                        csv_writer.writerow([timestamp, room_name, track_id, name, 'Entry', '', f'{best_similarity:.2f}', x1, y1, x2, y2])
                        csv_file.flush()
                    print(f"[{room_name} LOG] {name} (ID:{track_id}) Entry at {timestamp}")
                    active_names.add(name)

        # EXIT
        exited_names = active_names - current_frame_names
        for exited_name in exited_names:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            exited_track_ids = [tid for tid, n in track_id_to_name.items() if n == exited_name]
            exited_id = exited_track_ids[-1] if exited_track_ids else ''
            with csv_lock:
                csv_writer.writerow([timestamp, room_name, exited_id, exited_name, 'Exit', '', '', '', '', ''])
                csv_file.flush()
            print(f"[{room_name} LOG] {exited_name} (ID:{exited_id}) Exit at {timestamp}")
            active_names.remove(exited_name)

        cv2.imshow(f'Face Tracking - {room_name}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] {room_name} tracking stopped.")

# Start threads for each camera
cam1_thread = threading.Thread(target=camera_tracking, args=("Ruang A", 0))
cam2_thread = threading.Thread(target=camera_tracking, args=("Ruang B", 'http://10.65.128.76:8080//video'))

cam1_thread.start()
cam2_thread.start()

cam1_thread.join()
cam2_thread.join()

csv_file.close()
print("[INFO] Program selesai, log disimpan di logs.csv")
