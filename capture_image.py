import cv2
import os
from ultralytics import YOLO

# Load YOLOv8-Face model
model = YOLO('yolov8n-face.pt')

# Buat folder faces_db jika belum ada
save_dir = 'faces_db'
os.makedirs(save_dir, exist_ok=True)

# Mulai Kamera
cap = cv2.VideoCapture(0)
count = 0

print("[INFO] Tekan 's' untuk menyimpan wajah, 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Face Detection pakai YOLOv8
    results = model(frame)
    boxes = results[0].boxes

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Crop wajah
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        # Tampilkan hasil crop
        cv2.imshow('Face Crop', face_crop)

        # Gambar bounding box di frame utama
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Simpan crop wajah ke folder faces_db
        filename = f"{save_dir}/tes_{count}.jpg"
        cv2.imwrite(filename, face_crop)
        print(f"[INFO] Saved {filename}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
