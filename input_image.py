import cv2
import os
from ultralytics import YOLO

# Load YOLOv8-Face model
model = YOLO('yolov8n-face.pt')

# Folder gambar input
input_dir = 'input_images'  # ganti sesuai lokasi gambar
save_dir = 'faces_db'
os.makedirs(save_dir, exist_ok=True)

# Ambil semua nama file gambar dari folder input
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

count = 0

for img_name in image_files:
    img_path = os.path.join(input_dir, img_name)
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"[WARNING] Gagal membuka gambar {img_name}")
        continue

    # Face Detection pakai YOLOv8
    results = model(frame)
    boxes = results[0].boxes

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Crop wajah
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        # Simpan crop wajah ke folder faces_db
        filename = f"{save_dir}/face_{count}.jpg"
        cv2.imwrite(filename, face_crop)
        print(f"[INFO] Saved {filename}")
        count += 1

print("[INFO] Selesai memproses semua gambar.")
