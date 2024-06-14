import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import threading
from facenet_pytorch import MTCNN

app = FastAPI()

# Load YOLO model
model = YOLO("../models/best (3).pt", task='detect')
classNames = ["Fake", "Real"]

# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Create a lock for thread-safe operations
lock = threading.Lock()

def detect_faces(frame):
    boxes, _ = mtcnn.detect(frame)
    return boxes

def detect_objects(frame):
    with lock:
        results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)

    return frame

def put_centered_text(frame, text, y, font_scale=1, color=(0, 255, 0), thickness=1):
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_width = text_size[0]
    frame_width = frame.shape[1]
    x = (frame_width - text_width) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Detect faces
    face_boxes = detect_faces(img)
    num_faces = len(face_boxes) if face_boxes is not None else 0

    if num_faces == 1:
        processed_img = detect_objects(img)
        put_centered_text(processed_img, 'One face detected, real-time liveness prediction in process.', 60, font_scale=1, color=(0, 255, 0), thickness=2)
    elif num_faces > 1:
        processed_img = img
        put_centered_text(processed_img, 'Multiple faces in the view. Real-time liveness works on one face only.', 60, font_scale=1, color=(0, 0, 255), thickness=2)
    else:
        processed_img = img
        put_centered_text(processed_img, 'No face detected in the view. One face is required for real-time liveness.', 60, font_scale=1, color=(0, 0, 255), thickness=2)

    # Display the number of faces on the top right corner
    cv2.putText(processed_img, f'Total faces: {num_faces}', (img.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    _, img_encoded = cv2.imencode('.jpg', processed_img)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
