"""
Edge-to-network anomaly/defect detection streamer.

Captures live webcam frames, runs a YOLOv8 model on each frame, and streams
the annotated output as MJPEG over the local network to any browser on the
same network. Mirrors the capture-device -> inference -> review-app pattern
that AIVR runs at scale for rail video (One Big Circle).

Run:
    python app.py
Then open http://<this-machine-ip>:8000 in a browser on the same network
(or http://localhost:8000 on the same machine).
"""

import cv2
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn

# ---- config, change these to match your setup ----
MODEL_PATH = "yolov8n.pt"   # swap in your own trained weights (e.g. bridge-defect-detector's best.pt)
CAMERA_INDEX = 0            # 0 = default webcam, try 1/2 if you have multiple
INFER_EVERY_N_FRAMES = 2    # run detection every Nth frame to keep the stream smooth on CPU
JPEG_QUALITY = 80           # lower = faster stream, higher = better image
# ----------------------------------------------------

app = FastAPI(title="Edge Defect Detection Streamer")
model = YOLO(MODEL_PATH)
camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}. Check it's connected and not in use.")


def generate_frames():
    frame_count = 0
    last_annotated = None

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_count += 1

        if frame_count % INFER_EVERY_N_FRAMES == 0 or last_annotated is None:
            results = model(frame, verbose=False)
            annotated = results[0].plot()
            last_annotated = annotated
        else:
            annotated = last_annotated

        ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
      <head><title>Edge Defect Detection Stream</title></head>
      <body style="margin:0;background:#111;display:flex;align-items:center;justify-content:center;height:100vh;">
        <img src="/video" style="max-width:100%;max-height:100%;" />
      </body>
    </html>
    """


@app.get("/video")
def video():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
