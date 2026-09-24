# Edge Defect Detection Streamer

Small demo: webcam captures live video, a YOLOv8 model runs inference on
each frame, annotated output streams over the local network as MJPEG to
any browser. Same basic pattern as AIVR (capture device -> inference ->
review app), just scaled down to a laptop and a webcam.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Get a model. Two options:
   - Quick: leave `MODEL_PATH = "yolov8n.pt"` in `app.py`. Ultralytics
     downloads the stock COCO model automatically on first run (detects
     people, cars, etc, not defects, but proves the pipeline works).
   - Better fit for the story you're telling OBC: swap in your trained
     defect-detection weights from `bridge-defect-detector` (copy the
     `.pt` file into this folder and update `MODEL_PATH`).

3. Run it:
   ```
   python app.py
   ```

4. Open `http://localhost:8000` in a browser. You should see your webcam
   feed with live detection boxes drawn on it.

5. To view from another device on the same WiFi, find this machine's
   local IP (`ipconfig getifaddr en0` on Mac, `ipconfig` on Windows) and
   open `http://<that-ip>:8000` from the other device.

## Tuning

- `INFER_EVERY_N_FRAMES` in `app.py`: raise this if the stream stutters
  on a slower CPU (runs detection less often, reuses the last box overlay
  in between).
- `JPEG_QUALITY`: lower it for a smoother stream over weaker WiFi.

## Publishing to GitHub

```
cd obc-edge-stream-demo
git init
git add .
git commit -m "Edge-to-network defect detection streaming demo"
git branch -M main
git remote add origin https://github.com/Lakshan-D/obc-edge-stream-demo.git
git push -u origin main
```

(Or push it as a branch on `bridge-defect-detector` instead of a new repo,
if you want to visibly connect it to your existing detection model work.)
