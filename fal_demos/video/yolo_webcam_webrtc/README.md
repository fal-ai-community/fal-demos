# Webcam WebRTC YOLO Demo

WebRTC client for sending a webcam stream to a YOLO backend and receiving the
annotated stream back.

## Run the backend

Use `fal run` for local dev or `fal deploy` for a hosted endpoint.

```bash
fal run yolo.py
```

or

```bash
fal deploy yolo.py
```

## Run the frontend

```bash
cd frontend
FAL_KEY=myfalkey npm run dev
```

Open the Vite dev server in your browser and set the Endpoint field to the
full WebRTC endpoint (for example: `myuser/myapp/webrtc`).

## Model configuration

The backend loads the Ultralytics YOLOv8n model by default from
`/data/yolov8n.pt`. Override this with `YOLO_MODEL_PATH` if you want different
weights.
