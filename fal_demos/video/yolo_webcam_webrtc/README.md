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

## Run js client

```bash
cd frontend
FAL_KEY=myfalkey npm run dev
```

Open the Vite dev server in your browser and set the Endpoint field to the
full WebRTC endpoint (for example: `myuser/myapp/webrtc`).

## Run python client (alternative)

Alternatively, you can run the python client that will open 2 OpenCV windows for
the local webcam and the annotated stream.

```bash
python yolo_client.py  --endpoints myuser/myapp/realtime
```

## Model configuration

The backend loads the Ultralytics YOLOv8n model by default from
`/data/yolov8n.pt`. Override this with `YOLO_MODEL_PATH` if you want different
weights.
