# WebRTC Controls Demo

Minimal WebRTC demo with no model inference:

- server generates animated frames (color cycle, bouncing text, FPS counter)
- client sends key presses as control messages
- video is streamed over Fal realtime `/realtime`

## Run the backend

Use `fal run` for local dev or `fal deploy` for a hosted endpoint.

```bash
fal run webrtc.py
```

or

```bash
fal deploy webrtc.py
```

## Python client

```bash
python webrtc_client.py --endpoint myuser/myapp/realtime
```

Controls:

- focus the OpenCV window
- press any key to send it
- press `q` to quit

## Frontend client

```bash
cd frontend
FAL_KEY=myfalkey npm install
FAL_KEY=myfalkey npm run dev
```

Open the Vite app in your browser and set Endpoint to:

`myuser/myapp/realtime`

## TURN / ICE configuration

The backend sends ICE servers to clients over realtime signaling (`type: iceservers`).
So clients do not need TURN credentials configured locally.

Server-side ICE mode priority:

1. Metered secret key -> temporary API key -> credentials array (required)

### Metered REST mode (recommended)

Set on the server:

```bash
export METERED_TURN_SECRET_KEY="your_secret_key"
export METERED_TURN_LABEL="your-subdomain"
```

Notes:

- `METERED_TURN_LABEL` is your Metered subdomain (for example `fal-demos`).
- The credential request payload label and expiry are fixed in code.

The demo intentionally fails fast if required Metered secret env vars are
missing, instead of silently falling back.
