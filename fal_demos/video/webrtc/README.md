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

The server and Python client support three ICE modes (in this order):

1. Metered REST credentials (recommended)
2. Static Metered username/password
3. Public STUN fallback

### Metered REST mode (recommended)

Set on both sides (server and client process):

```bash
export METERED_TURN_CREDENTIALS_URL="https://<your-subdomain>.metered.live/api/v1/turn/credentials"
export METERED_TURN_API_KEY="your_api_key"
```

### Static Metered mode

```bash
export METERED_TURN_USERNAME="your_turn_username"
export METERED_TURN_CREDENTIAL="your_turn_credential"
```

If none of the vars above are set, the app uses `stun:stun.l.google.com:19302`.
