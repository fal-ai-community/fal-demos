# Matrix WebRTC Demo

WebRTC client for Matrix-Game. Model repo: https://github.com/SkyworkAI/Matrix-Game

## Run the backend

Use `fal run` for local dev or `fal deploy` for a hosted endpoint.

```bash
fal run fal_demo_matrix_webrtc/app.py
```

or

```bash
fal deploy fal_demo_matrix_webrtc/app.py
```

## Run the frontend

```bash
cd frontend
FAL_KEY=myfalkey npm run dev
```

Open the Vite dev server in your browser and set the Endpoint field to the
full WebRTC endpoint (for example: `myuser/myapp/realtime`).

## Game modes and seed image

The backend defaults to the `templerun` mode in `MatrixWebRTC2.setup()`. Other
supported modes include `universal` and `gta_drive`. To switch modes, update
`self._default_mode` and `self._mode_seed_dirs` in `fal_demo_matrix_webrtc/app.py`.

The seed image path is derived in `MatrixWebRTC2.setup()` from the selected mode.
If you want a different seed, update `self._default_seed_path` in `fal_demo_matrix_webrtc/app.py`.
