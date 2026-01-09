# Matrix WebRTC Demo

WebRTC client for Matrix-Game. Model repo: https://github.com/SkyworkAI/Matrix-Game

## Run the backend

Use `fal run` for local dev or `fal deploy` for a hosted endpoint.

```bash
fal run matrix.py
```

or

```bash
fal deploy matrix.py
```

## Run the frontend

```bash
cd frontend
FAL_KEY=myfalkey npm run dev
```

Open the Vite dev server in your browser and set the Endpoint field to the
full WebRTC endpoint (for example: `myuser/myapp/webrtc`).

## Game modes and seed image

The backend defaults to the `templerun` mode in `MatrixGame2.setup()`. Other
supported modes include `universal` and `gta_drive`. To switch modes, update
`self._default_mode` and `self._mode_seed_dirs` in `matrix.py`.

The seed image path is derived in `MatrixGame2.setup()` from the selected mode.
If you want a different seed, update `self._default_seed_path` in `matrix.py`.
