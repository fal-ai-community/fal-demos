# WMA World Model Examples

Four apps built on the [World Model Accelerator](https://fal.ai/docs/documentation/development/wma) (`fal.wma`).
Each app streams media over WebRTC (video, or audio for the voice app) and takes control messages from a client.
All four are experimental and pin `fal[wma]==1.82.1`.

| App | Pattern | Class | Python | Docs |
| --- | --- | --- | --- | --- |
| `matrix_game.py` | Keyboard-controlled world model (`CommandWorld` profile). Matrix-Game 2.0 reacts to move and look commands. | `MatrixGameApp` | 3.12 | [Deploy Matrix-Game 2.0 Real-time World Model](https://fal.ai/docs/examples/video-generation/deploy-realtime-world-model) |
| `video_restyle.py` | Camera in, video out. StreamDiffusionV2 restyles a webcam stream with a text prompt. | `RestyleApp` | 3.11 | [Deploy StreamDiffusionV2 Video Style Transfer](https://fal.ai/docs/examples/video-generation/deploy-realtime-video-restyle) |
| `robot_world_model.py` | Chunked robot policy and world model (`ConfiguredSession` profile). Cosmos3-Edge-Policy-DROID returns an action chunk and predicted video. Uses a custom container image. | `RobotWorldModel` | 3.13 | [Deploy NVIDIA Cosmos Robot World Model](https://fal.ai/docs/examples/video-generation/deploy-robot-world-model) |
| `voice_chat.py` | Microphone in, voice out. Moshi (Kyutai, full duplex) listens and speaks at the same time and streams its own words as `text` messages. | `VoiceChatApp` | 3.12 | [Deploy Moshi Real-time Voice Chat](https://fal.ai/docs/examples/audio-speech/deploy-realtime-voice-chat) |

Each app needs one H100 runner and a fal account with deployment access.
The app owner namespace (the `X` in `fal.run/X/app`) must also be enabled for fal-managed TURN. Request it from fal. Without it, sessions connect only when the runner is directly reachable, which fails on many runners and networks.

The camera and microphone apps need a browser page on HTTPS or `localhost`, because browsers block capture elsewhere. Use headphones with the voice app, so Moshi does not hear its own voice.

## Licenses

Each app downloads its upstream code and weights at a pinned revision. Check each license before you use an app in a product.

| App | Upstream code | Weights |
| --- | --- | --- |
| `matrix_game.py` | [SkyworkAI/Matrix-Game](https://github.com/SkyworkAI/Matrix-Game), MIT | `Skywork/Matrix-Game-2.0`, MIT |
| `video_restyle.py` | [chenfengxu714/StreamDiffusionV2](https://github.com/chenfengxu714/StreamDiffusionV2), Apache-2.0 | `Wan-AI/Wan2.1-T2V-1.3B` and `jerryfeng/StreamDiffusionV2`, Apache-2.0 |
| `robot_world_model.py` | [NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework), see its license file | `nvidia/Cosmos3-Edge-Policy-DROID`, NVIDIA OpenMDW 1.1 (commercial use allowed) |
| `voice_chat.py` | [`moshi`](https://github.com/kyutai-labs/moshi) Python package, MIT | `kyutai/moshiko-pytorch-bf16`, CC-BY-4.0 |

## Prerequisites

Create a uv environment with the Python version of the app you want to run, then install the SDK.
`fal run` needs a local Python that matches the app, so use 3.12 for `matrix_game.py` and `voice_chat.py`, 3.11 for `video_restyle.py` and 3.13 for `robot_world_model.py`.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install 'fal[wma]==1.82.1'
fal auth login
```

## Run and deploy

Run these commands from the repository root.
`fal run` starts an ephemeral runner. Use it as the development loop.
`fal deploy --auth private` creates a private app. Use the deployed app for the dashboard playground and for browser clients.

```bash
fal run fal_demos/video/wma/matrix_game.py::MatrixGameApp
fal run fal_demos/video/wma/video_restyle.py::RestyleApp
fal run fal_demos/video/wma/robot_world_model.py::RobotWorldModel
fal run fal_demos/video/wma/voice_chat.py::VoiceChatApp

fal deploy fal_demos/video/wma/matrix_game.py::MatrixGameApp --auth private
fal deploy fal_demos/video/wma/video_restyle.py::RestyleApp --auth private
fal deploy fal_demos/video/wma/robot_world_model.py::RobotWorldModel --auth private
fal deploy fal_demos/video/wma/voice_chat.py::VoiceChatApp --auth private
```

The first start of a runner is slow because it downloads weights (and, for the robot app, builds the container image).

## Cost

You pay per second for each runner, including setup time and idle time.

- Each app sets `min_concurrency=0`, so it scales to zero when no session is open. The first session after that waits for a cold start.
- Each app sets `keep_alive=300`. After the last session ends, the H100 runner stays up, and billed, for 300 seconds.
- `max_concurrency` limits the number of runners: 2 for Matrix-Game, the style transfer app and voice chat, and 1 for the robot app. Each runner serves one session at a time.

To stop paying for a deployed app, scale it to zero or delete it:

```bash
fal apps scale <app-name> --min-concurrency 0
fal apps delete <app-name>
```

## Try it in the dashboard

Open the deployed app in the fal dashboard. In the **Playground** tab, select the `/start-session` endpoint.
Select **Connect** to start a session.

- **Matrix-Game:** hold a movement or look control, or its keyboard key, to move. Release it to stop. Use **Restart world** to reset.
- **Video restyle:** allow the camera. Send a prompt example such as **Claymation** and read the `status` messages.
- **Robot world model:** send **Configure**, wait for `configured`, then send **Step from predicted state** and **New instruction**.
- **Voice chat:** allow the microphone and start talking. Moshi answers out loud, and its words arrive as `text` messages. Send **New conversation** to start over.

To connect from your own page, use the [WMA browser client](https://fal.ai/docs/documentation/development/wma-browser).
For the SDK reference, read the [WMA guide](https://fal.ai/docs/documentation/development/wma).
