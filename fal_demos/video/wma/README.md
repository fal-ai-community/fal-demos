# WMA World Model Examples

Three apps built on the [World Model Accelerator](https://fal.ai/docs/documentation/development/wma) (`fal.wma`).
Each app streams video over WebRTC and takes control messages from a client.
All three are experimental and pin `fal[wma]==1.82.1`.

| App | Pattern | Class | Python | Docs |
| --- | --- | --- | --- | --- |
| `matrix_game.py` | Keyboard-controlled world model (`CommandWorld` profile). Matrix-Game 2.0 reacts to move and look commands. | `MatrixGameApp` | 3.12 | [Deploy Real-time World Model](https://fal.ai/docs/examples/video-generation/deploy-realtime-world-model) |
| `video_restyle.py` | Camera in, video out. StreamDiffusionV2 restyles a webcam stream with a text prompt. | `RestyleApp` | 3.11 | [Deploy Real-time Video Restyle](https://fal.ai/docs/examples/video-generation/deploy-realtime-video-restyle) |
| `robot_world_model.py` | Chunked robot policy and world model (`ConfiguredSession` profile). Cosmos3-Edge-Policy-DROID returns an action chunk and predicted video. Uses a custom container image. | `RobotWorldModel` | 3.13 | [Deploy a Robot World Model](https://fal.ai/docs/examples/video-generation/deploy-robot-world-model) |

Each app needs one H100 runner and a fal account with deployment access and WMA bridge access.

## Prerequisites

Create a uv environment with the Python version of the app you want to run, then install the SDK.
`fal run` needs a local Python that matches the app, so use 3.12 for `matrix_game.py`, 3.11 for `video_restyle.py` and 3.13 for `robot_world_model.py`.

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

fal deploy fal_demos/video/wma/matrix_game.py::MatrixGameApp --auth private
fal deploy fal_demos/video/wma/video_restyle.py::RestyleApp --auth private
fal deploy fal_demos/video/wma/robot_world_model.py::RobotWorldModel --auth private
```

The first start of a runner is slow because it downloads weights (and, for the robot app, builds the container image).

## Try it in the dashboard

Open the deployed app in the fal dashboard and select its `/start-session` playground.
Select **Connect**, then send one of the example messages.

- **Matrix-Game:** send **Walk forward**, **Stop** or **Restart**.
- **Video restyle:** allow the camera. Send a prompt example such as **Claymation** and read the `status` messages.
- **Robot world model:** send **Configure**, wait for `configured`, then send **Step from predicted state** and **New instruction**.

To connect from your own page, use the [WMA browser client](https://fal.ai/docs/documentation/development/wma-browser).
For the SDK reference, read the [WMA guide](https://fal.ai/docs/documentation/development/wma).
