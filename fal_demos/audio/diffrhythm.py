import math
import os
import tempfile
from typing import Literal

import fal
from fal.exceptions import FieldException
from fal.toolkit import File, clone_repository, download_file
from fastapi import Response
from pydantic import BaseModel, Field


class TextToMusicInput(BaseModel):
    lyrics: str = Field(
        title="Lyrics",
        description="The prompt to generate the song from. Must have two sections. Sections start with either [chorus] or a [verse].",
        examples=[
            """[00:10.00]Moonlight spills through broken blinds
[00:13.20]Your shadow dances on the dashboard shrine
[00:16.85]Neon ghosts in gasoline rain
[00:20.40]I hear your laughter down the midnight train
[00:24.15]Static whispers through frayed wires
[00:27.65]Guitar strings hum our cathedral choirs
[00:31.30]Flicker screens show reruns of June
[00:34.90]I'm drowning in this mercury lagoon
[00:38.55]Electric veins pulse through concrete skies
[00:42.10]Your name echoes in the hollow where my heartbeat lies
[00:45.75]We're satellites trapped in parallel light
[00:49.25]Burning through the atmosphere of endless night
[01:00.00]Dusty vinyl spins reverse
[01:03.45]Our polaroid timeline bleeds through the verse
[01:07.10]Telescope aimed at dead stars
[01:10.65]Still tracing constellations through prison bars
[01:14.30]Electric veins pulse through concrete skies
[01:17.85]Your name echoes in the hollow where my heartbeat lies
[01:21.50]We're satellites trapped in parallel light
[01:25.05]Burning through the atmosphere of endless night
[02:10.00]Clockwork gears grind moonbeams to rust
[02:13.50]Our fingerprint smudged by interstellar dust
[02:17.15]Velvet thunder rolls through my veins
[02:20.70]Chasing phantom trains through solar plane
[02:24.35]Electric veins pulse through concrete skies
[02:27.90]Your name echoes in the hollow where my heartbeat lies
""",
        ],
        ui={
            "field": "textarea", # Set the input field to textarea for better user experience
        },
    )
    reference_audio_url: str = Field(
        title="Reference Audio URL",
        description="The URL of the reference audio to use for the music generation.",
        default=None,
        examples=[
            "https://storage.googleapis.com/falserverless/model_tests/diffrythm/rock_en.wav",
        ],
        ui={"important": True}, # Mark as important to not list it in the advanced options section
    )
    style_prompt: str = Field(
        title="Style Prompt",
        description="The style prompt to use for the music generation.",
        default=None,
        examples=[
            "pop",
        ],
    )
    music_duration: Literal["95s", "285s"] = Field(
        title="Music Duration",
        description="The duration of the music to generate.",
        default="95s",
    )
    cfg_strength: float = Field(
        title="CFG Strength",
        description="The CFG strength to use for the music generation.",
        default=4.0,
        le=10.0,
        ge=1.0,
    )
    scheduler: Literal["euler", "midpoint", "rk4", "implicit_adams"] = Field(
        title="Scheduler",
        description="The scheduler to use for the music generation.",
        default="euler",
    )
    num_inference_steps: int = Field(
        title="Number of Inference Steps",
        description="The number of inference steps to use for the music generation.",
        le=100,
        ge=10,
        default=32,
    )
    max_frames: int = Field(
        title="Max Frames",
        description="The maximum number of frames to use for the music generation.",
        default=2048,
        le=8192,
        ge=100,
    )


class Output(BaseModel):
    audio: File = Field(
        description="Generated music file.",
        examples=[
            File(
                **{
                    "url": "https://v3.fal.media/files/elephant/VV4wtKXBpZL1bNv6en36t_output.wav",
                    "content_type": "application/octet-stream",
                    "file_name": "output.wav",
                    "file_size": 33554520,
                }
            ),
        ],
    )


def extract_segments(text):
    result = []
    pos = 0
    while pos < len(text):
        # Find the opening '['
        start = text.find("[", pos)
        if start == -1:
            break

        # Find the closing ']'
        end = text.find("]", start)
        if end == -1:
            break

        # Extract the key inside the brackets
        key = text[start + 1 : end]

        # Find the next '[' or end of the text
        next_start = text.find("[", end)
        if next_start == -1:
            next_start = len(text)

        # Extract the content associated with the key
        content = text[end + 1 : next_start].rstrip()
        result.append((key, content))

        # Update the position
        pos = next_start

    return result

# Custom Docker Image to install apt packages like espeak-ng
DOCKER_STRING = """
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git espeak-ng ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*


# Install Python dependencies
RUN pip install accelerate==1.4.0 \
    inflect==7.5.0 \
    torchdiffeq==0.2.5 \
    torchaudio==2.6.0 \
    x-transformers==2.1.2 \
    transformers==4.49.0 \
    numba==0.61.0 \
    llvmlite==0.44.0 \
    librosa==0.10.2.post1 \
    pyarrow==19.0.1 \
    pandas==2.2.3 \
    pylance==0.23.2 \
    ema-pytorch==0.7.7 \
    prefigure==0.0.10 \
    bitsandbytes==0.45.3 \
    muq==0.1.0 \
    mutagen==1.47.0 \
    pyopenjtalk==0.4.1 \
    pykakasi==2.3.0 \
    jieba==0.42.1 \
    cn2an==0.5.23 \
    pypinyin==0.53.0 \
    onnxruntime==1.20.1 \
    Unidecode==1.3.8 \
    phonemizer==3.3.0 \
    LangSegment==0.2.0 \
    liger_kernel==0.5.4 \
    openai==1.65.2 \
    pydantic==2.10.6 \
    einops==0.8.1 \
    lazy_loader==0.4 \
    scipy==1.15.2 \
    ftfy==6.3.1 \
    torchdiffeq==0.2.5 \
    ffmpeg-python
"""


class DiffRhythm(
    fal.App,
    keep_alive=600,
    min_concurrency=1,
    max_concurrency=4,
    name="diffrhythm",
    kind="container", # Specify the kind of app as container
    image=fal.ContainerImage.from_dockerfile_str(DOCKER_STRING), # Use the custom Docker image
):
    machine_type = "GPU-H100"

    def setup(self):
        import numpy as np
        import torch

        # Clone the DiffRhythm repository
        repo_path = clone_repository(
            "https://huggingface.co/spaces/ASLP-lab/DiffRhythm",
            commit_hash="0d355fb2211e3f4f04112d8ed30cc9211a79c974",
            include_to_path=True,
            target_dir="/app",
            repo_name="diffrythm",
        )
        os.chdir(repo_path)

        # Download the negative prompt file with download_fike utility
        download_file(
            "https://huggingface.co/spaces/ASLP-lab/DiffRhythm/resolve/main/src/negative_prompt.npy",
            target_dir=f"{repo_path}/src",
        )
        self.negative_style_prompt = (
            torch.from_numpy(np.load("./src/negative_prompt.npy")).to("cuda").half()
        )

        download_file(
            "https://huggingface.co/spaces/ASLP-lab/DiffRhythm/resolve/main/diffrhythm/g2p/sources/g2p_chinese_model/poly_bert_model.onnx",
            target_dir=f"{repo_path}/diffrhythm/g2p/sources/g2p_chinese_model",
        )
        from diffrhythm.infer.infer_utils import prepare_model

        device = "cuda"
        self.cfm, self.cfm_full, self.tokenizer, self.muq, self.vae = prepare_model(
            device
        )
        self.warmup()

    def warmup(self):
        self._generate(
            TextToMusicInput(
                lyrics="""[00:10.00]Moonlight spills through broken blinds
[00:13.20]Your shadow dances on the dashboard shrine
[00:16.85]Neon ghosts in gasoline rain
[00:20.40]I hear your laughter down the midnight train
[00:24.15]Static whispers through frayed wires
[00:27.65]Guitar strings hum our cathedral choirs
""",
                reference_audio_url="https://storage.googleapis.com/falserverless/model_tests/diffrythm/rock_en.wav",
                num_inference_steps=32,
            ),
            Response(),
        )

    def _generate(
        self,
        input: TextToMusicInput,
        response: Response,
    ) -> Output:
        if not input.style_prompt and not input.reference_audio_url:
            raise FieldException(
                "style_prompt",
                "Either style prompt or reference audio URL must be provided.",
            )
        import torch
        import torchaudio
        from diffrhythm.infer.infer import inference
        from diffrhythm.infer.infer_utils import (
            get_audio_style_prompt,
            get_lrc_token,
            get_reference_latent,
            get_text_style_prompt,
        )

        # Create a temporary directory to save the intermediates along with the output file
        with tempfile.TemporaryDirectory() as output_dir:
            output_path = os.path.join(output_dir, "output.wav")

            if input.music_duration == "95s":
                max_frames = 2048
                cfm_model = self.cfm
            else:
                max_frames = 6144
                cfm_model = self.cfm_full

            sway_sampling_coef = -1 if input.num_inference_steps < 32 else None
            try:
                lrc_prompt, start_time = get_lrc_token(
                    max_frames, input.lyrics, self.tokenizer, "cuda"
                )
            except Exception as e:
                print("Error in lrc prompt", e)
                if "Unknown language" in str(e):
                    raise FieldException("lyrics", "Unsupported language in lyrics.")

            vocal_flag = False
            if input.reference_audio_url:
                try:
                    ref_audio_path = download_file(
                        input.reference_audio_url, target_dir=output_dir
                    )
                    style_prompt, vocal_flag = get_audio_style_prompt(
                        self.muq, ref_audio_path
                    )
                except Exception as e:
                    raise FieldException(
                        "reference_audio_url",
                        "The reference audio could not be processed.",
                    )
            else:
                try:
                    style_prompt = get_text_style_prompt(self.muq, input.style_prompt)
                except Exception as e:
                    raise FieldException(
                        "style_prompt", "The style prompt could not be processed."
                    )

            latent_prompt = get_reference_latent("cuda", max_frames)
            sample_rate, generated_song = inference(
                cfm_model=cfm_model,
                vae_model=self.vae,
                cond=latent_prompt,
                text=lrc_prompt,
                duration=max_frames,
                style_prompt=style_prompt,
                negative_style_prompt=self.negative_style_prompt,
                steps=input.num_inference_steps,
                sway_sampling_coef=sway_sampling_coef,
                start_time=start_time,
                file_type="wav",
                cfg_strength=input.cfg_strength,
                vocal_flag=vocal_flag,
                odeint_method=input.scheduler,
            )
            torchaudio.save(
                output_path,
                torch.from_numpy(generated_song).transpose(0, 1),
                sample_rate=sample_rate,
            )
            response.headers["x-fal-billable-units"] = str(
                max(math.ceil(generated_song.shape[0] // 441000), 1)
            )
            return Output(audio=File.from_path(output_path))

    @fal.endpoint("/")
    def generate(
        self,
        input: TextToMusicInput,
        response: Response,
    ) -> Output:
        return self._generate(input, response)
