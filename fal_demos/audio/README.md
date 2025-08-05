# Deploy a Text-to-Music Model

This example demonstrates how to build a sophisticated text-to-music generation service using DiffRhythm, showcasing advanced fal features including custom Docker images, repository cloning, file downloading, and complex audio processing workflows.

> **Source Code**: Find the complete implementation at [fal-ai-community/fal_demos](https://github.com/fal-ai-community/fal_demos/blob/main/fal_demos/audio/diffrhythm.py)

## Key Features

- **Custom Docker Images**: Build containers with system dependencies
- **Repository Cloning**: Clone and integrate external repositories
- **File Management**: Download models and handle temporary files
- **Audio Processing**: Generate music from lyrics and style prompts
- **Complex Validation**: Advanced input validation with custom error handling
- **Warmup Procedures**: Optimize model loading for better performance
- **Billing Integration**: Usage-based pricing tied to audio duration

## When to Use Custom Docker Images

Custom Docker images are essential when you need:
- System-level dependencies (apt packages, libraries)
- Complex installation procedures
- External tools or binaries
- Custom build processes

## Project Setup

```python
import math
import os
import tempfile
from typing import Literal

import fal
from fal.exceptions import FieldException
from fal.toolkit import File, clone_repository, download_file
from fastapi import Response
from pydantic import BaseModel, Field
```

## Input Model with Advanced Validation

Define comprehensive input schemas with custom UI elements:

```python
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
[00:49.25]Burning through the atmosphere of endless night""",
        ],
        ui={
            "field": "textarea",  # Custom UI field type
        },
    )
    reference_audio_url: str = Field(
        title="Reference Audio URL",
        description="The URL of the reference audio to use for the music generation.",
        default=None,
        examples=[
            "https://storage.googleapis.com/falserverless/model_tests/diffrythm/rock_en.wav",
        ],
        ui={"important": True},  # Mark as important in UI
    )
    style_prompt: str = Field(
        title="Style Prompt",
        description="The style prompt to use for the music generation.",
        default=None,
        examples=["pop"],
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
```

## Output Model for Audio Files

```python
class Output(BaseModel):
    audio: File = Field(
        description="Generated music file.",
        examples=[
            File(
                url="https://v3.fal.media/files/elephant/VV4wtKXBpZL1bNv6en36t_output.wav",
                content_type="application/octet-stream",
                file_name="output.wav",
                file_size=33554520,
            ),
        ],
    )
```

## Custom Docker Image Configuration

Create a Dockerfile string with all necessary dependencies:

```python
DOCKER_STRING = """
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git espeak-ng ffmpeg && \
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
```

## Application Configuration with Container Image

```python
class DiffRhythm(
    fal.App,
    keep_alive=600,
    min_concurrency=1,
    max_concurrency=4,
    name="diffrhythm",
    kind="container",  # Specify container-based deployment
    image=fal.ContainerImage.from_dockerfile_str(DOCKER_STRING),  # Custom image
):
    machine_type = "GPU-H100"
```

## Model Setup with Repository Cloning

The setup method handles complex initialization including repository cloning:

```python
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

    # Download required model files
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

    # Initialize models
    from diffrhythm.infer.infer_utils import prepare_model

    device = "cuda"
    self.cfm, self.cfm_full, self.tokenizer, self.muq, self.vae = prepare_model(device)
    
    # Perform warmup
    self.warmup()
```

## Warmup Strategy

Implement warmup to improve cold start performance:

```python
def warmup(self):
    self._generate(
        TextToMusicInput(
            lyrics="""[00:10.00]Moonlight spills through broken blinds
[00:13.20]Your shadow dances on the dashboard shrine
[00:16.85]Neon ghosts in gasoline rain
[00:20.40]I hear your laughter down the midnight train
[00:24.15]Static whispers through frayed wires
[00:27.65]Guitar strings hum our cathedral choirs""",
            reference_audio_url="https://storage.googleapis.com/falserverless/model_tests/diffrythm/rock_en.wav",
            num_inference_steps=32,
        ),
        Response(),
    )
```

## Core Generation Logic

The generation method handles complex audio processing workflows:

```python
def _generate(self, input: TextToMusicInput, response: Response) -> Output:
    # Custom validation logic
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

    # Use temporary directory for file handling
    with tempfile.TemporaryDirectory() as output_dir:
        output_path = os.path.join(output_dir, "output.wav")

        # Configure model based on duration
        if input.music_duration == "95s":
            max_frames = 2048
            cfm_model = self.cfm
        else:
            max_frames = 6144
            cfm_model = self.cfm_full

        sway_sampling_coef = -1 if input.num_inference_steps < 32 else None

        # Process lyrics with error handling
        try:
            lrc_prompt, start_time = get_lrc_token(
                max_frames, input.lyrics, self.tokenizer, "cuda"
            )
        except Exception as e:
            print("Error in lrc prompt", e)
            if "Unknown language" in str(e):
                raise FieldException("lyrics", "Unsupported language in lyrics.")

        # Handle reference audio or style prompt
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

        # Generate music
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

        # Save audio file
        torchaudio.save(
            output_path,
            torch.from_numpy(generated_song).transpose(0, 1),
            sample_rate=sample_rate,
        )

        # Calculate billing based on audio duration
        response.headers["x-fal-billable-units"] = str(
            max(math.ceil(generated_song.shape[0] // 441000), 1)
        )

        return Output(audio=File.from_path(output_path))

@fal.endpoint("/")
def generate(self, input: TextToMusicInput, response: Response) -> Output:
    return self._generate(input, response)
```

## Key Concepts and Best Practices

### Custom Docker Images

**When to use:**
- System dependencies required (espeak-ng, ffmpeg)
- Complex installation procedures
- External tools or binaries

**Best practices:**
```python
# Pin all versions for reproducibility
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Clean up after installations
RUN apt-get update && \
    apt-get install -y git espeak-ng ffmpeg && \
    rm -rf /var/lib/apt/lists/*

```

### Repository Cloning

```python
repo_path = clone_repository(
    "https://huggingface.co/spaces/ASLP-lab/DiffRhythm",
    commit_hash="0d355fb2211e3f4f04112d8ed30cc9211a79c974",  # Pin commit
    include_to_path=True,  # Add to Python path
    target_dir="/app",
    repo_name="diffrythm",
)
```

### File Management

```python
# Download external files
download_file(
    "https://example.com/model.bin",
    target_dir="/path/to/target"
)

# Use temporary directories for processing
with tempfile.TemporaryDirectory() as output_dir:
    output_path = os.path.join(output_dir, "output.wav")
    # Process files...
    return Output(audio=File.from_path(output_path))
```

### Custom Validation

```python
# Implement business logic validation
if not input.style_prompt and not input.reference_audio_url:
    raise FieldException(
        "style_prompt",
        "Either style prompt or reference audio URL must be provided.",
    )
```

### Error Handling for External Dependencies

```python
try:
    lrc_prompt, start_time = get_lrc_token(
        max_frames, input.lyrics, self.tokenizer, "cuda"
    )
except Exception as e:
    if "Unknown language" in str(e):
        raise FieldException("lyrics", "Unsupported language in lyrics.")
```

### Audio-Specific Billing

```python
# Calculate billing based on audio duration
audio_duration_seconds = generated_song.shape[0] / sample_rate
billable_units = max(math.ceil(audio_duration_seconds / 10), 1)  # Per 10-second blocks
response.headers["x-fal-billable-units"] = str(billable_units)
```

## Deployment and Usage

### Running the Service
```bash
# Development
fal run fal_demos/audio/diffrhythm.py::DiffRhythm

# Production deployment
fal deploy diffrhythm
```

### Making Requests
```python
import fal_client

result = await fal_client.submit_async(
    "your-username/diffrhythm",
    arguments={
        "lyrics": """[00:10.00]Moonlight spills through broken blinds
[00:13.20]Your shadow dances on the dashboard shrine""",
        "reference_audio_url": "https://example.com/reference.wav",
        "music_duration": "95s",
        "num_inference_steps": 32,
    }
)

# Download the generated audio
audio_url = result["audio"]["url"]
```

## Advanced Features Demonstrated

1. **Multi-Model Architecture**: Different models for different durations
2. **Conditional Logic**: Dynamic parameter adjustment based on input
3. **File Processing**: Audio file handling and format conversion
4. **Custom UI Elements**: Textarea fields and importance markers
5. **Comprehensive Validation**: Multiple validation layers with clear error messages
6. **Resource Optimization**: Warmup procedures and efficient model loading

## Use Cases

- **Music Production**: Generate backing tracks from lyrics
- **Content Creation**: Create custom music for videos or podcasts
- **Prototyping**: Quick musical sketches and demos
- **Education**: Teaching music composition and arrangement
- **Game Development**: Dynamic music generation for games

## Key Takeaways

- **Container deployment** enables complex system dependencies
- **Repository cloning** integrates external codebases seamlessly  
- **File management utilities** simplify model and data downloading
- **Custom validation** provides better user experience
- **Temporary directories** handle file processing safely
- **Warmup procedures** improve cold start performance
- **Audio-specific billing** aligns costs with resource usage

This pattern is ideal for sophisticated AI applications that require complex dependencies, external repositories, and custom processing workflows while maintaining the scalability and ease of deployment that fal provides.