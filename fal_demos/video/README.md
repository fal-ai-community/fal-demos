# Deploy a Text-to-Video Model


This example demonstrates how to build a sophisticated text-to-video generation service using Wan2.1, showcasing advanced techniques including custom safety checking with multiple AI models, prompt expansion using LLMs, video tensor processing, and comprehensive integration of multiple fal services within a single application.

> **Source Code**: Find the complete implementation at [fal-ai-community/fal_demos](https://github.com/fal-ai-community/fal_demos/blob/main/fal_demos/video/wan.py)

## Key Features

- **Advanced Video Generation**: High-quality text-to-video with Wan2.1 model
- **Multi-Model Safety Checking**: Text and image NSFW detection using multiple AI models
- **AI-Powered Prompt Expansion**: Enhance prompts using LLM services
- **Model Weight Management**: Efficient downloading and caching of large model weights
- **Service Integration**: Seamless integration of multiple fal AI services
- **Advanced Dependencies**: Complex wheel installations and GPU optimizations

## When to Use Service Integration

Service integration within fal apps is powerful for:
- **Multi-stage AI pipelines**: Combining multiple AI models in sequence
- **Safety and moderation**: Adding content filtering with specialized models
- **Content enhancement**: Using LLMs to improve or expand inputs
- **Quality assurance**: Multi-model validation and checking

## Project Setup

```python
import os
import random
import tempfile
from typing import Literal

import fal
from fal.exceptions import FieldException
from fal.toolkit import FAL_MODEL_WEIGHTS_DIR, File, clone_repository
from pydantic import BaseModel, Field
```

## Comprehensive Input Model

Define detailed input schemas with professional validation:

```python
class WanT2VRequest(BaseModel):
    prompt: str = Field(
        description="The text prompt to guide video generation.",
        examples=[
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
        ],
    )
    negative_prompt: str = Field(
        default="",
        description="""
            The negative prompt to use. Use it to address details that you don't want
            in the image. This could be colors, objects, scenery and even the small
            details (e.g. moustache, blurry, low resolution).
        """,
        examples=[""],
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility. If None, a random seed is chosen.",
    )
    aspect_ratio: Literal["9:16", "16:9"] = Field(
        default="16:9",
        description="Aspect ratio of the generated video (16:9 or 9:16).",
    )
    num_inference_steps: int = Field(
        default=30,
        description="Number of inference steps for sampling. Higher values give better quality but take longer.",
        ge=2,
        le=40,
    )
    inference_steps: int = Field(
        default=30,
        description="Number of inference steps for sampling. Higher values give better quality but take longer.",
        ge=2,
        le=40,
    )
    guidance_scale: float = Field(
        default=5.0,
        description="Classifier-free guidance scale. Controls prompt adherence vs. creativity",
        ge=0.0,
        le=20.0,
        title="Guidance scale (CFG)",
    )
    shift: float = Field(
        default=5.0,
        description="Noise schedule shift parameter. Affects temporal dynamics.",
        ge=0.0,
        le=10.0,
    )
    sampler: Literal["unipc", "dpm++"] = Field(
        default="unipc",
        description="The sampler to use for generation.",
    )
    enable_safety_checker: bool = Field(
        default=False,
        examples=[True],
        description="If set to true, the safety checker will be enabled.",
    )
    enable_prompt_expansion: bool = Field(
        default=False,
        examples=[False],
        description="Whether to enable prompt expansion.",
    )

class WanT2VResponse(BaseModel):
    video: File = Field(
        description="The generated video file.",
        examples=[
            File._from_url(
                "https://v3.fal.media/files/monkey/kqYTedbmW3W58-J_fsTIo_tmpbb3f9orp.mp4"
            )
        ],
    )
    seed: int = Field(description="The seed used for generation.")
```

## Application Configuration with Advanced Dependencies

```python
class Wan(
    fal.App,
    name="wan",
    min_concurrency=0,
    max_concurrency=1,
    keep_alive=300,
):
    machine_type = "GPU-H100"  # High-end GPU for video generation
    requirements = [
        "torch==2.6.0",
        "torchvision==0.21.0",
        "opencv-python>=4.9.0.80",
        "diffusers==0.32.2",
        "transformers==4.49.0",
        "tokenizers==0.21.0",
        "accelerate==1.4.0",
        "tqdm",
        "imageio",
        "easydict",
        "ftfy",
        "dashscope",
        "imageio-ffmpeg",
        # Custom wheel installation for optimized performance
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post1/flash_attn-2.7.0.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
        "numpy==1.24.4",
        "xfuser==0.4.1",
        "fal-client",  # For service integration
    ]
```

## Advanced Model Setup

The setup method demonstrates complex initialization workflows:

```python
def setup(self):
    """Initialize the app by cloning the repository, downloading models, and starting the server."""
    from pathlib import Path
    from huggingface_hub import snapshot_download

    # Clone repository with commit pinning
    target_dir = clone_repository(
        "https://github.com/Wan-Video/Wan2.1.git",
        commit_hash="6797c48002e977f2bc98ec4da1930f4cd46181a6",
        target_dir="/tmp",  # Use temp directory to avoid conflicts
        include_to_path=True,
        repo_name="wan-t2v",
    )
    os.chdir(target_dir)

    # Enable HuggingFace transfer optimization
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    # Download model weights to managed directory
    checkpoint_dir = Path(
        snapshot_download(
            "Wan-AI/Wan2.1-T2V-1.3B",
            local_dir=FAL_MODEL_WEIGHTS_DIR / "wan_t2v_1.3B_480p",
            local_dir_use_symlinks=True,  # Efficient storage
            local_files_only=False,
        )
    )

    # Initialize the model
    import wan
    from wan.configs import WAN_CONFIGS

    self.cfg = WAN_CONFIGS["t2v-1.3B"]
    self.wan_t2v = wan.WanT2V(
        config=self.cfg,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )

    # Define supported resolutions
    self.RESOLUTIONS = {
        "480p": {
            "16:9": (832, 480),
            "9:16": (480, 832),
        }
    }
```

## Custom Safety Checking with AI Services

Implement multi-layered safety checking using different AI models:

```python
def _is_nsfw_prompt(self, prompt: str) -> bool:
    """Check if text prompt contains NSFW content using LLM."""
    import fal_client

    try:
        response = fal_client.subscribe(
            "fal-ai/any-llm",
            {
                "prompt": prompt,
                "system_prompt": 'With just a single word of "yes" or "no" tell me the given user prompt contains any not safe for work material. Don\'t say anything other than yes or no. If the prompt contains unsafe material, say yes, otherwise say no.',
                "model": "google/gemini-flash-1.5",
            },
        )
    except Exception:
        return True  # Err on the side of caution
    else:
        return "yes" in response["output"].lower()

def _is_nsfw_request(self, image: str) -> bool:
    """Multi-stage image NSFW detection."""
    import fal_client

    try:
        # Upload image for analysis
        image_url = fal_client.upload_image(image)
        
        # Primary check with specialized NSFW detector
        response = fal_client.subscribe(
            "fal-ai/imageutils/nsfw", 
            {"image_url": image_url}
        )
    except Exception:
        return True
    else:
        if response["nsfw_probability"] >= 0.3:
            return True

    # Secondary check with vision LLM
    try:
        response = fal_client.subscribe(
            "fal-ai/any-llm/vision",
            {
                "prompt": 'With just a single word of "yes" or "no" tell me the given user image contains any not safe for work material. Don\'t say anything other than yes or no. If the image contains unsafe material, say yes, otherwise say no.',
                "image_url": image_url,
                "model": "google/gemini-flash-1.5",
            },
        )
    except Exception:
        return True
    else:
        return "yes" in response["output"]
```

## AI-Powered Prompt Expansion

Enhance user prompts using LLM services:

```python
def _expand_prompt(self, prompt: str) -> str:
    """Expand and enhance prompts using LLM for better video generation."""
    import fal_client

    try:
        response = fal_client.run(
            "fal-ai/any-llm",
            arguments={
                "model": "google/gemini-flash-1.5",
                "prompt": prompt,
                "system_prompt": 'Take the user\'s input prompt and enhance it by focusing on a single, central subject to create a coherent and impactful result. Structure the description from coarse to fine details, starting with broad elements and zooming into specific textures, colors, and finer characteristics. If relevant, include close-up and medium shots to describe both intimate details and the surrounding context, ensuring clarity in depth and framing. Append "high resolution 4k" at the end to reduce warping and ensure the highest level of detail. Specify vivid colors, rich textures, and appropriate moods to enhance the overall visual quality and cohesion of the output. Maximum output 200 tokens.',
            },
            timeout=10,
        )
    except Exception:
        import traceback
        traceback.print_exc()
        return prompt  # Return original if expansion fails
    else:
        return response["output"] or prompt
```

## Professional API Endpoint with Versioning

```python
@fal.endpoint("/v2.1/1.3b/text-to-video")
def generate_image_to_video(self, request: WanT2VRequest) -> WanT2VResponse:
    """WAN 1.3B model for fast text-to-video generation."""

    # Safety checking
    if request.enable_safety_checker and self._is_nsfw_prompt(request.prompt):
        raise FieldException("prompt", "NSFW content detected in the prompt.")

    # Prompt enhancement
    if request.enable_prompt_expansion:
        prompt = self._expand_prompt(request.prompt)
    else:
        prompt = request.prompt

    # Generate deterministic or random seed
    seed = request.seed or random.randint(0, 1000000)
    size = self.RESOLUTIONS["480p"][request.aspect_ratio]
    
    # Import video utilities
    from wan.utils.utils import cache_video

    # Generate video
    video = self.wan_t2v.generate(
        input_prompt=prompt,
        size=size,
        frame_num=81,  # ~2.7 seconds at 30fps
        n_prompt=request.negative_prompt,
        shift=request.shift,
        sample_solver=request.sampler,
        sampling_steps=request.inference_steps,
        guide_scale=request.guidance_scale,
        seed=seed,
        offload_model=False,
    )

    # Save video to temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        save_path = f.name
        cache_video(
            tensor=video[None],  # Add batch dimension
            save_file=save_path,
            fps=self.cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

        return WanT2VResponse(
            video=File.from_path(save_path), 
            seed=seed
        )
```

## Key Concepts and Best Practices

### Service Integration Patterns

**Text Analysis with LLMs:**
```python
response = fal_client.subscribe(
    "fal-ai/any-llm",
    {
        "prompt": user_prompt,
        "system_prompt": "Your analysis instructions...",
        "model": "google/gemini-flash-1.5",
    },
)
```

**Image Analysis:**
```python
image_url = fal_client.upload_image(image)
response = fal_client.subscribe(
    "fal-ai/imageutils/nsfw", 
    {"image_url": image_url}
)
```

**Vision-Language Models:**
```python
response = fal_client.subscribe(
    "fal-ai/any-llm/vision",
    {
        "prompt": "Analysis prompt...",
        "image_url": image_url,
        "model": "google/gemini-flash-1.5",
    },
)
```

### Model Weight Management

```python
# Use FAL_MODEL_WEIGHTS_DIR for managed storage
checkpoint_dir = Path(
    snapshot_download(
        "Wan-AI/Wan2.1-T2V-1.3B",
        local_dir=FAL_MODEL_WEIGHTS_DIR / "wan_t2v_1.3B_480p",
        local_dir_use_symlinks=True,  # Efficient storage
        local_files_only=False,
    )
)
```

### Video Processing Pipeline

```python
# Generate video tensor
video = model.generate(...)

# Convert tensor to video file
with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
    cache_video(
        tensor=video[None],  # Add batch dimension
        save_file=f.name,
        fps=cfg.sample_fps,
        normalize=True,
        value_range=(-1, 1),
    )
    return Response(video=File.from_path(f.name))
```

### Error Handling in Service Integration

```python
try:
    response = fal_client.subscribe("service-name", arguments)
except Exception:
    # Handle gracefully - don't break the main workflow
    return default_behavior()
else:
    return process_response(response)
```

### Advanced Dependencies

```python
requirements = [
    # Custom wheel for optimized performance
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post1/flash_attn-2.7.0.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
    # Other dependencies...
]
```


## Deployment and Usage

### Running the Service
```bash
# Development
fal run fal_demos/video/wan.py::Wan

# Production deployment
fal deploy wan
```

### Making Requests
```python
import fal_client

result = await fal_client.submit_async(
    "your-username/wan/v2.1/1.3b/text-to-video",
    arguments={
        "prompt": "A cat walking through a futuristic city",
        "aspect_ratio": "16:9",
        "num_inference_steps": 30,
        "enable_safety_checker": True,
        "enable_prompt_expansion": True,
        "seed": 42
    }
)

# The response includes the video file and seed
video_url = result["video"]["url"]  # Ready to embed in web players
seed_used = result["seed"]
```

## Advanced Features Demonstrated

1. **Multi-Model Workflows**: Combining text analysis, image analysis, and video generation
2. **Content Moderation**: Multi-layered safety checking with different AI models
3. **Dynamic Prompt Enhancement**: AI-powered prompt expansion for better results
4. **Efficient Resource Management**: Model weight caching and memory optimization
5. **Service Orchestration**: Seamless integration of multiple fal services


## Performance Optimizations

### Memory Management
```python
offload_model=False  # Keep model in VRAM for better performance
use_usp=False       # Optimize for single-user scenarios
```

### Storage Efficiency
```python
local_dir_use_symlinks=True  # Avoid duplicate model weights
FAL_MODEL_WEIGHTS_DIR       # Use managed storage location
```


## Key Takeaways

- **Service integration** enables sophisticated AI workflows within single applications
- **Multi-model safety checking** provides robust content moderation
- **AI-powered enhancement** can significantly improve user inputs
- **Resource management** is crucial for large model deployments
- **Error handling** in service integration must be graceful and not break main workflows

This pattern demonstrates how to build enterprise-grade AI services that combine multiple AI models, advanced safety features, and professional API design while maintaining the scalability and ease of deployment that fal provides. It represents the pinnacle of what's possible when combining all the techniques we've learned across the previous examples.