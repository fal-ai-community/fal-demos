# Text to Image Example

This example demonstrates how to build a production-ready text-to-image API using the fal SDK with the Sana diffusion models. We'll create two endpoints: a standard endpoint and a faster "sprint" endpoint that shares resources efficiently.

> **Source Code**: Find the complete implementation at [fal-ai-community/fal_demos](https://github.com/fal-ai-community/fal_demos/blob/main/fal_demos/image/sana.py)

## Key Features

- **Multiple endpoints** with shared model components
- **Input validation** using Pydantic models
- **Safety checking** for generated content
- **Flexible image generation** with customizable parameters

## Project Structure

```python
# Only import the python bundled packages, fal and fastapi in the global scope
import math
from typing import Literal

import fal
from fal.toolkit.image import ImageSizeInput, Image, ImageSize, get_image_size
from fal.toolkit.image.safety_checker import postprocess_images
from fastapi import Response
from pydantic import Field, BaseModel
from fal_demos.image.common import Output
```

## Input Models

Define your input schemas with clear documentation and examples:

```python
class BaseInput(BaseModel):
    prompt: str = Field(
        title="Prompt",
        description="The prompt to generate an image from.",
        examples=[
            "Underwater coral reef ecosystem during peak bioluminescent activity, multiple layers of marine life - from microscopic plankton to massive coral structures, light refracting through crystal-clear tropical waters, creating prismatic color gradients, hyper-detailed texture of marine organisms",
        ],
    )
    negative_prompt: str = Field(
        default="",
        description="""
            The negative prompt to use. Use it to address details that you don't want
            in the image. This could be colors, objects, scenery and even the small details
            (e.g. moustache, blurry, low resolution).
        """,
        examples=[""],
    )
    image_size: ImageSizeInput = Field(
        default=ImageSize(width=3840, height=2160),
        description="The size of the generated image.",
    )
    num_inference_steps: int = Field(
        default=18,
        description="The number of inference steps to perform.",
        ge=1,
        le=50,
    )
    seed: int | None = Field(
        default=None,
        description="""
            The same seed and the same prompt given to the same version of the model
            will output the same image every time.
        """,
    )
    guidance_scale: float = Field(
        default=5.0,
        description="""
            The CFG (Classifier Free Guidance) scale is a measure of how close you want
            the model to stick to your prompt when looking for a related image to show you.
        """,
        ge=0.0,
        le=20.0,
        title="Guidance scale (CFG)",
    )
    enable_safety_checker: bool = Field(
        default=True,
        description="If set to true, the safety checker will be enabled.",
    )
    num_images: int = Field(
        default=1,
        description="The number of images to generate.",
        ge=1,
        le=4,
    )
    output_format: Literal["jpeg", "png"] = Field(
        default="jpeg",
        description="The format of the generated image.",
    )

# Endpoint-specific input models
class TextToImageInput(BaseInput):
    pass

class SprintInput(BaseInput):
    # Override settings for the faster endpoint
    num_inference_steps: int = Field(
        default=2,
        description="The number of inference steps to perform.",
        ge=1,
        le=20,
    )
```

## Output Models

Create output models with example data for the playground:

```python
class SanaOutput(Output):
    images: list[Image] = Field(
        description="The generated image files info.",
        examples=[
            [
                Image(
                    url="https://fal.media/files/kangaroo/QAABS8yM6X99WhiMeLcoL.jpeg",
                    width=3840,
                    height=2160,
                    content_type="image/jpeg",
                )
            ],
        ],
    )

class SanaSprintOutput(Output):
    images: list[Image] = Field(
        description="The generated image files info.",
        examples=[
            [
                Image(
                    url="https://v3.fal.media/files/penguin/Q-i_zCk-Xf5EggWA9OmG2_e753bacc9b324050855f9664deda3618.jpg",
                    width=3840,
                    height=2160,
                    content_type="image/jpeg",
                )
            ],
        ],
    )
```

## Main Application Class

```python
class Sana(
    fal.App,
    keep_alive=600,  # Keep worker alive for 10 minutes
    min_concurrency=0,  # Scale to zero when idle
    max_concurrency=10,  # Limit concurrent requests
    name="sana",  # Endpoint served at username/sana
):
    requirements = [
        "torch==2.6.0",
        "accelerate==1.6.0",
        "transformers==4.51.3",
        "git+https://github.com/huggingface/diffusers.git@f4fa3beee7f49b80ce7a58f9c8002f43299175c9",
        "hf_transfer==0.1.9",
        "peft==0.15.0",
        "sentencepiece==0.2.0",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu124",
    ]
    local_python_modules = [
        "fal_demos",
    ]
    machine_type = "GPU-H100"

    def setup(self):
        """Load and cache models for all requests."""
        import torch
        from diffusers import SanaPipeline, SanaSprintPipeline

        self.pipes = {}
        
        # Load base pipeline
        self.pipes["base"] = SanaPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.pipes["base"].text_encoder.to(torch.bfloat16)

        # Load sprint pipeline, reusing text encoder for efficiency
        self.pipes["sprint"] = SanaSprintPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
            text_encoder=self.pipes["base"].text_encoder,  # Reuse component
            torch_dtype=torch.bfloat16,
        ).to("cuda")
```

## Shared Generation Logic

Create a reusable generation method for both endpoints:

```python
    async def _generate(
        self,
        input: TextToImageInput,
        response: Response,
        model_id: str,
    ) -> Output:
        import torch

        # Preprocess input
        image_size = get_image_size(input.image_size)
        seed = input.seed or torch.seed()
        generator = torch.Generator("cuda").manual_seed(seed)

        # Prepare model input
        model_input = {
            "prompt": input.prompt,
            "negative_prompt": input.negative_prompt,
            "num_inference_steps": input.num_inference_steps,
            "guidance_scale": input.guidance_scale,
            "height": image_size.height,
            "width": image_size.width,
            "num_images_per_prompt": input.num_images,
            "generator": generator,
        }
        
        # Handle model-specific differences
        if model_id == "sprint":
            model_input.pop("negative_prompt")  # Not supported in sprint

        # Generate images
        images = self.pipes[model_id](**model_input).images

        # Apply safety checking
        postprocessed_images = postprocess_images(
            images,
            input.enable_safety_checker,
        )

        # Calculate billing
        resolution_factor = math.ceil(
            (image_size.width * image_size.height) / (1024 * 1024)
        )
        response.headers["x-fal-billable-units"] = str(
            resolution_factor * input.num_images
        )

        return Output(
            images=[
                Image.from_pil(image, input.output_format)
                for image in postprocessed_images["images"]
            ],
            seed=seed,
            has_nsfw_concepts=postprocessed_images["has_nsfw_concepts"],
            prompt=input.prompt,
        )
```

## Endpoint Definitions

Define multiple endpoints using the shared generation logic:

```python
    @fal.endpoint("/")
    async def generate(
        self,
        input: TextToImageInput,
        response: Response,
    ) -> SanaOutput:
        return await self._generate(input, response, "base")

    @fal.endpoint("/sprint")
    async def generate_sprint(
        self,
        input: SprintInput,
        response: Response,
    ) -> SanaSprintOutput:
        return await self._generate(input, response, "sprint")
```

## Running the Application

### Development
```bash
fal run fal_demos/image/sana.py::Sana
```

### Using pyproject.toml
Add to your `pyproject.toml`:
```toml
[tool.fal.apps]
sana = "fal_demos.image.sana:Sana"
```

Then run:
```bash
fal run sana
```

## Testing Your Endpoints

Once deployed, your app will be available at URLs like:
- **Base endpoint**: `https://fal.ai/dashboard/sdk/username/app-id`
- **Sprint endpoint**: `https://fal.ai/dashboard/sdk/username/app-id/sprint`

## Best Practices Demonstrated

1. **Resource Sharing**: The text encoder is shared between pipelines to save memory
2. **Input Validation**: Comprehensive Pydantic models with examples and constraints
3. **Error Handling**: Safety checking and proper response formatting
4. **Billing Integration**: Resolution-based pricing
5. **Endpoint Flexibility**: Different configurations for different use cases
6. **Documentation**: Rich field descriptions and examples for auto-generated docs

## Key Takeaways

- Use `setup()` to load models once and cache them
- Share components between models when possible to optimize memory
- Create endpoint-specific input models for different use cases
- Implement proper billing with `x-fal-billable-units`
- Use the fal image toolkit for safety checking and processing
- Pin all dependency versions for reliability

This pattern works well for any multi-model or multi-endpoint image generation service where you want to optimize for both performance and cost efficiency. You can visit [fal_demos](https://github.com/fal-ai-community/fal_demos/blob/main/fal_demos/image/sana.py) to view the full file and other examples.