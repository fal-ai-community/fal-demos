# Only import the python bundled packages, fal and fastapi in the global scope, app specific imports must be inside a function
import math
from typing import Literal

import fal
from fal.toolkit.image import ImageSizeInput, Image, ImageSize, get_image_size
from fal.toolkit.image.safety_checker import postprocess_images
from fastapi import Response
from pydantic import Field, BaseModel
from fal_demos.image.common import Output


# The input model for the inference request, make sure to set the title and description for each field
# and set the examples for the fields that are not optional
class BaseInput(BaseModel):
    prompt: str = Field(
        title="Prompt",
        description="The prompt to generate an image from.",
        # Set the example to show it on the playground
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
        examples=[
            "",
        ],
    )
    # Use the ImageSizeInput to set the image size, it will be converted to ImageSize
    image_size: ImageSizeInput = Field(
        default=ImageSize(width=3840, height=2160),
        description="The size of the generated image.",
    )
    num_inference_steps: int = Field(
        default=18,
        description="The number of inference steps to perform.",
        # set the least and max values whenever possible to limit the input values
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


# For the base endpoint
class TextToImageInput(BaseInput):
    pass


# For the sprint endpoint, we can reuse the base input model and override the fields that we want to change
class SprintInput(BaseInput):
    num_inference_steps: int = Field(
        default=2,
        description="The number of inference steps to perform.",
        ge=1,
        le=20,
    )


class SanaOutput(Output):
    images: list[Image] = Field(
        description="The generated image files info.",
        # Set default examples to show a generated image when the user visits the playground
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
        # Set default examples to show a generated image when the user visits the playground
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


class Sana(
    fal.App,
    keep_alive=600,  # The worker will be kept alive for 10 minutes after the last request
    min_concurrency=0,  # The minimum number of concurrent workers to keep alive, if set to 0, the app will startup when the first request is received
    max_concurrency=10,  # The maximum number of concurrent workers to acquire, it helps limit the number of concurrent requests to the app
    name="sana",  # set the app name, the endpoint will be served at username/sana
):
    """
    Specify requirements as follows and make sure to pin the versions of packages and commit hashes to ensure reliability.
    """

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
    machine_type = "GPU-H100"  # Choose machine type from https://docs.fal.ai/private-serverless-models/resources/

    def setup(self):
        """
        This method is called once when the app is started. Use it to load your model and cache it for all requests.
        """
        # Import the libraries inside the setup method since these are installed in the worker environment as set in the requirements
        import torch
        from diffusers import SanaPipeline, SanaSprintPipeline

        self.pipes = {}
        self.pipes["base"] = SanaPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.pipes["base"].text_encoder.to(torch.bfloat16)

        self.pipes["sprint"] = SanaSprintPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
            text_encoder=self.pipes[
                "base"
            ].text_encoder,  # Reuse the text encoder from the base pipeline
            torch_dtype=torch.bfloat16,
        ).to("cuda")

    async def _generate(
        self,
        input: TextToImageInput,
        response: Response,
        model_id: str,
    ) -> Output:
        """
        Reuse the Inference code for both endpoints. Both the base and sprint pipelines have very similar inference code.
        """
        import torch

        # Preprocess the input
        image_size = get_image_size(
            input.image_size,
        )

        seed = input.seed or torch.seed()
        generator = torch.Generator("cuda").manual_seed(seed)

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
        if model_id == "sprint":
            # Negative prompt is not supported in the sprint pipeline
            model_input.pop("negative_prompt")

        # Generate the images
        images = self.pipes[model_id](**model_input).images

        # Perform the safety check
        postprocessed_images = postprocess_images(
            images,
            input.enable_safety_checker,
        )

        # Pricing
        resolution_factor = math.ceil(
            (image_size.width * image_size.height) / (1024 * 1024)
        )
        # The number of billable units is the resolution factor multiplied by the number of images
        response.headers["x-fal-billable-units"] = str(
            resolution_factor * input.num_images
        )
        # The cost is set in the billing dashboard which is calculated as the number of billable units multiplied by the cost per unit

        return Output(
            images=[
                Image.from_pil(image, input.output_format)
                for image in postprocessed_images["images"]
            ],
            seed=seed,
            has_nsfw_concepts=postprocessed_images["has_nsfw_concepts"],
            prompt=input.prompt,
        )

    @fal.endpoint("/")
    async def generate(
        self,
        input: TextToImageInput,  # This will be used to autgenerate the OpenAPI spec and the playground form
        response: Response,  # This is the response object that will be used to set the headers for setting the billing units
    ) -> (
        SanaOutput
    ):  # This is the output object that will be used to autgenerate the OpenAPI spec
        return await self._generate(input, response, "base")

    @fal.endpoint("/sprint")
    async def generate_sprint(
        self,
        input: SprintInput,  # Use a different input class for the sprint endpoint to change example values and remove the negative prompt
        response: Response,
    ) -> SanaSprintOutput:
        return await self._generate(input, response, "sprint")


# Run the app with fal run fal_demos/image/sana.py::Sana
# or fal run sana (needs to be defined in the pyproject.toml inside the tool.fal.apps section)

# The app will be served on an ephemeral URL, example: https://fal.ai/dashboard/sdk/fal-ai/9fe9b6fc-534d-4926-95b1-87b7f15a67de
# Visit https://fal.ai/dashboard/sdk/fal-ai/9fe9b6fc-534d-4926-95b1-87b7f15a67de to test the root endpoint
# To test the sprint endpoint, visit https://fal.ai/dashboard/sdk/fal-ai/9fe9b6fc-534d-4926-95b1-87b7f15a67de/sprint
