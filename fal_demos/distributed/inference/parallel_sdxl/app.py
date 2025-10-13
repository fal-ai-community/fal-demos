from math import floor, sqrt
from typing import TYPE_CHECKING, Any

import fal
from fal.distributed import DistributedRunner, DistributedWorker
from fal.toolkit import File, Image
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import torch
    from PIL import Image as PILImage

# Helper function


def tensors_to_image_grid(
    tensors: list["torch.Tensor"], blur_radius: int = 0
) -> "PILImage.Image":
    """
    Convert a list of tensors to a grid image.
    """
    import torchvision  # type: ignore[import-untyped]
    from PIL import Image as PILImage
    from PIL import ImageFilter

    # Create a grid of images
    image = (
        torchvision.utils.make_grid(
            tensors,
            nrow=floor(sqrt(len(tensors))),
            normalize=True,
            scale_each=True,
        )
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    image = (image * 255).astype("uint8")
    pil_image = PILImage.fromarray(image)

    if blur_radius > 0:
        pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return pil_image


# Distributed app code


class ExampleDistributedWorker(DistributedWorker):
    """
    This is a distributed worker that runs on multiple GPUs.

    It will run the Stable Diffusion XL model using random seeds on each GPU,
    then return the images as a grid to the main process.
    """

    def setup(self, **kwargs: Any) -> None:
        """
        On setup, we need to initialize the model.
        """
        import torch
        from diffusers import AutoencoderTiny, StableDiffusionXLPipeline

        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            low_cpu_mem_usage=False,
        ).to(self.device, dtype=torch.float16)
        self.tiny_vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesdxl",
            torch_dtype=torch.float16,
        ).to(self.device, dtype=torch.float16)
        if self.rank != 0:
            self.pipeline.set_progress_bar_config(disable=True)

    def pipeline_callback(
        self,
        pipeline: "torch.nn.Module",
        step: int,
        timestep: int,
        tensors: dict[str, "torch.Tensor"],
    ) -> dict[str, "torch.Tensor"]:
        """
        This callback is called after each step of the pipeline.
        """
        if step > 0 and step % 5 != 0:
            return tensors

        import torch
        import torch.distributed as dist

        latents = tensors["latents"]
        image = self.tiny_vae.decode(
            latents / self.tiny_vae.config.scaling_factor, return_dict=False
        )[0]
        image = self.pipeline.image_processor.postprocess(image, output_type="pt")[0]

        if self.rank == 0:
            gather_list = [
                torch.zeros_like(image, device=self.device)
                for _ in range(self.world_size)
            ]
        else:
            gather_list = None

        dist.gather(image, gather_list, dst=0)

        if gather_list:
            remaining = timestep / 1000
            image = tensors_to_image_grid(gather_list, blur_radius=int(remaining * 10))
            self.add_streaming_result({"image": image}, as_text_event=True)

        dist.barrier()
        return tensors

    def __call__(
        self,
        streaming: bool = False,
        width: int = 1024,
        height: int = 1024,
        prompt: str = "A fantasy landscape",
        negative_prompt: str = "A blurry image",
        num_inference_steps: int = 20,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run the model on the worker and return the image.
        """
        import torch
        import torch.distributed as dist

        image = self.pipeline(
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            output_type="pt",
            callback_on_step_end=self.pipeline_callback if streaming else None,
        ).images[0]

        if self.rank == 0:
            gather_list = [
                torch.zeros_like(image, device=self.device)
                for _ in range(self.world_size)
            ]
        else:
            gather_list = None

        # Gather the images from all workers
        # this will block until all workers are done
        dist.gather(image, gather_list, dst=0)

        # Clean memory on all workers
        torch.cuda.empty_cache()

        if not gather_list:
            # If we are not the main worker, we don't need to do anything
            return {}

        # The main worker will receive the images from all workers
        image = tensors_to_image_grid(gather_list)
        return {"image": image}


# Fal app code


class ExampleRequest(BaseModel):
    """
    This is the request model for the example app.

    There is only one required field, the prompt.
    """

    prompt: str = Field()
    negative_prompt: str = Field(default="blurry, low quality")
    num_inference_steps: int = Field(default=20)
    width: int = Field(default=1024)
    height: int = Field(default=1024)


class ExampleResponse(BaseModel):
    """
    This is the response model for the example app.

    The response contains the image as a file.
    """

    image: File = Field()


class ExampleDistributedApp(fal.App):
    machine_type = "GPU-H100"
    num_gpus = 2
    requirements = [
        "accelerate==1.4.0",
        "diffusers==0.30.3",
        "fal",
        "huggingface_hub==0.26.5",
        "opencv-python",
        "torch==2.6.0+cu124",
        "torchvision==0.21.0+cu124",
        "transformers==4.47.1",
        "pyzmq==26.0.0",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu124",
    ]

    async def setup(self) -> None:
        """
        On setup, create a distributed runner to run the model on multiple GPUs.
        """
        self.runner = DistributedRunner(
            worker_cls=ExampleDistributedWorker,
            world_size=self.num_gpus,
        )
        # Start and wait for ready
        await self.runner.start()
        # Warm-up
        warmup_result = await self.runner.invoke(
            ExampleRequest(prompt="a cat wearing a hat").dict()
        )
        assert (
            "image" in warmup_result
        ), "Warm-up failed, no image returned from the worker"

    @fal.endpoint("/")
    async def run(self, request: ExampleRequest, response: Response) -> ExampleResponse:
        """
        Run the model on the worker and return the image.
        """
        result = await self.runner.invoke(request.dict())
        assert "image" in result, "No image returned from the worker"
        return ExampleResponse(image=Image.from_pil(result["image"]))

    @fal.endpoint("/stream")
    async def stream(
        self, request: ExampleRequest, response: Response
    ) -> StreamingResponse:
        """
        Run the model on the worker and return the image as a stream.
        This will return a streaming response that reads the data the worker adds
        via `add_streaming_result`. Images are automatically encoded as data URIs.
        """
        return StreamingResponse(
            self.runner.stream(
                request.dict(),
                as_text_events=True,
            ),
            media_type="text/event-stream",
        )


if __name__ == "__main__":
    app = fal.wrap_app(ExampleDistributedApp)
    app()
