import os

import fal
from fal.exceptions import FieldException
from fal.toolkit import File
from fastapi import HTTPException
from httpx import HTTPStatusError
from pydantic import BaseModel, Field


class Hunyuan3DInput(BaseModel):
    input_image_url: str = Field(
        examples=[
            "https://storage.googleapis.com/falserverless/model_tests/video_models/robot.png"
        ],
        description="URL of image to use while generating the 3D model.",
    )
    textured_mesh: bool = Field(
        default=False,
        description="If set true, textured mesh will be generated and the price charged would be 3 times that of white mesh.",
        ui={"important": True},
    )
    seed: int | None = Field(
        default=None,
        description="""
            The same seed and the same prompt given to the same version of the model
            will output the same image every time.
        """,
    )
    num_inference_steps: int = Field(
        default=50,
        ge=1,
        le=50,
        description="Number of inference steps to perform.",
    )
    guidance_scale: float = Field(
        default=7.5,
        ge=0.0,
        le=20.0,
        description="Guidance scale for the model.",
    )
    octree_resolution: int = Field(
        default=256,
        ge=1,
        le=1024,
        description="Octree resolution for the model.",
    )


class ObjectOutput(BaseModel):
    model_mesh: File = Field(
        description="Generated 3D object file.",
        examples=[
            File(
                url="https://v3.fal.media/files/lion/WqIhtKPaSoeBtC30qzIGG_white_mesh.glb",
                content_type="application/octet-stream",
                file_name="white_mesh.glb",
                file_size=720696,
            )
        ],
    )
    seed: int = Field(
        description="Seed value used for generation.",
    )


class Hunyuan3D(
    fal.App,
    name="hunyuan3d",
    min_concurrency=0,
    max_concurrency=1,
    max_multiplexing=10,  # Multiplexing allows multiple requests to be handled by the same worker at the same time.
):
    machine_type = "M"
    requirements = ["fal-client"]

    def setup(self):
        # Get the secret key from the environment variable,
        # To set the secret key, run the following command:
        # fal secrets set MY_SECRET_KEY <your_secret_key>
        os.environ["FAL_KEY"] = os.environ.get("MY_SECRET_KEY", "")

    @fal.endpoint("/v2/mini/turbo")
    async def generate_image(self, input: Hunyuan3DInput) -> ObjectOutput:
        import fal_client
        from fal_client.client import Completed, InProgress, Queued

        handle = await fal_client.submit_async(
            "fal-ai/hunyuan3d/v2/mini/turbo",
            arguments={
                "input_image_url": input.input_image_url,
                "num_inference_steps": input.num_inference_steps,
                "guidance_scale": input.guidance_scale,
                "octree_resolution": input.octree_resolution,
                "seed": input.seed,
                "textured_mesh": input.textured_mesh,
            },
        )
        async for event in handle.iter_events(interval=0.1, with_logs=True):
            # Handle events correctly
            if type(event) == Queued:
                print("Queued:", event.position)
            elif type(event) == InProgress:
                if event.logs:
                    print(event.logs[-1]["message"], end="\r")
            elif type(event) == Completed:
                print(
                    f"Request {event.logs[-1]['labels']['fal_request_id']} took:",
                    event.metrics["inference_time"],
                )

        try:
            result = await handle.get()
        except FieldException as e:
            # Make sure to handle the exceptions correctly
            raise e
        except HTTPStatusError as e:
            print("HTTP status error:", e.response.status_code, e.response.text)
            raise e
        except fal_client.client.FalClientError as e:
            print("FalClient error:", e.args[0])
            e = e.args[0]
            if e["type"] == "UserError":
                raise HTTPException(
                    status_code=400,
                    detail=e["message"],
                )
            raise HTTPException(
                status_code=500,
                detail=e["message"],
            )

        return ObjectOutput(**result)
