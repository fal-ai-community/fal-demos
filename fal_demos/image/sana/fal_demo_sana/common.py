from fal.toolkit import Image
from pydantic import BaseModel, Field


# Base Output Model, it can be reused for image endpoints
class Output(BaseModel):
    images: list[Image] = Field(description="The generated image files info.")
    seed: int = Field(
        description="""
            Seed of the generated Image. It will be the same value of the one passed in the
            input or the randomly generated that was used in case none was passed.
        """
    )
    has_nsfw_concepts: list[bool] = Field(
        description="Whether the generated images contain NSFW concepts."
    )
    prompt: str = Field(
        description="The prompt used for generating the image.",
    )
