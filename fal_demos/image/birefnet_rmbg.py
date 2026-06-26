from typing import Literal

import fal
from fal.toolkit import Image
from pydantic import BaseModel, Field


class BackgroundRemovalInput(BaseModel):
    image_url: str = Field(
        title="Image URL",
        description="URL of the image to remove the background from.",
        examples=[
            "https://storage.googleapis.com/falserverless/model_tests/video_models/robot.png",
        ],
    )
    return_mask: bool = Field(
        default=False,
        description="If true, also return the grayscale alpha mask.",
    )
    output_format: Literal["png"] = Field(
        default="png",
        description="The output format. PNG is used to preserve transparency.",
    )


class BackgroundRemovalOutput(BaseModel):
    image: Image = Field(
        description="The foreground image with a transparent background.",
        examples=[
            Image(
                url="https://v3.fal.media/files/kangaroo/QAABS8yM6X99WhiMeLcoL.jpeg",
                width=1024,
                height=1024,
                content_type="image/png",
            )
        ],
    )
    mask: Image | None = Field(
        default=None,
        description="The optional grayscale alpha mask.",
    )


class BiRefNetRMBG(fal.App):
    def setup(self):
        import os

        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        torch.set_float32_matmul_precision("high")

        self.model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.model.to("cuda")
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _download_image(self, image_url: str):
        from io import BytesIO
        from urllib.request import urlopen

        from PIL import Image as PILImage

        with urlopen(image_url, timeout=30) as response:
            return PILImage.open(BytesIO(response.read())).convert("RGB")

    def _predict_mask(self, image):
        import torch
        from torchvision import transforms

        model_input = self.transform(image).unsqueeze(0).to("cuda", dtype=torch.float16)

        with torch.inference_mode():
            prediction = self.model(model_input)[-1].sigmoid().cpu()

        mask = prediction[0].squeeze()
        return transforms.ToPILImage()(mask).resize(image.size)

    @fal.endpoint("/")
    async def remove_background(
        self,
        input: BackgroundRemovalInput,
    ) -> BackgroundRemovalOutput:
        from PIL import Image as PILImage

        image = self._download_image(input.image_url)
        mask = self._predict_mask(image)

        output = PILImage.new("RGBA", image.size, (0, 0, 0, 0))
        output.paste(image.convert("RGBA"), mask=mask)

        return BackgroundRemovalOutput(
            image=Image.from_pil(output, input.output_format),
            mask=Image.from_pil(mask, input.output_format)
            if input.return_mask
            else None,
        )
