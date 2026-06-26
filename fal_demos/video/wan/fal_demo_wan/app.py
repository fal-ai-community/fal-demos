import os
import random
import tempfile
from typing import Literal

import fal
from fal.exceptions import FieldException
from fal.toolkit import FAL_MODEL_WEIGHTS_DIR, File, clone_repository
from pydantic import BaseModel, Field


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
        examples=[
            "",
        ],
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


class Wan(fal.App):
    def setup(self):
        """Initialize the app by cloning the repository, downloading models, and starting the server."""
        from pathlib import Path

        from huggingface_hub import snapshot_download

        # Clone the Wan Repository, make sure to pin the commit hash to avoid breaking changes
        # using a temporary directory ensures that there are fewer clashes with other workers while starting up
        target_dir = clone_repository(
            "https://github.com/Wan-Video/Wan2.1.git",
            commit_hash="6797c48002e977f2bc98ec4da1930f4cd46181a6",
            target_dir="/tmp",
            include_to_path=True,
            repo_name="wan-t2v",
        )
        os.chdir(target_dir)

        # Download model weights with HF transfer enabled
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        checkpoint_dir = Path(
            snapshot_download(
                "Wan-AI/Wan2.1-T2V-1.3B",
                local_dir=FAL_MODEL_WEIGHTS_DIR / "wan_t2v_1.3B_480p",
                local_dir_use_symlinks=True,
                local_files_only=False,
            )
        )

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

        self.RESOLUTIONS = {
            "480p": {
                "16:9": (832, 480),
                "9:16": (480, 832),
            }
        }

    # Custom NSFW checkers with any-llm
    def _is_nsfw_prompt(self, prompt: str) -> bool:
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
            return True
        else:
            return "yes" in response["output"].lower()

    # Better nsfw check to ensure tighter safety
    def _is_nsfw_request(self, image: str) -> bool:
        import fal_client

        try:
            image_url = fal_client.upload_image(image)
            response = fal_client.subscribe(
                "fal-ai/imageutils/nsfw", {"image_url": image_url}
            )
        except Exception:
            return True
        else:
            if response["nsfw_probability"] >= 0.3:
                return True

        # Secondary check
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

    # Prompt expansion using any-llm
    def _expand_prompt(self, prompt: str) -> str:
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
            prompt = prompt
        else:
            prompt = response["output"] or prompt
        return prompt

    # Specify the endpoint path to ensure that it is clearly distinguishable from the other variants of the model
    @fal.endpoint("/v2.1/1.3b/text-to-video")
    def generate_image_to_video(self, request: WanT2VRequest) -> WanT2VResponse:
        """WAN 1.3B model for fast text-to-video generation."""

        if request.enable_safety_checker and self._is_nsfw_prompt(request.prompt):
            raise FieldException("prompt", "NSFW content detected in the prompt.")

        if request.enable_prompt_expansion:
            prompt = self._expand_prompt(request.prompt)
        else:
            prompt = request.prompt

        seed = request.seed or random.randint(0, 1000000)
        size = self.RESOLUTIONS["480p"][request.aspect_ratio]
        from wan.utils.utils import cache_video

        video = self.wan_t2v.generate(
            input_prompt=prompt,
            size=size,
            frame_num=81,
            n_prompt=request.negative_prompt,
            shift=request.shift,
            sample_solver=request.sampler,
            sampling_steps=request.inference_steps,
            guide_scale=request.guidance_scale,
            seed=seed,
            offload_model=False,
        )
        # Save the video to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            save_path = f.name
            cache_video(
                tensor=video[None],
                save_file=save_path,
                fps=self.cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )

            return WanT2VResponse(video=File.from_path(save_path), seed=seed)
