"""
Flux Preprocessor App - Standalone Multi-GPU Preprocessing

This app preprocesses raw images for Flux LoRA training:
- Multi-GPU parallel processing using DistributedRunner
- Auto-captioning with moondream
- Trigger word injection
- VAE encoding (images → latents)
- T5/CLIP encoding (captions → embeddings)

Deployed separately from training app for clean separation and independent scaling.
"""

import fal
from fal.distributed import DistributedRunner
from fal.toolkit import File, download_file
from fastapi import Response
from pydantic import BaseModel, Field

from fal_demos.distributed.training.flux_lora.preprocessor.preprocessor_worker import (
    FluxPreprocessorWorker,
)


class PreprocessRequest(BaseModel):
    """Request model for preprocessing"""
    
    images_data_url: str = Field(
        description="URL to ZIP file containing training images"
    )
    trigger_word: str = Field(
        default="ohwx",
        description="Trigger word to inject into captions (e.g. 'ohwx', 'txcl')"
    )
    is_style: bool = Field(
        default=False,
        description="Reserved for future use (face detection not yet implemented)"
    )
    resolution: int = Field(
        default=512,
        description="Training resolution",
        ge=256,
        le=1024,
    )


class PreprocessResponse(BaseModel):
    """Response model for preprocessing"""
    
    preprocessed_data: File = Field(
        description="Preprocessed training data (.pt file)"
    )
    num_images: int = Field(
        description="Number of images processed"
    )
    message: str = Field(
        description="Status message"
    )


class FluxPreprocessorApp(fal.App):
    """
    Standalone Flux preprocessing app.
    
    Uses all GPUs in parallel to preprocess images:
    - Each GPU processes a subset of images
    - Results are gathered and saved
    - Returns URL to preprocessed data
    """
    
    machine_type = "GPU-H100"
    num_gpus = 2
    keep_alive = 300
    min_concurrency = 0
    max_concurrency = 2
    
    requirements = [
        "torch==2.4.0",
        "torchvision",  # Required by moondream
        "diffusers==0.30.3",
        "transformers==4.46.0",
        "tokenizers==0.20.1",
        "sentencepiece",
        "accelerate==1.4.0",
        "pyzmq==26.0.0",
        "huggingface_hub==0.26.5",
        "moondream==0.0.5",
        "einops",  # Required by moondream
        "pillow>=10.0.0",
        "timm",  # Required by moondream vision encoder
    ]
    
    async def setup(self) -> None:
        """
        Initialize the preprocessing runner.
        
        Downloads Flux weights and starts 8 GPU workers for preprocessing.
        """
        import os
        from huggingface_hub import snapshot_download
        
        os.environ["HF_HOME"] = "/data/models"
        
        # Download Flux weights
        print("Downloading Flux model weights...")
        model_path = snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev",
            local_dir="/data/flux_weights",
        )
        print(f"Model downloaded to {model_path}")
        
        # Create preprocessing runner
        self.runner = DistributedRunner(
            worker_cls=FluxPreprocessorWorker,
            world_size=self.num_gpus,
        )
        
        # Start workers
        print(f"Starting {self.num_gpus} preprocessing workers...")
        await self.runner.start(model_path=model_path)
        
        print("Preprocessing workers ready!")
    
    @fal.endpoint("/")
    async def preprocess(
        self,
        request: PreprocessRequest,
        response: Response,
    ) -> PreprocessResponse:
        """
        Preprocess images for Flux LoRA training.
        
        This endpoint:
        1. Downloads ZIP of images
        2. Generates/loads captions
        3. Injects trigger word
        4. Encodes images with VAE (parallel across GPUs)
        5. Encodes captions with T5/CLIP (parallel across GPUs)
        6. Returns preprocessed data file
        """
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download images ZIP
            print(f"Downloading images from {request.images_data_url}")
            images_zip_path = str(download_file(
                request.images_data_url,
                target_dir=temp_dir
            ))
            
            # Generate unique request ID for this preprocessing job
            import time
            request_id = str(int(time.time() * 1000000))  # Microsecond timestamp
            
            # Preprocess in parallel across all GPUs
            print(f"Preprocessing with {self.num_gpus} GPUs (request_id: {request_id})...")
            result = await self.runner.invoke({
                "images_zip_url": images_zip_path,
                "request_id": request_id,
                "trigger_word": request.trigger_word,
                "resolution": request.resolution,
            })
            
            # Upload preprocessed data
            preprocessed_file = File.from_path(result["preprocessed_data_path"])
            num_images = result["num_images"]
            
            print(f"Preprocessing complete! Processed {num_images} images")
            
            return PreprocessResponse(
                preprocessed_data=preprocessed_file,
                num_images=num_images,
                message=f"Preprocessed {num_images} images with trigger word '{request.trigger_word}'"
            )


if __name__ == "__main__":
    app = fal.wrap_app(FluxPreprocessorApp)
    app()

