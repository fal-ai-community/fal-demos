"""
Flux LoRA Training App - Calls Separate Preprocessor

Architecture:
- This app handles TRAINING only (8 GPUs with DDP)
- Calls flux-preprocessor-demo app for preprocessing (runs on separate 8 GPUs)
- Both apps stay warm, no reload overhead
- Clean separation of concerns
"""

import fal
from typing import ClassVar

from fal.distributed import DistributedRunner
from fal.toolkit import File, download_file
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from fal_demos.distributed.training.flux_lora.trainer.training_worker import (
    FluxLoRATrainingWorker,
)


class CompleteTrainingRequest(BaseModel):
    """Request model for complete training (preprocessing + training)"""
    
    SCHEMA_IGNORES: ClassVar[set[str]] = {"preprocessor_app"}
    
    images_data_url: str = Field(
        description="URL to ZIP file containing training images"
    )
    trigger_word: str = Field(
        default="ohwx",
        description="Trigger word to inject into captions (e.g. 'ohwx', 'txcl')"
    )
    steps: int = Field(
        default=250,
        description="Number of training steps",
        ge=100,
        le=10000,
    )
    learning_rate: float = Field(
        default=5e-4,
        description="Base learning rate for optimizer"
    )
    b_up_factor: float = Field(
        default=3.0,
        description="Learning rate multiplier for lora_B parameters"
    )
    batch_size: int = Field(
        default=4,
        description="Batch size per GPU"
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of steps to accumulate gradients"
    )
    resolution: int = Field(
        default=512,
        description="Training resolution",
        ge=256,
        le=1024,
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    preprocessor_app: str = Field(
        description="URL to the preprocessor app",
        default="alex-w67ic4anktp1/flux-preprocessor-demo/",
    )


class TrainingRequest(BaseModel):
    """Request model for training from preprocessed data"""
    
    training_data_url: File = Field(
        description="Preprocessed training data (.pt file)",
        media_type="application/octet-stream",
    )
    learning_rate: float = Field(
        default=5e-4,
        description="Base learning rate for optimizer"
    )
    b_up_factor: float = Field(
        default=3.0,
        description="Learning rate multiplier for lora_B parameters"
    )
    max_train_steps: int = Field(
        default=250,
        description="Number of training steps"
    )
    batch_size: int = Field(
        default=4,
        description="Batch size per GPU"
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of steps to accumulate gradients"
    )
    guidance_scale: float = Field(
        default=1.0,
        description="Guidance scale for training"
    )
    use_masks: bool = Field(
        default=True,
        description="Whether to use masks (for face-focused training)"
    )
    lr_scheduler: str = Field(
        default="linear",
        description="Learning rate scheduler type: 'constant', 'linear', 'cosine'"
    )
    lr_warmup_steps: int = Field(
        default=0,
        description="Number of warmup steps for learning rate"
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class TrainingResponse(BaseModel):
    """Response model for training"""
    
    checkpoint: File = Field(
        description="Trained LoRA checkpoint file"
    )
    final_loss: float = Field(
        description="Final average training loss"
    )
    num_steps: int = Field(
        description="Total number of training steps completed"
    )
    message: str = Field(
        description="Status message"
    )


class FluxLoRATrainingApp(fal.App):
    """
    Flux LoRA Training App using DistributedRunner.
    
    This app focuses on TRAINING only and uses 8 GPUs with DDP.
    For preprocessing, it calls the flux-preprocessor-demo app.
    
    Benefits:
    - Training runner stays warm (no reload)
    - Preprocessor runs on separate GPUs (no conflict)
    - Clean separation of concerns
    - Can scale independently
    """
    
    machine_type = "GPU-H100"
    num_gpus = 2
    keep_alive = 3000
    min_concurrency = 1
    max_concurrency = 1
    
    requirements = [
        "torch==2.4.0",
        "diffusers==0.30.3",
        "transformers==4.46.0",
        "tokenizers==0.20.1",
        "sentencepiece",
        "peft==0.12.0",
        "safetensors==0.4.4",
        "accelerate==1.4.0",
        "pyzmq==26.0.0",
        "huggingface_hub==0.26.5",
        "fal-client",  # For calling preprocessor app
    ]
    
    async def setup(self) -> None:
        """
        Initialize the training runner.
        
        Downloads Flux weights and starts 8 GPU workers for training.
        Preprocessing is handled by calling a separate app.
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
        
        # Create training runner (uses all 8 GPUs)
        self.runner = DistributedRunner(
            worker_cls=FluxLoRATrainingWorker,
            world_size=self.num_gpus,
        )
        
        # Start training workers
        print(f"Starting {self.num_gpus} training workers...")
        await self.runner.start(model_path=model_path)
        
        print("Training workers ready!")
    
    @fal.endpoint("/train")
    async def train(
        self,
        request: CompleteTrainingRequest,
    ) -> TrainingResponse:
        """
        Complete training pipeline: raw images → trained LoRA.
        
        This endpoint:
        1. Calls flux-preprocessor-demo app (runs on separate 8 GPUs)
        2. Downloads preprocessed data
        3. Trains LoRA on our 8 GPUs
        4. Returns checkpoint
        
        Both apps stay warm, so no model reload overhead!
        """
        import fal_client
        import tempfile
        
        # Step 1: Call preprocessor app (runs on separate instance with 8 GPUs)
        print(f"Calling preprocessor app: {request.preprocessor_app}")
        print(f"Preprocessing {request.images_data_url}...")
        
        try:
            preprocess_result = fal_client.submit(
                request.preprocessor_app,
                arguments={
                    "images_data_url": request.images_data_url,
                    "trigger_word": request.trigger_word,
                    "resolution": request.resolution,
                }
            ).get()
        except Exception as e:
            print(f"Preprocessing failed: {e}")
            print(f"Make sure {request.preprocessor_app} is deployed!")
            raise
        
        # Get preprocessed data URL
        preprocessed_data_url = preprocess_result["preprocessed_data"]["url"]
        num_images = preprocess_result["num_images"]
        
        print(f"✓ Preprocessed {num_images} images")
        
        # Step 2: Download preprocessed data
        with tempfile.TemporaryDirectory() as temp_dir:
            print("Downloading preprocessed data...")
            preprocessed_path = str(download_file(
                preprocessed_data_url,
                target_dir=temp_dir
            ))
            
            # Step 3: Train LoRA (on our 8 GPUs)
            print(f"Training LoRA with {self.num_gpus} GPUs...")
            train_result = await self.runner.invoke({
                "training_data_path": preprocessed_path,
                "learning_rate": request.learning_rate,
                "b_up_factor": request.b_up_factor,
                "max_train_steps": request.steps,
                "batch_size": request.batch_size,
                "gradient_accumulation_steps": request.gradient_accumulation_steps,
                "guidance_scale": 1.0,
                "use_masks": False,
                "lr_scheduler": "constant",
                "lr_warmup_steps": 0,
                "seed": request.seed,
                "streaming": False,
            })
            
            # Upload checkpoint
            checkpoint_file = File.from_path(train_result["checkpoint_path"])
            
            return TrainingResponse(
                checkpoint=checkpoint_file,
                final_loss=train_result["final_loss"],
                num_steps=train_result["num_steps"],
                message=f"Training complete! Processed {num_images} images, {request.steps} steps, final loss: {train_result['final_loss']:.6f}"
            )
    
    @fal.endpoint("/stream")
    async def stream(
        self,
        request: CompleteTrainingRequest,
    ) -> StreamingResponse:
        """
        Complete training pipeline with real-time streaming progress.
        
        This endpoint streams training metrics in real-time:
        - Step number
        - Current loss
        - Average loss
        - Learning rate
        - Progress status
        
        Returns Server-Sent Events (SSE) stream.
        """
        import fal_client
        import tempfile
        
        # Step 1: Call preprocessor app (runs on separate instance with 8 GPUs)
        print(f"Calling preprocessor app: {request.preprocessor_app}")
        print(f"Preprocessing {request.images_data_url}...")
        
        try:
            preprocess_result = fal_client.submit(
                request.preprocessor_app,
                arguments={
                    "images_data_url": request.images_data_url,
                    "trigger_word": request.trigger_word,
                    "resolution": request.resolution,
                }
            ).get()
        except Exception as e:
            print(f"Preprocessing failed: {e}")
            print(f"Make sure {request.preprocessor_app} is deployed!")
            raise
        
        # Get preprocessed data URL
        preprocessed_data_url = preprocess_result["preprocessed_data"]["url"]
        num_images = preprocess_result["num_images"]
        
        print(f"✓ Preprocessed {num_images} images")
        
        # Step 2: Download preprocessed data and stream training
        async def generate_stream():
            with tempfile.TemporaryDirectory() as temp_dir:
                print("Downloading preprocessed data...")
                preprocessed_path = str(download_file(
                    preprocessed_data_url,
                    target_dir=temp_dir
                ))
                
                # Step 3: Train LoRA with streaming (on our GPUs)
                print(f"Training LoRA with {self.num_gpus} GPUs (streaming)...")
                async for event in self.runner.stream(
                    {
                        "training_data_path": preprocessed_path,
                        "learning_rate": request.learning_rate,
                        "b_up_factor": request.b_up_factor,
                        "max_train_steps": request.steps,
                        "batch_size": request.batch_size,
                        "gradient_accumulation_steps": request.gradient_accumulation_steps,
                        "guidance_scale": 1.0,
                        "use_masks": False,
                        "lr_scheduler": "constant",
                        "lr_warmup_steps": 0,
                        "seed": request.seed,
                        "streaming": True,  # Enable streaming!
                    },
                    as_text_events=True,
                ):
                    yield event
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
        )
    
    @fal.endpoint("/train-from-preprocessed")
    async def train_from_preprocessed(
        self,
        request: TrainingRequest,
    ) -> TrainingResponse:
        """
        Train from already preprocessed data.
        
        Use this if you already have preprocessed .pt files.
        """
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Downloading preprocessed data from {request.training_data_url.url}")
            training_data_path = str(download_file(
                request.training_data_url.url,
                target_dir=temp_dir
            ))
            
            # Train on our 8 GPUs
            print(f"Training with {self.num_gpus} GPUs...")
            result = await self.runner.invoke({
                "training_data_path": training_data_path,
                "learning_rate": request.learning_rate,
                "b_up_factor": request.b_up_factor,
                "max_train_steps": request.max_train_steps,
                "batch_size": request.batch_size,
                "gradient_accumulation_steps": request.gradient_accumulation_steps,
                "guidance_scale": request.guidance_scale,
                "use_masks": request.use_masks,
                "lr_scheduler": request.lr_scheduler,
                "lr_warmup_steps": request.lr_warmup_steps,
                "seed": request.seed,
                "streaming": False,
            })
            
            # Check for errors
            if "error" in result:
                raise RuntimeError(f"Training failed: {result['error']}")
            
            # Upload checkpoint
            checkpoint_file = File.from_path(result["checkpoint_path"])
            
            return TrainingResponse(
                checkpoint=checkpoint_file,
                final_loss=result["final_loss"],
                num_steps=result["num_steps"],
                message=f"Training complete! {request.max_train_steps} steps, final loss: {result['final_loss']:.6f}"
            )
    
    @fal.endpoint("/train-from-preprocessed-stream")
    async def train_from_preprocessed_stream(
        self,
        request: TrainingRequest,
    ) -> StreamingResponse:
        """
        Train from already preprocessed data with real-time streaming progress.
        
        This endpoint streams training metrics in real-time as Server-Sent Events (SSE).
        """
        import tempfile
        
        async def generate_stream():
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"Downloading preprocessed data from {request.training_data_url.url}")
                training_data_path = str(download_file(
                    request.training_data_url.url,
                    target_dir=temp_dir
                ))
                
                # Train with streaming enabled
                print(f"Training with {self.num_gpus} GPUs (streaming)...")
                async for event in self.runner.stream(
                    {
                        "training_data_path": training_data_path,
                        "learning_rate": request.learning_rate,
                        "b_up_factor": request.b_up_factor,
                        "max_train_steps": request.max_train_steps,
                        "batch_size": request.batch_size,
                        "gradient_accumulation_steps": request.gradient_accumulation_steps,
                        "guidance_scale": request.guidance_scale,
                        "use_masks": request.use_masks,
                        "lr_scheduler": request.lr_scheduler,
                        "lr_warmup_steps": request.lr_warmup_steps,
                        "seed": request.seed,
                        "streaming": True,  # Enable streaming!
                    },
                    as_text_events=True,
                ):
                    yield event
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
        )


if __name__ == "__main__":
    app = fal.wrap_app(FluxLoRATrainingApp)
    app()

