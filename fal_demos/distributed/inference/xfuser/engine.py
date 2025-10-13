"""
Ray-based distributed inference engine for xFuser.
This module contains the Ray actors and Engine class that handle distributed image generation.
"""
import os
import time
import torch
import ray
import io
import logging
import base64
from typing import Optional

from xfuser import (
    xFuserPixArtAlphaPipeline,
    xFuserPixArtSigmaPipeline,
    xFuserFluxPipeline,
    xFuserStableDiffusion3Pipeline,
    xFuserStableDiffusionXLPipeline,
    xFuserHunyuanDiTPipeline,
    xFuserArgs,
)


@ray.remote(num_gpus=1)
class ImageGenerator:
    """Ray actor for distributed image generation using xFuser."""
    
    def __init__(self, xfuser_args_dict: dict, rank: int, world_size: int):
        """
        Initialize the ImageGenerator actor.
        
        Args:
            xfuser_args_dict: Dictionary containing xFuser configuration
            rank: Rank of this worker in the distributed setup
            world_size: Total number of workers
        """
        # Reconstruct xFuserArgs from dict (avoids Ray pickling issues with Pydantic)
        xfuser_args = xFuserArgs(**xfuser_args_dict)
        
        # Set PyTorch distributed environment variables
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        
        self.rank = rank
        self.setup_logger()
        self.initialize_model(xfuser_args)

    def setup_logger(self):
        """Configure logging for this worker."""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - Rank %(rank)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)

    def initialize_model(self, xfuser_args: xFuserArgs):
        """
        Initialize the xFuser pipeline for this worker.
        
        Args:
            xfuser_args: xFuser configuration object
        """
        # Initialize distributed environment in create_config
        self.engine_config, self.input_config = xfuser_args.create_config()
        
        model_name = self.engine_config.model_config.model.split("/")[-1]
        pipeline_map = {
            # PixArt models
            "PixArt-XL-2-1024-MS": xFuserPixArtAlphaPipeline,
            "PixArt-Sigma-XL-2-1024-MS": xFuserPixArtSigmaPipeline,
            "PixArt-Sigma-XL-2-2K-MS": xFuserPixArtSigmaPipeline,
            
            # SDXL models
            "stable-diffusion-xl-base-1.0": xFuserStableDiffusionXLPipeline,
            
            # SD3 models
            "stable-diffusion-3-medium-diffusers": xFuserStableDiffusion3Pipeline,
            
            # HunyuanDiT models
            "HunyuanDiT-v1.2-Diffusers": xFuserHunyuanDiTPipeline,
            
            # FLUX models
            "FLUX.1-schnell": xFuserFluxPipeline,
            "FLUX.1-dev": xFuserFluxPipeline,
        }
        
        PipelineClass = pipeline_map.get(model_name)
        if PipelineClass is None:
            raise NotImplementedError(f"{model_name} is currently not supported!")

        self.logger.info(f"Initializing model {model_name} from {xfuser_args.model}")

        self.pipe = PipelineClass.from_pretrained(
            pretrained_model_name_or_path=xfuser_args.model,
            engine_config=self.engine_config,
            torch_dtype=torch.float16,
        ).to("cuda")
        
        self.pipe.prepare_run(self.input_config)
        self.logger.info("Model initialization completed")

    def generate(self, request_dict: dict) -> Optional[dict]:
        """
        Generate an image based on the request.
        
        Args:
            request_dict: Dictionary containing generation parameters:
                - prompt (str): Text prompt
                - height (int): Image height
                - width (int): Image width
                - num_inference_steps (int): Number of denoising steps
                - seed (int): Random seed
                - cfg (float): Guidance scale
                - save_disk_path (str, optional): Path to save image
        
        Returns:
            Dictionary with generation results or None if not the last group
        """
        try:
            start_time = time.time()
            
            output = self.pipe(
                height=request_dict.get("height", 1024),
                width=request_dict.get("width", 1024),
                prompt=request_dict["prompt"],
                num_inference_steps=request_dict.get("num_inference_steps", 50),
                output_type="pil",
                generator=torch.Generator(device="cuda").manual_seed(request_dict.get("seed", 42)),
                guidance_scale=request_dict.get("cfg", 7.5),
                max_sequence_length=self.input_config.max_sequence_length
            )
            
            elapsed_time = time.time() - start_time

            if self.pipe.is_dp_last_group():
                save_disk_path = request_dict.get("save_disk_path")
                
                if save_disk_path:
                    # Save to disk
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"generated_image_{timestamp}.png"
                    file_path = os.path.join(save_disk_path, filename)
                    os.makedirs(save_disk_path, exist_ok=True)
                    output.images[0].save(file_path)
                    
                    return {
                        "message": "Image generated successfully",
                        "elapsed_time": f"{elapsed_time:.2f} sec",
                        "output": file_path,
                        "save_to_disk": True
                    }
                else:
                    # Return base64 encoded image
                    buffered = io.BytesIO()
                    output.images[0].save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    return {
                        "message": "Image generated successfully",
                        "elapsed_time": f"{elapsed_time:.2f} sec",
                        "output": img_str,
                        "save_to_disk": False
                    }
            
            return None

        except Exception as e:
            self.logger.error(f"Error generating image: {str(e)}")
            raise Exception(f"Image generation failed: {str(e)}")


class Engine:
    """Distributed inference engine that manages multiple Ray workers."""
    
    def __init__(self, world_size: int, xfuser_args_dict: dict):
        """
        Initialize the distributed engine.
        
        Args:
            world_size: Number of GPU workers to spawn
            xfuser_args_dict: Dictionary containing xFuser configuration
        """
        # Ensure Ray is initialized
        if not ray.is_initialized():
            ray.init()
        
        print(f"Initializing Ray engine with {world_size} workers...")
        
        # Create Ray actors for each GPU
        self.workers = [
            ImageGenerator.remote(xfuser_args_dict, rank=rank, world_size=world_size)
            for rank in range(world_size)
        ]
        
        print(f"Ray engine initialized with {len(self.workers)} workers")
    
    def generate(self, request_dict: dict) -> dict:
        """
        Generate an image using the distributed workers.
        
        Args:
            request_dict: Dictionary containing generation parameters
        
        Returns:
            Dictionary with generation results
        """
        # Execute generation on all workers in parallel
        results = ray.get([
            worker.generate.remote(request_dict)
            for worker in self.workers
        ])
        
        # Return the result from the last data parallel group
        result = next((r for r in results if r is not None), None)
        
        if result is None:
            raise RuntimeError("No result returned from workers")
        
        return result
