"""
Flux LoRA Training Worker - Production Implementation

This worker implements Flux LoRA training with:
- Flow matching loss for rectified flow models
- Sigmoid normal timestep sampling
- Proper image and text position IDs
- Dual learning rates for LoRA matrices (lora_A and lora_B)
- Flexible learning rate scheduling
- Optional torch.compile optimization for 20-40% speedup
"""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from fal.distributed import DistributedWorker


class FluxLoRATrainingWorker(DistributedWorker):
    """
    Distributed worker for Flux LoRA training.
    
    When torch.compile is enabled (default), these parameters are fixed to avoid recompilation:
    - BATCH_SIZE: 1
    - LATENT_DIMS: [16, 128, 128] (1024x1024 images)
    - GUIDANCE_SCALE: 1.0
    
    For different effective batch sizes, use gradient_accumulation_steps.
    """
    
    # Fixed constants for torch.compile optimization
    BATCH_SIZE = 1
    LATENT_CHANNELS = 16
    LATENT_HEIGHT = 128
    LATENT_WIDTH = 128
    GUIDANCE_SCALE = 1.0
    IMAGE_SIZE = LATENT_HEIGHT * 8  # 1024 pixels

    def setup(self, model_path: str = "/data/flux_weights", use_torch_compile: bool = True, **kwargs: Any) -> None:
        """
        Initialize the model on each GPU worker with proper LoRA configuration.
        
        Args:
            model_path: Path to the Flux model weights
            use_torch_compile: Whether to use torch.compile for 20-40% speedup (default: True)
        """
        from diffusers import FluxTransformer2DModel
        from peft import LoraConfig

        self.use_torch_compile = use_torch_compile
        self.rank_print(f"Loading Flux model on {self.device}")
        
        # Load the transformer model
        self.transformer = FluxTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        
        # Configure LoRA targeting all transformer blocks
        target_modules = []
        
        # Double stream blocks (19 blocks) - handle text-image interaction
        for block_num in range(19):
            target_modules.extend([
                f"transformer_blocks.{block_num}.attn.to_q",
                f"transformer_blocks.{block_num}.attn.to_k",
                f"transformer_blocks.{block_num}.attn.to_v",
                f"transformer_blocks.{block_num}.attn.to_out.0",
                f"transformer_blocks.{block_num}.attn.add_q_proj",
                f"transformer_blocks.{block_num}.attn.add_k_proj",
                f"transformer_blocks.{block_num}.attn.add_v_proj",
                f"transformer_blocks.{block_num}.attn.to_add_out",
                f"transformer_blocks.{block_num}.ff.net.0.proj",
                f"transformer_blocks.{block_num}.ff.net.2",
                f"transformer_blocks.{block_num}.ff_context.net.0.proj",
                f"transformer_blocks.{block_num}.ff_context.net.2",
            ])
        
        # Single stream blocks (38 blocks) - main image generation pathway
        for block_num in range(38):
            target_modules.extend([
                f"single_transformer_blocks.{block_num}.attn.to_q",
                f"single_transformer_blocks.{block_num}.attn.to_k",
                f"single_transformer_blocks.{block_num}.attn.to_v",
                f"single_transformer_blocks.{block_num}.proj_mlp",
                f"single_transformer_blocks.{block_num}.proj_out",
            ])
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            init_lora_weights="gaussian",
        )
        
        # Add LoRA adapters to the model
        self.transformer.add_adapter(lora_config)
        
        # Freeze base model parameters, enable LoRA parameters
        self.transformer.requires_grad_(False)
        for name, param in self.transformer.named_parameters():
            if "lora" in name:
                param.requires_grad = True
                # Initialize LoRA parameters
                if "lora_A" in name:
                    nn.init.normal_(param, mean=0.0, std=1.0 / 16)
                elif "lora_B" in name:
                    nn.init.zeros_(param)
        
        # Apply torch.compile optimization if enabled (must happen before DDP wrapping)
        if self.use_torch_compile:
            self.rank_print("Compiling transformer with torch.compile...")
            self.transformer = torch.compile(
                self.transformer, 
                mode="reduce-overhead", 
                dynamic=False
            )
            self.rank_print("Model compiled successfully")

        # Wrap with DistributedDataParallel for multi-GPU training
        self.transformer = DDP(
            self.transformer,
            device_ids=[self.rank],
            output_device=self.rank,
            find_unused_parameters=False,
        )

        if self.use_torch_compile:
            self.rank_print("Model compiled and wrapped with DDP")
            self.warmup()
        else:
            self.rank_print("Model loaded and wrapped with DDP")
    
    def warmup(self) -> None:
        """
        Warmup the compiled model to trigger torch.compile graph generation.
        
        Runs a single training step with a lightweight optimizer to compile the model
        graph before actual training begins. This prevents compilation overhead during
        the training phase.
        """
        self.rank_print("Running warmup to trigger torch.compile...")
        
        warmup_data_path = "/data/turbo_flux_trainer_warmup_data.pt"
        
        model = self.transformer.module
        
        # Collect trainable LoRA parameters
        params_A = []
        params_B = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "lora_A" in name:
                    params_A.append(param)
                elif "lora_B" in name:
                    params_B.append(param)
        
        # Create temporary optimizer for warmup
        warmup_optimizer = torch.optim.AdamW([
            {"params": params_A, "lr": 4e-4, "weight_decay": 0.1},
            {"params": params_B, "lr": 4e-4 * 2.0, "weight_decay": 0.1},
        ], betas=(0.9, 0.999), eps=1e-8)
        
        from diffusers.optimization import get_scheduler as get_diffusers_scheduler
        warmup_scheduler = get_diffusers_scheduler(
            "constant",
            optimizer=warmup_optimizer,
            num_warmup_steps=0,
            num_training_steps=1,
        )
        
        # Run a single training step to compile the model
        self._run_training_loop(
            training_data_path=warmup_data_path,
            max_train_steps=1,
            seed=42,
            optimizer=warmup_optimizer,
            lr_scheduler_obj=warmup_scheduler,
            gradient_accumulation_steps=1,
            streaming=False,
            save_checkpoint=False,
        )
        
        # Reset gradients to start training from clean state
        model.zero_grad()
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = None
        
        self.rank_print("Warmup complete - model compiled and ready for training")

    def create_img_ids(self, latent_height: int, latent_width: int) -> torch.Tensor:
        """
        Create image position IDs for Flux transformer.
        
        Args:
            latent_height: Height of latent tensors
            latent_width: Width of latent tensors
            
        Returns:
            Image position IDs tensor of shape [H*W/4, 3]
        """
        from diffusers import FluxPipeline
        
        img_ids_template = FluxPipeline._prepare_latent_image_ids(
            batch_size=1,
            height=latent_height,
            width=latent_width,
            device=self.device,
            dtype=torch.bfloat16,
        )
        return img_ids_template.squeeze(0)

    def run_training_step(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        pooled_embeddings: torch.Tensor,
        text_ids: torch.Tensor,
        img_ids: torch.Tensor,
        gpu_generator: torch.Generator,
        cpu_generator: torch.Generator,
        step: int,
        gradient_accumulation_steps: int = 1,
    ) -> torch.Tensor:
        """
        Execute one training step: batch preparation, forward pass, loss computation, and backward pass.
        
        Args:
            latents: Training latents [N, 16, 64, 64]
            text_embeddings: Text embeddings [N, 512, 4096]
            pooled_embeddings: Pooled embeddings [N, 768]
            text_ids: Text position IDs [1, 512, 3]
            img_ids: Image position IDs [H*W/4, 3]
            gpu_generator: Random generator for noise sampling
            cpu_generator: Random generator for timestep sampling
            step: Current training step number
            gradient_accumulation_steps: Gradient accumulation steps
            
        Returns:
            Loss value scaled by gradient_accumulation_steps
        """
        # Prepare batch
        batch_latents, batch_text_emb, batch_pooled_emb, batch_text_ids, batch_img_ids, timesteps, guidance = self.prepare_training_batch(
            latents=latents,
            text_embeddings=text_embeddings,
            pooled_embeddings=pooled_embeddings,
            text_ids=text_ids,
            img_ids=img_ids,
            batch_size=self.BATCH_SIZE,
            guidance_scale=self.GUIDANCE_SCALE,
            cpu_generator=cpu_generator,
            step=step,
        )
        
        # Compute loss
        loss = self.compute_flux_loss(
            latents=batch_latents,
            text_embeddings=batch_text_emb,
            pooled_embeddings=batch_pooled_emb,
            text_ids=batch_text_ids,
            img_ids=batch_img_ids,
            timesteps=timesteps,
            guidance=guidance,
            generator=gpu_generator,
        )
        
        # Scale loss by gradient accumulation steps
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        return loss

    def prepare_training_batch(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        pooled_embeddings: torch.Tensor,
        text_ids: torch.Tensor,
        img_ids: torch.Tensor,
        batch_size: int,
        guidance_scale: float,
        cpu_generator: torch.Generator,
        step: int = 0,
    ) -> tuple:
        """
        Prepare a training batch with random sampling and timestep generation.
        
        Uses numpy for random sampling to avoid torch.compile recompilation issues
        that would occur with dynamic tensor shapes in the computation graph.
        
        Args:
            step: Step counter for deterministic seed variation
        
        Returns:
            Tuple of (batch_latents, batch_text_emb, batch_pooled_emb, batch_text_ids, 
                     batch_img_ids, timesteps, guidance)
        """
        num_samples = latents.shape[0]
        
        # Use numpy RNG with step-based seeding for deterministic sampling
        base_seed = int(cpu_generator.initial_seed())
        rng = np.random.default_rng(seed=base_seed + step)
        
        # Sample batch indices
        replace = num_samples < batch_size
        indices = rng.choice(num_samples, size=batch_size, replace=replace)
        indices = torch.from_numpy(indices).to(latents.device)
        
        # Get batch
        batch_latents = latents[indices]
        batch_text_emb = text_embeddings[indices]
        batch_pooled_emb = pooled_embeddings[indices]
        
        # Sample timesteps using sigmoid-transformed normal distribution
        n = torch.normal(mean=0.0, std=1.0, size=(batch_size,), generator=cpu_generator)
        timesteps = torch.sigmoid(n).to(self.device, dtype=torch.bfloat16)
        
        # Create guidance tensor
        guidance = torch.full((batch_size,), guidance_scale, device=self.device, dtype=torch.bfloat16)
        
        # Expand position IDs to match batch size
        batch_text_ids = text_ids.expand(batch_size, -1, -1)
        batch_img_ids = img_ids.unsqueeze(0).expand(batch_size, -1, -1)
        
        return batch_latents, batch_text_emb, batch_pooled_emb, batch_text_ids, batch_img_ids, timesteps, guidance

    def compute_flux_loss(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        pooled_embeddings: torch.Tensor,
        text_ids: torch.Tensor,
        img_ids: torch.Tensor,
        timesteps: torch.Tensor,
        guidance: torch.Tensor,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        """
        Compute flow matching loss for Flux training.
        
        Implements the rectified flow objective:
        1. Sample random noise
        2. Interpolate between clean latents and noise based on timestep
        3. Predict velocity vector from noisy latents
        4. Compute MSE loss against true velocity
        
        Args:
            generator: Random generator for reproducible noise generation
        """
        from diffusers import FluxPipeline
        
        # Generate noise with explicit shape (uses class constants for torch.compile)
        noise = torch.randn(
            (self.BATCH_SIZE, self.LATENT_CHANNELS, self.LATENT_HEIGHT, self.LATENT_WIDTH),
            generator=generator,
            dtype=torch.bfloat16,
            device=latents.device
        )
        
        # Flow matching interpolation: noisy_latents = (1-t)*clean + t*noise
        weight = timesteps.view(self.BATCH_SIZE, 1, 1, 1)
        noisy_latents = (1.0 - weight) * latents + weight * noise
        
        # Pack latents for Flux transformer (converts to sequence format)
        packed_latents = FluxPipeline._pack_latents(
            noisy_latents,
            batch_size=self.BATCH_SIZE,
            num_channels_latents=self.LATENT_CHANNELS,
            height=self.LATENT_HEIGHT,
            width=self.LATENT_WIDTH,
        )
        
        # Forward pass through Flux transformer
        model_pred = self.transformer(
            hidden_states=packed_latents,
            timestep=timesteps,
            guidance=guidance,
            pooled_projections=pooled_embeddings,
            encoder_hidden_states=text_embeddings,
            txt_ids=text_ids,
            img_ids=img_ids,
            return_dict=False
        )[0]
        
        # Unpack predictions back to image space
        model_pred = FluxPipeline._unpack_latents(
            model_pred,
            height=self.LATENT_HEIGHT * 8 // 2,
            width=self.LATENT_WIDTH * 8 // 2,
            vae_scale_factor=8,
        )
        
        # Compute velocity target: v = noise - clean
        target = noise - latents
        
        # Compute MSE loss in float32 for numerical stability
        loss = torch.nn.functional.mse_loss(
            model_pred.float(),
            target.float()
        )
        return loss

    def _run_training_loop(
        self,
        training_data_path: str,
        max_train_steps: int,
        seed: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler_obj: Any,
        gradient_accumulation_steps: int,
        streaming: bool,
        save_checkpoint: bool = True,
    ) -> dict[str, Any]:
        """
        Core training loop shared by both warmup and actual training.
        
        Handles data loading, distributed broadcasting, training iteration,
        and checkpoint saving.
        
        Args:
            training_data_path: Path to preprocessed training data (.pt file)
            max_train_steps: Total number of training steps
            seed: Random seed for reproducibility
            optimizer: Optimizer instance
            lr_scheduler_obj: Learning rate scheduler
            gradient_accumulation_steps: Steps to accumulate gradients before update
            streaming: Whether to stream training progress
            save_checkpoint: Whether to save final checkpoint
            
        Returns:
            Dictionary with checkpoint path and training metrics
        """
        # Set seeds for reproducibility
        torch.manual_seed(seed + self.rank)
        torch.cuda.manual_seed(seed + self.rank)
        
        # Load training data on rank 0 and broadcast to all GPUs
        if self.rank == 0:
            self.rank_print(f"Loading training data from {training_data_path}")
            data = torch.load(training_data_path, map_location="cpu", weights_only=False)
            latents = data["latents"]
            text_embeddings = data["text_embeddings"]
            pooled_embeddings = data["pooled_embeddings"]
            text_ids = data["text_ids"]
        else:
            latents = torch.empty(0)
            text_embeddings = torch.empty(0)
            pooled_embeddings = torch.empty(0)
            text_ids = torch.empty(0, dtype=torch.long)
        
        # Broadcast data to all ranks
        torch.cuda.set_device(self.device)
        objects = [latents, text_embeddings, pooled_embeddings, text_ids]
        dist.broadcast_object_list(objects, src=0)
        latents, text_embeddings, pooled_embeddings, text_ids = objects
        
        # Transfer to GPU
        latents = latents.to(self.device, dtype=torch.bfloat16)
        text_embeddings = text_embeddings.to(self.device, dtype=torch.bfloat16)
        pooled_embeddings = pooled_embeddings.to(self.device, dtype=torch.bfloat16)
        text_ids = text_ids.to(self.device, dtype=torch.long)
        
        num_samples = latents.shape[0]
        self.rank_print(f"Loaded {num_samples} training samples")
        self.rank_print(f"Data shapes: latents={latents.shape}, text_emb={text_embeddings.shape}, pooled={pooled_embeddings.shape}, text_ids={text_ids.shape}")
        
        # Validate latent channels
        if latents.shape[1] != self.LATENT_CHANNELS:
            raise ValueError(
                f"Latent channels mismatch! Expected {self.LATENT_CHANNELS}, got {latents.shape[1]}. "
                f"This indicates incorrect VAE configuration. Flux requires 16-channel VAE."
            )
        
        # Rescale spatial dimensions if needed (for torch.compile compatibility)
        if latents.shape[2] != self.LATENT_HEIGHT or latents.shape[3] != self.LATENT_WIDTH:
            self.rank_print(
                f"⚠️  Rescaling latents from [{latents.shape[2]}, {latents.shape[3]}] "
                f"to [{self.LATENT_HEIGHT}, {self.LATENT_WIDTH}]. "
                f"For best quality, preprocess images to {self.IMAGE_SIZE}x{self.IMAGE_SIZE} pixels."
            )
            latents = torch.nn.functional.interpolate(
                latents,
                size=(self.LATENT_HEIGHT, self.LATENT_WIDTH),
                mode='bilinear',
                align_corners=False
            )
            self.rank_print(f"✓ Latents rescaled to {latents.shape}")
        else:
            self.rank_print(f"✓ Latent dimensions: {latents.shape}")
        
        # Prepare image position IDs
        img_ids = self.create_img_ids(
            latent_height=self.LATENT_HEIGHT,
            latent_width=self.LATENT_WIDTH
        )
        
        # Initialize random generators
        gpu_generator = torch.Generator(device=self.device).manual_seed(seed + self.rank)
        cpu_generator = torch.Generator(device='cpu').manual_seed(seed)
        
        # Training loop
        self.transformer.train()
        total_loss = 0.0
        
        for step in range(max_train_steps):
            # Run training step
            loss = self.run_training_step(
                latents=latents,
                text_embeddings=text_embeddings,
                pooled_embeddings=pooled_embeddings,
                text_ids=text_ids,
                img_ids=img_ids,
                gpu_generator=gpu_generator,
                cpu_generator=cpu_generator,
                step=step,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            
            # Update model weights
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler_obj.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            # Stream progress updates
            if streaming and self.rank == 0 and step % 10 == 0:
                avg_loss = total_loss / (step + 1)
                current_lr = optimizer.param_groups[0]['lr']
                self.add_streaming_result({
                    "step": step,
                    "loss": loss.item() * gradient_accumulation_steps,
                    "avg_loss": avg_loss,
                    "learning_rate": current_lr,
                    "progress": f"{step}/{max_train_steps}",
                }, as_text_event=True)
            
            # Print progress
            if step % 20 == 0:
                avg_loss = total_loss / (step + 1)
                self.rank_print(f"Step {step}/{max_train_steps}, Loss: {loss.item() * gradient_accumulation_steps:.6f}, Avg: {avg_loss:.6f}")
            

        
        # Calculate final metrics
        avg_loss = total_loss / max_train_steps if max_train_steps > 0 else 0.0
        self.rank_print(f"Training complete! Average loss: {avg_loss:.6f}")
        
        # Save checkpoint (rank 0 only)
        if save_checkpoint and self.rank == 0:
            self.rank_print("Saving checkpoint...")
            
            from peft import get_peft_model_state_dict
            from diffusers import FluxPipeline
            import shutil
            
            # Unwrap model from DDP
            model_to_save = self.transformer.module
            
            # Unwrap from torch.compile if applicable
            if self.use_torch_compile and hasattr(model_to_save, '_orig_mod'):
                model_to_save = model_to_save._orig_mod
            
            # Extract LoRA weights
            lora_state_dict = get_peft_model_state_dict(model_to_save)
            
            # Save in Flux-compatible format
            with tempfile.TemporaryDirectory() as temp_dir:
                FluxPipeline.save_lora_weights(
                    save_directory=temp_dir,
                    transformer_lora_layers=lora_state_dict,
                    text_encoder_lora_layers=None,
                )
                
                source_path = Path(temp_dir) / "pytorch_lora_weights.safetensors"
                
                with tempfile.NamedTemporaryFile(
                    mode='wb',
                    suffix='.safetensors',
                    delete=False,
                    dir='/tmp'
                ) as f:
                    checkpoint_path = f.name
                
                shutil.copy(source_path, checkpoint_path)
            
            file_size = Path(checkpoint_path).stat().st_size / (1024 * 1024)
            self.rank_print(f"Checkpoint saved: {checkpoint_path} ({file_size:.2f} MB)")
            
            return {
                "checkpoint_path": checkpoint_path,
                "final_loss": avg_loss,
                "num_steps": max_train_steps,
            }
        
        return {}

    def __call__(
        self,
        streaming: bool = False,
        training_data_path: str = None,
        learning_rate: float = 4e-4,
        b_up_factor: float = 2.0,
        max_train_steps: int = 100,
        gradient_accumulation_steps: int = 1,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 0,
        seed: int = 42,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Train a Flux LoRA model.
        
        Note: When torch.compile is enabled (default), batch_size is fixed at 1 and
        guidance_scale at 1.0. Use gradient_accumulation_steps for larger effective batch sizes.
        
        Args:
            streaming: Whether to stream training progress
            training_data_path: Path to preprocessed training data
            learning_rate: Base learning rate for lora_A parameters
            b_up_factor: Multiplier for lora_B learning rate (typically 2.0)
            max_train_steps: Number of training steps
            gradient_accumulation_steps: Steps to accumulate gradients
            lr_scheduler: Learning rate scheduler type
            lr_warmup_steps: Number of warmup steps for scheduler
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing checkpoint path and training metrics
        """
        self.rank_print(f"Starting training: lr={learning_rate}, steps={max_train_steps}, batch_size={self.BATCH_SIZE}")
        
        # Configure optimizer with dual learning rates for LoRA matrices
        params_A = []
        params_B = []
        
        model = self.transformer.module
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "lora_A" in name:
                    params_A.append(param)
                elif "lora_B" in name:
                    params_B.append(param)
        
        # lora_B typically trains with higher LR than lora_A
        optimizer = torch.optim.AdamW([
            {"params": params_A, "lr": learning_rate, "weight_decay": 0.1},
            {"params": params_B, "lr": learning_rate * b_up_factor, "weight_decay": 0.1},
        ], betas=(0.9, 0.999), eps=1e-8)
        
        # Configure learning rate scheduler
        from diffusers.optimization import get_scheduler as get_diffusers_scheduler
        
        lr_scheduler_obj = get_diffusers_scheduler(
            lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=max_train_steps,
        )
        
        # Run training
        return self._run_training_loop(
            training_data_path=training_data_path,
            max_train_steps=max_train_steps,
            seed=seed,
            optimizer=optimizer,
            lr_scheduler_obj=lr_scheduler_obj,
            gradient_accumulation_steps=gradient_accumulation_steps,
            streaming=streaming,
            save_checkpoint=True,
        )
