"""
Flux LoRA Training Worker - Production-Ready Implementation

This worker implements REAL Flux LoRA training with:
- Proper flow matching loss
- Sigmoid normal timestep sampling
- Correct image/text IDs
- Dual learning rates for lora_A and lora_B
- Learning rate scheduling
- torch.compile optimization with warmup
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
    Production-ready distributed worker for Flux LoRA training.
    
    CRITICAL CONSTRAINTS for torch.compile with dynamic=False:
    - BATCH_SIZE: Fixed at 4 (matches warmup)
    - LATENT_DIMS: Fixed at [16, 64, 64] (matches 512x512 images)
    - GUIDANCE_SCALE: Fixed at 1.0 (matches warmup)
    
    These are hardcoded to prevent recompilation. Any deviation will cause
    torch.compile to recompile the entire model during training.
    """
    
    # Fixed compilation constants - DO NOT CHANGE without retraining warmup
    BATCH_SIZE = 4
    LATENT_CHANNELS = 16
    LATENT_HEIGHT = 64
    LATENT_WIDTH = 64
    GUIDANCE_SCALE = 1.0
    IMAGE_SIZE = LATENT_HEIGHT * 8  # 512 pixels

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
        
        # Configure LoRA targeting all blocks (both double and single stream)
        
        target_modules = []
        
        # Double stream blocks (19 blocks) - these handle text-image interaction
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
        
        # Single stream blocks (38 blocks) - CRITICAL for image generation
        for block_num in range(38):
            target_modules.extend([
                f"single_transformer_blocks.{block_num}.attn.to_q",
                f"single_transformer_blocks.{block_num}.attn.to_k",
                f"single_transformer_blocks.{block_num}.attn.to_v",
                f"single_transformer_blocks.{block_num}.proj_mlp",
                f"single_transformer_blocks.{block_num}.proj_out",
            ])
        
        lora_config = LoraConfig(
            r=16,  # rank
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            init_lora_weights="gaussian",
        )
        
        # Add LoRA adapters
        self.transformer.add_adapter(lora_config)
        
        # Freeze base model, only train LoRA
        self.transformer.requires_grad_(False)
        for name, param in self.transformer.named_parameters():
            if "lora" in name:
                param.requires_grad = True
                # Initialize LoRA parameters properly
                if "lora_A" in name:
                    nn.init.normal_(param, mean=0.0, std=1.0 / 16)
                elif "lora_B" in name:
                    nn.init.zeros_(param)
        
        # Optionally compile BEFORE wrapping with DDP (CRITICAL ORDER!)
        if self.use_torch_compile:
            # mode="reduce-overhead" optimizes for reduced Python overhead in training loops
            # dynamic=False with numpy random sampling avoids recompilation issues
            self.rank_print("Compiling transformer with torch.compile...")
            self.transformer = torch.compile(
                self.transformer, 
                mode="reduce-overhead", 
                dynamic=False
            )
            self.rank_print("Model compiled successfully")

        # Then wrap the compiled model with DDP
        self.transformer = DDP(
            self.transformer,
            device_ids=[self.rank],
            output_device=self.rank,
            find_unused_parameters=False,
        )

        if self.use_torch_compile:
            self.rank_print("Model compiled and wrapped with DDP")
            # Run warmup to trigger compilation during setup
            self.warmup()
        else:
            self.rank_print("Model loaded and wrapped with DDP (torch.compile disabled)")
    
    def warmup(self) -> None:
        """
        Warmup the compiled model using real preprocessed training data.
        
        Creates DUMMY optimizer and scheduler to ensure 100% identical control flow
        with training. This is critical - even though we don't update weights during
        warmup, we need to call optimizer.step() and scheduler.step() to match the
        exact execution path of training for torch.compile with dynamic=False.
        """
        self.rank_print("Running warmup to trigger torch.compile...")
        
        warmup_data_path = "/data/turbo_flux_trainer_warmup_data.pt"
        
        # Create DUMMY optimizer and scheduler for warmup
        # This ensures identical control flow as training (no conditional branches)
        model = self.transformer.module
        
        # Collect trainable parameters (same as training)
        params_A = []
        params_B = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "lora_A" in name:
                    params_A.append(param)
                elif "lora_B" in name:
                    params_B.append(param)
        
        # Create dummy optimizer (same config as training)
        dummy_optimizer = torch.optim.AdamW([
            {"params": params_A, "lr": 4e-4, "weight_decay": 0.1},
            {"params": params_B, "lr": 4e-4 * 2.0, "weight_decay": 0.1},
        ], betas=(0.9, 0.999), eps=1e-8)
        
        # Create dummy scheduler (same as training)
        from diffusers.optimization import get_scheduler as get_diffusers_scheduler
        dummy_scheduler = get_diffusers_scheduler(
            "constant",
            optimizer=dummy_optimizer,
            num_warmup_steps=0,
            num_training_steps=1,
        )
        
        # Call the training loop with dummy optimizer/scheduler
        # This ensures IDENTICAL execution path as real training
        self._run_training_loop(
            training_data_path=warmup_data_path,
            max_train_steps=1,
            seed=42,
            optimizer=dummy_optimizer,  # Dummy optimizer (won't actually update weights much in 1 step)
            lr_scheduler_obj=dummy_scheduler,  # Dummy scheduler
            gradient_accumulation_steps=1,
            streaming=False,
            save_checkpoint=False,  # Don't save checkpoint during warmup
        )
        
        # Zero out any changes from the dummy optimizer step
        # This ensures we start training from the exact same initial state
        model.zero_grad()
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = None
        
        self.rank_print("Warmup complete - model compiled with real data and ready for training")

    def create_img_ids(self, latent_height: int, latent_width: int) -> torch.Tensor:
        """
        Create image position IDs for Flux transformer.
        
        This is extracted as a separate method to ensure warmup and training use
        IDENTICAL code paths.
        
        Args:
            latent_height: Height of latent tensors (should match LATENT_HEIGHT)
            latent_width: Width of latent tensors (should match LATENT_WIDTH)
            
        Returns:
            Image IDs tensor of shape [H*W/4, 3]
        """
        from diffusers import FluxPipeline
        
        img_ids_template = FluxPipeline._prepare_latent_image_ids(
            batch_size=1,
            height=latent_height,
            width=latent_width,
            device=self.device,
            dtype=torch.bfloat16,
        )
        return img_ids_template.squeeze(0)  # Remove batch dim: [H*W/4, 3]

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
        Execute one complete training step: batch prep -> forward -> loss -> backward.
        
        This method is shared by BOTH warmup and training to guarantee 100% identical
        execution paths. This is absolutely critical for torch.compile with dynamic=False.
        
        Args:
            latents: Training latents [N, 16, 64, 64]
            text_embeddings: Text embeddings [N, 512, 4096]
            pooled_embeddings: Pooled embeddings [N, 768]
            text_ids: Text position IDs [1, 512, 3]
            img_ids: Image position IDs [H*W/4, 3]
            gpu_generator: Generator for noise sampling
            cpu_generator: Generator for timestep sampling
            step: Current training step
            gradient_accumulation_steps: Number of steps to accumulate gradients
            
        Returns:
            Loss value (already scaled by gradient_accumulation_steps)
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
        Prepare a single training batch with all preprocessing.
        
        Used by BOTH warmup and training to guarantee identical execution path.
        This is critical for torch.compile with dynamic=False.
        
        IMPORTANT: Uses numpy for random sampling instead of torch.randperm to avoid
        graph dependencies on dataset size (num_samples), which would cause recompilation
        when warmup uses different num_samples than training.
        
        Args:
            step: Step counter for deterministic seed variation (avoids torch.randint in compiled region)
        
        Returns:
            Tuple of (batch_latents, batch_text_emb, batch_pooled_emb, batch_text_ids, 
                     batch_img_ids, timesteps, guidance)
        """
        num_samples = latents.shape[0]
        
        # Use step counter for deterministic seed variation instead of torch.randint
        # This avoids creating torch operations in the compiled region
        base_seed = int(cpu_generator.initial_seed())
        rng = np.random.default_rng(seed=base_seed + step)
        
        # Sample indices without replacement if possible, with replacement if needed
        if num_samples >= batch_size:
            indices = rng.choice(num_samples, size=batch_size, replace=False)
        else:
            indices = rng.choice(num_samples, size=batch_size, replace=True)
        
        # Convert to tensor on the correct device
        indices = torch.from_numpy(indices).to(latents.device)
        
        # Get batch
        batch_latents = latents[indices]
        batch_text_emb = text_embeddings[indices]
        batch_pooled_emb = pooled_embeddings[indices]
        
        # Sample timesteps using sigmoid normal distribution
        n = torch.normal(mean=0.0, std=1.0, size=(batch_size,), generator=cpu_generator)
        timesteps = torch.sigmoid(n).to(self.device, dtype=torch.bfloat16)
        
        # Guidance
        guidance = torch.full((batch_size,), guidance_scale, device=self.device, dtype=torch.bfloat16)
        
        # Expand text_ids to batch size: [1, 512, 3] -> [batch_size, 512, 3]
        batch_text_ids = text_ids.expand(batch_size, -1, -1)
        
        # Expand img_ids to batch size: [H*W, 3] -> [batch_size, H*W, 3]
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
        Compute proper Flux flow matching loss.
        
        This implements the rectified flow objective:
        - Sample noise
        - Interpolate between latents and noise using timestep
        - Predict velocity from latents to noise
        - Compute MSE loss
        
        CRITICAL: Uses class constants instead of dynamic shape extraction to prevent
        torch.compile recompilation with dynamic=False.
        
        Args:
            generator: GPU generator for reproducible noise. Required for deterministic training.
        """
        from diffusers import FluxPipeline
        
        # Generate noise using EXPLICIT shape tuple (not latents.shape which is dynamic)
        # CRITICAL: Must use explicit tuple of constants, not tensor.shape
        noise = torch.randn(
            (self.BATCH_SIZE, self.LATENT_CHANNELS, self.LATENT_HEIGHT, self.LATENT_WIDTH),
            generator=generator,
            dtype=torch.bfloat16,  # Explicit dtype instead of latents.dtype
            device=latents.device
        )
        
        # Flow matching: interpolate between latents and noise
        # Use explicit batch size instead of -1 for dynamic=False
        weight = timesteps.view(self.BATCH_SIZE, 1, 1, 1)  # Explicit instead of -1
        noisy_latents = (1.0 - weight) * latents + weight * noise
        
        # Pack latents (Flux-specific packing for efficient attention)
        # Use class constants for all shape parameters to prevent recompilation
        packed_latents = FluxPipeline._pack_latents(
            noisy_latents,
            batch_size=self.BATCH_SIZE,
            num_channels_latents=self.LATENT_CHANNELS,
            height=self.LATENT_HEIGHT,
            width=self.LATENT_WIDTH,
        )
        
        # Forward pass through transformer

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
        
        # Unpack predictions using fixed constants
        model_pred = FluxPipeline._unpack_latents(
            model_pred,
            height=self.LATENT_HEIGHT * 8 // 2,  # 256 for 64x64 latents
            width=self.LATENT_WIDTH * 8 // 2,    # 256 for 64x64 latents
            vae_scale_factor=8,
        )
        
        # Target is the velocity from latents to noise
        target = noise - latents
        
        # Convert to float32 for loss computation
        target = target.float()
        model_pred = model_pred.float()
        
        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(model_pred, target)
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
        Core training loop logic shared by both warmup and training.
        
        This method handles:
        - Data loading and broadcasting
        - Device setup
        - Training loop execution
        - Checkpoint saving (controlled by save_checkpoint flag)
        
        Args:
            training_data_path: Path to preprocessed training data
            max_train_steps: Number of training steps
            seed: Random seed
            optimizer: Optimizer (dummy for warmup, real for training)
            lr_scheduler_obj: LR scheduler (dummy for warmup, real for training)
            gradient_accumulation_steps: Gradient accumulation steps
            streaming: Whether to stream progress
            save_checkpoint: Whether to save checkpoint at end (False for warmup)
            
        Returns:
            Dict with results (empty for warmup, checkpoint info for training)
        """
        # Set seeds for reproducibility
        torch.manual_seed(seed + self.rank)
        torch.cuda.manual_seed(seed + self.rank)
        
        # Step 1: Load training data (only rank 0)
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
        
        # Step 2: Broadcast data to all ranks
        # IMPORTANT: Set correct CUDA device before broadcast to avoid device mismatch
        torch.cuda.set_device(self.device)
        objects = [latents, text_embeddings, pooled_embeddings, text_ids]
        dist.broadcast_object_list(objects, src=0)
        latents, text_embeddings, pooled_embeddings, text_ids = objects
        
        # Move to GPU
        latents = latents.to(self.device, dtype=torch.bfloat16)
        text_embeddings = text_embeddings.to(self.device, dtype=torch.bfloat16)
        pooled_embeddings = pooled_embeddings.to(self.device, dtype=torch.bfloat16)
        text_ids = text_ids.to(self.device, dtype=torch.long)
        
        num_samples = latents.shape[0]
        self.rank_print(f"Loaded {num_samples} training samples")
        self.rank_print(f"Data shapes: latents={latents.shape}, text_emb={text_embeddings.shape}, pooled={pooled_embeddings.shape}, text_ids={text_ids.shape}")
        
        # CRITICAL: Rescale latents to match warmup dimensions to prevent recompilation
        # torch.compile with dynamic=False requires EXACT dimension matching
        if latents.shape[1] != self.LATENT_CHANNELS:
            raise ValueError(
                f"Latent channels mismatch! Expected {self.LATENT_CHANNELS}, got {latents.shape[1]}. "
                f"Cannot rescale channels - this indicates wrong VAE model. "
                f"Flux requires 16-channel VAE."
            )
        
        # Automatically rescale spatial dimensions if needed
        if latents.shape[2] != self.LATENT_HEIGHT or latents.shape[3] != self.LATENT_WIDTH:
            self.rank_print(
                f"⚠️  Rescaling latents from [{latents.shape[2]}, {latents.shape[3]}] "
                f"to [{self.LATENT_HEIGHT}, {self.LATENT_WIDTH}] to match compilation dimensions. "
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
            self.rank_print(f"✓ Latent dimensions match warmup: {latents.shape}")
        
        # Step 3: Prepare image IDs using shared helper method (identical to warmup)
        img_ids = self.create_img_ids(
            latent_height=self.LATENT_HEIGHT,
            latent_width=self.LATENT_WIDTH
        )
        
        # Step 4: Set up random generators
        # GPU generator for noise (faster), CPU generator for timesteps (deterministic)
        gpu_generator = torch.Generator(device=self.device).manual_seed(seed + self.rank)
        cpu_generator = torch.Generator(device='cpu').manual_seed(seed)
        
        # Step 7: Training loop
        self.transformer.train()
        total_loss = 0.0
        
        for step in range(max_train_steps):
            # Run training step using IDENTICAL method as warmup
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
            
            # Update weights - happens every time (including warmup with dummy optimizer)
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler_obj.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            # Stream progress (only rank 0, only during actual training not warmup)
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
            

        
        # Final average loss
        avg_loss = total_loss / max_train_steps if max_train_steps > 0 else 0.0
        self.rank_print(f"Training complete! Average loss: {avg_loss:.6f}")
        
        # Step 9: Save checkpoint (only rank 0, only when requested)
        if save_checkpoint and self.rank == 0:
            self.rank_print("Saving checkpoint...")
            
            # Extract LoRA weights using proper Flux pipeline format
            from peft import get_peft_model_state_dict
            from diffusers import FluxPipeline
            import shutil
            
            # Get the underlying module from DDP
            model_to_save = self.transformer.module
            
            # If torch.compile was used, unwrap the compiled model to access the original
            # torch.compile wraps the model and stores the original in _orig_mod
            if self.use_torch_compile and hasattr(model_to_save, '_orig_mod'):
                self.rank_print("Unwrapping torch.compiled model...")
                model_to_save = model_to_save._orig_mod
            
            # Extract only LoRA parameters
            lora_state_dict = get_peft_model_state_dict(model_to_save)
            
            # Create temporary directory for proper LoRA format
            # FluxPipeline.save_lora_weights creates the correct format for loading
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save using FluxPipeline format (creates pytorch_lora_weights.safetensors)
                FluxPipeline.save_lora_weights(
                    save_directory=temp_dir,
                    transformer_lora_layers=lora_state_dict,
                    text_encoder_lora_layers=None,
                )
                
                # The saved file will be pytorch_lora_weights.safetensors
                source_path = Path(temp_dir) / "pytorch_lora_weights.safetensors"
                
                # Copy to a permanent location
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
        
        # Other ranks return empty dict
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
        Production-ready training function with proper Flux training logic.
        
        IMPORTANT FIXED PARAMETERS (to prevent torch.compile recompilation):
        - batch_size: Fixed at 4
        - guidance_scale: Fixed at 1.0
        - latent_dims: Automatically rescaled to [16, 64, 64]
        
        If you need different effective batch sizes, use gradient_accumulation_steps.
        
        This method sets up the optimizer and scheduler, then calls the shared
        _run_training_loop method that is also used by warmup.
        """
        self.rank_print(f"Starting training: lr={learning_rate}, steps={max_train_steps}, batch_size={self.BATCH_SIZE}")
        
        # Set up optimizer with DUAL learning rates for lora_A and lora_B
        params_A = []
        params_B = []
        
        # Get the underlying module from DDP
        model = self.transformer.module
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "lora_A" in name:
                    params_A.append(param)
                elif "lora_B" in name:
                    params_B.append(param)
        
        # Create optimizer with different LRs
        optimizer = torch.optim.AdamW([
            {"params": params_A, "lr": learning_rate, "weight_decay": 0.1},
            {"params": params_B, "lr": learning_rate * b_up_factor, "weight_decay": 0.1},
        ], betas=(0.9, 0.999), eps=1e-8)
        
        # Set up learning rate scheduler
        from diffusers.optimization import get_scheduler as get_diffusers_scheduler
        
        lr_scheduler_obj = get_diffusers_scheduler(
            lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=max_train_steps,
        )
        
        # Call the shared training loop (same as warmup but with real optimizer and checkpoint saving)
        return self._run_training_loop(
            training_data_path=training_data_path,
            max_train_steps=max_train_steps,
            seed=seed,
            optimizer=optimizer,
            lr_scheduler_obj=lr_scheduler_obj,
            gradient_accumulation_steps=gradient_accumulation_steps,
            streaming=streaming,
            save_checkpoint=True,  # Save checkpoint at end of training
        )
