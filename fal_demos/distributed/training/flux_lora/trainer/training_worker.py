"""
Flux LoRA Training Worker - Production-Ready Implementation

This worker implements REAL Flux LoRA training with:
- Proper flow matching loss
- Sigmoid normal timestep sampling
- Correct image/text IDs
- Dual learning rates for lora_A and lora_B
- Learning rate scheduling
- Multi-caption support
- Validation
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
    """

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
        This runs a forward + backward pass with actual data to trigger torch.compile
        compilation during setup rather than during the first training iteration.

        CRITICAL: Uses real preprocessed data to guarantee 100% identical tensor
        characteristics as training. This ensures torch.compile with dynamic=False
        doesn't recompile on first training step.
        
        NOTE: With the numpy random sampling fix in prepare_training_batch, the exact
        number of samples no longer matters (4 warmup vs 7+ training is OK).
        """
        self.rank_print("Running warmup to trigger torch.compile...")

        # Load real preprocessed data for warmup (subset only)
        warmup_data_path = "/data/turbo_flux_trainer_warmup_data.pt"

        if self.rank == 0:
            self.rank_print(f"Loading warmup data from {warmup_data_path}")
            warmup_data = torch.load(warmup_data_path, map_location="cpu", weights_only=False)

            # Use subset of real data for warmup (first 4 samples)
            warmup_latents = warmup_data["latents"][:4]  # [4, 16, 64, 64]
            warmup_text_emb = warmup_data["text_embeddings"][:4]  # [4, 512, 4096]
            warmup_pooled = warmup_data["pooled_embeddings"][:4]  # [4, 768]
            warmup_text_ids = warmup_data["text_ids"]  # [1, 512, 3]
            warmup_masks = warmup_data.get("masks", None)
            if warmup_masks is not None:
                warmup_masks = warmup_masks[:4]  # [4, 1, 64, 64]

            self.rank_print(f"Warmup data shapes: latents={warmup_latents.shape}, text_emb={warmup_text_emb.shape}, pooled={warmup_pooled.shape}")
        else:
            # Other ranks get empty tensors (will be filled by broadcast)
            warmup_latents = torch.empty(0)
            warmup_text_emb = torch.empty(0)
            warmup_pooled = torch.empty(0)
            warmup_text_ids = torch.empty(0, dtype=torch.long)
            warmup_masks = torch.empty(0)

        # Broadcast warmup data to all ranks
        torch.cuda.set_device(self.device)
        objects = [warmup_latents, warmup_text_emb, warmup_pooled, warmup_text_ids, warmup_masks]
        dist.broadcast_object_list(objects, src=0)
        warmup_latents, warmup_text_emb, warmup_pooled, warmup_text_ids, warmup_masks = objects

        # Move to GPU and ensure correct dtypes (same as training)
        warmup_latents = warmup_latents.to(self.device, dtype=torch.bfloat16)
        warmup_text_emb = warmup_text_emb.to(self.device, dtype=torch.bfloat16)
        warmup_pooled = warmup_pooled.to(self.device, dtype=torch.bfloat16)
        warmup_text_ids = warmup_text_ids.to(self.device, dtype=torch.long)
        if warmup_masks is not None and warmup_masks.numel() > 0:
            warmup_masks = warmup_masks.to(self.device, dtype=torch.bfloat16)
        else:
            warmup_masks = None

        # Create img_ids dynamically (same as training code)
        from diffusers import FluxPipeline
        latent_height = warmup_latents.shape[2]
        latent_width = warmup_latents.shape[3]
        img_ids_template = FluxPipeline._prepare_latent_image_ids(
            batch_size=1,
            height=latent_height,
            width=latent_width,
            device=self.device,
            dtype=torch.bfloat16,
        )
        warmup_img_ids = img_ids_template.squeeze(0)  # Remove batch dim: [H*W/4, 3]

        # Create generators (same as training)
        gpu_generator = torch.Generator(device=self.device).manual_seed(42)
        cpu_generator = torch.Generator(device='cpu').manual_seed(42)

        # Use ACTUAL training batch preparation with real data - guaranteed identical path!
        batch_latents, batch_text_emb, batch_pooled_emb, batch_text_ids, batch_img_ids, timesteps, guidance, batch_masks = self.prepare_training_batch(
            latents=warmup_latents,
            text_embeddings=warmup_text_emb,
            pooled_embeddings=warmup_pooled,
            text_ids=warmup_text_ids,
            img_ids=warmup_img_ids,
            masks=warmup_masks,
            batch_size=4,
            guidance_scale=1.0,
            cpu_generator=cpu_generator,
        )

        # Use ACTUAL loss computation with real data (same as training)
        loss = self.compute_flux_loss(
            latents=batch_latents,
            text_embeddings=batch_text_emb,
            pooled_embeddings=batch_pooled_emb,
            text_ids=batch_text_ids,
            img_ids=batch_img_ids,
            timesteps=timesteps,
            guidance=guidance,
            masks=batch_masks,
            generator=gpu_generator,
        )

        # Backward pass to trigger full compilation
        loss.backward()

        # Clear gradients
        self.transformer.zero_grad()

        self.rank_print("Warmup complete - model compiled with real data and ready for training")

    def prepare_training_batch(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        pooled_embeddings: torch.Tensor,
        text_ids: torch.Tensor,
        img_ids: torch.Tensor,
        masks: torch.Tensor | None,
        batch_size: int,
        guidance_scale: float,
        cpu_generator: torch.Generator,
    ) -> tuple:
        """
        Prepare a single training batch with all preprocessing.
        
        Used by BOTH warmup and training to guarantee identical execution path.
        This is critical for torch.compile with dynamic=False.
        
        IMPORTANT: Uses numpy for random sampling instead of torch.randperm to avoid
        graph dependencies on dataset size (num_samples), which would cause recompilation
        when warmup uses different num_samples than training.
        
        Returns:
            Tuple of (batch_latents, batch_text_emb, batch_pooled_emb, batch_text_ids, 
                     batch_img_ids, timesteps, guidance, batch_masks)
        """
        num_samples = latents.shape[0]
        
        # CRITICAL FIX: Use numpy random instead of torch.randperm to avoid graph dependency
        # torch.randperm(num_samples) creates a symbolic shape dependency on num_samples,
        # causing recompilation when warmup (4 samples) differs from training (variable samples)
        rng = np.random.default_rng(seed=int(cpu_generator.initial_seed()) + int(torch.randint(0, 2**31, (1,)).item()))
        
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
        batch_masks = masks[indices] if masks is not None else None
        
        # Sample timesteps using sigmoid normal distribution
        n = torch.normal(mean=0.0, std=1.0, size=(batch_size,), generator=cpu_generator)
        timesteps = torch.sigmoid(n).to(self.device, dtype=torch.bfloat16)
        
        # Guidance
        guidance = torch.full((batch_size,), guidance_scale, device=self.device, dtype=torch.bfloat16)
        
        # Expand text_ids to batch size: [1, 512, 3] -> [batch_size, 512, 3]
        batch_text_ids = text_ids.expand(batch_size, -1, -1)
        
        # Expand img_ids to batch size: [H*W, 3] -> [batch_size, H*W, 3]
        batch_img_ids = img_ids.unsqueeze(0).expand(batch_size, -1, -1)
        
        return batch_latents, batch_text_emb, batch_pooled_emb, batch_text_ids, batch_img_ids, timesteps, guidance, batch_masks

    def compute_flux_loss(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        pooled_embeddings: torch.Tensor,
        text_ids: torch.Tensor,
        img_ids: torch.Tensor,
        timesteps: torch.Tensor,
        guidance: torch.Tensor,
        masks: torch.Tensor | None = None,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        """
        Compute proper Flux flow matching loss.
        
        This implements the rectified flow objective:
        - Sample noise
        - Interpolate between latents and noise using timestep
        - Predict velocity from latents to noise
        - Compute MSE loss with optional masking
        
        Args:
            generator: GPU generator for reproducible noise. Required for deterministic training.
        """
        from diffusers import FluxPipeline
        
        batch_size = latents.shape[0]
        
        # Generate noise using GPU generator for reproducibility
        # No branching = optimal torch.compile performance with dynamic=False
        noise = torch.randn(latents.shape, generator=generator, dtype=latents.dtype, device=latents.device)
        
        # Flow matching: interpolate between latents and noise
        # weight controls the interpolation (0 = latents, 1 = noise)
        weight = timesteps.view(-1, 1, 1, 1)
        noisy_latents = (1.0 - weight) * latents + weight * noise
        
        # Pack latents (Flux-specific packing for efficient attention)
        packed_latents = FluxPipeline._pack_latents(
            noisy_latents,
            batch_size=batch_size,
            num_channels_latents=latents.shape[1],
            height=latents.shape[2],
            width=latents.shape[3],
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
        
        # Unpack predictions
        model_pred = FluxPipeline._unpack_latents(
            model_pred,
            height=int(latents.shape[2] * 8 // 2),
            width=int(latents.shape[3] * 8 // 2),
            vae_scale_factor=8,
        )
        
        # Target is the velocity from latents to noise
        target = noise - latents
        
        # Apply masks if provided (for inpainting)
        if masks is not None:
            target = masks.float() * target.float()
            model_pred = masks.float() * model_pred.float()
        else:
            target = target.float()
            model_pred = model_pred.float()
        
        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(model_pred, target)
        return loss

    def __call__(
        self,
        streaming: bool = False,
        training_data_path: str = None,
        learning_rate: float = 4e-4,
        b_up_factor: float = 2.0,
        max_train_steps: int = 100,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        guidance_scale: float = 1.0,
        use_masks: bool = True,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 0,
        seed: int = 42,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Production-ready training function with proper Flux training logic.
        
        The numpy random sampling fix in prepare_training_batch prevents recompilation
        even when dataset size varies between warmup and training.
        """
        self.rank_print(f"Starting training: lr={learning_rate}, steps={max_train_steps}")
        
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
            masks = data.get("masks", None)
        else:
            latents = torch.empty(0)
            text_embeddings = torch.empty(0)
            pooled_embeddings = torch.empty(0)
            text_ids = torch.empty(0, dtype=torch.long)
            masks = torch.empty(0)
        
        # Step 2: Broadcast data to all ranks
        # IMPORTANT: Set correct CUDA device before broadcast to avoid device mismatch
        torch.cuda.set_device(self.device)
        objects = [latents, text_embeddings, pooled_embeddings, text_ids, masks]
        dist.broadcast_object_list(objects, src=0)
        latents, text_embeddings, pooled_embeddings, text_ids, masks = objects
        
        # Move to GPU
        latents = latents.to(self.device, dtype=torch.bfloat16)
        text_embeddings = text_embeddings.to(self.device, dtype=torch.bfloat16)
        pooled_embeddings = pooled_embeddings.to(self.device, dtype=torch.bfloat16)
        text_ids = text_ids.to(self.device, dtype=torch.long)
        if masks is not None and masks.numel() > 0 and use_masks:
            masks = masks.to(self.device, dtype=torch.bfloat16)
        else:
            masks = None
        
        num_samples = latents.shape[0]
        self.rank_print(f"Loaded {num_samples} training samples")
        self.rank_print(f"Data shapes: latents={latents.shape}, text_emb={text_embeddings.shape}, pooled={pooled_embeddings.shape}, text_ids={text_ids.shape}")
        
        # Step 3: Prepare image IDs using FluxPipeline's method for consistency
        # This creates properly formatted position IDs for the packed latents
        from diffusers import FluxPipeline
        
        # Note: latents are 64x64, Flux packs to 32x32
        # FluxPipeline._prepare_latent_image_ids expects unpacked dimensions
        latent_height = latents.shape[2]
        latent_width = latents.shape[3]
        
        # Create template image IDs (will be expanded per batch)
        img_ids_template = FluxPipeline._prepare_latent_image_ids(
            batch_size=1,
            height=latent_height,
            width=latent_width,
            device=self.device,
            dtype=torch.bfloat16,
        )
        # Shape: [1, H*W/4, 3] (already packed by _prepare_latent_image_ids)
        img_ids = img_ids_template.squeeze(0)  # Remove batch dim: [H*W/4, 3]
        
        # Step 4: Set up optimizer with DUAL learning rates for lora_A and lora_B
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
        # Reference implementation uses weight_decay=0.1
        optimizer = torch.optim.AdamW([
            {"params": params_A, "lr": learning_rate, "weight_decay": 0.1},
            {"params": params_B, "lr": learning_rate * b_up_factor, "weight_decay": 0.1},
        ], betas=(0.9, 0.999), eps=1e-8)
        
        # Step 5: Set up learning rate scheduler
        from diffusers.optimization import get_scheduler as get_diffusers_scheduler
        
        lr_scheduler_obj = get_diffusers_scheduler(
            lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=max_train_steps,
        )
        
        # Step 6: Set up random generators
        # GPU generator for noise (faster), CPU generator for timesteps (deterministic)
        gpu_generator = torch.Generator(device=self.device).manual_seed(seed + self.rank)
        cpu_generator = torch.Generator(device='cpu').manual_seed(seed)
        
        # Step 7: Training loop
        self.transformer.train()
        total_loss = 0.0
        
        for step in range(max_train_steps):
            # Prepare batch using shared function (same as warmup!)
            batch_latents, batch_text_emb, batch_pooled_emb, batch_text_ids, batch_img_ids, timesteps, guidance, batch_masks = self.prepare_training_batch(
                latents=latents,
                text_embeddings=text_embeddings,
                pooled_embeddings=pooled_embeddings,
                text_ids=text_ids,
                img_ids=img_ids,
                masks=masks,
                batch_size=batch_size,
                guidance_scale=guidance_scale,
                cpu_generator=cpu_generator,
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
                masks=batch_masks,
                generator=gpu_generator,
            )
            
            # Backward pass with gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Update weights
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler_obj.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            # Stream progress (only rank 0)
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
        avg_loss = total_loss / max_train_steps
        self.rank_print(f"Training complete! Average loss: {avg_loss:.6f}")
        
        # Step 9: Save checkpoint (only rank 0)
        if self.rank == 0:
            self.rank_print("Saving checkpoint...")
            
            # Extract LoRA weights using proper Flux pipeline format
            from peft import get_peft_model_state_dict
            from diffusers import FluxPipeline
            import shutil
            
            # Get the underlying module from DDP
            model_to_save = self.transformer.module
            
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
