"""
Flux Preprocessing Worker - Multi-GPU Parallel Preprocessing

This worker preprocesses raw images for Flux LoRA training:
- Auto-caption generation (moondream)
- Trigger word injection
- VAE encoding (images → latents)
- Text encoding (T5 + CLIP)
- Face detection and masks (optional)

Uses multi-GPU parallelization with DistributedSampler for speed.
"""

import os
import zipfile
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
from PIL import Image
import numpy as np

from fal.distributed import DistributedWorker


class FluxPreprocessorWorker(DistributedWorker):
    """
    Multi-GPU preprocessing worker for Flux LoRA training.

    Each GPU processes a subset of images in parallel, then results are gathered.
    """

    def setup(self, model_path: str = "/data/flux_weights", **kwargs: Any) -> None:
        """
        Load preprocessing models on each GPU:
        - VAE for image encoding
        - T5-XXL for text encoding
        - CLIP for pooled embeddings
        - Moondream for auto-captioning
        """
        self.rank_print("Loading preprocessing models...")

        # Load VAE
        from diffusers import AutoencoderKL

        self.vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        self.rank_print("VAE loaded")

        # Load text encoders
        from transformers import T5EncoderModel, T5TokenizerFast
        from transformers import CLIPTextModel, CLIPTokenizer

        self.text_encoder = T5EncoderModel.from_pretrained(
            model_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

        self.tokenizer = T5TokenizerFast.from_pretrained(
            model_path,
            subfolder="tokenizer_2",
        )

        self.rank_print("T5 loaded")

        self.clip_encoder = CLIPTextModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.clip_encoder.eval()
        self.clip_encoder.requires_grad_(False)

        self.clip_tokenizer = CLIPTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer",
        )

        self.rank_print("CLIP loaded")

        # Load captioning model on ALL GPUs for parallel caption generation
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.captioner = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                revision="2024-08-26",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).to(self.device)
            self.captioner.eval()

            self.caption_tokenizer = AutoTokenizer.from_pretrained(
                "vikhyatk/moondream2",
                revision="2024-08-26",
            )

            self.rank_print("Moondream captioner loaded")
        except Exception as e:
            self.rank_print(f"Warning: Could not load captioner: {e}")
            self.captioner = None

        self.rank_print("All preprocessing models loaded")

    def extract_zip(self, zip_path: str, extract_dir: str) -> list[Path]:
        """Extract images from zip file (rank 0 only)"""
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = []

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Find all images
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_paths.append(Path(root) / file)

        return sorted(image_paths)

    def load_or_generate_caption(
        self,
        image_path: Path,
        image_pil: Image.Image,
    ) -> str:
        """Load caption from .txt file or generate with moondream"""
        # Check for .txt file
        txt_path = image_path.with_suffix(".txt")
        if txt_path.exists():
            return txt_path.read_text().strip()

        # Generate caption if captioner available
        if self.captioner is not None:
            try:
                # Moondream captioning
                enc_image = self.captioner.encode_image(image_pil)
                caption = self.captioner.answer_question(
                    enc_image,
                    "Describe this image in detail.",
                    self.caption_tokenizer,
                )
                return caption
            except Exception as e:
                self.rank_print(f"Caption generation failed: {e}")
                return "an image"

        return "an image"

    def generate_captions_parallel(
        self,
        image_paths: list[Path],
        trigger_word: str = "",
    ) -> list[str]:
        """Generate captions in parallel across GPUs using DistributedSampler"""

        # Split image paths across ranks using DistributedSampler logic
        num_images = len(image_paths)
        images_per_rank = num_images // self.world_size
        start_idx = self.rank * images_per_rank

        # Last rank takes any remaining images
        if self.rank == self.world_size - 1:
            end_idx = num_images
        else:
            end_idx = start_idx + images_per_rank

        local_image_paths = image_paths[start_idx:end_idx]

        self.rank_print(
            f"Generating captions for {len(local_image_paths)} images (indices {start_idx}:{end_idx})..."
        )

        # Generate captions for local subset
        local_captions = []
        for img_path in local_image_paths:
            img_pil = Image.open(img_path).convert("RGB")
            caption = self.load_or_generate_caption(img_path, img_pil)

            # Inject trigger word
            if trigger_word:
                caption = f"{trigger_word} {caption}"

            local_captions.append(caption)

        self.rank_print(f"Generated {len(local_captions)} captions")

        # Gather all captions to rank 0 in the correct order
        # Use all_gather_object to collect from all ranks
        # IMPORTANT: Set the correct CUDA device before gather to avoid device mismatch
        torch.cuda.set_device(self.device)
        all_captions_nested = [None] * self.world_size
        dist.all_gather_object(all_captions_nested, local_captions)

        # Only rank 0 needs to flatten and return
        if self.rank == 0:
            # Flatten the nested list to get all captions in order
            all_captions = []
            for rank_captions in all_captions_nested:
                all_captions.extend(rank_captions)
            return all_captions

        return []  # Non-rank-0 returns empty (will get broadcast later)

    def encode_images_parallel(
        self,
        image_paths: list[Path],
        resolution: int,
    ) -> torch.Tensor:
        """Encode images to latents using VAE in parallel across GPUs"""

        # Manual splitting (same logic as caption generation to avoid DistributedSampler padding)
        num_images = len(image_paths)
        images_per_rank = num_images // self.world_size
        start_idx = self.rank * images_per_rank

        # Last rank takes any remaining images
        if self.rank == self.world_size - 1:
            end_idx = num_images
        else:
            end_idx = start_idx + images_per_rank

        local_image_paths = image_paths[start_idx:end_idx]

        self.rank_print(
            f"Encoding {len(local_image_paths)} images (indices {start_idx}:{end_idx})..."
        )

        latents_list = []

        with torch.no_grad():
            for img_path in local_image_paths:
                # Load and preprocess image
                image = Image.open(img_path).convert("RGB")
                image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)

                # Convert to tensor [C, H, W] normalized to [0, 1]
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
                image_tensor = image_tensor.unsqueeze(0).to(
                    self.device, dtype=torch.bfloat16
                )

                # Normalize to [-1, 1] for VAE
                image_tensor = image_tensor * 2.0 - 1.0

                # Encode with VAE
                latent_dist = self.vae.encode(image_tensor).latent_dist
                latents = latent_dist.sample() * self.vae.config.scaling_factor

                # Keep on GPU for gathering (NCCL requires GPU tensors)
                latents_list.append(latents)

        # Concatenate local results
        if latents_list:
            local_latents = torch.cat(latents_list, dim=0)
        else:
            # Empty tensor if this rank got no data
            local_latents = torch.empty(
                0, 16, resolution // 8, resolution // 8, device=self.device
            )

        self.rank_print(f"Encoded {local_latents.shape[0]} latents")

        # Gather from all ranks
        gathered_latents = self.gather_tensors(local_latents)

        return gathered_latents

    def encode_text_parallel(
        self,
        captions: list[str],
        batch_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions to embeddings using T5 and CLIP in parallel"""

        # Split captions across ranks
        captions_per_rank = len(captions) // self.world_size
        start_idx = self.rank * captions_per_rank
        end_idx = (
            start_idx + captions_per_rank
            if self.rank < self.world_size - 1
            else len(captions)
        )

        local_captions = captions[start_idx:end_idx]

        t5_embeddings_list = []
        clip_embeddings_list = []

        self.rank_print(f"Encoding {len(local_captions)} captions...")

        with torch.no_grad():
            for i in range(0, len(local_captions), batch_size):
                batch_captions = local_captions[i : i + batch_size]

                # T5 encoding
                t5_tokens = self.tokenizer(
                    batch_captions,
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                t5_outputs = self.text_encoder(**t5_tokens)
                # Keep on GPU for gathering (NCCL requires GPU tensors)
                t5_embeddings_list.append(t5_outputs.last_hidden_state)

                # CLIP encoding
                clip_tokens = self.clip_tokenizer(
                    batch_captions,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                clip_outputs = self.clip_encoder(**clip_tokens)
                # Keep on GPU for gathering (NCCL requires GPU tensors)
                clip_embeddings_list.append(clip_outputs.pooler_output)

        # Concatenate local results
        if t5_embeddings_list:
            local_t5 = torch.cat(t5_embeddings_list, dim=0)
            local_clip = torch.cat(clip_embeddings_list, dim=0)
        else:
            local_t5 = torch.empty(0, 512, 4096)
            local_clip = torch.empty(0, 768)

        # Gather from all ranks
        gathered_t5 = self.gather_tensors(local_t5)
        gathered_clip = self.gather_tensors(local_clip)

        return gathered_t5, gathered_clip

    def gather_tensors(self, local_tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensors from all ranks to rank 0"""
        if self.world_size == 1:
            return local_tensor.cpu()

        # IMPORTANT: Set correct device for NCCL operations
        torch.cuda.set_device(self.device)

        # Ensure tensor is on the correct GPU
        if local_tensor.device != self.device:
            local_tensor = local_tensor.to(self.device)

        # Get shapes from all ranks using all_gather_object (doesn't require same-sized tensors)
        all_shapes: list[Optional[list[int]]] = [None] * self.world_size
        dist.all_gather_object(all_shapes, list(local_tensor.shape))

        # Pad tensors to same size for all_gather (NCCL requires same-sized tensors)
        if any(shape is None for shape in all_shapes):
            raise RuntimeError("Failed to gather shapes from all ranks")
        shapes = [shape for shape in all_shapes if shape is not None]
        max_size = max(shape[0] for shape in shapes)

        if local_tensor.shape[0] < max_size:
            # Pad with zeros ON GPU (NCCL requires GPU tensors)
            padding_size = max_size - local_tensor.shape[0]
            padding_shape = [padding_size] + list(local_tensor.shape[1:])
            padding = torch.zeros(
                padding_shape, dtype=local_tensor.dtype, device=self.device
            )
            local_tensor_padded = torch.cat([local_tensor, padding], dim=0)
        else:
            local_tensor_padded = local_tensor

        # Gather to all ranks (tensors must be on GPU for NCCL)
        gathered_list = [
            torch.zeros_like(local_tensor_padded, device=self.device)
            for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_list, local_tensor_padded)

        # Only rank 0 needs to concatenate, remove padding, and move to CPU
        if self.rank == 0:
            # Remove padding and concatenate
            result_list = []
            for gathered, shape in zip(gathered_list, shapes):
                actual_size = shape[0]
                if actual_size > 0:
                    result_list.append(gathered[:actual_size])

            result = torch.cat(result_list, dim=0) if result_list else local_tensor
            return result.cpu()  # Move to CPU after gathering

        return torch.empty(0)  # Non-rank-0 return empty tensor

    def __call__(
        self,
        images_zip_url: str = None,
        request_id: str = None,
        trigger_word: str = "ohwx",
        resolution: int = 512,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Preprocess images for Flux LoRA training.

        Args:
            images_zip_url: URL to ZIP file with images
            request_id: Unique identifier for this preprocessing request
            trigger_word: Token to inject into captions
            resolution: Target resolution

        Returns:
            dict with preprocessed_data_path
        """
        import time as time_module

        # Debug: Check if distributed is initialized
        if dist.is_initialized():
            self.rank_print(
                f"[{time_module.time():.2f}] torch.distributed IS initialized, backend={dist.get_backend()}, rank={dist.get_rank()}, world_size={dist.get_world_size()}"
            )
        else:
            self.rank_print(
                f"[{time_module.time():.2f}] WARNING: torch.distributed is NOT initialized!"
            )

        self.rank_print(
            f"[{time_module.time():.2f}] Starting preprocessing: request_id={request_id}, trigger_word='{trigger_word}', resolution={resolution}"
        )

        # Step 1: Extract images to shared directory (rank 0 only)
        # IMPORTANT: Use /data (shared across all workers), not /tmp (isolated per worker)!
        extract_dir = Path("/data") / "flux_preprocessing" / f"request_{request_id}"

        self.rank_print(
            f"[{time_module.time():.2f}] Extract dir will be: {extract_dir}"
        )

        # Rank 0 extracts the ZIP and gets image paths
        if self.rank == 0:
            self.rank_print(
                f"[{time_module.time():.2f}] Rank 0: Creating directory {extract_dir}"
            )
            extract_dir.mkdir(exist_ok=True, parents=True)
            self.rank_print(f"[{time_module.time():.2f}] Rank 0: Directory created")

            self.rank_print(
                f"[{time_module.time():.2f}] Rank 0: Extracting images from {images_zip_url}"
            )
            image_paths = self.extract_zip(images_zip_url, str(extract_dir))
            self.rank_print(
                f"[{time_module.time():.2f}] Rank 0: Found {len(image_paths)} images"
            )
            self.rank_print(
                f"[{time_module.time():.2f}] Rank 0: Image paths: {[str(p) for p in image_paths[:3]]}..."
            )
        else:
            image_paths = None

        # Step 2: Broadcast image_paths to all ranks (no filesystem handoff!)
        self.rank_print(
            f"[{time_module.time():.2f}] Rank {self.rank}: Broadcasting image paths..."
        )
        # IMPORTANT: Set the correct CUDA device before broadcast to avoid device mismatch
        torch.cuda.set_device(self.device)
        image_paths_list = [image_paths]
        dist.broadcast_object_list(image_paths_list, src=0)
        image_paths = image_paths_list[0]
        self.rank_print(
            f"[{time_module.time():.2f}] Rank {self.rank}: Received {len(image_paths)} image paths"
        )

        # Step 3: Generate captions in PARALLEL across all GPUs
        self.rank_print(
            f"[{time_module.time():.2f}] Rank {self.rank}: Starting parallel caption generation..."
        )
        captions = self.generate_captions_parallel(image_paths, trigger_word)

        # Step 4: Broadcast captions to all ranks (rank 0 has the full list, others get it)
        self.rank_print(
            f"[{time_module.time():.2f}] Rank {self.rank}: Broadcasting captions..."
        )
        torch.cuda.set_device(self.device)
        captions_list = [captions]
        dist.broadcast_object_list(captions_list, src=0)
        captions = captions_list[0]
        self.rank_print(
            f"[{time_module.time():.2f}] Rank {self.rank}: Received {len(captions)} captions"
        )

        # Step 5: Encode images in parallel across all GPUs
        self.rank_print("Encoding images with VAE...")
        latents = self.encode_images_parallel(image_paths, resolution)

        # Step 6: Encode text in parallel across all GPUs
        self.rank_print("Encoding captions with T5/CLIP...")
        text_embeddings, pooled_embeddings = self.encode_text_parallel(
            captions, batch_size=1
        )

        # Step 7: Generate text IDs - create template that training will replicate per batch
        # Shape: [1, 512, 3] - will be expanded to [batch_size, 512, 3] during training
        text_ids = torch.zeros(1, 512, 3, dtype=torch.long)
        text_ids[0, :, 1] = torch.arange(512)

        # Step 8: Create masks (all ones - no face detection implemented)
        masks = torch.ones_like(latents)

        # Step 9: Save preprocessed data (rank 0 only)
        if self.rank == 0:
            self.rank_print("Saving preprocessed data...")

            output_data = {
                "latents": latents,
                "text_embeddings": text_embeddings,
                "pooled_embeddings": pooled_embeddings,
                "text_ids": text_ids,
                "masks": masks,
            }

            # Save to /data (shared across workers)
            output_path = (
                Path("/data") / "flux_preprocessing" / f"preprocessed_{request_id}.pt"
            )
            torch.save(output_data, output_path)

            self.rank_print(f"Preprocessing complete! Saved to {output_path}")

            # Cleanup extract directory
            import shutil

            try:
                shutil.rmtree(extract_dir)
                self.rank_print(f"Cleaned up {extract_dir}")
            except Exception as e:
                self.rank_print(f"Warning: Could not cleanup {extract_dir}: {e}")

            return {
                "preprocessed_data_path": str(output_path),
                "num_images": len(image_paths),
            }

        return {}
