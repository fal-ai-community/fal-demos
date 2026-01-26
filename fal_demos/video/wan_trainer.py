"""
WAN LoRA Training Demo
"""

import json
import math
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import fal
from fal.exceptions import FieldException
from fal.toolkit import File, clone_repository, download_file
from fal.toolkit.file.providers.fal import FalFileRepositoryV3
from fal.toolkit.utils.download_utils import FAL_MODEL_WEIGHTS_DIR, _hash_url
from fastapi import Request, Response
from pydantic import BaseModel, Field

MODEL_TYPE = Literal["T2V 1.3B"]
VIDEO_CLIP_MODE_TYPE = Literal[
    "single_beginning", "single_middle", "multiple_overlapping"
]
SIZE_BUCKET_TYPE = tuple[int, int, int]

APP_NAME = "wan-lora-trainer-demo"
MACHINE_TYPE = "GPU-H100"
NUM_GPUS = 1
ALLOWED_MEDIA_FILES = [".png", ".jpg", ".jpeg", ".gif", ".mp4"]

DEFAULT_CUDA_HOME = "/usr/local/cuda-12.4"
DEFAULT_VIDEO_CLIP_MODE: VIDEO_CLIP_MODE_TYPE = "single_beginning"
DEFAULT_RANK = 16
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_NUM_STEPS = 400
MAXIMUM_NUM_STEPS = 5000
MAXIMUM_NUMBER_OF_FILES = 200

MODEL_DIR = os.path.join(FAL_MODEL_WEIGHTS_DIR, f"{APP_NAME}-weights")
DATASET_DIR = tempfile.mkdtemp()
OUTPUT_DIR = tempfile.mkdtemp()
REPO_DIR = tempfile.mkdtemp()

IMAGE_240: SIZE_BUCKET_TYPE = (416, 240, 1)
VIDEO_240: SIZE_BUCKET_TYPE = (416, 240, 81)
MODEL_SIZE_BUCKETS: dict[MODEL_TYPE, list[SIZE_BUCKET_TYPE]] = {
    "T2V 1.3B": [IMAGE_240, VIDEO_240],
}

VAE_FILE_NAME = "Wan2.1_VAE.pth"
TEXT_ENCODER_FILE_NAME = "models_t5_umt5-xxl-enc-bf16.pth"
IMAGE_ENCODER_FILE_NAME = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
DIFFUSION_PIPE_REPO_URL = "https://github.com/painebenjamin/diffusion-pipe-trainer.git"
DIFFUSION_PIPE_REVISION = "dffaafadcc92385606762bd116c9c3205b97ff37"
MODEL_REVISIONS: dict[MODEL_TYPE, str] = {
    "T2V 1.3B": "37ec512624d61f7aa208f7ea8140a131f93afc9a",
}
SHARED_MODEL_REPO_ID = "Wan-AI/Wan2.1-I2V-14B-720P"
SHARED_MODEL_REVISION = "8823af45fcc58a8aa999a54b04be9abc7d2aac98"

DEPENDENCIES = [
    "accelerate==1.4.0",
    "av==14.2.0",
    "bitsandbytes==0.45.3",
    "datasets==3.3.2",
    "deepspeed==0.14.5",
    "diffusers==0.32.2",
    "easydict==1.13",
    "einops==0.8.1",
    "ftfy==6.3.1",
    "flash-attn@https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
    "imageio==2.37.0",
    "imageio-ffmpeg==0.6.0",
    "omegaconf==2.3.0",
    "peft==0.14.0",
    "pillow==10.4.0",
    "protobuf==6.31.1",
    "psutil==7.0.0",
    "safetensors==0.5.3",
    "sentencepiece==0.2.0",
    "tensorboard==2.19.0",
    "termcolor==2.5.0",
    "toml==0.10.2",
    "torch==2.6.0",
    "torch-optimi==0.2.1",
    "torchvision==0.21.0",
    "tqdm==4.67.1",
    "transformers==4.49.0",
    "triton==3.2.0",
    "wandb",
]

DATASET_CONFIG_TEMPLATE = """
size_buckets = [{size_buckets:s}]
enable_ar_bucket = false

[[directory]]
path = '{dataset_path:s}'
num_repeats = 1
""".strip()

TRAINING_CONFIG_TEMPLATE = """
output_dir = '{output_dir:s}'
dataset = '{dataset_config_file:s}'
epochs = 10000
save_every_n_epochs = 10000

num_steps = {num_steps:d}
save_every_n_steps = 100000000

activation_checkpointing = true
caching_batch_size = 1
gradient_accumulation_steps = 1
gradient_clipping = 1.0
micro_batch_size_per_gpu = 1
partition_method = 'parameters'
pipeline_stages = 1
save_dtype = 'float32'
video_clip_mode = '{video_clip_mode:s}'
warmup_steps = 5

[model]
type = 'wan'
ckpt_path = '{model_dir:s}'
dtype = 'bfloat16'
timestep_sample_method = 'logit_normal'

[adapter]
type = 'lora'
dtype = 'bfloat16'
rank = {rank:d}

[optimizer]
type = 'adamw_optimi'
lr = {learning_rate:0.1e}
betas = [0.9, 0.99]
weight_decay = 0.01
eps = 1e-8
""".strip()


class Input(BaseModel):
    training_data_url: str = Field(
        description="URL to zip archive with images/videos. Captions optional.",
        title="Training Data URL",
        ui={"field": "multivideo"},
    )
    rank: int = Field(description="LoRA rank.", default=DEFAULT_RANK, ge=1, le=128)
    number_of_steps: int = Field(
        description="Training steps.",
        default=DEFAULT_NUM_STEPS,
        ge=100,
        le=MAXIMUM_NUM_STEPS,
    )
    learning_rate: float = Field(
        description="Learning rate.",
        default=DEFAULT_LEARNING_RATE,
        ge=1e-6,
        le=1.0,
    )
    trigger_phrase: str = Field(
        description="Trigger phrase to prepend to captions.",
        default="",
        ui={"important": True},
    )
    auto_scale_input: bool = Field(
        description="If true, scale videos to 81 frames at 16fps.",
        default=False,
        examples=[True],
        title="Auto-Scale Input",
    )
    video_clip_mode: VIDEO_CLIP_MODE_TYPE = Field(
        description="Video clip sampling mode.",
        default=DEFAULT_VIDEO_CLIP_MODE,
    )


class Output(BaseModel):
    lora_file: File = Field(description="Trained LoRA weights.")
    config_file: File = Field(
        description="Config to help set up inference endpoints."
    )


def ensure_nvcc_executable() -> None:
    """
    Ensures an 'nvcc' executable exists in CUDA_HOME for DeepSpeed.
    """
    cuda_home = os.getenv("CUDA_HOME", DEFAULT_CUDA_HOME)
    cuda_bin = os.path.join(cuda_home, "bin")
    nvcc = os.path.join(cuda_bin, "nvcc")

    os.makedirs(cuda_bin, exist_ok=True)
    if not os.path.exists(nvcc):
        with open(nvcc, "w") as fp:
            fp.write(
                """#!/usr/bin/env python
print('''nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0''')"""
            )
        os.chmod(nvcc, stat.S_IEXEC)


@fal.cached
def get_repository() -> FalFileRepositoryV3:
    """
    Gets a singleton repository instance.
    """
    return FalFileRepositoryV3()


def upload_file(path: str, request: Request) -> File:
    """
    Uploads a file to the FAL repository.
    """
    fal_file = File.from_path(
        path=path,
        request=request,
        repository=get_repository(),
    )

    presign_removed_url = fal_file.url
    try:
        weights_dir = Path(FAL_MODEL_WEIGHTS_DIR / _hash_url(presign_removed_url))
        weights_dir.mkdir(parents=True, exist_ok=True)
        cache_path = weights_dir / os.path.basename(path)
        shutil.copy2(path, cache_path)
    except Exception as e:
        print("Failed to precache file", e)

    return fal_file


def download_diffusion_pipe() -> Path:
    """
    Downloads diffusion pipe.
    """
    try:
        return clone_repository(
            DIFFUSION_PIPE_REPO_URL,
            commit_hash=DIFFUSION_PIPE_REVISION,
            target_dir=REPO_DIR,
        )
    except Exception as e:
        raise FieldException(
            "trainer_repo",
            "Failed to download diffusion-pipe-trainer repository.",
        ) from e


def download_model_weights() -> None:
    """
    Downloads WAN model weights.
    """
    import huggingface_hub

    model_dir = os.path.join(MODEL_DIR, "Wan2.1-T2V-1.3B")
    shared_dir = os.path.join(MODEL_DIR, "shared")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(shared_dir, exist_ok=True)
    try:
        huggingface_hub.snapshot_download(
            repo_id="Wan-AI/Wan2.1-T2V-1.3B",
            local_dir=model_dir,
            revision=MODEL_REVISIONS["T2V 1.3B"],
            allow_patterns=[
                "diffusion_pytorch_model*",
                "config.json",
                "xlm-roberta-large/*",
                "google/*",
            ],
        )
        huggingface_hub.snapshot_download(
            repo_id=SHARED_MODEL_REPO_ID,
            local_dir=shared_dir,
            revision=SHARED_MODEL_REVISION,
            allow_patterns=[
                VAE_FILE_NAME,
                TEXT_ENCODER_FILE_NAME,
                IMAGE_ENCODER_FILE_NAME,
            ],
        )

        for shared_file in [
            VAE_FILE_NAME,
            TEXT_ENCODER_FILE_NAME,
            IMAGE_ENCODER_FILE_NAME,
        ]:
            shared_path = os.path.join(shared_dir, shared_file)
            symlink_path = os.path.join(model_dir, shared_file)
            if os.path.exists(shared_path) and not os.path.exists(symlink_path):
                os.symlink(shared_path, symlink_path)
    except Exception as e:
        raise FieldException(
            "model",
            "Failed to download WAN model weights from Hugging Face.",
        ) from e


def get_media_files_from_dir(directory: str) -> list[str]:
    """
    Finds all media files in a directory.
    """
    media_files: list[str] = []
    for dirname, subdirs, filenames in os.walk(directory):
        if "__macosx" in dirname.lower():
            continue
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext.lower() in ALLOWED_MEDIA_FILES:
                media_files.append(os.path.join(dirname, filename))
    return media_files


def recursively_unpack_archive(
    archive_path: str,
    target_dir: str,
    archive_format: str | None = None,
    remove_after_unpack: bool = True,
) -> None:
    """
    Unpacks an archive, then recursively unpacks nested archives.
    """
    if archive_format is None:
        archive_format = os.path.splitext(os.path.basename(archive_path))[1][1:]

    try:
        shutil.unpack_archive(archive_path, target_dir, format=archive_format)
    except Exception as ex:
        print(f"Failed to unpack archive {os.path.basename(archive_path)}: {ex}")
        return

    child_archives: list[str] = []
    for dirname, subdirs, filenames in os.walk(target_dir):
        if "__macosx" in dirname.lower():
            continue
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() == ".zip":
                child_archives.append(os.path.join(dirname, filename))

    for child_archive in child_archives:
        child_archive_name = os.path.splitext(os.path.basename(child_archive))[0]
        child_archive_dir = os.path.dirname(child_archive)
        child_target_dir = os.path.join(child_archive_dir, child_archive_name)
        os.makedirs(child_target_dir, exist_ok=True)
        recursively_unpack_archive(
            archive_path=child_archive,
            target_dir=child_target_dir,
            remove_after_unpack=remove_after_unpack,
        )

    if remove_after_unpack:
        os.remove(archive_path)


def fit_to_frame_count_fps(
    input_file: str,
    output_file: str,
    target_frames: int = 81,
    target_fps: int = 16,
) -> None:
    """
    Convert a video to exactly target_frames at target_fps.
    """
    duration_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_file,
    ]
    duration = float(subprocess.check_output(duration_cmd).decode("utf-8").strip())

    frame_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_file,
    ]
    try:
        frame_count = int(subprocess.check_output(frame_cmd).decode("utf-8").strip())
    except Exception:
        fps_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            input_file,
        ]
        fps_str = subprocess.check_output(fps_cmd).decode("utf-8").strip()
        num, den = (
            map(int, fps_str.split("/")) if "/" in fps_str else (float(fps_str), 1)
        )
        fps = num / den
        frame_count = int(duration * fps)

    if frame_count < target_frames:
        extract_cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-vf",
            f"fps={target_frames}/{duration}",
            "-vsync",
            "vfr",
            "-q:v",
            "2",
            "-y",
            f"{output_file}.frames/frame_%04d.jpg",
        ]
    else:
        extract_cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-vf",
            f"select=not(mod(n\\,{max(int(frame_count/target_frames), 1)})),setpts=N/({target_fps}*TB)",
            "-vsync",
            "vfr",
            "-q:v",
            "2",
            "-frames:v",
            str(target_frames),
            "-y",
            f"{output_file}.frames/frame_%04d.jpg",
        ]

    temp_dir = Path(f"{output_file}.frames")
    temp_dir.mkdir(exist_ok=True)

    subprocess.run(extract_cmd, check=False)

    create_cmd = [
        "ffmpeg",
        "-framerate",
        str(target_fps),
        "-i",
        f"{temp_dir}/frame_%04d.jpg",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-y",
        output_file,
    ]
    subprocess.run(create_cmd, check=False)

    for file in temp_dir.glob("frame_*.jpg"):
        file.unlink()


def prepare_dataset_files(
    source_directory: str,
    target_directory: str = DATASET_DIR,
    raise_when_no_files: bool = True,
    copy_files: bool = False,
    trigger_phrase: str = "",
    fit_videos: bool = True,
) -> None:
    """
    Flattens media files into a dataset directory and writes captions.
    """
    os.makedirs(target_directory, exist_ok=True)
    num_media = 0

    for media_file in get_media_files_from_dir(source_directory):
        media_filename = os.path.basename(media_file)
        media_name = os.path.splitext(media_filename)[0]

        num_media += 1
        target_path = os.path.join(target_directory, media_filename)
        target_caption_path = os.path.join(target_directory, f"{media_name}.txt")

        if copy_files:
            shutil.copy(media_file, target_path)
        else:
            os.rename(media_file, target_path)

        caption = ""
        maybe_caption_file = os.path.join(
            os.path.dirname(media_file), f"{media_name}.txt"
        )
        if os.path.exists(maybe_caption_file):
            try:
                caption = open(maybe_caption_file).read().strip()
            except Exception as ex:
                raise FieldException(
                    "training_data_url",
                    f"Caught exception reading caption file {maybe_caption_file}: {ex}",
                )
        if trigger_phrase:
            caption = f"{trigger_phrase} {caption}".strip()
        with open(target_caption_path, "w") as fp:
            fp.write(caption)

        if (
            fit_videos
            and os.path.splitext(media_file)[1].lower()
            in [".mp4", ".mov", ".avi", ".mkv"]
        ):
            fit_to_frame_count_fps(
                target_path, target_path, target_frames=81, target_fps=16
            )

    if num_media == 0 and raise_when_no_files:
        raise FieldException(
            "training_data_url",
            f"No trainable media was found. Acceptable file types are {ALLOWED_MEDIA_FILES}",
        )


def download_dataset(dataset_url: str) -> str:
    """
    Downloads a dataset archive.
    """
    try:
        return str(download_file(dataset_url, target_dir=tempfile.gettempdir()))
    except Exception as e:
        raise FieldException(
            "training_data_url",
            "Failed to download training dataset.",
        ) from e


def write_diffusion_pipe_config(
    model_type: MODEL_TYPE,
    num_steps: int = DEFAULT_NUM_STEPS,
    num_steps_per_save: int = DEFAULT_NUM_STEPS,
    video_clip_mode: VIDEO_CLIP_MODE_TYPE = DEFAULT_VIDEO_CLIP_MODE,
    output_dir: str = OUTPUT_DIR,
    rank: int = DEFAULT_RANK,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    config_dir: str = tempfile.gettempdir(),
    config_name: str = "config.toml",
    dataset_config_name: str = "dataset.toml",
) -> str:
    """
    Writes the diffusion pipe configuration.
    """
    size_buckets = ",".join(
        [str(list(bucket)) for bucket in MODEL_SIZE_BUCKETS[model_type]]
    )
    dataset_config = DATASET_CONFIG_TEMPLATE.format(
        dataset_path=DATASET_DIR, size_buckets=size_buckets
    )
    dataset_config_name = f"{uuid4().hex}_{dataset_config_name}"
    dataset_config_path = os.path.join(config_dir, dataset_config_name)

    with open(dataset_config_path, "w") as fp:
        fp.write(dataset_config)

    training_config = TRAINING_CONFIG_TEMPLATE.format(
        output_dir=output_dir,
        dataset_config_file=dataset_config_path,
        num_steps=num_steps,
        num_steps_per_save=num_steps_per_save,
        video_clip_mode=video_clip_mode,
        model_dir=os.path.join(MODEL_DIR, "Wan2.1-T2V-1.3B"),
        rank=rank,
        learning_rate=learning_rate,
    )
    config_name = f"{uuid4().hex}_{config_name}"
    training_config_path = os.path.join(config_dir, config_name)

    with open(training_config_path, "w") as fp:
        fp.write(training_config)

    return training_config_path


def train(
    repo_dir: str,
    dataset_directory_archive_or_url: str,
    num_steps: int = DEFAULT_NUM_STEPS,
    num_steps_per_save: int = -1,
    rank: int = DEFAULT_RANK,
    video_clip_mode: VIDEO_CLIP_MODE_TYPE = DEFAULT_VIDEO_CLIP_MODE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    return_last_n_checkpoints: int = 1,
    trigger_phrase: str = "",
    fit_videos: bool = True,
    min_files: int = 1,
) -> list[tuple[str, str]]:
    """
    Executes a training run and returns the final trained checkpoint.
    """
    copy_files = True

    if dataset_directory_archive_or_url.startswith("http://") or dataset_directory_archive_or_url.startswith("https://"):
        copy_files = False
        dataset_directory_archive_or_url = download_dataset(
            dataset_directory_archive_or_url
        )

    if os.path.isfile(dataset_directory_archive_or_url):
        copy_files = False
        archive_filename = os.path.basename(dataset_directory_archive_or_url)
        archive_name = os.path.splitext(archive_filename)[0]
        archive_dir = os.path.join(os.path.dirname(archive_filename), archive_name)
        os.makedirs(archive_dir, exist_ok=True)
        recursively_unpack_archive(
            dataset_directory_archive_or_url, target_dir=archive_dir
        )
        dataset_directory_archive_or_url = archive_dir

    if not os.path.isdir(dataset_directory_archive_or_url):
        raise FieldException(
            "training_data_url",
            "Could not determine format of passed training dataset.",
        )

    media_files = get_media_files_from_dir(dataset_directory_archive_or_url)
    num_files = len(media_files)
    if num_files > MAXIMUM_NUMBER_OF_FILES:
        raise FieldException(
            "training_data_url",
            "The number of files in the dataset is greater than the maximum number of files.",
        )
    if num_files < min_files:
        num_copies = math.ceil(min_files / max(num_files, 1))
        for _ in range(num_copies):
            for source_path in media_files:
                source_dir = os.path.dirname(source_path)
                stub, ext = os.path.splitext(os.path.basename(source_path))
                if ext.lower() not in ALLOWED_MEDIA_FILES:
                    continue
                target_path = os.path.join(
                    source_dir, f"{stub}-copy-{uuid4().hex}{ext}"
                )
                shutil.copy2(source_path, target_path)
                maybe_caption_file = os.path.join(source_dir, f"{stub}.txt")
                if os.path.exists(maybe_caption_file):
                    caption_target_path = os.path.join(
                        source_dir, f"{stub}-copy-{uuid4().hex}.txt"
                    )
                    shutil.copy2(maybe_caption_file, caption_target_path)

    prepare_dataset_files(
        dataset_directory_archive_or_url,
        copy_files=copy_files,
        trigger_phrase=trigger_phrase,
        fit_videos=fit_videos,
    )

    try:
        config_file_path = write_diffusion_pipe_config(
            learning_rate=learning_rate,
            model_type="T2V 1.3B",
            num_steps=num_steps,
            num_steps_per_save=(
                num_steps if num_steps_per_save <= 0 else num_steps_per_save
            ),
            rank=rank,
            video_clip_mode=video_clip_mode,
        )
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.chdir(repo_dir)

        process = subprocess.Popen(
            f"deepspeed train.py --config {config_file_path} --deepspeed",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in iter(process.stdout.readline, ""):  # type: ignore
            print(line, end="")

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(
                return_code,
                f"deepspeed train.py --config {config_file_path} --deepspeed",
            )

        run_output_dir_names = os.listdir(OUTPUT_DIR)
        run_output_dir_names.sort()
        if not run_output_dir_names:
            raise RuntimeError("Training failed to start.")

        run_output_dir = os.path.join(OUTPUT_DIR, run_output_dir_names[-1])
        checkpoints: dict[int, tuple[str, str]] = {}
        for subfolder in os.listdir(run_output_dir):
            if subfolder.startswith("step"):
                step_num = int(subfolder[4:])
                model_path = os.path.join(
                    run_output_dir, subfolder, "adapter_model.safetensors"
                )
                config_path = os.path.join(
                    run_output_dir, subfolder, "adapter_config.json"
                )
                checkpoints[step_num] = (model_path, config_path)

        saved_steps = sorted(checkpoints.keys())
        if not saved_steps:
            raise RuntimeError("Training failed - no results were produced.")

        return [checkpoints[step] for step in saved_steps][-return_last_n_checkpoints:]
    finally:
        shutil.rmtree(DATASET_DIR)


class WanLoRATrainerDemo(
    fal.App,
    keep_alive=0,
    request_timeout=43200,
    name=APP_NAME,
    _scheduler_options={"region": ["us-central", "us-west", "us-east", "eu-north"]},
):  # type: ignore
    machine_type = MACHINE_TYPE
    num_gpus = NUM_GPUS
    requirements = DEPENDENCIES

    repo_dir: Path

    def setup(self) -> None:
        """
        Setup dependencies and model weights.
        """
        self.repo_dir = download_diffusion_pipe()
        download_model_weights()
        ensure_nvcc_executable()
        os.environ["CUDA_HOME"] = os.getenv("CUDA_HOME", DEFAULT_CUDA_HOME)

    @fal.endpoint("/")
    def run(self, request: Input, response: Response, http_request: Request) -> Output:
        """
        Runs training on the WAN 1.3B model.
        """
        minimum_media_files = NUM_GPUS
        step_count = max(1, request.number_of_steps // max(NUM_GPUS, 1))
        learning_rate = request.learning_rate * max(NUM_GPUS, 1) ** 0.5

        [(model_path, _cfg)] = train(
            repo_dir=str(self.repo_dir),
            dataset_directory_archive_or_url=request.training_data_url,
            num_steps=request.number_of_steps,
            rank=request.rank,
            num_steps_per_save=step_count,
            learning_rate=learning_rate,
            trigger_phrase=request.trigger_phrase,
            fit_videos=request.auto_scale_input,
            min_files=minimum_media_files,
            video_clip_mode=request.video_clip_mode,
        )

        response.headers["x-fal-billable-units"] = str(
            int(max(100, request.number_of_steps))
        )

        config: dict[str, Any] = {"instance_prompt": request.trigger_phrase}
        config_file_path = os.path.join(
            tempfile.gettempdir(), f"{uuid4().hex}-config.json"
        )
        with open(config_file_path, "w") as fp:
            fp.write(json.dumps(config))

        return Output(
            lora_file=upload_file(model_path, request=http_request),
            config_file=upload_file(config_file_path, request=http_request),
        )


if __name__ == "__main__":
    app = fal.wrap_app(WanLoRATrainerDemo)
    app()
