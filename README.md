# Private Model Hosting Demos with Fal

Welcome to the Fal Private Serverless model hosting demos! This collection of examples is designed to guide you through the process of deploying various types of machine learning models on the fal platform. Each demo showcases different features and best practices for creating robust, scalable, and user-friendly model APIs. We will explore image generation, 3D object creation, text-to-speech synthesis, text-to-video generation, and music generation.

## Core Concepts in fal private deployments 

Before diving into specific demos, let's touch upon some core concepts you'll see repeatedly:

* **`fal.App`**: This is the fundamental class you'll use to define your app. It encapsulates your model, its dependencies, and the inference logic. You configure aspects like `name`, `machine_type`, `requirements`, `keep_alive`, `min_concurrency`, and `max_concurrency` directly on this class.
* **Input and Output Models**: Using `pydantic.BaseModel`, you define structured inputs and outputs for your API. The `Field` object allows you to provide rich metadata like titles, descriptions, examples, and validation rules (e.g., `ge`, `le` for numerical ranges, `Literal` for specific choices). This metadata is automatically used to generate an interactive playground and OpenAPI specification for your model.
* **`setup()` Method**: This special method within your `fal.App` class is called once when a new worker instance starts. It's the ideal place to load your models, download any necessary assets, and perform one-time initializations. This ensures that these heavy operations don't slow down individual inference requests.
* **`@fal.endpoint()` Decorator**: This decorator is used to expose methods of your `fal.App` class as HTTP endpoints. You can define multiple endpoints within a single app, each potentially serving a different variant or functionality of your model.
* **Requirements**: You specify your Python package dependencies in a `requirements` list within the `fal.App` class. It's crucial to pin package versions (e.g., `torch==2.6.0`) and even commit hashes for dependencies installed from Git repositories to ensure consistent and reproducible environments.
* **Machine Types**: fal offers various machine types (e.g., `GPU-H100`, `M` for CPU-medium). You select the appropriate `machine_type` in your `fal.App` configuration based on your model's computational needs.
* **Billing**: For many models, especially generative ones, you might want to implement custom billing logic. This can often be done by setting the `x-fal-billable-units` header in the `fastapi.Response` object. The actual cost per unit is configured in your Fal dashboard.
* **Error Handling**: Fal provides `FieldException` for input validation errors that are nicely displayed in the UI, and you can use FastAPI's `HTTPException` for other types of errors.
* **`fal.toolkit`**: This toolkit provides helpful utilities for common tasks, such as image handling (`Image`, `ImageSizeInput`), file operations (`File`), and more.
* **Secrets Management**: For sensitive information like API keys, fal allows you to set secrets (e.g., via `fal secrets set MY_SECRET_KEY <value>`) which can then be accessed as environment variables within your app.
* **Custom Docker Images**: For complex dependencies that go beyond Python packages (e.g., system libraries via `apt-get`), fal supports deploying applications using custom Docker images.

Now, let's explore each demo in detail.

---

## 1. Image Generation with Sana

The [`fal_demos/image/sana.py`](fal_demos/image/sana.py) demo showcases how to host a text-to-image generation model, specifically using the Sana and SanaSprint pipelines. It demonstrates robust input/output definitions, model loading, safety checking, and custom billing.

**Overview**:
This application provides two endpoints for generating images from text prompts: a standard quality endpoint and a "sprint" endpoint optimized for faster generation with fewer inference steps.

**Key fal Concepts Demonstrated**:

The definition of input data structures is a crucial first step. The `BaseInput` class, found at [`fal_demos/image/sana.py#L13`](fal_demos/image/sana.py#L13), serves as a common foundation for image generation parameters. Each field, like `prompt` ([`fal_demos/image/sana.py#L15`](fal_demos/image/sana.py#L15)) or `num_inference_steps` ([`fal_demos/image/sana.py#L31`](fal_demos/image/sana.py#L31)), is defined using `pydantic.Field`, complete with titles, descriptions, and illustrative examples. This significantly enhances the usability of the auto-generated API playground. For instance, the `image_size` field ([`fal_demos/image/sana.py#L26`](fal_demos/image/sana.py#L26)) utilizes `ImageSizeInput` from `fal.toolkit.image`, which is then converted to an `ImageSize` object internally.

The application defines specific input models for its two endpoints: `TextToImageInput` ([`fal_demos/image/sana.py#L65`](fal_demos/image/sana.py#L65)) for the standard generation and `SprintInput` ([`fal_demos/image/sana.py#L69`](fal_demos/image/sana.py#L69)) for the faster "sprint" version. The `SprintInput` class inherits from `BaseInput` but overrides `num_inference_steps` with a different default and validation range suitable for quicker inference.

Similarly, output structures are defined with `SanaOutput` ([`fal_demos/image/sana.py#L78`](fal_demos/image/sana.py#L78)) and `SanaSprintOutput` ([`fal_demos/image/sana.py#L95`](fal_demos/image/sana.py#L95)). These classes specify the format of the response, including a list of generated `Image` objects, also from `fal.toolkit.image`, and other metadata like the `seed` used.

The core of the application is the `Sana` class ([`fal_demos/image/sana.py#L111`](fal_demos/image/sana.py#L111)), which inherits from `fal.App`. Here, serverless execution parameters are configured: `keep_alive=600` ensures a worker stays warm for 10 minutes after the last request, `min_concurrency=0` allows the app to scale down to zero workers, `max_concurrency=10` limits the number of concurrent workers, and `name="sana"` defines the application's name, influencing its endpoint URL.

Dependencies are explicitly listed in the `requirements` attribute ([`fal_demos/image/sana.py#L119`](fal_demos/image/sana.py#L119)), including `torch`, `diffusers` (pinned to a specific commit hash for stability), and other necessary packages. The `local_python_modules` attribute ([`fal_demos/image/sana.py#L126`](fal_demos/image/sana.py#L126)) ensures that local modules like `fal_demos` are correctly packaged. The application specifies `machine_type = "GPU-H100"` ([`fal_demos/image/sana.py#L128`](fal_demos/image/sana.py#L128)) to leverage powerful GPU acceleration.

The model loading logic resides in the `setup()` method ([`fal_demos/image/sana.py#L129`](fal_demos/image/sana.py#L129)). Inside this method, `SanaPipeline` and `SanaSprintPipeline` are imported and initialized. Notably, the text encoder from the base `SanaPipeline` is reused for the `SanaSprintPipeline` ([`fal_demos/image/sana.py#L142`](fal_demos/image/sana.py#L142)) to optimize memory usage and loading time. These initialized pipelines are stored in `self.pipes` for access during inference.

A private helper method, `_generate()` ([`fal_demos/image/sana.py#L148`](fal_demos/image/sana.py#L148)), encapsulates the common image generation logic for both endpoints. This method handles input preprocessing, seed generation, model invocation, and postprocessing. An important aspect demonstrated here is the safety checker integration using `postprocess_images` from `fal.toolkit.image.safety_checker` ([`fal_demos/image/sana.py#L176`](fal_demos/image/sana.py#L176)).

Custom billing is implemented by calculating `resolution_factor` based on the image dimensions and then setting the `x-fal-billable-units` header on the `fastapi.Response` object ([`fal_demos/image/sana.py#L181-L186`](fal_demos/image/sana.py#L181-L186)).

Finally, two public endpoints are defined: `generate` at the root path ([`fal_demos/image/sana.py#L198`](fal_demos/image/sana.py#L198)) and `generate_sprint` at the `/sprint` path ([`fal_demos/image/sana.py#L207`](fal_demos/image/sana.py#L207)). Both endpoints utilize the `_generate` method, passing the appropriate input model and model identifier.

The comments at the end of the file ([`fal_demos/image/sana.py#L214-L219`](fal_demos/image/sana.py#L214-L219)) provide instructions on how to run the app using `fal run` and example URLs for testing the deployed endpoints.

---

## 2. 3D Object Generation with Hunyuan3D (via Fal Client)

The [`fal_demos/image/hunyuan3d.py`](fal_demos/image/hunyuan3d.py) demo illustrates a different pattern: hosting a "proxy" application that, in turn, calls another pre-existing Fal function (in this case, `fal-ai/hunyuan3d`) to perform the actual 3D model generation. This is useful for adding custom preprocessing, postprocessing, or simplified interfaces around existing models.

**Overview**:
This application takes an input image URL and parameters, submits a job to an internal Hunyuan3D model on Fal, and then streams back the results, including the generated 3D object file.

**Key fal Concepts Demonstrated**:

The input parameters for the 3D generation are defined in the `Hunyuan3DInput` model ([`fal_demos/image/hunyuan3d.py#L9`](fal_demos/image/hunyuan3d.py#L9)), including the `input_image_url`, whether to generate a `textured_mesh`, and other model-specific settings. The `ObjectOutput` model ([`fal_demos/image/hunyuan3d.py#L49`](fal_demos/image/hunyuan3d.py#L49)) describes the expected output, primarily a `File` object representing the 3D mesh.

The `Hunyuan3D` class ([`fal_demos/image/hunyuan3d.py#L65`](fal_demos/image/hunyuan3d.py#L65)) is configured with `max_multiplexing=10`. This setting allows a single worker instance to handle up to 10 concurrent requests by interleaving their processing, which is particularly efficient for I/O-bound tasks or when waiting for other services.

A key feature demonstrated is the use of Fal secrets. The `setup()` method ([`fal_demos/image/hunyuan3d.py#L72`](fal_demos/image/hunyuan3d.py#L72)) retrieves a secret key named `MY_SECRET_KEY` from environment variables and sets it as `FAL_KEY`. This is a common pattern for authenticating with other services or Fal functions. The comment indicates how to set this secret using the `fal secrets set` command.

The main logic is within the `generate_image` endpoint ([`fal_demos/image/hunyuan3d.py#L77`](fal_demos/image/hunyuan3d.py#L77)). This function uses `fal_client.submit_async` ([`fal_demos/image/hunyuan3d.py#L80`](fal_demos/image/hunyuan3d.py#L80)) to make an asynchronous call to the `fal-ai/hunyuan3d/v2/mini/turbo` function, passing along the user-provided arguments.

The application then iterates over events from the submitted job using `handle.iter_events` ([`fal_demos/image/hunyuan3d.py#L90`](fal_demos/image/hunyuan3d.py#L90)). This allows for real-time feedback, printing status updates like "Queued" or "InProgress" along with any logs.

Comprehensive error handling for the Fal client interaction is shown in the `try-except` block ([`fal_demos/image/hunyuan3d.py#L102-L120`](fal_demos/image/hunyuan3d.py#L102-L120)). It catches `FieldException` (for user input errors), `HTTPStatusError` (for network issues during the call), and `fal_client.client.FalClientError` (for other errors from the Fal platform). For user errors, it re-raises them as an `HTTPException` with a 400 status code, and for server errors, a 500 status code, providing meaningful error messages to the client.

---

## 3. Text-to-Speech with Kokoro

The [`fal_demos/tts/kokoro.py`](fal_demos/tts/kokoro.py) demo focuses on hosting a text-to-speech (TTS) model, Kokoro, which supports multiple languages and voices. It demonstrates how to manage different input schemas for various language options and how to load multiple model variants.

**Overview**:
This application provides several TTS endpoints, each tailored for a specific language or accent (American English, British English, Japanese). Users provide text and select a voice, and the app generates an audio file.

**Key fal Concepts Demonstrated**:

To cater to different languages and their specific voices, multiple Pydantic input models are defined: `AmEnglishRequest` ([`fal_demos/tts/kokoro.py#L9`](fal_demos/tts/kokoro.py#L9)), `BrEnglishRequest` ([`fal_demos/tts/kokoro.py#L48`](fal_demos/tts/kokoro.py#L48)), and `JapaneseRequest` ([`fal_demos/tts/kokoro.py#L71`](fal_demos/tts/kokoro.py#L71)). A notable feature here is the use of `typing.Literal` for the `voice` field in each request model (e.g., [`fal_demos/tts/kokoro.py#L21`](fal_demos/tts/kokoro.py#L21)). This restricts the input to a predefined set of valid voice IDs, providing clear options to the user and simplifying validation. Corresponding output models (`AmEngOutput` ([`fal_demos/tts/kokoro.py#L92`](fal_demos/tts/kokoro.py#L92)), `BrEngOutput`, `JapaneseOutput`) define the structure of the response, which includes an `audio` field of type `fal.toolkit.File`.

The `Kokoro` application class is defined at [`fal_demos/tts/kokoro.py#L125`](fal_demos/tts/kokoro.py#L125). The `requirements` list ([`fal_demos/tts/kokoro.py#L132`](fal_demos/tts/kokoro.py#L132)) includes the `kokoro` library and its language-specific components like `misaki[en]` and `misaki[ja]`, as well as a Spacy model downloaded from a URL. The `machine_type` is set to `"L"` (CPU-Large) ([`fal_demos/tts/kokoro.py#L141`](fal_demos/tts/kokoro.py#L141)), as Kokoro is relatively lightweight and runs efficiently on CPU.

In the `setup()` method ([`fal_demos/tts/kokoro.py#L143`](fal_demos/tts/kokoro.py#L143)), instances of `kokoro.KPipeline` are created for each supported language ("American English", "British English", "Japanese") and stored in a dictionary `self.pipelines`. This pre-loads the necessary models for each language variant.

The shared TTS generation logic is in the `_generate()` method ([`fal_demos/tts/kokoro.py#L149`](fal_demos/tts/kokoro.py#L149)). This method first checks if the input prompt length exceeds a certain limit (20,000 characters) and raises a `FieldException` if it does ([`fal_demos/tts/kokoro.py#L154-L158`](fal_demos/tts/kokoro.py#L154-L158)), providing user-friendly feedback. It then uses the appropriate pipeline from `self.pipelines` to generate the audio. The generated audio, a NumPy array, is concatenated if produced in chunks.

Custom billing is implemented by setting the `x-fal-billable-units` header based on the length of the input prompt, with a minimum of 1 unit and scaling every 1000 characters ([`fal_demos/tts/kokoro.py#L173`](fal_demos/tts/kokoro.py#L173)).

The generated audio is saved to a temporary WAV file using `soundfile.sf.write`. This temporary file is then returned as a `fal.toolkit.File` object, specifying `content_type="audio/wav"` and `repository="cdn"` to serve it efficiently via Fal's CDN ([`fal_demos/tts/kokoro.py#L176-L180`](fal_demos/tts/kokoro.py#L176-L180)).

Multiple endpoints are defined to serve the different language versions: `/` and `/american-english` ([`fal_demos/tts/kokoro.py#L182`](fal_demos/tts/kokoro.py#L182), [`fal_demos/tts/kokoro.py#L188`](fal_demos/tts/kokoro.py#L188)) for American English, `/british-english` ([`fal_demos/tts/kokoro.py#L194`](fal_demos/tts/kokoro.py#L194)) for British English, and `/japanese` ([`fal_demos/tts/kokoro.py#L200`](fal_demos/tts/kokoro.py#L200)) for Japanese. Each calls the `_generate` method with the appropriate language identifier.

---

## 4. Text-to-Video with Wan

The [`fal_demos/video/wan.py`](fal_demos/video/wan.py) demo tackles a more complex scenario: hosting the Wan text-to-video model. This involves cloning a Git repository, managing model weights, and integrating custom safety and prompt enhancement features by calling other Fal functions.

**Overview**:
This application allows users to generate short video clips from text prompts using the Wan 2.1 model (1.3B version). It includes optional safety checks and prompt expansion features.

**Key fal Concepts Demonstrated**:

The `WanT2VRequest` input model ([`fal_demos/video/wan.py#L10`](fal_demos/video/wan.py#L10)) defines parameters such as `prompt`, `negative_prompt`, `aspect_ratio`, and controls for enabling `enable_safety_checker` and `enable_prompt_expansion`. The `WanT2VResponse` model ([`fal_demos/video/wan.py#L67`](fal_demos/video/wan.py#L67)) specifies that the output will be a `video` (as a `File` object) and the `seed` used.

The `Wan` application class is defined at [`fal_demos/video/wan.py#L79`](fal_demos/video/wan.py#L79). Its `requirements` list ([`fal_demos/video/wan.py#L86`](fal_demos/video/wan.py#L86)) is extensive, including specific versions of `torch`, `diffusers`, and other libraries, as well as a FlashAttention wheel directly from a GitHub releases URL. This highlights Fal's flexibility in handling diverse dependency sources.

The `setup()` method ([`fal_demos/video/wan.py#L100`](fal_demos/video/wan.py#L100)) is particularly involved. It begins by cloning the Wan 2.1 Git repository using `fal.toolkit.clone_repository` ([`fal_demos/video/wan.py#L107`](fal_demos/video/wan.py#L107)). This function clones the repo to a temporary directory (specified as `/tmp/wan-t2v`), pins it to a specific `commit_hash` for reproducibility, and adds the repository's path to `sys.path` (`include_to_path=True`). The current working directory is then changed to this cloned repository's path ([`fal_demos/video/wan.py#L114`](fal_demos/video/wan.py#L114]) because the Wan library scripts expect to be run from their root directory.

Next, model weights are downloaded from Hugging Face Hub using `huggingface_hub.snapshot_download` ([`fal_demos/video/wan.py#L118`](fal_demos/video/wan.py#L118)). The weights are downloaded to a directory managed by Fal (`FAL_MODEL_WEIGHTS_DIR`) to leverage caching across worker restarts if `local_dir_use_symlinks=True` is effective and the underlying storage persists. Finally, the `WanT2V` model object is initialized using configurations and checkpoints from the cloned repository ([`fal_demos/video/wan.py#L129`](fal_demos/video/wan.py#L129)).

This demo showcases advanced features by calling other Fal functions for auxiliary tasks. The `_is_nsfw_prompt` method ([`fal_demos/video/wan.py#L140`](fal_demos/video/wan.py#L140)) uses `fal_client.subscribe` to call `fal-ai/any-llm` with a system prompt designed to classify the user's text prompt as NSFW or not. Similarly, `_is_nsfw_request` ([`fal_demos/video/wan.py#L158`](fal_demos/video/wan.py#L158)) first uploads an image (if this were an image-to-video model, or a frame from a generated video for post-check) using `fal_client.upload_image` and then calls two different Fal functions (`fal-ai/imageutils/nsfw` and `fal-ai/any-llm/vision`) for a multi-layered NSFW image check. The `_expand_prompt` method ([`fal_demos/video/wan.py#L188`](fal_demos/video/wan.py#L188)) calls `fal-ai/any-llm` with a specific system prompt to enhance and elaborate on the user's original prompt.

The main inference endpoint is `generate_image_to_video` ([`fal_demos/video/wan.py#L210`](fal_demos/video/wan.py#L210]), specifically path-versioned as `/v2.1/1.3b/text-to-video`. Inside this function, if `enable_safety_checker` is true, the `_is_nsfw_prompt` method is called. If `enable_prompt_expansion` is true, `_expand_prompt` is invoked ([`fal_demos/video/wan.py#L213-L217`](fal_demos/video/wan.py#L213-L217)). The Wan T2V model's `generate` method is then called to create the video tensor ([`fal_demos/video/wan.py#L221`](fal_demos/video/wan.py#L221)). The resulting video tensor is saved to a temporary MP4 file using `wan.utils.utils.cache_video`, and this file is returned as a `fal.toolkit.File` ([`fal_demos/video/wan.py#L230-L239`](fal_demos/video/wan.py#L230-L239)).

---

## 5. Music Generation with DiffRhythm (Custom Docker Image)

The [`fal_demos/audio/diffrhythm.py`](fal_demos/audio/diffrhythm.py) demo illustrates how to deploy a model that requires system-level dependencies beyond standard Python packages. It achieves this by using a custom Docker image. DiffRhythm generates music based on lyrics and an optional reference audio or style prompt.

**Overview**:
This application takes lyrics (with timestamps), an optional reference audio URL or style prompt, and other parameters to generate a piece of music using the DiffRhythm model.

**Key fal Concepts Demonstrated**:

The `TextToMusicInput` model ([`fal_demos/audio/diffrhythm.py#L11`](fal_demos/audio/diffrhythm.py#L11)) defines the inputs. The `lyrics` field is notable for its example, which shows a structured format with timestamps, and for its `ui` hint ([`fal_demos/audio/diffrhythm.py#L34`](fal_demos/audio/diffrhythm.py#L34)) suggesting a `textarea` for better user experience in the playground. The `reference_audio_url` also has a `ui={"important": True}` hint ([`fal_demos/audio/diffrhythm.py#L44`](fal_demos/audio/diffrhythm.py#L44)) to prioritize it in the UI. The `Output` model is straightforward, expecting an `audio` `File` ([`fal_demos/audio/diffrhythm.py#L72`](fal_demos/audio/diffrhythm.py#L72)).

A utility function `extract_segments` is defined ([`fal_demos/audio/diffrhythm.py#L88`](fal_demos/audio/diffrhythm.py#L88)) for parsing text with bracketed keys, though the main inference logic uses `get_lrc_token` from the DiffRhythm library for lyrics processing.

The most significant feature of this demo is the use of a custom Docker image. The `DOCKER_STRING` variable ([`fal_demos/audio/diffrhythm.py#L112`](fal_demos/audio/diffrhythm.py#L112)) contains the Dockerfile instructions. It starts `FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`, then uses `apt-get update && apt-get install -y` to install system packages like `git`, `espeak-ng`, and `ffmpeg`. After that, it installs a long list of Python dependencies using `pip install`.

The `DiffRhythm` application class ([`fal_demos/audio/diffrhythm.py#L162`](fal_demos/audio/diffrhythm.py#L162)) specifies `kind="container"` and uses `image=fal.ContainerImage.from_dockerfile_str(DOCKER_STRING)` to tell Fal Serverless to build and use this custom environment.

Inside the `setup()` method ([`fal_demos/audio/diffrhythm.py#L169`](fal_demos/audio/diffrhythm.py#L169)), the DiffRhythm Hugging Face Space repository is cloned using `clone_repository` ([`fal_demos/audio/diffrhythm.py#L172`](fal_demos/audio/diffrhythm.py#L172)). Specific files, like `negative_prompt.npy` and an ONNX model, are downloaded using `download_file` ([`fal_demos/audio/diffrhythm.py#L179`](fal_demos/audio/diffrhythm.py#L179), [`fal_demos/audio/diffrhythm.py#L186`](fal_demos/audio/diffrhythm.py#L186]) into the cloned repository structure. Finally, the `prepare_model` utility from DiffRhythm is called to load and initialize the various model components (`cfm`, `cfm_full`, `tokenizer`, `muq`, `vae`).

A `warmup()` method ([`fal_demos/audio/diffrhythm.py#L195`](fal_demos/audio/diffrhythm.py#L195)) is included, which calls the internal `_generate` method with a sample input. This is a good practice to ensure the model is fully loaded before handling actual user requests, reducing cold start latency for the first request after a worker starts.

The `_generate()` method ([`fal_demos/audio/diffrhythm.py#L206`](fal_demos/audio/diffrhythm.py#L206)) contains the main inference logic. It first validates that either a `style_prompt` or `reference_audio_url` is provided ([`fal_demos/audio/diffrhythm.py#L209`](fal_demos/audio/diffrhythm.py#L209)). It conditionally loads either the `cfm` or `cfm_full` model based on the requested `music_duration` ([`fal_demos/audio/diffrhythm.py#L221-L226`](fal_demos/audio/diffrhythm.py#L221-L226)). Lyrics are processed using `get_lrc_token`. If a `reference_audio_url` is given, it's downloaded, and `get_audio_style_prompt` is used; otherwise, `get_text_style_prompt` processes the `style_prompt` text ([`fal_demos/audio/diffrhythm.py#L230-L250`](fal_demos/audio/diffrhythm.py#L230-L250)). Error handling within this section uses `FieldException` to report issues related to lyrics, reference audio, or style prompt processing.

The core `inference` function from the DiffRhythm library is then called ([`fal_demos/audio/diffrhythm.py#L253`](fal_demos/audio/diffrhythm.py#L253)). The generated audio is saved to a temporary WAV file within a temporary directory, and custom billing units are calculated based on the audio duration ([`fal_demos/audio/diffrhythm.py#L270`](fal_demos/audio/diffrhythm.py#L270)). The output is returned as a `File` object.

The primary endpoint `/` is defined at [`fal_demos/audio/diffrhythm.py#L274`](fal_demos/audio/diffrhythm.py#L274)}, which simply calls the `_generate` method.

---

## 6. Distributed Computing Examples

The distributed computing examples demonstrate how to leverage multiple GPUs for both inference and training workloads on the fal platform. These examples are organized into two categories: distributed inference and distributed training.

### Distributed Inference

Distributed inference examples show how to scale inference across multiple GPUs for faster generation or higher throughput.

#### 6.1 Parallel SDXL Inference with DistributedRunner

The [`fal_demos/distributed/inference/parallel_sdxl/app.py`](fal_demos/distributed/inference/parallel_sdxl/app.py) demo showcases data parallelism using `DistributedRunner`. This approach runs independent model instances on each GPU, generating multiple images simultaneously.

**Overview**:
This application uses `fal.distributed.DistributedRunner` to run Stable Diffusion XL across multiple GPUs. Each GPU generates images independently with different random seeds, and the results are gathered into a grid.

**Key fal Concepts Demonstrated**:

The `ExampleDistributedWorker` class ([`fal_demos/distributed/inference/parallel_sdxl/app.py#L51`](fal_demos/distributed/inference/parallel_sdxl/app.py#L51)) extends `DistributedWorker` and implements the worker logic. In the `setup()` method ([`fal_demos/distributed/inference/parallel_sdxl/app.py#L59`](fal_demos/distributed/inference/parallel_sdxl/app.py#L59)), each worker initializes its own SDXL pipeline on its assigned GPU (`self.device`).

The `__call__` method ([`fal_demos/distributed/inference/parallel_sdxl/app.py#L117`](fal_demos/distributed/inference/parallel_sdxl/app.py#L117)) implements the inference logic. Each worker generates an image independently, and `torch.distributed.gather` ([`fal_demos/distributed/inference/parallel_sdxl/app.py#L153`](fal_demos/distributed/inference/parallel_sdxl/app.py#L153)) collects all images on rank 0 (the main worker).

The `ExampleDistributedApp` class ([`fal_demos/distributed/inference/parallel_sdxl/app.py#L194`](fal_demos/distributed/inference/parallel_sdxl/app.py#L194)) configures the multi-GPU setup with `num_gpus = 2` and initializes the `DistributedRunner` ([`fal_demos/distributed/inference/parallel_sdxl/app.py#L215`](fal_demos/distributed/inference/parallel_sdxl/app.py#L215)) with the worker class and world size.

A streaming endpoint is also provided ([`fal_demos/distributed/inference/parallel_sdxl/app.py#L238`](fal_demos/distributed/inference/parallel_sdxl/app.py#L238)) that demonstrates real-time progress updates during generation using `add_streaming_result`.

**When to use**: Data parallelism is ideal when you need to generate multiple variations simultaneously or achieve higher throughput by processing multiple requests in parallel.

#### 6.2 xFuser Distributed Inference

The [`fal_demos/distributed/inference/xfuser/`](fal_demos/distributed/inference/xfuser/) demo showcases model parallelism using xFuser with Ray. This approach splits the model across GPUs for faster single-image generation.

**Overview**:
This application uses xFuser's PipeFusion and Ulysses parallelism strategies to distribute a Diffusion Transformer model across multiple GPUs. Unlike data parallelism, model parallelism splits the model itself, allowing faster generation of a single image.

**Key fal Concepts Demonstrated**:

The `XFuserApp` class ([`fal_demos/distributed/inference/xfuser/app.py#L35`](fal_demos/distributed/inference/xfuser/app.py#L35)) configures the application with `num_gpus = 2` and supports various DiT-based models like SD3 Medium, PixArt-Sigma, and HunyuanDiT.

In the `setup()` method ([`fal_demos/distributed/inference/xfuser/app.py#L88`](fal_demos/distributed/inference/xfuser/app.py#L88)), the xFuser distributed engine is initialized using Ray. The xFuser configuration ([`fal_demos/distributed/inference/xfuser/app.py#L159`](fal_demos/distributed/inference/xfuser/app.py#L159)) specifies parallelism strategies:
- `pipefusion_parallel_degree`: Split model across pipeline stages
- `ulysses_degree`: Split sequence dimension for attention
- `use_cfg_parallel`: Parallel classifier-free guidance

The `Engine` class ([`fal_demos/distributed/inference/xfuser/engine.py`](fal_demos/distributed/inference/xfuser/engine.py)) manages Ray actors that run xFuser workers, coordinating distributed inference across GPUs.

**Supported Models**: SD3 Medium (default), SDXL, PixArt-Alpha/Sigma, HunyuanDiT, FLUX.1

**When to use**: Model parallelism is ideal when you need faster generation of individual images, especially for large DiT-based models, or when the model doesn't fit on a single GPU.

### Distributed Training

Distributed training examples demonstrate multi-GPU training patterns using PyTorch DistributedDataParallel (DDP).

#### 6.3 Flux LoRA Training with Multi-GPU Preprocessing and Training

The [`fal_demos/distributed/training/flux_lora/`](fal_demos/distributed/training/flux_lora/) demo showcases a complete, production-ready training pipeline matching [fal.ai/flux-lora-fast-training](https://fal.ai/models/fal-ai/flux-lora-fast-training).

**Overview**:
This is a complete Flux LoRA training pipeline that accepts raw images in a ZIP file and produces a trained LoRA checkpoint. It includes multi-GPU preprocessing (VAE encoding, caption generation) and multi-GPU training using DDP.

---

## 7. Matrix WebRTC

The [`fal_demos/video/matrix_webrtc/`](fal_demos/video/matrix_webrtc/) demo streams a Matrix-Game session over WebRTC with a small Vite frontend. It shows how to expose a websocket WebRTC endpoint with `fal.App`, forward keyboard actions, and stream frames to a browser client. Run instructions are in [`fal_demos/video/matrix_webrtc/README.md`](fal_demos/video/matrix_webrtc/README.md).

**Key fal Concepts Demonstrated**:

The pipeline consists of two main components:

**Preprocessor** ([`fal_demos/distributed/training/flux_lora/preprocessor/`](fal_demos/distributed/training/flux_lora/preprocessor/)):
- `FluxPreprocessorWorker` distributes images across GPUs for parallel processing
- Auto-generates captions using the moondream model
- Injects trigger words into captions (e.g., "ohwx", "txcl")
- Parallel VAE encoding and T5/CLIP text encoding
- Gathers preprocessed tensors from all GPUs

**Trainer** ([`fal_demos/distributed/training/flux_lora/trainer/`](fal_demos/distributed/training/flux_lora/trainer/)):
- `FluxLoRATrainingWorker` extends `DistributedWorker` for DDP training
- Each GPU loads the Flux transformer and adds LoRA adapters
- Models are wrapped with `torch.nn.parallel.DistributedDataParallel` for gradient synchronization
- Implements proper Flux flow matching loss
- Supports real-time progress streaming
- Only rank 0 saves the final checkpoint

The `FluxLoRATrainingApp` ([`fal_demos/distributed/training/flux_lora/trainer/app.py`](fal_demos/distributed/training/flux_lora/trainer/app.py)) provides multiple endpoints:
- `/train-complete`: Full pipeline from raw images
- `/train-from-preprocessed`: Training from preprocessed data
- `/train-complete-stream`: Streaming progress updates

**Complete Pipeline**:
```
Raw Images (ZIP) + trigger_word
        ↓
Multi-GPU Preprocessing (parallel VAE/T5/CLIP encoding)
        ↓
preprocessed_data.pt
        ↓
Multi-GPU Training (DDP with gradient sync)
        ↓
lora_checkpoint.safetensors
```

**When to use**: This pattern is essential for production training workloads that require:
- Multi-GPU acceleration for faster training
- Preprocessing large datasets in parallel
- Synchronized gradient updates across GPUs
- Streaming training metrics in real-time

For detailed documentation on the training architecture, synchronization patterns, and troubleshooting, see [`fal_demos/distributed/training/flux_lora/README.md`](fal_demos/distributed/training/flux_lora/README.md).

---

These demos provide a solid foundation for understanding how to deploy a wide variety of models on fal. By going through them, you can learn how to structure your applications, manage dependencies, handle inputs and outputs effectively, and leverage the platform's features to build powerful and scalable AI services—from single-GPU inference to multi-GPU distributed training. Remember to consult the official [Fal documentation](docs.fal.ai) for more in-depth information on any of the concepts discussed.
