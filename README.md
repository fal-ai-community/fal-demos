# Demos for Fal Serverless

Welcome\! This repository is a curated collection of production-ready examples designed to help you master the `fal` SDK and deploy powerful AI models on our serverless platform.

Whether you're building a simple text-to-image API or a complex, multi-stage video generation pipeline, these demos provide the patterns and best practices you need. We'll cover everything from image generation and text-to-speech to 3D object creation and music synthesis, showcasing the flexibility and power of Fal.

## Getting Started

1.  **Sign up for Fal:** If you haven't already, create an account at [fal.ai](https://fal.ai).
2.  **Install the Client:**
    ```bash
    pip install fal
    ```
3.  **Authenticate:**
    ```bash
    fal auth login
    ```
4.  **Clone this Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

## Core Concepts of the Fal SDK

Before diving into the demos, let's review the fundamental building blocks of a `fal` application. You'll see these concepts in every example.

| Concept | Description |
| :--- | :--- |
| **`fal.App`** | The heart of your application. This class defines the entire serverless environment, including the `machine_type`, Python `requirements`, concurrency settings (`min_concurrency`, `max_concurrency`), and `keep_alive` duration. |
| **`pydantic.BaseModel`** | Defines the structured `input` and `output` schemas for your API. Using `pydantic.Field`, you can add rich metadata (descriptions, examples, validation rules) that automatically powers your API's interactive playground. |
| **`setup()` method** | A special method in your `fal.App` class that runs **once** when a new worker instance boots up. This is the perfect place for heavy, one-time operations like downloading and loading your ML models into memory. |
| **`@fal.endpoint()` decorator** | Exposes a method within your `fal.App` class as a secure, scalable HTTP endpoint. A single app can have multiple endpoints. |
| **`fal.toolkit`** | A collection of high-level utilities that simplify common tasks. Includes helpers for handling images (`fal.toolkit.Image`), files (`fal.toolkit.File`), cloning Git repositories, and more. |
| **Secrets Management** | Securely manage API keys and other credentials using `fal secrets set MY_KEY <value>`. These are exposed as environment variables within your running application. |
| **Custom Docker Images** | For applications with complex system-level dependencies (e.g., `ffmpeg`, `apt-get` packages), `fal` supports deploying from a custom `Dockerfile` or Dockerfile string. |

-----

## The Demos

Each demo is a self-contained application that highlights specific features and design patterns.

### 1\. Advanced Text-to-Image with Sana

This demo hosts the Sana text-to-image model and demonstrates how to build a robust, user-friendly API with multiple endpoints and custom billing.

  * **File:** [`fal_demos/image/sana.py`](https://www.google.com/search?q=fal_demos/image/sana.py)
  * **What You'll Learn:**
      * Defining rich API schemas with `pydantic`.
      * Creating multiple endpoints (`/` and `/sprint`) from a single `fal.App`.
      * Efficiently loading multiple model variants in `setup()`.
      * Pinning dependencies (including Git commits) for reproducible builds.
      * Implementing custom, usage-based billing with response headers.
      * Integrating safety checkers and helper utilities from `fal.toolkit`.

#### Code Walkthrough

The application defines structured inputs like `TextToImageInput` ([L65](https://www.google.com/search?q=fal_demos/image/sana.py%23L65)) and outputs like `SanaOutput` ([L78](https://www.google.com/search?q=fal_demos/image/sana.py%23L78)) using Pydantic. Notice how `Field` objects are used to provide examples and validation, and `fal.toolkit.ImageSizeInput` ([L26](https://www.google.com/search?q=fal_demos/image/sana.py%23L26)) simplifies handling image dimensions.

The `Sana` class ([L111](https://www.google.com/search?q=fal_demos/image/sana.py%23L111)) configures the serverless environment, setting `keep_alive`, concurrency limits, and the `GPU-H100` `machine_type` ([L128](https://www.google.com/search?q=fal_demos/image/sana.py%23L128)). In `setup()` ([L129](https://www.google.com/search?q=fal_demos/image/sana.py%23L129)), both the standard and sprint pipelines are loaded, with the sprint version reusing the base model's text encoder to save memory ([L142](https://www.google.com/search?q=fal_demos/image/sana.py%23L142)).

A private `_generate()` method ([L148](https://www.google.com/search?q=fal_demos/image/sana.py%23L148)) encapsulates the core logic. Custom billing is calculated based on image resolution and passed via the `x-fal-billable-units` header ([L181-L186](https://www.google.com/search?q=fal_demos/image/sana.py%23L181-L186)). Finally, two public endpoints, `generate` ([L198](https://www.google.com/search?q=fal_demos/image/sana.py%23L198)) and `generate_sprint` ([L207](https://www.google.com/search?q=fal_demos/image/sana.py%23L207)), are exposed.

-----

### 2\. Proxying APIs & Managing Secrets with Hunyuan3D

This demo illustrates a common pattern: creating a "proxy" or "wrapper" application that calls another Fal function. This is perfect for adding a simplified interface, custom logic, or authentication layer on top of an existing model.

  * **File:** [`fal_demos/image/hunyuan3d.py`](https://www.google.com/search?q=fal_demos/image/hunyuan3d.py)
  * **What You'll Learn:**
      * Using the `fal_client` to call other Fal functions asynchronously (`fal_client.submit_async`).
      * Managing API keys with `fal secrets`.
      * Handling I/O-bound concurrency with `max_multiplexing`.
      * Streaming real-time logs and results from a background job.
      * Implementing robust error handling for client-side exceptions.

#### Code Walkthrough

The `setup()` method ([L72](https://www.google.com/search?q=fal_demos/image/hunyuan3d.py%23L72)) shows how to retrieve a secret (`MY_SECRET_KEY`) from environment variables to authenticate the `fal_client`. The `max_multiplexing=10` setting ([L67](https://www.google.com/search?q=fal_demos/image/hunyuan3d.py%23L67)) allows a single worker to efficiently handle up to 10 concurrent requests that are waiting on the downstream API.

The `generate_image` endpoint ([L77](https://www.google.com/search?q=fal_demos/image/hunyuan3d.py%23L77)) uses `fal_client.submit_async` ([L80](https://www.google.com/search?q=fal_demos/image/hunyuan3d.py%23L80)) to trigger the `fal-ai/hunyuan3d` model. It then iterates over the job's events with `handle.iter_events` ([L90](https://www.google.com/search?q=fal_demos/image/hunyuan3d.py%23L90)) to stream progress. The extensive `try-except` block ([L102-L120](https://www.google.com/search?q=fal_demos/image/hunyuan3d.py%23L102-L120)) demonstrates how to handle `FieldException`, `HTTPStatusError`, and `FalClientError` gracefully.

-----

### 3\. Multi-Language Text-to-Speech with Kokoro

This demo hosts the Kokoro TTS model, showcasing how to serve multiple languages and voice options from a single application on a CPU worker.

  * **File:** [`fal_demos/tts/kokoro.py`](https://www.google.com/search?q=fal_demos/tts/kokoro.py)
  * **What You'll Learn:**
      * Managing multiple model variants (one for each language) within one app.
      * Using `typing.Literal` to create constrained, dropdown-style inputs in the UI.
      * Deploying to CPU-only machine types (`"L"`).
      * Handling audio data and returning it as a `fal.toolkit.File`.
      * Validating input length and providing user-friendly errors with `FieldException`.

#### Code Walkthrough

This app defines distinct input models for each language, such as `AmEnglishRequest` ([L9](https://www.google.com/search?q=fal_demos/tts/kokoro.py%23L9)). The `voice` field uses `typing.Literal` ([L21](https://www.google.com/search?q=fal_demos/tts/kokoro.py%23L21)) to provide a fixed list of valid choices. In `setup()` ([L143](https://www.google.com/search?q=fal_demos/tts/kokoro.py%23L143)), a dictionary of `kokoro.KPipeline` instances is created, loading a model for each language.

The `_generate()` method ([L149](https://www.google.com/search?q=fal_demos/tts/kokoro.py%23L149)) contains the shared logic. It validates prompt length ([L154](https://www.google.com/search?q=fal_demos/tts/kokoro.py%23L154)), selects the correct pipeline, and generates the audio. The output audio is saved to a temporary WAV file and returned as a `fal.toolkit.File` with the correct `content_type` ([L176-L180](https://www.google.com/search?q=fal_demos/tts/kokoro.py%23L176-L180)), which Fal can serve efficiently from its CDN.

-----

### 4\. Complex Workflows with Wan Text-to-Video

This text-to-video demo tackles a more complex scenario, composing multiple Fal functions to create a feature-rich pipeline that includes prompt safety checks and automatic prompt enhancement.

  * **File:** [`fal_demos/video/wan.py`](https://www.google.com/search?q=fal_demos/video/wan.py)
  * **What You'll Learn:**
      * Cloning Git repositories during the `setup` phase with `fal.toolkit.clone_repository`.
      * Handling complex dependencies, including pre-compiled wheels from URLs.
      * Orchestrating multiple `fal_client` calls to build advanced features (e.g., prompt safety, prompt expansion).
      * Managing the Python path and working directory for non-standard library structures.
      * Handling video file I/O.

#### Code Walkthrough

The `setup()` method ([L100](https://www.google.com/search?q=fal_demos/video/wan.py%23L100)) is a great example of a complex initialization. It uses `fal.toolkit.clone_repository` ([L107](https://www.google.com/search?q=fal_demos/video/wan.py%23L107)) to fetch the model's source code, `os.chdir` ([L114](https://www.google.com/search?q=fal_demos/video/wan.py%23L114)) to work within that repository, and `huggingface_hub` to download weights.

The key pattern here is composition. Helper methods like `_is_nsfw_prompt` ([L140](https://www.google.com/search?q=fal_demos/video/wan.py%23L140)) and `_expand_prompt` ([L188](https://www.google.com/search?q=fal_demos/video/wan.py%23L188)) call `fal-ai/any-llm` to perform auxiliary tasks. The main `generate_image_to_video` endpoint ([L210](https://www.google.com/search?q=fal_demos/video/wan.py%23L210)) orchestrates these calls before running the core video generation, providing a much richer user experience.

-----

### 5\. Custom Environments with DiffRhythm (Docker)

This demo shows how to deploy a model with heavy system-level dependencies (`ffmpeg`, `espeak-ng`) by defining a custom container environment directly in the Python script.

  * **File:** [`fal_demos/audio/diffrhythm.py`](https://www.google.com/search?q=fal_demos/audio/diffrhythm.py)
  * **What You'll Learn:**
      * Defining a custom runtime environment using `fal.ContainerImage.from_dockerfile_str`.
      * Installing system packages (`apt-get`) and Python packages (`pip`) in a Docker context.
      * Using a `warmup()` method to reduce cold-start latency for the first request.
      * Downloading model assets on the fly with `fal.toolkit.download_file`.
      * Providing UI hints (`ui={"widget": "textarea"}`) in Pydantic models for a better playground experience.

#### Code Walkthrough

The most important feature is the `DOCKER_STRING` ([L112](https://www.google.com/search?q=fal_demos/audio/diffrhythm.py%23L112)), which contains a multi-stage Dockerfile. This string is passed to `fal.ContainerImage.from_dockerfile_str` ([L164](https://www.google.com/search?q=fal_demos/audio/diffrhythm.py%23L164)) to tell Fal to build and use this specific environment.

The `setup()` method ([L169](https://www.google.com/search?q=fal_demos/audio/diffrhythm.py%23L169)) clones the model's repository and downloads necessary asset files like `.npy` and `.onnx` models. A `warmup()` method ([L195](https://www.google.com/search?q=fal_demos/audio/diffrhythm.py%23L195)) is defined to pre-run inference, ensuring the model is fully JIT-compiled and ready before serving traffic. The inference logic in `_generate()` ([L206](https://www.google.com/search?q=fal_demos/audio/diffrhythm.py%23L206)) is intricate, handling conditional logic based on user inputs (e.g., using a reference audio URL vs. a text style prompt).

## Running the Demos

To deploy and run any of these demos, use the `fal run` command followed by the path to the Python file.

```bash
# Example: Deploying the Sana image generation app
fal run fal_demos/image/sana.py
```

The CLI will stream the build logs and, upon successful deployment, provide you with the URL to your application's API playground.

## Next Steps

You're now equipped with the patterns to build and deploy almost any AI model on Fal Serverless.

  * **Dive Deeper:** Explore the official [Fal Documentation](https://docs.fal.ai) for in-depth guides on every feature.
  * **Experiment:** Modify these demos or start from scratch with your own model.
  * **Join the Community:** Have questions? Join our [Discord server](https://www.google.com/search?q=https://fal.ai/discord) to chat with the team and other developers.