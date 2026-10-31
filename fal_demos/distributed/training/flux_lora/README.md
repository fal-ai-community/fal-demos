# Flux LoRA Training on Multiple GPUs

A complete multi-GPU LoRA training pipeline on fal Serverless, split into
separately deployed apps:

| App | What it does | GPUs |
|-----|--------------|------|
| [`preprocessor/`](preprocessor/) | Auto-captioning, trigger word injection, VAE encoding, T5/CLIP encoding | `num_gpus` |
| [`trainer/`](trainer/) | DDP LoRA training, launched with `fal.distributed` | `num_gpus` |
| [`trainer_torchrun/`](trainer_torchrun/) | The same training, launched with a plain spawn | `num_gpus` |

The trainer calls the preprocessor over the network with `fal_client`, so the
two scale independently and the training box does not sit idle while images are
being encoded.

The two trainers are interchangeable. Both import the identical training code
from [`core.py`](core.py) and differ only in how the ranks are started, which
is the cleanest way to show that the choice of launcher is yours.

## Is `fal.distributed` required for multi-GPU?

**No.** A fal app with `num_gpus = 8` is a single container with all 8 GPUs
visible, exactly like an 8x node in your own cluster. Anything that works there
works here: `torchrun`, `accelerate launch`, Lightning Fabric, or a plain
`torch.multiprocessing` spawn.

`fal.distributed` is a convenience layer over `torch.multiprocessing.spawn` plus
NCCL on localhost, with ZMQ to move payloads between your app process and the
ranks. It saves you writing the spawn, teardown, and IPC glue, and it gives you
streaming intermediate results for free. It is a convenience, not a
requirement.

Pick based on what you already have:

| | [`trainer/`](trainer/) (`fal.distributed`) | [`trainer_torchrun/`](trainer_torchrun/) (own launcher) |
|---|---|---|
| Rank startup | `DistributedRunner` | `multiprocessing` spawn you control |
| App ↔ ranks | ZMQ, handled for you | HTTP to rank 0 on localhost, yours to write |
| Streaming progress | Built in, via `add_streaming_result` | Write it yourself |
| Failure supervision | Handled for you | Yours to write |
| Porting an existing trainer | Restructure into `DistributedWorker` | Keep your launcher as-is |

Start fresh with `fal.distributed`. Bring an existing DDP trainer, and keep
your launcher: shell out to it from `setup()` and leave your training code
untouched. That second shape is what fal's own production trainers use.

`fal.distributed` *is* required for
[multi-node](https://fal.ai/docs/documentation/serverless/distributed/multi-node)
(one app spanning several GPU nodes), which is a separate, beta feature.

### If you bring your own launcher

Two things will cost you an afternoon otherwise:

- **Use the `spawn` start method, not `fork`.** The app process may have already
  initialized CUDA, and forking after that breaks the child ranks.
- **Set `NCCL_NVLS_ENABLE=0` on 8x Hopper.** On NCCL 2.20.5 + H100/H200 the
  first collective can fail with CUDA error 401 while NVLS sets up its memory
  pools. Falling back to ring/tree collectives is safe on a single node.

[`trainer_torchrun/app.py`](trainer_torchrun/app.py) is the full working
version. In outline:

```python
class Trainer(fal.App, min_concurrency=1, max_concurrency=1):
    machine_type = "GPU-H100"
    num_gpus = 8
    startup_timeout = 3600

    def setup(self):
        env = {**os.environ, "NCCL_NVLS_ENABLE": "0"}
        self.process = subprocess.Popen(
            ["torchrun", "--nproc_per_node=8", "train.py", "--ready-file", READY],
            env=env,
        )
        # Block until rank 0 signals ready, and fail loudly if a rank dies.
        while not os.path.exists(READY):
            if (rc := self.process.poll()) is not None:
                raise RuntimeError(f"ranks exited with {rc} before becoming ready")
            time.sleep(1.0)

    @fal.endpoint("/")
    def train(self, input: Input) -> Output:
        r = requests.post("http://127.0.0.1:8001/", json=input.dict(), timeout=None)
        r.raise_for_status()
        return Output(**r.json())
```

Rank 0 serves a small local HTTP server and broadcasts each job to the other
ranks, which sit in `dist.broadcast_object_list` between requests. Give
`init_process_group` a long `timeout` for that reason.

## Deploying

Deploy the preprocessor first, since the trainer calls it by name:

```bash
fal deploy fal_demos/distributed/training/flux_lora/preprocessor/app.py
fal deploy fal_demos/distributed/training/flux_lora/trainer/app.py

# ...or the same trainer with your own launcher instead of fal.distributed
fal deploy fal_demos/distributed/training/flux_lora/trainer_torchrun/app.py
```

Then point the trainer at your own preprocessor deployment via the
`preprocessor_app` field (it defaults to the author's namespace):

```bash
curl -X POST "https://fal.run/<your-username>/flux-trainer-demo/train" \
  -H "Authorization: Key $FAL_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "images_data_url": "https://.../images.zip",
    "trigger_word": "ohwx",
    "steps": 250,
    "preprocessor_app": "<your-username>/flux-preprocessor-demo/"
  }'
```

`/stream` takes the same payload and returns Server-Sent Events with per-step
loss.

## Scaling

Both GPU apps pin `min_concurrency = max_concurrency = 1` so a single training
job owns the whole node for its duration. That is also the setup to use if you
want your own scheduler to control capacity: keep concurrency pinned and drive
the replica count yourself with the
[scale CLI](https://fal.ai/docs/api-reference/cli/apps/scale) or the
[Python SDK](https://fal.ai/docs/api-reference/python-sdk/client).

The trainer sets a long `keep_alive` because reloading Flux is expensive; the
preprocessor scales to zero between batches.

## Cold starts

Both trainers keep base weights on `/data`, and `trainer_torchrun/` also starts
its ranks before downloading so the two overlap. Beyond that the demo stays
deliberately simple. For production training apps, four more levers do most of
the work:

- **Keep weights on `/data`, never baked into the image.** Declare `data_mounts`
  and download in `setup()`. `/data` is shared across all your runners and
  survives deploys.
  → [Persistent Storage](https://fal.ai/docs/documentation/development/use-persistent-storage)
- **Persist the `torch.compile` Inductor cache** so one worker compiles and the
  rest unpack. Bump the cache key whenever the torch version changes; a stale
  cache unzips fine but never hits, and you pay a full recompile on the first
  request instead of at setup.
  → [Compiled Caches](https://fal.ai/docs/documentation/serverless/optimizations/optimize-startup-with-compiled-caches)

  ```python
  from fal.toolkit import synchronized_inductor_cache

  with synchronized_inductor_cache("flux-lora-trainer/v1"):
      model = torch.compile(model)
      warmup()
  ```

- **Use FlashPack for large weight blobs.**
  → [FlashPack](https://fal.ai/docs/documentation/serverless/optimizations/flashpack)
- **Prefetch `/data` files in parallel** to warm the page cache while the ranks
  are still importing torch.
  → [Parallel File Loading](https://fal.ai/docs/documentation/serverless/optimizations/parallel-file-loading)

Start your ranks *before* downloading weights where you can. Importing torch and
diffusers in 8 processes takes tens of seconds, and that time is free if it
overlaps the download rather than queueing behind it. Both designs need a
readiness gate for this: a file the app touches once weights are complete and
the ranks wait on, since `DistributedRunner.start()` and a subprocess readiness
poll both block on every rank finishing `setup()`.

Overview: [Optimizing Cold Starts](https://fal.ai/docs/documentation/serverless/optimizations/optimize-cold-starts)
and [Scaling Parameters](https://fal.ai/docs/documentation/serverless/optimizations/cold-start-scaling).

## Storing and loading the trained LoRA

The trainer uploads the safetensors and returns a URL. At inference, pull it
with `download_model_weights`, which caches under `/data/.fal/model_weights`
keyed by a hash of the URL — so the second request for a given LoRA on that
runner is a local read, not a download.

```python
from fal.toolkit.utils.download_utils import download_model_weights

local_path = download_model_weights(lora_url)
```

## Further reading

- [fal.distributed overview](https://fal.ai/docs/documentation/serverless/distributed/overview)
- [fal.distributed API reference](https://fal.ai/docs/documentation/serverless/distributed/api-reference)
- [App lifecycle](https://fal.ai/docs/documentation/development/app-lifecycle)
- [Migrating an external Docker server](https://fal.ai/docs/documentation/development/migrate-external-docker-server)
