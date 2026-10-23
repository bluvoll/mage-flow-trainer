# Remote training: the GUI here, the GPU there

Run the GUI on your laptop and train on a GPU box you reach through a notebook: Colab, a JupyterHub
server, or a rented pod. The pod runs a small job server, and the GUI connects to it with one link.
The workflow follows [LoRA_Easy_Training_scripts](https://github.com/67372a/LoRA_Easy_Training_scripts):
copy the link, press Connect, and the settings go to the pod.

There are two ways to use it. Pick per session.

| | Jobs mode (`serve`) | Notebook mode (`receive_config` + `run`) |
| --- | --- | --- |
| What Start Training does | Caches latents, then trains, **on the pod** | Sends the config to the pod. The waiting cell returns its path |
| Where training output appears | The GUI's console and live graphs | The notebook cell |
| Stop / Save now / Save & stop | From the GUI | `touch output/<run>/save` (or `save_quit`) in the notebook |
| Laptop may sleep or close | Yes. The job keeps running, and Connect again reattaches | Yes |

## 1. On the GPU machine

Clone and install once (Linux, CUDA). In a notebook, prefix each command with `!`:

```bash
git clone https://github.com/RicemanT/mage-flow-trainer
cd mage-flow-trainer && bash install.sh
pip install -r requirements-tracking.txt   # into venv/, if you use wandb/TensorBoard/Trackio
```

**Jobs mode:** start the server and leave the cell running:

```bash
venv/bin/python -m trainer.remote serve
```

**Notebook mode:** wait for the config, then train it in the next cell:

```bash
venv/bin/python -m trainer.remote receive          # prints the saved config path when it arrives
venv/bin/python -m trainer.remote run configs/remote/<run_name>.toml
```

If the notebook kernel itself has the trainer's dependencies, the Python API does the same thing:
`from trainer.remote import serve, receive_config, run`. Then use `serve()`, or
`config = receive_config()` followed by `run(config)`. `serve(block=False)` returns immediately and
keeps the server alive with the kernel. [notebooks/mageflow-remote.ipynb](../notebooks/mageflow-remote.ipynb)
has all of this as cells.

The server prints a block like this:

```
Mage-Flow remote server ready -- paste this link into the GUI's Remote bar and press Connect:

    https://example-words-here.trycloudflare.com/#token=Qm9...
```

## 2. In the GUI

On a laptop that only runs the GUI, install it without CUDA: `install.bat --gui-only` (Windows) or
`./install.sh --gui-only` (Linux). That is CPU PyTorch plus the GUI's few packages, pinned like
`requirements.txt` (`requirements-gui.txt`): about 1 GB instead of the full CUDA stack. Then `start-gui.bat` /
`./start-gui.sh`. The GUI imports the trainer's config classes to edit and check configs, which is why PyTorch is
still needed; it never loads a model.

1. Paste the link into the **Remote** bar at the bottom and press **Connect**. A new quick-tunnel
   hostname takes a few seconds to become reachable, and Connect retries for about 30 seconds.
2. The bar shows the pod's hostname and mode. Local GPU checkboxes are replaced by **Remote GPUs**:
   leave it blank for every GPU on the pod, one process each, or enter a device list like `0,1`.
3. Fill in the config with **paths on the pod**: model, dataset or subsets file, output folder.
4. **Start Training.** The config is checked by the trainer's own loader on the pod first. A bad
   key, a missing subsets file or a stage list not totalling 100% is reported there before anything
   runs, and missing dataset folders or model files appear as warnings. In jobs mode it then runs
   `cache-config` (a no-op when the cache is warm) and training, tailing the log into the console and
   the Metrics tab. In receive mode it only sends the config.
5. **Stop** stops the job on the pod. **Save now** and **Save & stop** write the trainer's signal
   files in the run folder on the pod. **Cache latents** and **Cache (dry run)** also run on the pod.
   **Audit dataset** doesn't, because it reads images; use the dry run, or run
   `cache_latents audit` there.

The link is remembered in `.gui_remote.json` beside the GUI (git-ignored, since it contains the
token). Closing the GUI never stops a remote job. Connect again and the GUI attaches to a job that's
still running and replays its log, graphs included.

## Data, models and the latent cache through Hugging Face

[notebooks/mageflow-remote.ipynb](../notebooks/mageflow-remote.ipynb) has a cell for each step. The images never
leave the machine that holds them; the training machine gets only the latent cache:

1. **Prepare the dataset** (on the Illustration Scrapping Studio server). The Studio's training-layout export
   (`folders.csv`, `subsets.toml`, `mageflow-512.toml`, `mageflow-1024.toml`) becomes the root of a local dataset
   folder, and every `<group>/<artist>` folder it lists is hard-linked into it from the library (no copy; the
   latents written beside them stay out of the library). `folders.csv` lists those folders relative to itself, so
   it works wherever the dataset ends up.
2. **Models.** The text encoder and tokenizer from `mage-flow-community/Mage-Flow`, the FLUX.2 VAE (diffusers format;
   the configs set `flux2_vae = true`), and the transformer you
   finetune as a single file (`train.transformer_path`).
3. **Download the latent cache** on the training machine (no images).
4. **Write the configs.** Fills the exported configs' `@MODEL_PATH@`, `@TRANSFORMER_PATH@`, `@DATA_DIR@` and
   `@OUTPUT_DIR@` with that machine's paths, into `configs/magetrail/`.
5. **Cache** on the server, where GPU time is cheap: latents for each stage with `cache-config`, and the caption
   embeddings with `trainer.training.train <config> --text-cache-only`, which builds `train.caption_cache_path` and
   exits without loading the transformer.
6. **Upload the latent cache**: latents, `.txt`/`_nl.txt` captions and the layout files, never the
   images, to a dataset that is public by default (`CACHE_PUBLIC`): public repos do not count against the private
   storage quota, and the cell makes the repo public before uploading so a large upload cannot run into it. Not the caption-embedding cache: it stores whole Qwen3-VL hidden states, about 80 GB for a
   6,500-image dataset at 8 variations (terabytes at 360k images), so the training machine builds its own at
   startup, which takes minutes on large GPUs.
7. **Train** on the expensive machine from that cache alone (`dataset.source = "auto"` reads the latents when a
   folder has no images). Cell 7C uploads the checkpoint `.safetensors` files to `RicemanT/MageTrail` `V0.4/`
   (set in the cell; files still being written skipped). It stands alone, so it can run in a second notebook during
   training; run it before the pod is deleted.
8. **Publish the dataset** (optional): the images with their tags, captions and artist metadata as WebDataset tar
   shards of about 1 GB, which Hugging Face streams natively, plus a dataset card.

The caption cache is portable: its entries are keyed by each folder's name, not its absolute path, so a cache built
under `/home/jovyan/...` is used as it is under `/workspace/...`.

## Connection options

- **Cloudflare quick tunnel** (default): the pod makes an outbound connection, so no ports need to
  be open and no account is needed. `cloudflared` comes from PATH or `MAGEFLOW_CLOUDFLARED`, or is
  downloaded once into `~/.cache/mageflow-trainer/`.
- **Your own address:** `serve --tunnel none --url https://<pod-id>-8765.proxy.runpod.net` (or an
  exposed port, or an `ssh -L` tunnel). Without a tunnel the server binds `0.0.0.0:8765`.
  `--port` and `--host` change that.
- **Fixed token:** `--token` or `MAGEFLOW_REMOTE_TOKEN`, if you'd rather not paste a new link each
  session.

## Security

The tunnel URL is public, so every API call requires the session token. The connect link carries
it after `#`, which browsers and proxies don't send to servers. The landing page shows nothing
about the machine. A request without the token gets a 401.

The server never runs a command line it receives. The GUI sends a config, and the pod runs its own
fixed `cache-config` and `accelerate launch ... trainer.training.train` commands, with GPU lists
validated as plain device indices. Treat the link like a password anyway: whoever has it can train,
and can read the training log, on that machine. Per-job configs and logs are kept in
`remote_jobs/<time>-<run>/` on the pod.

## API

JSON over HTTP, for scripting: `GET /api/status`, `POST /api/validate {toml}`,
`POST /api/config {toml, name}`, `POST /api/run {toml, name, steps, gpus, num_processes}` (steps
from `cache`, `cache_dry`, `train`), `GET /api/log?offset=N`, `POST /api/stop`,
`POST /api/signal {name: save|save_quit}`, `POST /api/shutdown`. Authenticate with
`Authorization: Bearer <token>`. `trainer/remote/client.py` wraps all of these.
