# MageTrail fork: added features

This fork of [bluvoll/mage-flow-trainer](https://github.com/bluvoll/mage-flow-trainer) adds what the
MageTrail finetune used from the DeepSpeed-based
[diffusion-pipe-mageflow-ft](https://github.com/RicemanT/diffusion-pipe-mageflow-ft), plus a few
things a 10,000-artist dataset needs. With every new option at its default, training behaves exactly
like upstream.

| Feature | Config | GUI |
| --- | --- | --- |
| [StageLR schedule](#stagelr) | `[schedule] kind = "stage"`, `stages` | Optimizer → StageLR |
| [Artist attribution](#artist-attribution) | `[dataset.caption] attribution_*` | Dataset → Artist Attribution |
| [TensorBoard / wandb / Trackio](#experiment-tracking) | `[tracking]` | Monitoring |
| [Validation samples](#validation-samples) | `[sampling]` | Monitoring |
| [Held-out eval loss](#held-out-eval-loss) | `[eval]` | Monitoring |
| [Save now / save and stop](#save-now-and-save-and-stop) | files in the run folder | Save now, Save & stop |
| [Subset manifest](#thousands-of-folders-subsets_file) | `dataset.subsets_file`, `subsets_root` | Dataset Source |
| [Half-precision latent cache](#latent-cache-size-and-speed) | `dataset.latent_dtype`, `cache_latents convert` | Dataset Source |
| [Remote training](remote-training.md) | `python -m trainer.remote serve` on the GPU box | Remote bar |
| [Startup at scale](#startup-at-scale) | automatic | -- |

Training from cached latents with the images deleted was already in upstream: set
`dataset.source = "latents"` (or leave `"auto"`) and keep each latent's `.txt`/`_nl.txt`. Run
`python -m trainer.tools.cache_latents audit <folder>` before deleting anything.

## StageLR

Chained schedule segments, each a share of the run, in the same table format as the diffusion-pipe
fork's `[StageLR] stages`, so those lists paste in unchanged:

```toml
[optimizer]
lr = 1e-7            # where the first stage starts

[schedule]
kind = "stage"
warmup_steps = 0     # optional; added before the stages
stages = [
  { type = "linear",   end_lr = 7e-6, percent = 0.1 },              # ramp 1e-7 -> 7e-6
  { type = "constant", lr = 7e-6, percent = 0.5 },
  { type = "rex",      max_val = 7e-6, min_val = 0, percent = 0.4 },
]
```

- **Types:** `linear` and `cosine` move from where the previous stage ended to `end_lr`. `constant`
  holds `lr`. `rex` decays from `max_val` toward `min_val`. The values are absolute learning rates.
- **Exact lengths:** the trainer knows its optimizer-step count before step 1, so stage lengths are
  integers that sum to it exactly. Every stage gets at least one step, and the rounding remainder
  goes to the last one. The update ranges are printed at startup:
  ```
  sched    stage (StageLR), 3 stage(s) over 850 updates
           linear    updates 1..85 (85), lr 1e-07 -> 7e-06
           constant  updates 86..510 (425), lr 7e-06 -> 7e-06
           rex       updates 511..850 (340), lr 7e-06 -> 2.01e-07
  ```
- **Every param group is scheduled.** The `stage-lr` package computes a single LR from group 0, and
  PyTorch applies it to group 0 only. With `[component_lr]` overrides, every other group sat at its
  initial LR for the whole run. Here a group with its own LR follows the same curve, scaled by its
  ratio to `optimizer.lr`.
- **Same curves as the package.** The formulas are the package's, including its REX endpoint (a REX
  stage stops just above `min_val`) and its warmup factor (`1/w + (1 - 1/w) * step / w`). The tests
  check exact agreement with `stage-lr` at the commit diffusion-pipe pinned.
- **Multi-GPU and resume:** under DDP the plan is evaluated per optimizer update, so the boundaries
  don't drift with world size. On resume, the plan is rebuilt for the resumed run's length and
  continues from the saved step.

## Artist attribution

Studio sidecars carry `Drawn by <artist>` as a tag, and NL captions usually repeat it as a sentence.
Attribution handling keeps that trigger intact through augmentation:

```toml
[dataset.caption]
caption_mode = "mixed"
shuffle_tags = true
tag_dropout_percent = 0.1
attribution_patterns = ['^drawn by\s']   # regex per line in the GUI; empty = off
attribution_position = "fixed"            # or "random"
attribution_dropout_immune = true
attribution_dedupe_on_combine = true
```

- An attribution is any tag or NL sentence matching a pattern (case-insensitive), wherever it sits.
  A post crediting **several** artists has several, and all of them are handled. The diffusion-pipe
  version only pulled out the first, so a second credited artist could still be shuffled away or
  dropped.
- `fixed` puts attributions first in their field, in their original order. `random` places each one
  anywhere in its field, still exactly once.
- `attribution_dropout_immune` keeps them out of tag dropout and shuffling. `min_tags_kept` counts
  them.
- `attribution_dedupe_on_combine`: the `tags_nl` and `nl_tags` variants join both fields. The
  trailing field's copy of an attribution is removed only when the leading field **already carries
  it**, so the artist always appears exactly once. The diffusion-pipe version stripped the trailing
  copy unconditionally, so a caption whose tag had been dropped lost the artist entirely.
- Works with every text-encoding mode. With `cache_text_embeddings` and no `caption_variations`,
  captions must be fixed, so use `fixed` position there. `random` needs variations or the live encoder.
- Existing caption-variation caches stay valid while `attribution_patterns` is empty. Enabling it
  changes the cache key, so new caption slots are built.

Also fixed here: shuffling NL sentences no longer produces `..` when the final sentence (the only
one carrying its own period) lands in the middle.

## Protected tags file

`[dataset.caption] protected_tags_file` is a real setting now (upstream popped it before the GUI
could see it, so loading and saving a config in the GUI lost it). One tag per line, `#` comments;
its tags are added to the inline `protected_tags` list. Illustration Scrapping Studio's training
export writes `protected_tags.txt` with every character and copyright tag in the exported folders.

- The GUI shows it as a file field and does not require the file to exist on the machine running
  the GUI: it is usually a path on the pod.
- The path is not part of the caption cache key; the tags it holds are, so editing the file builds
  new caption slots.

## MOSS-Image captions and layouts

For MOSS-Image, the MageTrail captioning script writes MOSS's own caption format next to each image:
`<stem>_moss.json` (captions at three lengths, watermark boxes, borders) and, for a share of images,
`<stem>_layout.json` (a summary plus `character` / `object` / `text` items with boxes; text boxes from OCR).
Boxes are `[y_min, x_min, y_max, x_max]` normalized to 0-1000 of the original image.

- **Variants.** `caption_mode` and `mixed_weights` accept `short`, `medium`, `long` and `layout` next to
  `tags`, `nl`, `tags_nl` and `nl_tags`. A sample without the file a variant needs falls back to its NL
  caption, then to its tags. The files travel with the latent cache (they match `*.json`).
- **Layouts per crop.** `layout` compiles the prompt MOSS-Image was trained on for each training crop: the
  dataset scales an image to cover its bucket and centre-crops it, so boxes are moved into the bucket's
  pixels, items mostly cut off by the crop are dropped (text needs 90% visible, other items 50%), and the
  coordinates are written as `abs_centralized` / `yxyx` with MOSS's rounding. One file serves every
  resolution. The layout variant needs the original image size, which latents carry in their metadata.
- **The artist.** The annotation captions never name the artist. `annotation_attribution_template`
  (default `"{attribution}. {caption}"`) adds the caption's attribution tags, e.g. `@foo bar`, to the
  short / medium / long captions and to the layout's summary caption; `""` leaves the artist out.
- **MOSS tag order.** MOSS recommends quality, meta, year, character, series, artist, general.
  `keep_order_through_attribution = true` (with `attribution_patterns = ["^@"]`) keeps every tag up to and
  including the last attribution tag in place and never drops them; shuffling and dropout apply only to the
  general tags after it.
- **Layout granularity.** `layout_item_dropout` (default 0, off) makes each layout sample leave out a random
  share of its items, drawn between 0 and the setting, so the model learns both complete and sparse layouts
  (Ideogram 4 likewise trains with optional boxes). MOSS requires a box on every item, so whole items go; their
  order is kept and at least one stays. With caption variations each slot draws its own subset; 0.5 gives
  everything from the full layout down to about half of it.
- **Panels.** Beyond MOSS's `character` / `object` / `text`, MageTrail layouts carry `panel` items: one box
  per comic or collage panel, captioned with what happens in it, so the tune learns panel composition (base
  MOSS-Image only has the summary's `canvas: "multi_panel"`). `layout_panels = false` trains on MOSS's
  vocabulary only. Prompt it as `{"label":"panel","caption":"...","bbox":[...]}` next to the other items.
  `layout_panels_as_objects = true` gives panels as `object` items captioned "panel: ..." instead (a caption
  whose first words already say "panel" is kept as is). That is what a box node that only draws objects, such
  as ComfyUI's default, sends, and base MOSS already follows object boxes captioned "panel". The layout files
  keep their `panel` items either way.
- **Short item captions.** Layout items carry a full one-sentence `caption` and a `short` one of a few words,
  the way people prompt a box ("Reimu", "Marisa holding a spoon on a stool"). `layout_short_captions` (default
  0) is the share of layout samples whose items use the short caption, so both prompting styles are learned;
  0.4-0.5 is a reasonable start. Items without one keep their full caption.
- **Character details.** Layouts box up to four distinctive details per character (accessories, hair
  ornaments, hats, worn weapons, standout outfit pieces) as `object` items with `part_of` (their character's
  index). The model sees plain MOSS objects, so standard object prompts gain fine placement; `layout_parts =
  false` leaves the details out (to compare, or to back out without re-captioning).
- Weights left at 0 for the new variants are dropped from the config, so configs that do not use them keep
  their caption-variation cache keys. When they are used, the annotation files and the crop are part of the
  key, so editing a caption builds new slots.

## Kahan + AdaLN warning

SDNQ's Kahan summation (`optimizer.use_kahan`) damages the shared AdaLN projection when AdaLN is trained. In
an audit of the MageTrail V0.4 pilot (FLUX.2 VAE, AdamW 8-bit) the shared projection moved 60.7% with Kahan
against 1.1% without, and block modulation 8.9% against 0.15%. Samples went unstable and the loss rose
instead of falling. Upstream regards it as an SDNQ issue and does not change it; its MOSS-Image and Flux3
presets set `use_kahan = false`.

The fork warns instead of refusing, since the combination is legal: the GUI shows it as an advisory, the
remote server's config check returns it, and the trainer prints it at startup. It fires for an SDNQ optimizer
(`adamw8bit`, `adafactor`, `came`, `lion`, `muon`) with `use_kahan = true` in a full finetune that trains
AdaLN (an unset `component_lr.adaln` means it trains at the main LR). It applies to every model family with a
shared modulation, MOSS-Image's `shared_block_modulation` included. Fix: `use_kahan = false`, or freeze AdaLN
with `[component_lr] adaln = 0`.

## Experiment tracking

```toml
[tracking]
backends = ["tensorboard", "wandb"]   # any of tensorboard, wandb, trackio
project = "magetrail"
# wandb_mode = "offline"              # write locally; `wandb sync` later
# trackio_space_id = "user/board"     # Trackio on a Hugging Face Space; empty = local
```

Install the ones you use with `pip install -r requirements-tracking.txt`.

- **Logged at every `log_every`:** `train/loss` (plus `train/mse` and `train/hf_loss` with the HF
  loss), `train/lr` and `lr/<component>` for each component group, `train/grad_norm` (the total norm
  before clipping), `train/samples_per_second` (counts actual bucket sizes, all ranks),
  `train/seconds_per_step`, `train/peak_memory_gb` (worst rank), `train/epoch`, `train/progress` and
  `train/eta_seconds`. Eval and sampling add `eval/*` and `samples`.
- **Never takes the run down.** A backend that fails to start is skipped. One that fails mid-run is
  disabled with one warning, and training and the other backends continue. If an online wandb start
  fails, it falls back to offline instead of losing the run's data.
- **Resumes.** With `train.resume_from`, wandb continues the same run (its id is kept in
  `<run>/tracking_run.json`) and Trackio continues the run of the same name. Turn this off with
  `resume_run = false`.
- **No API keys in the config.** The whole config is embedded in every exported checkpoint's
  metadata, so a key there would ship with every LoRA you share. Use `WANDB_API_KEY` or
  `wandb login`. Trackio Spaces use your Hugging Face login.
- **Files:** TensorBoard and local Trackio data go under `<output_dir>/<run_name>/tracking/`. View
  them with `tensorboard --logdir <that folder>` or `trackio show --project <project>`. On a rented
  GPU, tunnel the port (`ssh -L 6006:localhost:6006 ...`) or use wandb or a Trackio Space.

## Validation samples

```toml
[sampling]
prompts = [
  "Drawn by emily (pure dream), 1girl, smile, outdoors",
  { prompt = "Drawn by alice, 1girl, night city", width = 832, height = 1216, seed = 7 },
]
every_n_steps = 500      # and/or every_n_epochs
at_start = true          # baseline of the untrained model
steps = 30
cfg = 5.0
shift = 6.0
width = 1024
height = 1024
negative_prompt = " "    # the reference blank
```

- **No resident text encoder.** Blu's concern was that sampling needs the text encoder loaded. Here
  prompt and negative embeddings are encoded once at startup, with the training captions, into
  whichever text cache the run uses. With a warm caption-variation cache, they're read from SQLite
  and the encoder never loads. With the live encoder, they go through it as usual.
- **Decoder only.** Training loads Mage-VAE without its decoder. Sampling loads a decoder-only copy
  once, keeps it in host RAM and moves it to the GPU only to decode. FLUX.2-VAE runs are decoded
  through FLUX.2 by inverting the batch-norm packing.
- **Reference sampler:** Microsoft's pipeline settings. Static-shift sigmas, Euler steps,
  dual-forward CFG (one fused batch-2 forward, skipped entirely at `cfg <= 1`), optional CFG
  renormalisation, CPU-seeded noise. `seed_strategy = "fixed"` reuses each prompt's seed every round,
  so differences between rounds come from training, not noise.
- **Multi-GPU:** each rank renders its share of the prompts, then rank 0 gathers and logs them.
- **Output:** `<run>/samples/<step or epoch>/NN_<label>_seed<seed>.png` plus `prompts.json`, and the
  `samples` gallery in each tracker.
- Sampling never consumes the training RNG stream, and the model returns to train mode afterwards. A
  failing prompt is logged and skipped; it never stops the run.

## Held-out eval loss

```toml
[eval]
path = "/data/magetrail/heldout"      # cached like any subset; Cache latents includes it
every_n_steps = 250
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
batch_size = 4
max_samples = 500                      # optional fixed spread of the folder
```

Each round scores the same images at the same timesteps against the same noise, with un-augmented
captions (mixed mode evaluates on tags). Timesteps are quantiles of the training timestep
distribution, shift included. Only the weights change between rounds, so `eval/loss` and the
per-quantile `eval/loss_qX.XX` move only when the model does. Training loss can't do this: it
re-draws timestep and noise every step. Low quantiles track detail, high ones composition. Work is
shared across ranks.

## Save now and Save & stop

While a run is going, the GUI shows **Save now** (checkpoint and continue) and **Save & stop** (a
resumable checkpoint with optimizer state, then a clean exit). Both drop a file in the run folder,
and anything else can do the same, such as a notebook cell or `ssh`:

```bash
touch output/<run_name>/save        # save at the next optimizer step, keep training
touch output/<run_name>/save_quit   # save resumable state, then stop
```

Rank 0 checks after every optimizer step and broadcasts the decision, so all ranks save together.
Resume with `train.resume_from = "output/<run>/<run>-stepNNNNNN-state"`.

## Thousands of folders: `subsets_file`

The Studio's training-layout export lists one folder per artist with its repeat count. Point the
config at it instead of writing thousands of `[[dataset.subsets]]` blocks:

```toml
[dataset]
subsets_file = "/data/magetrail/exports/run-3/training/folders.csv"   # or folders.json / dataset.toml
subsets_root = "/workspace/library/images"   # optional: re-root paths written on another machine
```

- **Formats:** CSV with a `path` column and optional `num_repeats`/`repeats` and `texture`. JSON
  lists of the same keys. TOML `[[subsets]]` or diffusion-pipe `[[directory]]` tables. Relative
  paths resolve against the file's folder. Rows with 0 repeats or 0 images are skipped.
- **`subsets_root`** keeps each folder's name and swaps everything above it. For example,
  `/home/jovyan/Artist-Collection-Dataset/images/emily` becomes `/workspace/library/images/emily`.
- **Checkpoint metadata** records the file path, not ten thousand expanded rows. The startup report
  lists the 15 largest subsets and summarises the rest.
- **Combining:** inline `subsets` rows can be added alongside the file. `dataset.path` can't.

## Latent cache size and speed

- **`dataset.latent_dtype = "float16"`** halves new caches. A 1024 px latent is 2 MB in float32 and
  1 MB in float16, at about 5e-4 relative rounding, which is well below the VAE's own error.
  Training reads every precision and computes in float32. A value that would overflow float16 is
  kept in float32 automatically.
- **`python -m trainer.tools.cache_latents convert --config run.toml --dtype float16`** rewrites an
  existing cache in place, with no VAE and no GPU. Each file is swapped in atomically, so an
  interrupted run leaves every file either converted or untouched. Add `--dry-run` to see the saving
  first.
- **`cache-config` now runs in one process.** It used to start one Python process and load the VAE
  once per subset, which takes hours at 10,000 subsets. It now loads the VAE once, covers inline
  subsets, `subsets_file` and `[eval]`, and with `--devices 0,1,...` splits all folders' images
  across GPUs together. The GUI's Cache button and the cache step before Start Training use it too.
- A folder whose images were deleted after caching counts as "nothing to do", not an error.
- Fixed: the planned cache size printed by `cache` was half the real float32 size.

## Caption cache built ahead of time

`python -m trainer.training.train <config> --text-cache-only` builds the persistent caption cache
(`train.caption_cache_path`, needs `cache_text_embeddings = true` and `caption_variations > 0`) and exits before the
transformer loads, so a smaller GPU can prepare it. Under `accelerate launch --num_processes N` the captions are
split across the processes as in a training start. Entries are keyed by each image's folder **name** and file stem
(not the absolute folder path), so the database stays valid when the dataset is downloaded somewhere else.
Caches written before this change (`caption-slots-v1`) plan their slots again once; embeddings of captions that
come out the same are reused.

## Startup at scale

- **Rank 0 scans, the others receive.** Every rank used to scan the whole dataset: an 8-GPU job read about 600k image (or cache) headers, captions and cache files eight times over the same volume. Now rank 0 scans and broadcasts the result. If the scan fails, the error is raised on every rank instead of leaving the others waiting until the one-hour process-group timeout.
- **Folders are scanned in parallel.** From 8 subsets up, rank 0 scans on up to 16 threads, and results are merged in folder order, so entries and batches are identical to a sequential scan. Measured on 300 folders x 6 images: 7.2 s vs 27.9 s with a cold file cache, 2.1 s vs 2.9 s warm. `MAGEFLOW_SCAN_WORKERS` overrides the thread count, and 1 turns threading off.

## Not ported, and why

| diffusion-pipe feature | Status here |
| --- | --- |
| AdamW8bitKahan (bitsandbytes) | Not needed: `optimizer.kind = "adamw8bit"` is SDNQ's 8-bit AdamW with stochastic rounding, plus `use_kahan = true`. |
| Block compile, selective checkpointing, packed attention | Already upstream: `train.compile`, `checkpoint_blocks`, `torch_varlen` (the batched execution and packed attention came from the fork at `40bf63a`; see NOTICE). |
| Live progress bar, ETA, training plan | Already upstream: tqdm bar, ETA, exact step count in the startup report. |
| Lium notebook relay | Specific to one host's dataset staging. |
| automagic, fftdescent, ocgoptv2, wiwiopt optimizers | Not requested. SDNQ and Optimi cover the used cases. |

## Testing

New tests: `tests/test_stage_lr.py`, `test_attribution.py`, `test_tracking.py`,
`test_sampling_eval.py`, `test_dataset_scale.py`, `test_gui_magetrail.py`, and
`test_trainer_e2e.py`. The last one runs a real `Trainer` on CPU with every feature above enabled,
stops it with `save_quit` and resumes it to completion. They run on CPU (`MAGE_FLOW_ALLOW_CPU=1` is
set by the e2e test). Multi-GPU paths (prompt and eval sharding, the signal broadcast) follow the
same collective patterns as upstream but haven't been run on more than one GPU yet.
