# Experimental Self-Flow training

Self-Flow is available in the normal CLI trainer and in **Method → Experimental
Self-Flow** in the GUI. Restart an already-open GUI to load the new controls.

The local starter config is
`configs/Overfit-mageflow-Finetune-SelfFlow-Adafactor-swap-vsf3-512px.toml`.
It uses the original synthetic-1 dataset and caption-cache paths, the VSF3-adapted
base, SDNQ INT8 training, Adafactor RMS clipping, LR 1e-5, REX with 100 warmup
steps, two epochs, and batch size 2 per GPU. It keeps the original six cached
caption variations. Gradient accumulation can increase effective batch size.

```toml
[self_flow]
enabled = true
ema_dtype = "bfloat16"
ema_device = "cuda"
adaln_fp32 = true
stochastic_rounding = true
decay = 0.99
weight = 0.8
```

The student feature comes from block 4, the teacher from block 8. Teacher EMA
includes the input and timestep projections and shared compressed modulation.
Modulation and timestep EMA stay FP32; other teacher weights use BF16 stochastic
rounding. EMA updates once per completed optimizer update, including under
gradient accumulation. DDP synchronizes EMA rounding RNG across ranks.

The implementation requires Mage-Flow **full finetuning**, packed cached latents,
Cached Text Encoder, and dual-timestep noising. Batch size is configurable per GPU;
increase it in the GUI's normal batch-size control or `train.batch_size` until you
reach your desired VRAM usage. Mixed-resolution batches use packed attention with
independent sample boundaries, timestep pairs and teacher features. Flow and
alignment losses each weight images equally rather than weighting by token count.
RTI, curriculum, preservation, HF loss, OT, optimizer gradient release, frozen
quantization and quantized matmul are rejected with a clear configuration error.
The GUI exposes all EMA
settings, including CPU storage and full FP32 EMA.

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer.training.train \
  configs/Overfit-mageflow-Finetune-SelfFlow-Adafactor-swap-vsf3-512px.toml

# Two GPUs; alternatively select both GPUs in the GUI.
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 --module trainer.training.train \
  configs/Overfit-mageflow-Finetune-SelfFlow-Adafactor-swap-vsf3-512px.toml
```

Inference exports contain the ordinary student weights and keep their existing
Mage-Flow/VSF3 loading requirements. They exclude the feature projector and teacher.
No additional Self-Flow inference node is needed.

With `train.save_optimizer_state = true`, the sibling `-state` folder includes
the student/projector, optimizer, scheduler, RNG and teacher EMA for resume.
SDNQ model resume state uses PyTorch serialization to preserve tensor wrappers;
inference exports remain safetensors. Use the checkpoint or its `-state` folder
as `train.resume_from`. Keep EMA decay, rounding and precision settings unchanged
when resuming. A fresh run from inference weights starts a new teacher and projector.

The `sf` log value is the unweighted cosine alignment loss; total loss already
includes `self_flow.weight * sf`. Loss magnitude alone does not demonstrate better
generation quality or faster convergence. Compare held-out prompts and checkpoints.

For measured GPU/CPU EMA and DDP feasibility results, see
[the experimental measurements](self-flow-finetune-probe.md).

Integration validation on September 21, 2026: the normal trainer completed two
single-GPU updates, saved native inference and full resume checkpoints, restored
step 2 and completed step 3. The normal two-GPU training path also completed two
updates. The native export had 399 tensors and no feature-projector keys. A full
offscreen GUI load/collect/validation round trip retained the Self-Flow settings.
Focused tests reported 19 passes and two GPU-dependent skips. Smoke-test logs and
metadata verification remain under `benchmarks/self-flow-finetune-512/`; temporary
smoke-test model/state files were removed after verification.

Packed-batch validation on the same date used cached 512-area latents, SDNQ INT8,
Adafactor, mixed GPU EMA with FP32 AdaLN, checkpointing, compilation and torch
varlen attention. Batch size 2 completed three single-GPU updates, peaking at
14.0 GB allocated; the two post-compilation steps took 0.78 and 0.76 seconds.
Two-GPU DDP completed two updates at batch size 2 per GPU, peaking at 14.3/14.4 GB
allocated, with the second step taking 2.02 seconds. These are short feasibility
checks on ten images, not convergence or maximum-batch measurements; memory uses
the trainer's decimal GB units. Packed mixed-resolution output, sample isolation
and backward equivalence tests passed (12 passed, one skipped across the focused
Self-Flow and dual-timestep suites). Logs are `batch2.log` and `batch2-ddp.log`
under the benchmark directory above.
