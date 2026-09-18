# RTI implementation and resume plan

Status: experimental foundation implemented and CPU-tested; production integration
and real-model validation remain pending. Last updated 2026-09-16.

Stable starting point: `9782acb` pushed to `origin/main`. Dual-timestep training
is available there. Leave the GPUs free for the user's LoKr training unless
they explicitly make one available for RTI tests.

## Objective

Adapt Region Token Interface (RTI) to Mage-Flow, preserving dense inference as
a reference, reducing middle-block image-token counts gradually, and measuring
quality before claiming faster training or useful compression. RTI and
dual-timestep noising must be mutually exclusive, with explicit validation.

References: [paper](https://arxiv.org/abs/2608.29281),
[authors' implementation](https://github.com/eduardzamfir/RTI), inspected commit
`239d580`. Local read-only reference clone: `/tmp/rti-paper-review`; PDF:
`/tmp/paper-2608.29281.pdf`. These temporary files may disappear; production
code must not depend on either path.

## Decisions

- New work starts under `trainer/experimental/`, not in the active training loop.
- Keep a dense prelude and coda. A middle span is a hypothesis, not a chosen
  production default: audit Mage-Flow features before fixing it.
- Explicit 100% token retention calls the original forward unchanged.
- Read starts as mean pooling, size embeddings at zero; Write initially
  broadcasts the core's **delta**, rather than zeroing out the core contribution.
- Retain pre-core dense image features and add individualized region updates.
- Region positions average the original complex RoPE phasors, without unit
  renormalization. Text tokens and their primary timestep remain unchanged.
- Support rectangular latent grids. Our 1344x768 images produce 84x48 tokens;
  divisible-by-16 images do not imply square/power-of-two token grids.
- The first rectangular strategy filters a enclosing-square Hilbert traversal.
  Mandatory cuts at any resulting spatial discontinuity keep each region
  connected. Enforce/report the minimum feasible region count; never count a
  disconnected region as valid compression. Compare against a rectangular
  space-filling traversal if that floor becomes material.
- Begin compressed learning at 98% token retention and anneal by **optimizer
  step**, initially toward 75%. These numbers are proposed pilot settings.
  Do not confuse this with learning a policy over denoising timesteps.
- Quantize/cache the existing base as usual; the new interface stays trainable
  with FP32 pooling/reductions. No EMA teacher is required by this design.
- Region routing is nondifferentiable, computed outside checkpointed blocks;
  Read/Write and the region-core feature path remain differentiable.
- No claim that a 50% token budget halves total time, VRAM, or optimizer state.

## Milestones

### 1. Isolated foundation (current)

- [x] Rectangular connected partitions and explicit budget floor.
- [x] Read/Write with size embeddings, pooled RoPE and correct delta initialization.
- [x] Deterministic step-based cosine budget annealing, serializable settings.
- [x] Single-image Mage-Flow prototype with exact original-forward dense mode.
- [x] CPU tests: rectangular coverage/connectivity, pooling reconstruction,
  dense parity, frozen-base/interface gradients, conflict checks, schedule resume.

### 2. Measure redundancy before choosing a core

- [ ] Audit frozen MageTrail block features on a fixed cached image/caption set
  at several noise levels and budgets; record reconstruction error by block.
- [ ] Include rectangular examples and detail-heavy images, not just flat scenes.
- [ ] Select candidate core spans and initial target budgets from that evidence.
- [ ] Run GPU forward/backward parity and measure Read/Write overhead separately.

### 3. Production integration

- [ ] Add `[rti]` config validation and GUI controls; reject RTI + dual timestep.
- [ ] Native-resolution packed execution: per-image regions and RoPE, correct
  varlen boundaries, unchanged text lengths, restore dense layout before loss.
- [ ] SDNQ frozen base + LoKr integration; do not freeze/quantize the new interface
  accidentally. Test compressed FP32 AdaLN and ordinary Mage-Flow separately.
- [ ] Put interface parameters into optimizer groups, synchronize under DDP,
  and handle unused interface parameters on dense steps explicitly.
- [ ] Use a small discrete budget set for compile reuse; anneal sampling toward
  the target, retaining nearby budgets. Synchronize budget choice across ranks.
- [ ] Store RTI version/config/interface weights and exact budget-schedule progress
  with checkpoints and optimizer resume state. Reject incomplete RTI exports.
- [ ] Save/load roundtrip and resumed-vs-uninterrupted parity tests before GUI use.
- [ ] ComfyUI custom-node support; an ordinary merged LoKr alone is insufficient.

### 4. Controlled pilot and release decision

- [ ] Compare dense LoKr, fixed mild RTI, and annealed RTI on identical data,
  seeds, optimizer settings and effective batch. Dual timestep off throughout.
- [ ] Evaluate held-out prompts, anatomy, fine detail, text and style separately.
- [ ] Compare matched updates AND matched wall-clock budgets; report peak VRAM,
  optimizer/DDP costs and inference throughput separately.
- [ ] Only lower retention toward 50% when quality measurements justify it.
- [ ] Decide whether a learned timestep-dependent budget is worth a new experiment.

## Resume instructions

1. Read this plan and `git status`; do not include unrelated untracked datasets,
   checkpoints, PDFs or personal configs in commits.
2. Run the foundation tests listed in the completed-work section below.
3. Continue with the first unchecked milestone. Confirm GPU availability before
   any GPU work; user training takes precedence.
4. Update this file with changes, exact commands/results and remaining limitations.

## Completed work and next action

Implemented in `trainer/experimental/rti.py`:

- `rectangular_hilbert_order` / `partition_regions`: complete token coverage,
  mandatory cuts across disconnected jumps, requested and actual region counts.
- `RegionInterface`: differentiable FP32 reductions, initialized mean Read,
  region-size embeddings, mean complex RoPE and delta-broadcast Write.
- `BudgetSchedule`: deterministic cosine schedule; reconstruct from serialized
  dataclass settings and completed optimizer step. Production resume not wired.
- `RTIMageFlow`: opt-in **single unpacked image** wrapper around an existing
  backbone. `keep_fraction=1` uses the original forward exactly. Compressed
  forward preserves text, timestep/AdaLN and the dense image skip. It rejects
  dual-timestep inputs. It does not alter backbone parameter trainability.

Validation command (CPU only; no GPUs used):

```bash
.venv/bin/python -m unittest discover -s tests -p test_rti.py
```

Result: **6/6 passed**. Includes actual 84x48 and 48x84 token-grid partitions,
ordinary and compressed tiny Mage-Flow models, exact dense output parity,
interface-only optimization with a frozen base, checkpointed vs uncheckpointed
gradients, tensor state roundtrip and budget-schedule reconstruction.

RTI is not exposed in production config or GUI. No real MageTrail RTI checkpoint
has been trained/exported. Packed batches, GPU/compile/DDP/SDNQ integration,
quality testing and ComfyUI loading are explicitly unvalidated/pending.
The prototype's Python routing and per-block metadata construction are not
optimized, so do not use it to claim production throughput.

**Next action:** implement a read-only feature-redundancy audit for the real
MageTrail checkpoint and a fixed cached dataset. Record reconstruction error
per layer at multiple noise levels and 98/90/75/50% retention. Do not train a
compressed span until those measurements justify its placement. Run that GPU
audit only after the user makes a GPU available.
