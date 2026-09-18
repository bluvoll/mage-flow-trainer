# RTI port plan — Mage-Flow, packed/varlen first

Status: **initial packed/varlen implementation landed locally.** The core interface,
Gilbert traversal with mandatory discontinuity cuts, packed Mage-Flow wiring, config
validation, quantization exclusion, checkpoint metadata/loading, trainer scheduling,
and GUI controls are implemented. `tests/test_rti_packed.py` is the CPU structural
suite; a CUDA packed-kernel smoke test remains the next validation milestone. Supersedes
milestone 3 of `docs/rti-plan.md`, which stays the research record. Last updated
2026-09-18.

Source material: `trainer/experimental/rti.py` (the CPU-tested single-image prototype,
6/6 green) and the Anima implementation at `/home/bluvoll/cabal-SDXL-trainer`
(`anima_utils/elastic_tokens.py` plus its wiring in `model.py`, `compiled_runtime.py`,
`save.py`, `optimizer_groups.py`, `train-anima.py`, `main-anima.py`).

## 0. Scope locked for this landing

| Decision | Value | Consequence |
|---|---|---|
| Forward paths | **Packed/varlen only** | `[rti].enabled` requires `train.pack_resolutions = true`. The uniform-batch `forward` gets a `b == 1` CPU test seam only. `batched.py`'s compiled hot path is not touched. |
| Training mode | **Full finetune only** | `[rti].enabled` with a LoRA/LyCORIS adapter is refused at config load. LoKr-on-RTI comes after a trained RTI base exists. |

Both follow from the same fact: LyCORIS/PEFT state dicts structurally cannot carry
non-adapter tensors, so a trained interface would be **silently discarded at every
save**; and per-sample region RoPE would require editing the single compiled
`block_forward` used by every run, RTI or not.

## 1. Evidence base (measured, not assumed)

Everything below was measured directly against the two repos before planning.

**Traversal.** Anima's gilbert2d beats the prototype's enclosing-square Hilbert filter
on rectangles: 0 discontinuities on 84×48, 48×84, 33×45, 84×33, 99×45, versus 6 for the
prototype on 84×48. Sweeping every grid from 1×1 to 80×80: even×even and odd×odd are
**always** contiguous; mixed-parity grids are contiguous ~54% of the time and otherwise
have **exactly one** discontinuity. Worst case anywhere is 1.

Anima nonetheless *raises* `ValueError` on any discontinuity
(`elastic_tokens.py:97-107`) — it failed 191 of 990 shapes I tried. That converts a
one-region cost into a crashed step. Under varlen, arbitrary `(h,w)` is the norm rather
than the exception, so this must not be ported as-is.

**Identity initialisation.** Exact: bit-exact (max err 0.0) dense round-trip at budgets
1.0, 0.5 and 0.25, and a core-side delta broadcasts exactly to every region member.

**Gradient reachability**, with a core whose output genuinely depends on its input:

| keep | `read_score` | `size_embedding` | `write_map` |
|---|---|---|---|
| 1.0 | **exactly 0** | 4.6e-2 | 1.7e-1 |
| 0.75 | 5.1e-2 | 2.8e-2 | 1.5e-1 |
| 0.5 | 8.3e-2 | 2.7e-2 | 1.6e-1 |
| 0.25 | 9.1e-2 | 4.2e-2 | 1.2e-1 |

A one-member segment softmax is identically 1.0, so the identity phase is *structurally
unable* to train READ, while still training WRITE and the size embedding. This is a
property to document and assert, not a bug.

## 2. Design decisions

| # | Decision | Why |
|---|---|---|
| **D1** | **Attach, don't wrap**: `transformer.region_interface`. Discard the prototype's `RTIMageFlow` wrapper (`rti.py:160`). | A wrapper breaks `export.py:16` (`model.params`), `export.py:22` (`hasattr(model,"_lycoris_config")` — a LyCORIS run would silently take the PEFT branch), `loader.py:94-98` (prefix strip), `train.py:336` (`.patch_size`), `train.py:365` (`configure_execution`), `train.py:411` (`quantize_module`). |
| **D2** | **Packed-only** (§0). | `batched.py:36` hard-codes rank-2 RoPE (`freqs_complex.view(1,1,L,Dh/2)`); per-sample region RoPE needs `[B,R,64]`. |
| **D3** | **Gilbert2d traversal + the prototype's discontinuity policy.** Never raise; force a mandatory cut at each jump, report the floor. | §1. Neither implementation is correct alone: anima has the better curve and the wrong failure mode; the prototype has the right failure mode and the worse curve. |
| **D4** | **No `keep_fraction == 1` short-circuit.** Reverses `docs/rti-plan.md:28`. | Identity through the region path is exact, so the bypass buys no accuracy. Two of three interface tensors *do* train during the identity phase (§1), so the phase is a genuine WRITE warm-up. `read_score` gets no gradient there; `find_unused_parameters=True` (`train.py:159`) already covers it. |
| **D5** | **One flat `RegionPlan` over the concatenated packed sequence**, not a list of per-image states. Per-image offsets are baked into `order` and `labels`. | READ/WRITE become single `index_select`/`index_add`/`index_copy` calls over `[1, Σnᵢ, D]`. The only Python loop is in the no-grad integer plan construction. This is the core varlen design element and has no counterpart in either reference. |
| **D6** | **Partition at the READ point, inside the block loop** — not before it. | Both references do this (`rti.py:206-209`; anima `compiled_runtime.py:413-419` with labels from `curve_tokens.detach()`, `elastic_tokens.py:311`). Routing on pre-prefix features would score cut boundaries in one representation while pooling another, and would remove the entire reason the dense prefix exists. |

## 3. New module: `trainer/modeling/region_tokens.py`

`trainer/experimental/rti.py` is left byte-identical (its 6 tests stay green), plus a
docstring note pointing here.

### 3.1 Traversal

```python
@lru_cache(maxsize=512)
def gilbert_order(height, width) -> tuple[tuple[int, ...], tuple[bool, ...], int]:
    """Row-major flat indices along a generalized Hilbert curve, the per-step
    discontinuity mask, and the connectivity floor (jumps + 1)."""
```

Port `_sign` and `_generate_gilbert_2d` verbatim (`elastic_tokens.py:17-74`), the
orientation choice at `:84-87`, and the coverage assertion at `:89-95`. **Delete the
`ValueError` at `:97-107`**; compute the jump mask and `minimum = sum(jumps) + 1` in
Python at cache-fill time.

That single change also kills `rti.py:76`'s `int(jumps.sum().item())` — a **GPU sync per
image per step inside the block loop**. `minimum` becomes a cached Python int.

```python
def gilbert_permutation(height, width, device) -> tuple[Tensor, Tensor, int]
```

Module-global dict keyed `(h, w, str(device))`, **LRU-bounded to ~256 entries per
device**. Anima's `_DEVICE_ORDER_CACHE` (`elastic_tokens.py:111-124`) is unbounded — a
real leak under native-resolution packing. Deliberately a module global rather than a
buffer or module attribute: `loader.py:91-105` shows that anything created under
`torch.device("meta")` that is not a Parameter cannot be materialised by
`load_state_dict(assign=True)`.

### 3.2 The ragged plan

```python
@dataclass
class RegionPlan:
    order: Tensor          # int64 [Σn] curve permutation of the CONCATENATED sequence
    labels: Tensor         # int64 [Σn] GLOBAL region id, per-image offsets baked in
    counts: Tensor         # int64 [Σr]
    image_lengths: list[int]
    region_lengths: list[int]
    requested: list[int]
    minimum: list[int]
    atom_to_token: Tensor | None = None   # reserved seam for K>1 atoms; must be None today
```

`build_region_plan(features, shapes, image_lengths, keep_fraction, device)` is
`@torch.no_grad()` (routing is nondifferentiable, `rti-plan.md:46-47`). Per image:

```python
perm, jumps, minimum = gilbert_permutation(h, w, device)
requested = max(1, min(n, int(math.floor(n * keep_fraction + 0.5))))
r = max(requested, minimum)
if r == n:
    local = torch.arange(n, device=device)
else:
    ordered = features[off : off + n].index_select(0, perm).float()
    scores  = (ordered[1:] - ordered[:-1]).square().sum(-1).masked_fill(jumps, torch.inf)
    cuts    = scores.argsort(descending=True, stable=True)[: r - 1]
    starts  = torch.zeros(n, dtype=torch.long, device=device)
    starts[cuts + 1] = 1
    local   = starts.cumsum(0)
```

Three deliberate choices:

- **`argsort(descending=True, stable=True)` replaces the prototype's `topk`**
  (`rti.py:84`). `topk` has no tie-stability guarantee, so it loses anima's **nesting
  property**: with a budget-independent stable ranking, `cuts(R₁) ⊂ cuts(R₂)` for
  `R₁ < R₂`, so annealing only *merges* regions and the interface learns a stationary
  function instead of chasing a reshuffling partition. `O(N log N)` vs `O(N log k)` is
  noise at N≈4032.
- **`masked_fill(jumps, inf)` plus `r = max(requested, minimum)`** is kept from the
  prototype (`rti.py:75-77,83`). The `inf` entries sort first, so every discontinuity is
  guaranteed to become a cut. Anima has no equivalent.
- **`round`-half-up as explicit integer arithmetic** replaces `ceil` (`rti.py:115`), so
  every DDP rank derives the same `rᵢ` from the same `(keep, nᵢ)`.

`counts` uses `zeros(R, long).scatter_add_(0, labels, ones_like(labels))`, **not
`torch.bincount`** — bincount computes a device-side max to size its output and
therefore syncs. With this, plan construction performs **zero device syncs and zero H2D
copies** after warm-up, honouring the rule `mageflow_attention.py:48-52` documents.

### 3.3 `RegionInterface`

```python
class RegionInterface(nn.Module):
    def __init__(self, width, size_buckets=17, *, core_start, core_end):
        self.read_score     = nn.Linear(width, 1)            # zero-init
        self.size_embedding = nn.Embedding(size_buckets, width)  # zero-init
        self.write_map      = nn.Linear(2 * width, width)    # [0 | I], bias 0
```

**Submodule names are load-bearing.** They must never be named `img_in`, `txt_in`,
`proj_out`, `norm_out`, `pos_embed`, `time_text_embed` or `txt_norm`: `_LORA_TARGETS`
matched by `name.endswith("." + s)` (`params.py:44`, `:225-227`) would wrap them with
LoRA, and `_HIGH_PRECISION` (`quant.py:17-25`) matched by dotted-component equality
would silently skip them from quantization.

**`size_buckets = 17` fixed** — reject anima's resolution-derived
`ceil(log2(max_spatial_tokens))+1` (`elastic_tokens.py:230`). Under varlen there is no
bucket list and no `max_spatial_tokens`; a derived bin count bakes a tensor shape into
the checkpoint and forces the zero-pad-on-load workaround anima needs in two places.
17 bins covers 2¹⁶ tokens.

READ, flat over the whole packed batch (`D = img.shape[-1]`,
`buckets = self.size_embedding.num_embeddings`):

```python
ordered = img[0].index_select(0, plan.order)
scores  = self.read_score(ordered).squeeze(-1).float()
maxima  = scores.detach().new_full((R,), -torch.inf).scatter_reduce_(
              0, plan.labels, scores.detach(), reduce="amax", include_self=True)
weights = (scores - maxima[plan.labels]).exp()
sums    = scores.new_zeros(R).index_add(0, plan.labels, weights).clamp_min_(1e-12)
weights = weights / sums[plan.labels]
pooled  = ordered.new_zeros((R, D), dtype=torch.float32).index_add(
              0, plan.labels, ordered.float() * weights[:, None])
size_ids = plan.counts.clamp_min(1).float().log2().floor().long().clamp_max(buckets - 1)
pooled   = pooled + self.size_embedding(size_ids).float()
phases   = torch.view_as_real(dense_freqs.index_select(0, plan.order)).float()
reduced  = phases.new_zeros((R, *phases.shape[1:])).index_add(0, plan.labels, phases)
reduced  = reduced / plan.counts.clamp_min(1)[:, None, None]
return pooled.to(img.dtype)[None], torch.view_as_complex(reduced.contiguous())
```

- **`maxima` is detached.** The shift is mathematically a no-op (softmax is
  shift-invariant), so differentiating the `amax` scatter gains nothing and `-inf` is the
  one value in READ that could turn a backward into NaN. Anima leaves it live
  (`elastic_tokens.py:286-289`); do not copy that.
- **Keep the prototype's complex-phasor RoPE pooling** (`rti.py:147-150`); **reject
  anima's `(cos, sin)` angle tuple** (`elastic_tokens.py:326-343`). Mage-Flow RoPE is
  already unit-magnitude complex (`torch.polar`, `mage_layers.py:142`), so `view_as_real`
  *is* `(cos θ, sin θ)`. Pooled region freqs concatenate to rank-2 `[Σrᵢ, 64]` exactly
  like the dense table — **no `batched.py` change at all in the packed path.**
- Anima's numerical guards adopted: `clamp_min_(1e-12)` on the denominator, `clamp_min(1)`
  before `log2` and on the RoPE divisor — but computed **out-of-place**; anima's
  `counts.unsqueeze(-1).clamp_min_(1.0)` (`:335`) mutates `counts` through a view.

**No magnitude renormalisation** (`rti-plan.md:33`). The reason is region *spatial
spread*, not coordinate centring: with `axes_dim=[16,56,56]` and `theta=10000`, the
fastest spatial channel has ω = 1.0 rad/token, and a contiguous run of k positions pools
to `|sin(kω/2) / (k sin(ω/2))|` = 0.878 / 0.474 / 0.197 for k = 2 / 4 / 8, and 0.018 at
k = 64. It can also cross zero and flip sign (k=8, ω=1 gives sin(4) < 0), which is a π
rotation rather than attenuation. Because the pooled phasor multiplies both q and k,
logits scale by the *product* of two magnitudes, so heavily-merged regions are
systematically down-weighted as keys. `scale_rope` centring is irrelevant here — the
coordinate run is contiguous and `|mean exp(i(p₀+k)ω)|` is translation-invariant.

The frame axis is unaffected (all tokens have frame 0, so those 8 channels pool to unit
magnitude). **Log `|pooled_freq|` per axis group and its minimum over regions** so the
effect is observable. If a later audit says it matters, the fix is one line — divide
`reduced` by its per-channel modulus — and it preserves the keep=1.0 identity exactly.

WRITE, with the delta computed internally (anima's encapsulation,
`elastic_tokens.py:366-393`; the prototype's caller-computed delta at `rti.py:214` can be
silently violated and gets worse under varlen):

```python
delta   = region_out[0] - region_in[0]
member  = delta.index_select(0, plan.labels)
ordered = dense_img[0].index_select(0, plan.order)
update  = self.write_map(torch.cat((ordered, member), dim=-1))
return dense_img[0].index_copy(0, plan.order, ordered + update)[None]
```

`plan.order` is a full permutation of `range(Σnᵢ)`, so every row is overwritten and the
base tensor contributes no gradient path. Out-of-place `index_copy` is autograd-safe and
restores row-major order.

**Interface parameters stay in trunk dtype (bf16).** `rti-plan.md:45-46` asks for FP32
pooling/reductions, which the code above does; it does not require fp32 *parameters*.
Honest accounting: bf16 params avoid retaining the `cat(ordered, member)` fp32 buffer for
backward, roughly one of the two ~100 MB buffers at 4×4032 tokens — not both, since
`ordered.float() * weights` is materialised either way.

### 3.4 `BudgetSchedule`

```python
@dataclass(frozen=True)
class BudgetSchedule:
    start_keep: float = 0.98
    target_keep: float = 0.75
    identity_steps: int = 0
    warmup_steps: int = 0
    anneal_steps: int = 1000
    budget_steps: tuple[float, ...] = ()   # discrete grid; () == continuous

    def resolve(self, optimizer_step: int) -> tuple[float, str]:   # (keep, phase)
```

Phases `identity` / `warmup` / `anneal` / `target`; cosine over
`[warmup, warmup+anneal)`, clamped at both ends (the loop can overshoot `total_steps` by
one).

- **Keep the prototype's frozen, stateless, absolute-step form** (`rti.py:92-115`).
  **Reject `ElasticBudgetCurriculum`** (`elastic_tokens.py:165-195`): its mutable
  `update_index` is never serialised, so a resumed anima run replays the identity phase.
  Ours is a pure function of `self.global_step`, already restored at `train.py:1159` —
  **no new resume state**, exactly like `Curriculum.resolve`.
- **Adopt anima's named `identity` phase and `(ratio, phase)` return** for logging.
- `budget_steps` snaps to the **nearest grid value — deterministic, never sampled.**
  Deliberate narrowing of `rti-plan.md:78-79`: sampling would need a generator seeded
  from `(seed, global_step)` because `set_seed(..., device_specific=True)`
  (`train.py:178`) desynchronises per-rank RNG by design, and it buys nothing — under
  varlen `Σnᵢ` and `metadata[2]` already vary per microbatch, so `compile_dynamic=True`
  is mandatory with or without RTI. Ship the knob, default it off.

## 4. Model wiring: `trainer/modeling/mage_flow.py`

### 4.1 Params and construction

`MageFlowParams` gains three fields on the `modulation_rank: int = 0` precedent:

```python
rti_size_buckets: int = 0   # 0 == no interface; preserves the original architecture
rti_core_start:  int = 0
rti_core_end:    int = 0
```

They ride into `model_config` via `asdict(model.params)` (`export.py:73`) and back
through the field-filtered reconstruction at `loader.py:81-88`, which drops unknown keys
and defaults missing ones — **bidirectionally compatible with every existing
checkpoint**, and a future ComfyUI node gets them free.

`MageFlow.__init__` sets `self.region_interface = None`, then builds a real
`RegionInterface` when `params.rti_size_buckets`. Assigning `None` stores in `__dict__`;
a later `nn.Module` assignment registers a proper submodule. A dense checkpoint builds no
interface, so `load_state_dict(strict=True, assign=True)` still passes; an RTI checkpoint
builds it *before* the load, so one strict load covers everything — cleaner than anima's
two-stage prefix-stripped second load (`model.py:751-770`).

**Validate the span in `__init__`**, beside the existing `modulation_rank` guard
(`mage_flow.py:69-71`): require `0 <= rti_core_start < rti_core_end <= depth` and
`rti_core_end < depth`. Without this, `loader.py:76-77` will happily accept a stale
sidecar; `rti_core_start >= rti_core_end` makes READ fire while WRITE never does, and the
failure surfaces as an unrelated broadcast error at `mage_flow.py:325-329`.
`rti_core_start >= depth` is worse — nothing fires, the model runs dense, the interface is
dead weight and `find_unused_parameters=True` absorbs it silently.

`configure_rti(*, dense_prefix_blocks=2, dense_suffix_blocks=2, size_buckets=17)` is the
dense→RTI retrofit path, mirroring `configure_elastic_tokens` including its "at least one
core block" guard. **Parameterised by prefix/suffix, not absolute indices**, so it
survives depth changes. Note the prototype's floating-point-`img_in` guard
(`rti.py:176-178`) does not carry over: we size the interface from `params.hidden_size`
and never inspect `img_in.weight`, so SDNQ bases work.

### 4.2 Varlen metadata builder

Lift `mage_flow.py:289-301` verbatim into a module-level
`_packed_varlen_metadata(text_lengths, img_lengths, device)`, called once with
`image_lengths` and once with `region_lengths`.

**Correct by construction for the compressed span**: block tensors stay
`[all text | all images]`, the compressed image tensor is `[1, Σrᵢ, D]` while `txt` is
untouched at `[1, Σtᵢ, D]`, so `ioff = sum(text_lengths)` remains the image base and
substituting `rᵢ` for `nᵢ` yields the right interleave, the right `cu_seqlens`, and
`max_seqlen = max(tᵢ + rᵢ)`. **Text lengths are unchanged.** All outputs are integer
tensors carrying no autograd graph.

This holds against the kernel: `_packed_attention` derives its scatter target from
`q.shape` (`mageflow_attention.py:81`, `:108`) and treats `max_seqlen` as an upper bound
(`:97-98`), so a shortened block tensor needs no kernel change.

### 4.3 `forward_packed` — the edit

Signature gains `keep_fraction=1.0`.

At `:282-286`, keep a complex handle and bind the block-ready form **once, under its own
name**, so WRITE has something to restore:

```python
dense_freqs        = torch.cat([self.pos_embed([(1, h, w)], device=device) for h, w in shapes])
dense_freqs_blocks = torch.view_as_real(dense_freqs) if self.compiled_blocks else dense_freqs
freqs              = dense_freqs_blocks
```

The per-shape loop stays — `MageFlowEmbedRope.forward` silently keeps only `video_fhw[0]`
when handed a list (`mage_layers.py:164-167`), which is why the `torch.cat` exists. Do
not "optimize" it.

At `:296-301` → `dense_metadata = _packed_varlen_metadata(text_lengths, image_lengths, device)`;
`metadata, ids = dense_metadata, img_ids`.

**Inside the block loop, at `i == rti.core_start`, before `args`** — per D6, the plan is
built here, from the live post-prefix hidden state:

```python
plan            = build_region_plan(img[0].detach(), shapes, image_lengths, keep_fraction, device)
region_metadata = _packed_varlen_metadata(text_lengths, plan.region_lengths, device)
region_ids      = torch.repeat_interleave(
                      torch.arange(len(images), device=device),
                      torch.tensor(plan.region_lengths, device=device))
dense_img       = img
region_in, region_freqs = rti.read(img, dense_freqs, plan)
img      = region_in
freqs    = torch.view_as_real(region_freqs) if self.compiled_blocks else region_freqs
metadata = region_metadata
ids      = region_ids
```

All three constructions stay in the Python loop, outside `checkpoint(...)` and outside the
compiled `block_forward`, so the zero-sync property and `fullgraph=True` safety are
unaffected. `region_ids` is 1-D int64 of length `Σrᵢ`, exactly what `select()` in
`batched.py:77-87` expects.

**After the block call, at `i + 1 == rti.core_end`:**

```python
img = rti.write(dense_img, img, region_in, plan)
freqs, metadata, ids = dense_freqs_blocks, dense_metadata, img_ids
```

READ before block `core_start`, WRITE after block `core_end - 1`; core span
`[core_start, core_end)` — anima's convention. `args[10]` becomes `(ids, txt_ids)`.
**Everything from `:320` down is untouched**, because WRITE restores `Σnᵢ` tokens in
row-major order.

Four tensors change together at READ and all four are restored at WRITE: **`img`,
`freqs`, `ids`, `metadata`**. Missing any one is a *silent* correctness bug — a stale
`ids` applies the wrong sample's AdaLN modulation, and stale `cu_seqlens` lets tokens
attend across image boundaries with no error from the varlen kernel. This enumeration is
exhaustive for the packed path: `txt` and `block_temb` are layout-independent and
`attn_mask` is `None` there.

Add a post-loop assertion mirroring anima's `RuntimeError` at
`compiled_runtime.py:461-462`: if the interface exists and `img.shape[1] != sum(image_lengths)`,
raise naming `core_end`.

**Compile/checkpoint.** `use_reentrant=False` (`:317`) means anima's
`detach().requires_grad_(True)` fixup (`compiled_runtime.py:435-443`) is unnecessary — that
workaround exists only for reentrant checkpointing. Cost to document, not hide:
**`dense_img` stays alive across the whole core span, outside any checkpoint** — ~25 MB
per 4032-token sample at bf16, a new long-lived activation that partially offsets the
saving.

### 4.4 `forward` (uniform) — test seam only

Gated `b == 1` (region freqs are then rank-2, so `batched.py:36` is untouched); `b > 1`
raises naming `_apply_rope_batched`. Reached only by tests, since §5 refuses RTI without
packing. Its value: `packed_attention` hard-requires CUDA + fp16/bf16
(`mageflow_attention.py:69`), so this is the only CPU-runnable RTI forward.

**Here `mask` is a fifth tensor that must track the token count.** Rebuild `mask` and
`metadata` inside the loop from `img.shape[1]` on every iteration, exactly as
`rti.py:216-218` does, rather than at the READ/WRITE boundaries. A stale dense mask over
compressed `img` fails loudly under sdpa, but under a varlen backend it would select the
wrong token positions with no shape error.

## 5. Config: `trainer/training/config.py`

New `RTIConfig` dataclass; field on `Config` (`:250-263`); entry in `_SECTIONS`
(`:272-282`) — mandatory, because `from __future__ import annotations` makes field types
plain strings.

```toml
[rti]
enabled = false            # the single sentinel; everything else is inert when false
dense_prefix_blocks = 2
dense_suffix_blocks = 2
size_buckets = 17
start_keep = 0.98
target_keep = 0.75
identity_steps = 0
warmup_steps = 0
anneal_steps = 1000
budget_steps = []          # discrete grid; [] == continuous

[component_lr]
rti = 1e-5                 # None inherits optimizer.lr; 0.0 freezes
```

`__post_init__`: short-circuit `if not self.enabled: return` (the `PreserveConfig`
pattern); explicit casts (`_build` does **no** type coercion); `type(x) is not int` for
integer fields (plain `isinstance` accepts `True`); range rules lifted from
`rti.py:100-104`; `dense_prefix_blocks >= 1 and dense_suffix_blocks >= 1`; `budget_steps`
→ sorted tuple, every value in `(0, 1]`. Depth validation cannot live here (depth is a
model fact) — it goes at the attach site, like `preserve`'s check at `train.py:377-386`.

`enabled: bool = False` is required for `bridge.prune_defaults` to emit nothing when off,
which `trainer/parity/test_gui.py:170-171` asserts.

Cross-section rules, immediately after `config.py:392`:

```python
if cfg.rti.enabled and cfg.flow.dual_timestep:
    raise ValueError("RTI and dual-timestep noising are mutually exclusive")
if cfg.rti.enabled and not cfg.train.pack_resolutions:
    raise ValueError("[rti] requires train.pack_resolutions = true; the uniform-batch "
                     "forward needs per-sample RoPE support that is not implemented")
if cfg.rti.enabled and cfg.preserve.enabled:
    raise ValueError("[rti] and [preserve] are incompatible: ConceptPreserver's reference "
                     "pass runs the transformer dense, spanning two architectures")
if cfg.rti.enabled and cfg.is_lora:
    raise ValueError("[rti] is full-finetune only in this landing: LyCORIS/PEFT state "
                     "dicts cannot carry region_interface.* weights, so a trained "
                     "interface would be silently dropped at save. Train the interface "
                     "in a full finetune first, then LoRA on that base.")
if cfg.rti.enabled and cfg.train.compile and not cfg.train.compile_dynamic:
    raise ValueError("[rti] requires train.compile_dynamic = true")
```

The LoRA rule is now unconditional on `cfg.is_lora` (not on `component_lr.rti`), which
matches the §0 scope and removes the need to exempt `"rti"` from the `noop` comprehension
at `:400-404` — leave that check alone, since a LR on a component with no adapter really
is a no-op and should still be rejected.

## 6. Optimizer groups and trainability: `trainer/training/params.py`

`classify()` (`:48-55`) raises `KeyError` on unmatched names and `build_param_groups` maps
it over **every** `named_parameter` (`:130`, unfiltered). Registering `region_interface`
without this edit crashes at startup — highest-probability port failure. Note the LoRA
builder is *not* the crash site: `build_adapter_param_groups` filters on `p.requires_grad`
(`:175-177`) after `apply_adapter` has frozen the transformer (`:330`).

1. `_COMPONENT_PATTERNS` += `("rti", re.compile(r"^region_interface\."))`.
2. `COMPONENTS` → `(..., "base", "rti")` — **appended, never inserted.** Group order
   follows this tuple through `buckets` (`:126`, `:148`); reordering pairs saved optimizer
   moments with the wrong parameters, and the size check in the optimizer will not catch a
   swap between two components of equal parameter count.
3. `ComponentLRs` += `rti: float | None = None`.
4. Zero weight decay in both group literals (`:141-150`, `:186-195`):
   `"weight_decay": 0.0 if component == "rti" else weight_decay`. No new per-group key, so
   `_BOOKKEEPING_KEYS` is untouched and SDNQ's group-key assertion is safe. This replaces
   anima's `_anima_disable_weight_decay` per-parameter marker
   (`optimizer_groups.py:126`), which is fragile — `accelerate`'s
   `set_module_tensor_to_device` replaces Parameter objects and drops the attribute, so
   anima has to re-stamp it (`model.py:773-777`).

Trainability: full finetune only (§0). The interface is an ordinary component;
`component_lr.rti = 0.0` freezes it, unset inherits `optimizer.lr`. Anima's
`core_only` / `core_plus_conditioning` scopes are **not in this landing** (§11).

## 7. Quantization: `trainer/training/quant.py`

- `_HIGH_PRECISION` (`:17-25`) += `"region_interface"`. Skip keys match on dotted-component
  equality, so the bare key excludes the whole subtree.
- Belt-and-braces in `quantize_module`, **after** the `modulation_rank` block at
  `:137-141` so it appends to the already-extended list:
  ```python
  if getattr(module, "region_interface", None) is not None:
      cfg = replace(cfg, extra_skip=cfg.extra_skip + ["region_interface"])
  ```

Without this, `write_map` at `nn.Linear(2*3072, 3072)` = 18.9 M elements passes both SDNQ
thresholds and becomes int8 with no error. `read_score` (3072 elements) and
`size_embedding` (`quant_embedding` defaults False) are already safe. Verification is
free: `quantized_layer_report` (`:173-187`) lists skipped Linears at startup.

Ordering is not the mechanism — the skip key is. The interface exists before `_quantize()`
(`train.py:341`) and before `_prepare()` (`:231`), the latter required in SDNQ+DDP mode
where only `sync_quantized_model` broadcasts from rank 0.

## 8. Checkpoint I/O

### `trainer/modeling/export.py`

Guarded exactly like the `modulation_rank` block at `:16-21`. **Flat snake_case string
keys, no dots** — repo convention; anima's `anima.elastic_*` prefix is dropped, not
transliterated.

```python
if getattr(model, "region_interface", None) is not None:
    metadata.update(architecture="mageflow-rti-v1", experimental="true",
                    rti="true", rti_metadata_version="1", rti_order="gilbert2d+cuts",
                    rti_core_start=str(p.rti_core_start), rti_core_end=str(p.rti_core_end),
                    rti_size_buckets=str(p.rti_size_buckets))
```

Setting `architecture` makes the export **self-identifying**, so a stock loader that
gates on it refuses rather than loading silently-dense. Until the ComfyUI node exists
(§11), that is the only protection an RTI checkpoint has — state it in the README.

- `rti_order = "gilbert2d+cuts"` **must differ from anima's `"gilbert2d"`** (different
  discontinuity policy) and from the prototype's traversal, or mutually incompatible
  checkpoints cross-load silently.
- **No `rti_config` JSON blob** — flat string keys mean `inspect_checkpoint.py:12-13`'s
  hard-coded decode tuple needs no edit, and `training_config`
  (`checkpoint_metadata.py:67`) already carries the whole `[rti]` section.
- `cfg` here is **duck-typed** — four existing test files pass `SimpleNamespace`. Every
  RTI read off `cfg` must be `getattr`-guarded. The block above reads only `model`.
- The full-FT branch (`:59`) picks up `region_interface.*` from `model.state_dict()` with
  zero code change, and `asdict(model.params)` (`:73`) carries the three new fields.

### `trainer/modeling/loader.py`

Immediately before `load_state_dict(..., strict=True, assign=True)` at `:98`:

```python
has_rti = any(k.startswith("region_interface.") for k in state)
if has_rti and not params.rti_size_buckets:
    raise ValueError("checkpoint carries region_interface.* weights but its model_config "
                     "reports rti_size_buckets=0; the embedded config or sidecar is stale")
if params.rti_size_buckets and not has_rti:
    raise ValueError("model_config declares RTI but no region_interface.* weights are "
                     "present; this is an incomplete RTI export")
```

Without this the failures are raw `Unexpected key(s)` / `Missing key(s)` tracebacks with
no mention of RTI.

**Anima's `anima.elastic_default_budget_ratio` signature asymmetry is deliberately not
copied**: it is enforced on LoRA load (`save.py:149-158`) but absent from
`expected_layout` (`model.py:686-690`), so an anima LoRA trained at `budget_end=0.5`
refuses to load on a base configured at `0.75`. The budget is a *runtime* choice, not an
architecture fact. Only `rti_core_start`/`rti_core_end`/`rti_size_buckets`/`rti_order`
gate loading.

Anima's LoRA-side `requires_rti` refusal (`save.py:145-148`, `model.py:672-676`) is **not
ported**: this repo has no adapter-load path for it to live in, and §0 refuses RTI+LoRA at
config load anyway. Revisit when LoKr-on-RTI lands.

## 9. Training loop: `trainer/training/train.py`

**Attach** in `_build_model`, after `load_components` (`:311-318`) and **before**
`_quantize()` (`:341`):

```python
self.rti_interface = None
if cfg.rti.enabled:
    existing = getattr(self.transformer, "region_interface", None)
    self.rti_interface = existing or self.transformer.configure_rti(...)
    # depth check + prefix/suffix-vs-checkpoint mismatch refusal here
self.budget = cfg.rti.schedule() if cfg.rti.enabled else None
```

Caching `self.rti_interface` avoids `_unwrap` on every log step; DDP preserves submodule
object identity.

**Clock** — one method beside `_set_phase()`, called at the same place (`train.py:733`,
*outside* the `accumulate` context):

```python
def _set_budget(self):
    if self.budget is None: return
    self.keep_fraction, self.budget_phase = self.budget.resolve(self.global_step)
```

Being a pure function of `self.global_step` it is (a) constant across every microbatch of
an accumulation group, (b) identical on every rank with zero communication, (c) correct
after resume.

Three traps, avoided by construction:

- **Never count `optimizer.step()`** — it runs on every microbatch (`:750/752`) and is a
  no-op off a sync boundary. `self.global_step` (`:759`) is the only count of optimizer
  updates.
- **Never multiply the anneal horizon by `num_processes`.** `build_scheduler` does that at
  `:431-433` only because `AcceleratedScheduler` advances `num_processes` times per call.
  Copying it would finish the anneal at 1/N of the run.
- **Never resolve the budget inside `accumulate`.** The region count is a *shape*;
  changing it mid-window sums gradients from two compression topologies into one update.

**Threading** — a per-call kwarg, never model state: `train.py:641` (`_step_packed`) gets
`**({"keep_fraction": self.keep_fraction} if self.budget else {})`.

**Resume.** `save_full_state` (`:1119-1129`) calls `accelerator.save_state`, which writes
model weights alongside optimizer state; resuming into a differently-shaped model crashes
cryptically inside `load_state`. Add a guard in `_resume` (`:1131-1164`): before
`accelerator.load_state`, check the recorded RTI flags in `state.json` (written at
`:1048-1051`) against `cfg.rti.enabled` / `rti_core_start` / `rti_core_end`, and refuse
with a message naming RTI and directing the user to start a fresh run from the exported
checkpoint (`train.model_path`) rather than `train.resume_from`.

**Reporting.** `_report()` (`:874-985`) gains an `rti` line next to `flow` / `sched`:
`rti  blocks [2,10) of 12  keep 0.98→0.75 over 1000 steps (identity 0, warmup 0)`.
The `runtime` dict (`:1034-1040`) gains `rti_keep_fraction`, `rti_phase`, `rti_regions`,
`rti_tokens`, `rti_min_regions`, JSON-encoded into `training_state` — that is
`rti-plan.md:80-81`'s "exact budget-schedule progress", satisfied with no new resume
state.

`last_stats` holds **Python ints only**, computed host-side from the cached traversal. No
`.item()`, no live device tensor. Anima's `max_region_size` is a live device tensor both
its consumers must defensively convert; do not repeat that.

**DDP**: `find_unused_parameters=True` is already unconditional (`:159`), which covers
`read_score` receiving no gradient during the identity phase (§1). `trainable` is captured
once before the loop (`:710`), so the interface must exist by then — it does.

## 10. GUI — four coordinated edits or the parity test fails

1. `bridge.py:45-57` — `SECTIONS["rti"] = RTIConfig`.
2. `schema.py` — one `_spec(...)` per field, modelled on the `flow.dual_timestep` pair at
   `:404-413`. `BoolEditor` + `inline_label=True` for `enabled`; `FloatEditor(0.01, 1.0,
   0.01, 3)` for the keeps; `IntEditor` for step counts; `NumListEditor(cast=float,
   as_tuple=True)` for `budget_steps` — **`as_tuple` matters** (`fields.py:310-313`) or the
   drift check fires on every load. One `("RTI (experimental)", [...])` group in the
   `("Method", ...)` tab, plus the title in `TRANSFORMED_GROUP_TITLES` (`:758`).
3. `app.py:82` — bulk-derive `_RULES` from `SPEC` so a new key cannot be left ungated
   (the `dataset.texture.*` idiom at `:144-150`), and gate `flow.dual_timestep` on
   `not rti.enabled`. The `load_config` `ValueError` reaches the status bar via
   `bridge.validate` and disables Start at `app.py:671`.
4. `trainer/parity/test_gui.py:137-138` — add `"rti"` to the round-trip `SECTIONS` tuple,
   or an RTI section that fails to round-trip passes the gate silently.

**`metrics.py`** — appending an RTI token to `_log`'s `parts` (`train.py:824-834`) without
a matching optional group in `STEP_RE` (`:36-48`) makes the GUI silently drop **every**
step line. That bug is live today for `pres` (emitted at `train.py:833-834`, dropped at
`metrics.py:180`). Fix both: add optional `pres` and `rti` groups to `STEP_RE`, and `rti`
to `REPORT_RE` (`:51`).

## 11. Tests — `tests/test_rti_packed.py`

| # | Test | Pins |
|---|---|---|
| 1 | Traversal completeness + bijection + ≤1 discontinuity over `84×48, 48×84, 33×45, 84×33, 99×45, 4×5, 6×7, 48×85, 8×8, 1×9` | §1; that the port never raises |
| 2 | Within a label run, consecutive curve steps are L1-distance 1 | connectivity floor |
| 3 | **Nesting**: `cuts(R₁) ⊂ cuts(R₂)` for 0.25 ⊂ 0.5 ⊂ 0.75 on fixed features | the stable-argsort replacement of `topk` |
| 4 | **Bit-exact round-trip** `atol=0, rtol=0` at keep 1.0/0.5/0.25 over **ragged** spans `[(6,7),(4,5),(3,3)]`, fp32 and bf16 | the measured identity property |
| 5 | **Exact delta broadcast** at init | `write_map = [0|I]` |
| 6 | **`forward_packed` identity at keep=1.0** vs `region_interface = None`, `rtol=1e-6` fp32 / few ulps bf16 | end-to-end varlen swap |
| 7 | `_packed_varlen_metadata`: every index once; boundaries == cumulative `tᵢ+rᵢ`; `max_seqlen == max(tᵢ+rᵢ)`; text lengths unchanged | the varlen contract |
| 8 | `region_ids` length == `Σrᵢ`, bincount == `region_lengths` | AdaLN indexing |
| 9 | Schedule phase boundaries + `resolve` purity ⇒ resumed-vs-uninterrupted parity | `rti-plan.md:82` |
| 10 | `classify("region_interface.read_score.weight") == "rti"`; `COMPONENTS[-1] == "rti"`; group `weight_decay == 0.0`; disabled RTI emits no group; `build_adapter_param_groups` after `apply_adapter` yields no `rti` group and does not raise | §6 |
| 11 | Config: RTI×dual-timestep, RTI without packing, RTI×preserve, RTI+LoRA each raise; `[rti]` defaults prune to zero lines | §5 |
| 12 | Export→load roundtrip: `region_interface.*` present, `rti`/`rti_order`/`architecture` metadata set, strict load succeeds, RTI weights with `rti_size_buckets=0` raise | §8 |
| 13 | **Gradient reachability**: after a compressed step all three interface params have finite non-`None` grads with `write_map.grad.abs().max() > 0`; after keep=1.0, `read_score.grad` is exactly zero while the other two are non-zero | §1, D4 |
| 14 | `forward_packed` identity at keep=1.0 with `modulation_rank > 0` | `rti-plan.md:74-75` compressed AdaLN |

**Test 6 must patch `trainer.modeling.batched.packed_attention`, not
`resolve_flash_attention`.** `batched.py:5` does `from .mageflow_attention import
packed_attention` and calls it at `:134`, so the name is bound in the *batched* namespace;
patching there bypasses the CUDA/fp16 guard at `mageflow_attention.py:69` entirely.
Implement the stand-in as a per-segment reference over `cu_seqlens`.

Test 6 cannot be bit-exact: at keep=1.0 READ hands the core the same tokens in **Hilbert
curve order**. Attention is permutation-equivariant and the permutation never crosses an
image boundary, so the result is mathematically identical, but the reductions accumulate
in a different order. Only tests 4 and 5 are genuinely `atol=0, rtol=0`, because
`index_copy` inverts the permutation with no intervening reduction.

No CUDA test is possible for the real varlen kernel; GPU parity is a documented manual
step.

## 12. Not in this landing

1. **Uniform-batch RTI for `B > 1`.** Needs both branches of `_apply_rope_batched`
   (`batched.py:22-38`) to accept rank-3 freqs. Deferred because it modifies the hot
   compiled kernel for a path not trained here.
2. **Sub-patch / atom overprovisioning (K > 1).** The *seam* is built, not the feature:
   `RegionPlan.atom_to_token` is `None` (identity, K=1, bit-exact) and validated as such;
   nothing in §3–§4 assumes one token per latent cell. A future atomizer produces `KN`
   atoms with their own traversal and RoPE in front of READ. **Caveat**: Mage-Flow is
   `patch_size=1` over a /16 128-channel VAE, so sub-tokens would be channel/feature
   atoms, not recovered spatial detail — the `4N atoms → RTI → N tokens` variant is the
   interesting one. Do not implement the atomizer.
3. **LoKr/LyCORIS on an RTI base**, and anima's `core_only` /
   `core_plus_conditioning` trainability scopes (`model.py:444-500`).
   `build_param_groups` discards the parameter name (`:130`), so a block-index scope needs
   that loop widened.
4. **ComfyUI RTI node** (`rti-plan.md:83`). Separate GPL-3.0 package mirroring
   `ComfyUI-MageFlow-Compressed/`: metadata gate on `rti`/`rti_metadata_version`/`rti_order`,
   key translation for the three new `model_config` fields, a budget widget, and READ/WRITE
   in the copied block loop. Until it exists, the `architecture="mageflow-rti-v1"` marker
   (§8) is the only thing standing between an RTI export and a silently-dense load.
5. **`tools/benchmark_rti.py`** — port `_explained_variance`
   (`tools/benchmark_anima_rti.py:116-140`) and the depth probe (`:143-170`).
   This is `rti-plan.md` milestone 2 and a **prerequisite for trusting
   `dense_prefix_blocks=2 / dense_suffix_blocks=2`**, which ships as an unvalidated
   placeholder.
6. **Discrete-budget *sampling*** (`rti-plan.md:78-79`) — replaced by deterministic
   nearest-grid snapping (§3.4).
7. **`torch.use_deterministic_algorithms`** — `index_add` / `scatter_reduce_` are
   nondeterministic on CUDA. Documented limitation; the repo does not enable determinism.

## 13. `docs/rti-plan.md` decisions overridden

| Plan line | Override |
|---|---|
| `:28` "Explicit 100% token retention calls the original forward unchanged" | **Reversed** (D4). The bypass buys no accuracy, and the identity phase genuinely trains WRITE and the size embedding. |
| `:36-40` enclosing-square Hilbert filter; "compare against a rectangular space-filling traversal if that floor becomes material" | **The comparison has been run; gilbert2d wins** (§1). The *policy* — mandatory cuts, report the floor, never count a disconnected region as compression — is kept exactly. |
| `:45-46` "the new interface stays trainable with FP32 pooling/reductions" | Satisfied by fp32 *reductions*. fp32 *parameters* rejected on activation-memory grounds (§3.3), with honest accounting. |
| `:78-79` discrete budget set + anneal sampling + cross-rank sync | **Narrowed** to deterministic snapping; sampling rejected (§3.4). |
| `:80-81` "Store exact budget-schedule progress with … optimizer resume state" | Satisfied **without new state**: the schedule is a pure function of `global_step`. Recorded in `training_state` for observability only. |
| `:72-73` packed execution listed third in milestone 3 | **Promoted to first** — the only execution path this landing implements. |

## 14. Licensing

`gilbert2d` is **BSD-2-Clause, © 2018 Jakub Červený**, retained by anima at
`third_party/gilbert/LICENSE`. `trainer/experimental/LICENSE.RTI` covers only Zamfir's MIT
read/write design and does **not** cover the traversal. Add
`trainer/modeling/LICENSE.gilbert` (verbatim copy), a notice block in
`region_tokens.py`'s docstring naming both origins, and an entry in `NOTICE`.

## 15. Implementation order

1. `trainer/modeling/region_tokens.py` + `LICENSE.gilbert` + `NOTICE`
2. `trainer/modeling/mage_flow.py` (params, `__init__` validation, `configure_rti`,
   `_packed_varlen_metadata`, `forward_packed`, uniform test seam)
3. `trainer/training/config.py` (`RTIConfig`, `_SECTIONS`, cross-rules)
4. `trainer/training/params.py` (component, `ComponentLRs`, zero WD)
5. `trainer/training/quant.py` (skip keys)
6. `trainer/modeling/export.py`, `loader.py` (metadata, gating)
7. `trainer/training/train.py` (attach, clock, threading, resume guard, report, runtime)
8. GUI: `bridge.py`, `schema.py`, `app.py`, `metrics.py`, `parity/test_gui.py`
9. `tests/test_rti_packed.py`
10. `configs/mageflow-rti.toml` example + README section

Steps 1–2 are independently testable against tests 1–8 before any trainer wiring exists.
