"""TOML training config.

Unknown keys are a hard error. A typo'd `learing_rate` that silently keeps the default would cost
a full training run to notice, and this trainer exists specifically for runs that are expensive.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from ..data.caption import CaptionConfig, load_protected_tags
from ..data.dataset import DatasetConfig, SubsetConfig
from ..data.texture import TextureConfig
from .curriculum import Curriculum, Phase
from .flow import FlowConfig
from .params import AdapterConfig, ComponentLRs
from .preserve import PreserveConfig
from .quant import QuantConfig
from ..modeling.region_tokens import BudgetSchedule
from .optimizer_specs import OPTIMIZERS


@dataclass
class OptimizerConfig:
    kind: str = "adamw"                 # See optimizer_specs. Optimi kinds use optimi_ prefix.
    lr: float = 1e-5
    # Omitted values use the selected optimizer's upstream defaults.
    betas: tuple[float, ...] | None = None
    eps: float | None = None
    weight_decay: float | None = None
    max_grad_norm: float = 1.0
    # sdnq.optim: quantize and/or offload optimizer state. For a 2B full finetune this is not
    # optional -- fp32 AdamW state alone is 16GB.
    quantize_state: bool = False
    offload_state: bool = False
    # Kahan summation over the bf16 master weight. sdnq keeps a per-parameter residual buffer and
    # folds the part of each update that the bf16 cast threw away back into the next one, so
    # updates below an ulp accumulate instead of vanishing. This is *not* an alternative to
    # stochastic rounding -- sdnq applies both (`optim/utils.py:57`), SR on the cast and Kahan on
    # what the cast lost. Costs one extra buffer per trainable parameter.
    use_kahan: bool = False
    # SDNQ Adafactor/CAME update normalization; None preserves upstream defaults.
    norm_mode: str | None = None
    use_first_moment: bool = False  # SDNQ Adafactor; smoothing uses betas[1].

    # Optimi's name is deliberately separate from SDNQ's use_kahan.
    kahan_sum: bool | str | None = None
    momentum: float = 0.0
    gradient_release: bool = False

    def __post_init__(self):
        self.kind = self.kind.lower()
        if self.gradient_release and not self.kind.startswith("optimi_"):
            raise ValueError("optimizer.gradient_release requires an Optimi optimizer")
        if self.gradient_release and self.max_grad_norm != 0:
            raise ValueError("optimizer.gradient_release requires optimizer.max_grad_norm=0; global clipping needs persistent gradients")
        if self.kind not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer: {self.kind!r}")
        spec = OPTIMIZERS[self.kind]
        if type(self.use_first_moment) is not bool:
            raise ValueError("optimizer.use_first_moment must be true or false")
        if self.use_first_moment and self.kind != "adafactor":
            raise ValueError("optimizer.use_first_moment is only supported by SDNQ Adafactor")
        if self.norm_mode is not None:
            if self.kind not in ("adafactor", "came"):
                raise ValueError("optimizer.norm_mode is only supported by SDNQ Adafactor/CAME")
            if self.norm_mode not in ("relative", "rms_clip", "rms", "clip", "none"):
                raise ValueError(f"unknown optimizer.norm_mode: {self.norm_mode!r}")
        self.betas = spec.betas if self.betas is None else tuple(self.betas)
        self.eps = spec.eps if self.eps is None else self.eps
        self.weight_decay = spec.weight_decay if self.weight_decay is None else self.weight_decay
        if spec.family == "Optimi" and self.kahan_sum is None:
            self.kahan_sum = False if self.kind == "optimi_adan" else "auto"
        if self.kahan_sum is not None and type(self.kahan_sum) is not bool and self.kahan_sum != "auto":
            raise ValueError("optimizer.kahan_sum must be true, false, or 'auto'")
        if len(self.betas) != len(spec.betas):
            raise ValueError(f"optimizer.kind = {self.kind!r} takes {len(spec.betas)} betas; "
                             f"defaults are {spec.betas}, got {self.betas}")
        for i, beta in enumerate(self.betas):
            valid = beta < 0 if self.kind == "adafactor" and i == 0 else 0 <= beta < 1
            if not valid:
                raise ValueError(f"optimizer.betas[{i}] is invalid for {self.kind}: {beta}")
        if self.eps is not None and not self.eps > 0:
            raise ValueError("optimizer.eps must be positive")
        if spec.family != "SDNQ" and (self.quantize_state or self.offload_state or self.use_kahan):
            raise ValueError("optimizer.use_kahan/quantize_state/offload_state need an sdnq optimizer")
        if spec.family != "Optimi" and self.kahan_sum is not None:
            raise ValueError("optimizer.kahan_sum is only supported by Optimi")
        if self.kind != "optimi_sgd" and self.momentum != 0:
            raise ValueError("optimizer.momentum is only supported by optimi_sgd")
        if not 0 <= self.momentum < 1:
            raise ValueError("optimizer.momentum must be in [0, 1)")


@dataclass
class ScheduleConfig:
    kind: str = "constant"              # "constant" | "cosine" | "linear" | "rex" | "rerex"
    warmup_steps: int = 0
    # Floor for every decaying schedule, as a fraction of the group's peak LR. sd-scripts hardcodes
    # 0.001 for REX/ReREX; leaving this at 0.0 makes them decay all the way to zero instead.
    min_lr_ratio: float = 0.0

    # --- rex ---
    d: float = 0.9                      # decay sharpness; 0 == linear, higher holds the peak longer

    # --- rerex ---
    global_d: float = 0.78              # `d` of the outer curve the segment endpoints ride
    local_d: float = 0.85               # `d` of the decay inside each segment
    weight_power: float = 1.5           # step-budget skew toward early segments; 0 == equal lengths
    num_segments: int = 8

    def __post_init__(self):
        if self.kind not in ("constant", "cosine", "linear", "rex", "rerex"):
            raise ValueError(f"unknown schedule.kind: {self.kind!r}")
        if not 0.0 <= self.min_lr_ratio < 1.0:
            raise ValueError(f"schedule.min_lr_ratio must be in [0, 1), got {self.min_lr_ratio}")
        # d == 1 makes the REX denominator collapse to the numerator: a flat curve that falls off a
        # cliff at the very last step, and a 0/0 there. Rejected rather than clamped.
        for name in ("d", "global_d", "local_d"):
            v = getattr(self, name)
            if not 0.0 <= v < 1.0:
                raise ValueError(f"schedule.{name} must be in [0, 1), got {v}")
        if self.num_segments < 1:
            raise ValueError(f"schedule.num_segments must be >= 1, got {self.num_segments}")
        if self.weight_power < 0.0:
            raise ValueError(f"schedule.weight_power must be >= 0, got {self.weight_power}")


@dataclass
class RTIConfig:
    enabled: bool = False
    dense_prefix_blocks: int = 2
    dense_suffix_blocks: int = 2
    size_buckets: int = 17
    start_keep: float = .98
    target_keep: float = .75
    identity_steps: int = 0
    warmup_steps: int = 0
    anneal_steps: int = 1000
    budget_steps: list[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.enabled:
            return
        if self.dense_prefix_blocks < 0 or self.dense_suffix_blocks < 1 or self.size_buckets < 1:
            raise ValueError("RTI requires non-negative prefix, positive suffix, and positive size_buckets")
        BudgetSchedule(self.start_keep, self.target_keep, self.identity_steps, self.warmup_steps, self.anneal_steps, tuple(self.budget_steps))

    def schedule(self):
        return BudgetSchedule(self.start_keep, self.target_keep, self.identity_steps, self.warmup_steps, self.anneal_steps, tuple(self.budget_steps))


# The converted diffusers repo (see README section 0). `MAGE_FLOW_MODEL` overrides it, so a machine
# that keeps the model somewhere fixed does not have to set `model_path` in every config; the
# relative fallback resolves for the common layout where the model sits beside this checkout.
DEFAULT_MODEL_PATH = os.environ.get("MAGE_FLOW_MODEL", "mage-flow")


@dataclass
class TrainConfig:
    model_family: str = "auto"  # auto | mage_flow
    model_path: str = DEFAULT_MODEL_PATH
    transformer_path: str | None = None
    text_encoder_path: str | None = None
    vae_path: str | None = None
    flux2_vae: bool = False  # Experimental: pack FLUX.2 32c/8x latents to Mage's 128c/16x layout.
    tokenizer_path: str | None = None
    output_dir: str = "output"
    run_name: str = "mageflow"

    epochs: int = 1
    max_steps: int | None = None
    # int, or {longest_side_threshold: micro_batch_size} e.g. {512 = 8, 1024 = 4, 1536 = 1}.
    batch_size: int | dict[int, int] = 1
    gradient_accumulation_steps: int = 1
    pack_resolutions: bool = False
    checkpoint_blocks: list[int] | None = None
    attention_backend: str = "sdpa"
    cache_text_embeddings: bool = False
    text_cache_batch_size: int = 4
    caption_variations: int = 0
    caption_cache_path: str | None = None
    offload_text_encoder: bool = False
    compile_text_encoder: bool = False
    text_encoder_embedding_only: bool = False
    max_text_tokens: int = 512
    gradient_checkpointing: bool = True
    dtype: str = "bfloat16"
    compressed_adaln_dtype: str = "float32"
    seed: int = 42
    num_workers: int = 2
    # Images per VAE forward under `dataset.source = "encode"`. The encode runs under no_grad and
    # is freed before the transformer allocates, so the step peak is max(train, encode) -- this
    # knob decides which of the two wins. Raising it buys nothing: measured at 1024px on 413
    # images, mean step time is 6.52 / 6.52 / 6.53 s/it at chunk 1 / 2 / 6 while peak goes
    # 18.2 / 19.3 / 23.1 GB, and chunk 12 OOMs outright. The VAE encode is not launch-bound, so
    # there is no throughput to recover by batching it. Leave this at 1.
    vae_encode_chunk: int = 1

    save_every_steps: int | None = None
    save_every_epochs: int | None = 1
    save_native: bool = True            # also write a ComfyUI-loadable single file
    keep_last_n: int | None = None
    resume_from: str | None = None
    # Optimizer + scheduler state alongside each checkpoint. Required to resume; costs roughly the
    # optimizer-state size on disk per checkpoint, which for a quantized full FT is ~3.5GB.
    save_optimizer_state: bool = False
    skip_final_save: bool = False       # benchmarking only
    # Texture curricula are refused on more than one process. Two runs degraded anatomy
    # progressively against single-GPU at MATCHED steps -- so it is not the halved step count, and
    # more steps do not fix it. The cause is not identified: ranks seeding identically was found
    # and fixed (`set_seed(device_specific=True)`), but was never shown to be the mechanism.
    #
    # This exists so re-testing that fix is a deliberate act rather than the default. It is not a
    # "I know better" switch -- if a run under it comes out clean against a single-GPU control at
    # matched steps, the gate should be removed, not the override left on.
    allow_multi_gpu_texture: bool = False

    log_every: int = 1
    # Progress display. "auto" draws a tqdm bar when stdout is a terminal and falls back to the
    # plain per-step line when it is not -- a bar redirected to a file is thousands of \r-separated
    # fragments. "bar"/"plain" force it either way; "off" suppresses the per-step line entirely.
    progress: str = "auto"              # "auto" | "bar" | "plain" | "off"

    # torch.compile, via Accelerate's TorchDynamoPlugin so the compile/DDP ordering is handled.
    # None = off. Measure before enabling: bucketing means many input shapes, and every distinct
    # shape is a potential recompile.
    compile: str | None = None          # "default" | "reduce-overhead" | "max-autotune"
    # Dynamic shapes. Effectively mandatory here -- with static shapes each bucket compiles its
    # own graph and dynamo's recompile limit (8) is exhausted almost immediately, after which it
    # silently falls back to eager for the rest of the run.
    compile_dynamic: bool = True
    # Compile the repeated transformer block once instead of the whole trunk. 28 identical blocks,
    # so this cuts compile time by roughly that factor for nearly all of the benefit.
    compile_regional: bool = True

    def __post_init__(self):
        if self.model_family not in ("auto", "mage_flow"):
            raise ValueError("train.model_family must be auto or mage_flow")
        if self.compile is False:
            self.compile = None
        if self.pack_resolutions and (self.attention_backend == "sdpa" or type(self.batch_size) is not int or self.batch_size < 1):
            raise ValueError("pack_resolutions requires packed attention and a positive integer batch_size")
        if type(self.text_cache_batch_size) is not int or self.text_cache_batch_size < 1:
            raise ValueError("train.text_cache_batch_size must be a positive integer")
        for name in ("compile_text_encoder", "text_encoder_embedding_only"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"train.{name} must be a boolean")
        if self.max_text_tokens < 1:
            raise ValueError("max_text_tokens must be positive")
        if self.attention_backend not in ("sdpa", "torch_varlen", "flash_attn_2", "flash_attn_3"):
            raise ValueError("Unsupported attention_backend")
        if self.checkpoint_blocks is not None:
            if not self.gradient_checkpointing or not isinstance(self.checkpoint_blocks, list):
                raise ValueError("checkpoint_blocks requires gradient_checkpointing and a list")
            if any(type(i) is not int or i < 0 for i in self.checkpoint_blocks) or len(set(self.checkpoint_blocks)) != len(self.checkpoint_blocks):
                raise ValueError("checkpoint_blocks must contain unique nonnegative indices")
        if self.compile and (not self.compile_regional or self.attention_backend not in ("sdpa", "torch_varlen") or self.compile not in ("default", "max-autotune-no-cudagraphs")):
            raise ValueError("Mage-Flow compile requires regional SDPA and default/max-autotune-no-cudagraphs mode")
        if isinstance(self.batch_size, dict):
            self.batch_size = {int(k): int(v) for k, v in self.batch_size.items()}
        if self.compressed_adaln_dtype not in ("float32", "bfloat16"):
            raise ValueError("train.compressed_adaln_dtype must be float32 or bfloat16")
        if self.dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"unknown dtype: {self.dtype}")
        if self.vae_encode_chunk < 1:
            raise ValueError(
                f"train.vae_encode_chunk must be >= 1, got {self.vae_encode_chunk}")
        if self.progress not in ("auto", "bar", "plain", "off"):
            raise ValueError(
                f"train.progress must be auto|bar|plain|off, got {self.progress!r}")
        if self.compile is not None and self.compile not in (
            "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
        ):
            raise ValueError(
                f"unknown train.compile mode: {self.compile!r}. Use one of 'default', "
                f"'reduce-overhead', 'max-autotune', or omit it to disable."
            )


@dataclass
class SelfFlowConfig:
    enabled: bool = False
    ema_dtype: str = 'bfloat16'
    ema_device: str = 'cuda'
    adaln_fp32: bool = True
    stochastic_rounding: bool = True
    decay: float = .99
    weight: float = .8

    def __post_init__(self):
        if self.ema_dtype not in ('float32', 'bfloat16') or self.ema_device not in ('cpu', 'cuda'):
            raise ValueError('Invalid Self-Flow EMA dtype/device')
        if not 0 <= self.decay < 1 or not 0 <= self.weight < float('inf'):
            raise ValueError('Self-Flow needs 0 <= decay < 1 and finite nonnegative weight')


@dataclass
class Config:
    train: TrainConfig = field(default_factory=TrainConfig)
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(path=""))
    flow: FlowConfig = field(default_factory=FlowConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    component_lr: ComponentLRs = field(default_factory=ComponentLRs)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    preserve: PreserveConfig = field(default_factory=PreserveConfig)
    rti: RTIConfig = field(default_factory=RTIConfig)
    self_flow: SelfFlowConfig = field(default_factory=SelfFlowConfig)
    # `[[curriculum]]` -- an array of tables, so it is built by hand in `load_config` rather than
    # through `_SECTIONS`. Empty by default; an empty curriculum is exactly today's behaviour.
    curriculum: Curriculum = field(default_factory=Curriculum)

    @property
    def is_lora(self) -> bool:
        return self.adapter.kind != "none"


# `from __future__ import annotations` makes dataclass field types plain strings, so the section
# classes are resolved from this table rather than from field introspection.
_SECTIONS = {
    "train": TrainConfig,
    "dataset": DatasetConfig,
    "flow": FlowConfig,
    "optimizer": OptimizerConfig,
    "schedule": ScheduleConfig,
    "component_lr": ComponentLRs,
    "adapter": AdapterConfig,
    "quant": QuantConfig,
    "preserve": PreserveConfig,
    "rti": RTIConfig,
    "self_flow": SelfFlowConfig,
}


def _build(cls, data: dict, path: str):
    """Instantiate a flat dataclass from a dict, rejecting unknown keys."""
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"[{path}] unknown key(s): {unknown}. Valid: {sorted(known)}")
    return cls(**{k: v for k, v in data.items() if k in known})


def validate_model_options(cfg):
    from ..modeling.loader import resolve_model_family
    cfg.train.model_family = resolve_model_family(cfg.train.model_path, cfg.train.model_family)
    if cfg.self_flow.enabled:
        if (cfg.train.model_family != 'mage_flow' or cfg.is_lora
                or not cfg.train.pack_resolutions or not cfg.flow.dual_timestep
                or cfg.rti.enabled or cfg.curriculum.phases or cfg.preserve.enabled
                or cfg.flow.hf_scale or cfg.flow.use_ot or cfg.optimizer.gradient_release
                or cfg.quant.mode == 'frozen' or cfg.quant.use_quantized_matmul is True
                or cfg.dataset.source != 'latents' or not cfg.train.cache_text_embeddings):
            raise ValueError('Experimental Self-Flow requires Mage-Flow full finetuning, '
                             'packed cached latents, Cached Text Encoder, and dual timestep. '
                             'RTI, curriculum, preservation, HF loss, OT, gradient release, frozen quantization '
                             'and quantized matmul are not supported together with it.')


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    raw = tomllib.loads(p.read_text(encoding="utf-8"))
    if raw.get("adapter", {}).get("kind") == "lycoris_lora":
        raw.setdefault("train", {}).setdefault("compile", "default")

    unknown = sorted(set(raw) - {f.name for f in fields(Config)})
    if unknown:
        raise ValueError(f"unknown top-level section(s): {unknown}")

    sections = {}
    for f in fields(Config):
        if f.name not in raw:
            continue
        if f.name == "curriculum":
            phases = raw["curriculum"]
            if not isinstance(phases, list):
                raise ValueError(
                    "[curriculum] must be an array of tables -- write `[[curriculum]]` (double "
                    "brackets), one block per phase."
                )
            sections[f.name] = Curriculum([_build(Phase, dict(p), "curriculum") for p in phases])
            continue
        data = dict(raw[f.name])
        if f.name == "dataset":
            # Both keys set is always a mistake: `resolutions` wins, so a leftover
            # `resolution = 1024` above `resolutions = [768, 1280]` would be silently ignored.
            # Checked here rather than in the dataclass because only the raw TOML distinguishes
            # "explicitly set" from "left at the default".
            if "resolution" in data and "resolutions" in data:
                raise ValueError(
                    "[dataset] sets both `resolution` and `resolutions`. Use `resolution` for a "
                    "single tier or `resolutions` for several, not both -- `resolution` would be "
                    "ignored."
                )
            # `caption` and `texture` are nested inside `[dataset]` in TOML but are their own
            # dataclasses. Without this they arrive as plain dicts, which fails nowhere at load
            # time and then raises deep inside the first texture batch.
            caption = data.pop("caption", {})
            texture = data.pop("texture", {})
            # `[[dataset.subsets]]` arrives as a list of dicts and needs the same treatment: built
            # per element so an unknown key is named against `dataset.subsets` rather than being
            # swallowed. Passed through the constructor, not assigned after, so `DatasetConfig`'s
            # path/subsets exclusivity check sees both at once.
            if "subsets" in data:
                data["subsets"] = [_build(SubsetConfig, dict(s), "dataset.subsets")
                                   for s in data["subsets"]]
            protected_file = caption.pop("protected_tags_file", None)
            cfg = _build(DatasetConfig, data, "dataset")
            cfg.caption = _build(CaptionConfig, caption, "dataset.caption")
            cfg.texture = _build(TextureConfig, texture, "dataset.texture")
            if protected_file:
                cfg.caption.protected_tags = load_protected_tags(protected_file)
            sections[f.name] = cfg
        else:
            if f.name == "flow" and data.get("flux_shift") and "shift" not in data:
                data = {**data, "shift": None}
            sections[f.name] = _build(_SECTIONS[f.name], data, f.name)

    cfg = Config(**sections)
    validate_model_options(cfg)
    if not cfg.dataset.path and not cfg.dataset.subsets:
        raise ValueError("dataset.path (or a [[dataset.subsets]] list) is required")

    # A per-tier `batch_size` map must name exactly the tiers that exist. Checked here because it
    # is the one rule that spans two sections, and because getting it wrong is expensive: the old
    # threshold semantics let `{512 = 32, 1024 = 12}` hand a 960x960 bucket 32 images and OOM
    # several minutes into a run. One second at load beats that.
    if isinstance(cfg.train.batch_size, dict):
        tiers = set(cfg.dataset.tiers)
        keys = set(cfg.train.batch_size)
        declared = "resolutions" if cfg.dataset.resolutions else "resolution"
        if unknown := sorted(keys - tiers):
            raise ValueError(
                f"train.batch_size has key(s) {unknown} that are not declared resolutions. "
                f"dataset.{declared} declares {sorted(tiers)}; the map is keyed by tier, matched "
                f"exactly -- it is not a threshold ladder. Use `batch_size = <int>` for one size "
                f"everywhere."
            )
        if missing := sorted(tiers - keys):
            raise ValueError(
                f"train.batch_size gives no size for tier(s) {missing}. dataset.{declared} "
                f"declares {sorted(tiers)}, so every one of them needs an entry (or use a plain "
                f"int for the same size everywhere)."
            )
    # Texture crops are chosen per step from image content, so no cache can hold them. Checked at
    # load rather than at the first texture batch -- which could be several hundred steps in, and
    # which the GUI's status bar would never reach, since it validates the config without running.
    if any(p.mode == "texture" for p in cfg.curriculum.phases) and cfg.dataset.source != "encode":
        raise ValueError(
            f"a curriculum phase uses mode = 'texture', which crops per step from the source "
            f"image, but dataset.source is {cfg.dataset.source!r}. Set "
            f'dataset.source = "encode" -- cached latents are a frozen centre crop and cannot '
            f"express a per-step crop."
        )

    if cfg.train.pack_resolutions and (cfg.curriculum.phases or cfg.flow.use_ot):
        raise ValueError("pack_resolutions currently requires no curriculum and flow.use_ot=false")
    if cfg.rti.enabled:
        if not cfg.train.pack_resolutions:
            raise ValueError("RTI requires train.pack_resolutions=true")
        if cfg.flow.dual_timestep:
            raise ValueError("RTI and flow.dual_timestep cannot be combined")
        if cfg.preserve.enabled:
            raise ValueError("RTI and concept preservation cannot be combined")
        if cfg.is_lora:
            raise ValueError("RTI currently supports full finetuning only; adapters cannot export its interface")
        if cfg.train.compile and not cfg.train.compile_dynamic:
            raise ValueError("RTI compilation requires train.compile_dynamic=true")
    if cfg.adapter.kind == "lycoris_lora" and cfg.preserve.enabled:
        raise ValueError("Concept preservation is not supported with lycoris_lora")
    if cfg.is_lora:
        # component_lr now applies to adapters too, so the check is no longer "is it allowed" but
        # "would it do anything". The previous guard only looked at adaln/base, which meant a
        # `component_lr.mlp` under LoRA was accepted and then silently ignored.
        adapter_components = set(cfg.adapter.components)

        noop = {
            c: lr for c, lr in cfg.component_lr.explicit().items()
            if lr not in (None, 0.0) and c not in adapter_components
        }
        if noop:
            raise ValueError(
                f"[component_lr] sets {noop} but no adapter is injected there, so it would have "
                f"no effect. Add the component to adapter.components "
                f"(currently {sorted(adapter_components)}), or remove the LR."
            )
    if cfg.optimizer.gradient_release and cfg.train.gradient_accumulation_steps != 1:
        raise ValueError("optimizer.gradient_release requires train.gradient_accumulation_steps=1")
    if cfg.optimizer.kind.startswith("optimi_") and cfg.quant.mode == "training":
        raise ValueError("Optimi does not yet support SDNQ training tensors. Use quant.mode='none' "
                         "for full finetuning, or 'frozen' for adapters.")
    if cfg.quant.mode == "frozen" and not cfg.is_lora:
        # A frozen-quantized base has no trainable weights at all, so this would run and produce
        # nothing. Caught here rather than after the model loads.
        raise ValueError(
            "quant.mode='frozen' quantizes the base and trains nothing; it requires an adapter. "
            "For a quantized full finetune use quant.mode='training'."
        )
    if cfg.quant.mode == "training" and cfg.is_lora:
        raise ValueError(
            "quant.mode='training' is for full finetuning; with an adapter use 'frozen'"
        )
    if cfg.train.caption_variations < 0:
        raise ValueError("caption_variations must be >= 0")
    if cfg.train.caption_variations and not cfg.train.cache_text_embeddings:
        raise ValueError("caption_variations requires cache_text_embeddings=true")
    if cfg.train.cache_text_embeddings:
        from ..data.text_cache import validate_static_captions
        validate_static_captions(cfg)
        if cfg.train.offload_text_encoder:
            raise ValueError("Cached text embeddings already remove the encoder; offload_text_encoder is redundant")
    return cfg
