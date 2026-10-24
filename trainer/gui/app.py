"""The window.

Shell structure (tab bar / stacked pages / footer) follows Aozora's; every page is ours.

The one idea worth stating: **the GUI reimplements no validation.** Every keystroke re-serialises
the form to TOML and hands it to `trainer.training.config.load_config` -- the same function the
trainer calls -- and shows whatever it raises. Start stays disabled until that passes. So the
config-level guards this trainer accumulated (both `resolution` and `resolutions` set;
`sigmoid_scale` under `uniform`; a `component_lr` on a component with no adapter injected; `frozen`
quantization with nothing to train) are enforced here for free and cannot drift.

On top of that, `_RULES` greys out controls that the config would *accept* but that would do
nothing in the current combination. That is one step ahead of the validators: a disabled box says
"this knob is inert right now" before you spend a run finding out.
"""

from __future__ import annotations

import sys
import hashlib
from dataclasses import asdict, replace
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from . import bridge
from ..training.optimizer_specs import OPTIMIZERS, optimizer_defaults, supports
from .metrics import LiveMetricsWidget
from .timestep_plot import TimestepDistributionWidget
from .run_queue import RunQueueWidget
from .preferences import Preferences
from .process import (Job, ProcessRunner, audit_launch, cache_launch, concat_launch,
                      train_launch, training_env)
from .schema import LAYOUT, SPEC
from .widgets import (
    DANGER,
    PROJECT_ROOT,
    STYLESHEET,
    SUCCESS,
    THEME,
    WARN,
    SearchableComboBox,
    VirtualConsoleWidget,
    group_box,
    make_btn,
    make_label,
    prevent_sleep,
    safe_stem,
    usable_dialog_start,
    set_role,
)

CONFIG_DIR = PROJECT_ROOT / "configs"

# DDP here goes over NCCL, which has no Windows build -- torch falls back to gloo, which does not
# support the GPU collectives this needs. So on Windows the GPU picker is a one-of-N choice rather
# than a multi-select. Every VRAM and throughput measurement in this repo is single-GPU anyway;
# what is lost is the 1.75x, not the ability to train.
MULTI_GPU_SUPPORTED = sys.platform != "win32"


def detect_gpus() -> list[tuple[int, str]]:
    """[(index, name)], via nvidia-smi.

    Deliberately not via torch: importing torch here would create a CUDA context in the GUI
    process and hold a few hundred MB for the whole session, so the trainer would start with less
    VRAM than every measurement in this repo assumes.
    """
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        gpus = []
        for line in out.stdout.splitlines():
            if not line.strip():
                continue
            idx, _, name = line.partition(",")
            gpus.append((int(idx.strip()), name.strip()))
        return gpus or [(0, "GPU 0")]
    except Exception:
        return [(0, "GPU 0")]


def _has_mask_folder(c: dict) -> bool:
    return bool(c.get("dataset.mask_path")
                or any(s.get("mask_path") for s in c.get("dataset.subsets") or []))


# Controls that exist but would do nothing in the current combination. Each entry is
# key -> predicate(flat_config) -> enabled.
_RULES = {
    "dataset.use_masks": _has_mask_folder,
    **{f"dataset.{option}": (lambda c: _has_mask_folder(c) and bool(c.get("dataset.use_masks")))
       for option in ("mask_outside_weight", "mask_dilate_px", "mask_feather_px")},
    **{f'paired_texture.{option}': (lambda c: float(c.get('paired_texture.scale') or 0) > 0) for option in (
        'levels', 'patch', 'tokens', 'features', 'seed', 'snr_cut', 'min_gate')},
    **{f'flow.{option}': (lambda c: float(c.get('flow.hf_scale') or 0) > 0) for option in (
        'hf_high_noise_snr_cut', 'hf_high_noise_min_weight', 'hf_high_noise_power')},
    **{f'adversarial.{option}': (lambda c: bool(c.get('adversarial.enabled'))) for option in (
        'weight', 'discriminator_lr', 'warm_start')},
    **{f'perceptual.{option}': (lambda c: bool(c.get('perceptual.enabled'))) for option in (
        'weight', 'max_t', 'every', 'cache_dir', 'eye_manifest', 'dedicated_low_noise')},
    **{f'self_flow.{option}': (lambda c: bool(c.get('self_flow.enabled'))) for option in (
        'ema_dtype', 'ema_device', 'adaln_fp32', 'stochastic_rounding', 'decay', 'weight')},
    # Measured: under `uniform`, scale 0.5/1.0/2.0 give byte-identical distributions.
    "flow.sigmoid_scale": lambda c: c.get("flow.timestep_sample_method") == "logit_normal",
    "flow.shift": lambda c: c.get("flow.timestep_sample_method") != "moss",
    "flow.flux_shift": lambda c: c.get("flow.timestep_sample_method") != "moss",
    **{f"flow.moss_{option}": (lambda c: c.get("flow.timestep_sample_method") == "moss")
       for option in ("mu", "std", "cutoff", "uniform_mix", "p_full", "steps")},
    "flow.dual_timestep_mask_ratio": lambda c: bool(c.get("flow.dual_timestep")),
    "flow.hf_exponent": lambda c: float(c.get("flow.hf_scale") or 0) > 0,
    **{f"rti.{option}": (lambda c: bool(c.get("rti.enabled"))) for option in (
        "dense_prefix_blocks", "dense_suffix_blocks", "size_buckets", "start_keep", "target_keep",
        "identity_steps", "warmup_steps", "anneal_steps", "budget_steps")},

    "train.moss_qkv_format": lambda c: (c.get("train.model_family") in ("auto", "moss_image")
        and (c.get("adapter.kind") in ("none", "lora") or (c.get("adapter.kind") == "lycoris_lora" and c.get("adapter.lycoris_algo") == "locon"))),
    "train.activation_offload_budget_mb": lambda c: bool(c.get("train.activation_offload")),
    "train.activation_offload_pin_memory": lambda c: bool(c.get("train.activation_offload")),
    "train.activation_offload_prefetch": lambda c: bool(c.get("train.activation_offload")) and bool(c.get("train.activation_offload_pin_memory")),
    "dataset.resolution": lambda c: not c.get("dataset.resolutions"),
    "dataset.tier_collapse": lambda c: bool(c.get("dataset.resolutions")),
    "dataset.min_source_area": lambda c: bool(c.get("dataset.resolutions")),
    # Inert on the no_upscale path -- verified, 256/128/64 give identical buckets.
    "dataset.min_bucket_reso": lambda c: not c.get("dataset.bucket_no_upscale", True),
    "dataset.caption.mixed_weights": lambda c: c.get("dataset.caption.caption_mode") == "mixed",
    "dataset.caption.shuffle_keep_first_n": lambda c: bool(c.get("dataset.caption.shuffle_tags")),
    "dataset.caption.nl_keep_first_sentence":
        lambda c: bool(c.get("dataset.caption.nl_shuffle_sentences")),

    # Texture settings only mean anything if some phase asks for texture. There is no separate
    # "texture mode" switch to gate on -- the curriculum IS the switch -- so the rule reads the
    # phase list. `flow.phase_mapping` is inert without any curriculum at all, texture or not.
    "flow.phase_mapping": lambda c: bool(c.get("curriculum")),

    **{f"optimizer.{option}": (lambda c, option=option: supports(c.get("optimizer.kind"), option))
       for option in ("betas", "eps", "use_kahan", "kahan_sum", "momentum",
                      "quantize_state", "offload_state", "gradient_release", "norm_mode",
                      "use_first_moment")},

    "schedule.min_lr_ratio": lambda c: c.get("schedule.kind") != "constant",
    "schedule.d": lambda c: c.get("schedule.kind") == "rex",
    "schedule.global_d": lambda c: c.get("schedule.kind") == "rerex",
    "schedule.local_d": lambda c: c.get("schedule.kind") == "rerex",
    "schedule.weight_power": lambda c: c.get("schedule.kind") == "rerex",
    "schedule.num_segments": lambda c: c.get("schedule.kind") == "rerex",

    "train.text_cache_batch_size": lambda c: bool(c.get("train.cache_text_embeddings")),
    "train.compile_text_encoder": lambda c: not bool(c.get("train.cache_text_embeddings")),
    "train.text_encoder_embedding_only": lambda c: not bool(c.get("train.cache_text_embeddings")),
    "train.caption_variations": lambda c: bool(c.get("train.cache_text_embeddings")),
    "train.caption_cache_path": lambda c: bool(c.get("train.cache_text_embeddings")) and bool(c.get("train.caption_variations")),
    "adapter.lycoris_algo": lambda c: c.get("adapter.kind") == "lycoris_lora",
    "adapter.lycoris_bypass": lambda c: c.get("adapter.kind") == "lycoris_lora" and c.get("adapter.lycoris_algo") not in ("dora",),
    "adapter.lycoris_wd_on_output": lambda c: c.get("adapter.kind") == "lycoris_lora" and c.get("adapter.lycoris_algo") in ("dora",),
    "adapter.dtype": lambda c: c.get("adapter.kind") != "none",
    "adapter.rank": lambda c: c.get("adapter.kind") != "none",
    "adapter.alpha": lambda c: c.get("adapter.kind") != "none",
    "adapter.dropout": lambda c: c.get("adapter.kind") != "none",
    "adapter.components": lambda c: c.get("adapter.kind") != "none",
    "adapter.lokr_factor": lambda c: c.get("adapter.kind") == "lokr" or (c.get("adapter.kind") == "lycoris_lora" and c.get("adapter.lycoris_algo") in ("lokr",)),
    "adapter.lokr_decompose_both": lambda c: c.get("adapter.kind") == "lokr" or (c.get("adapter.kind") == "lycoris_lora" and c.get("adapter.lycoris_algo") in ("lokr",)),

    "train.compile_dynamic": lambda c: bool(c.get("train.compile")),
    "train.compile_regional": lambda c: bool(c.get("train.compile")),

    "quant.text_encoder_weights_dtype": lambda c: bool(c.get("quant.quantize_text_encoder")),
    "quant.weights_dtype": lambda c: (c.get("quant.backend", "sdnq") == "sdnq" and c.get("quant.mode") != "none") or c.get("quant.quantize_text_encoder"),
    "quant.use_quantized_matmul": lambda c: c.get("quant.backend", "sdnq") == "sdnq" and c.get("quant.mode") != "none",
    "quant.skip_policy": lambda c: c.get("quant.mode") != "none",
    "quant.extra_skip": lambda c: c.get("quant.mode") != "none",
    "quant.group_size": lambda c: (c.get("quant.backend", "sdnq") == "sdnq" and c.get("quant.mode") != "none") or c.get("quant.quantize_text_encoder"),
    "quant.dynamic_loss_threshold": lambda c: (c.get("quant.backend", "sdnq") == "sdnq" and c.get("quant.mode") != "none") or c.get("quant.quantize_text_encoder"),
    "quant.use_stochastic_rounding": lambda c: c.get("quant.backend", "sdnq") == "sdnq" and c.get("quant.mode") == "training",
}


def _has_texture_phase(c: dict) -> bool:
    return any(p.get("mode") == "texture" for p in (c.get("curriculum") or []))


# Derived from the schema rather than listed, so adding a texture key cannot leave it ungated --
# which is exactly what happened when the oversize cascade added four of them at once.
for _tex in (k for k in SPEC if k.startswith("dataset.texture.")):
    _RULES[_tex] = lambda c: _has_texture_phase(c)


def _adapter_targets(c: dict) -> set[str]:
    t = set(c.get("adapter.components") or [])
    return t


for _comp in ("image_attn", "text_attn", "mlp", "adaln", "base"):
    # Under an adapter, a per-component LR only reaches parameters that actually have one injected
    # -- which is exactly what `load_config` rejects. Full FT can address every component.
    _RULES[f"component_lr.{_comp}"] = (
        lambda c, comp=_comp: c.get("adapter.kind") == "none" or comp in _adapter_targets(c)
    )


_RULES["component_lr.adaln"] = lambda c: c.get("adapter.kind") == "none"


class TrainingGUI(QtWidgets.QWidget):
    def __init__(self, *, preferences=None, config_dir=None):
        super().__init__()
        self.setWindowTitle("Mage-Flow Trainer")
        self.setMinimumSize(1100, 820)
        self.resize(1600, 1000)

        self.editors: dict[str, object] = {}
        self.rows: dict[str, tuple] = {}
        self.runner = None
        self._retired = None
        # Launches still to run after the current one -- one per dataset folder (see `_run_each`).
        self._queue: list[Job] = []
        self._batch_queue_active = False
        self._queue_cancelled = False
        self._active_queue_index = None
        self._queue_generation = 0
        self._on_failure = "stop"
        self._applying = False
        self._current_path: Path | None = None
        # Configs opened from outside `configs/`, kept for the session so the preset
        # dropdown can list them alongside the built-ins.
        self._external: list[Path] = []
        self.preferences = preferences if preferences is not None else Preferences()
        saved_dir = self.preferences.values.get("config_dir")
        self.config_dir = Path(saved_dir).expanduser() if isinstance(saved_dir, str) and saved_dir else CONFIG_DIR
        self.sections = {}
        self._launch_configs = []

        self._setup_ui()
        self._on_gpu_selection()
        if config_dir is not None:
            self._set_config_dir(config_dir)
        self._load_presets()

    # ---------------------------------------------------------------- build

    def _setup_ui(self):
        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(6, 6, 6, 6)
        main.setSpacing(0)

        nav = QtWidgets.QWidget()
        set_role(nav, "navigation")
        nav_lay = QtWidgets.QHBoxLayout(nav)
        nav_lay.setContentsMargins(0, 0, 8, 0)
        nav_lay.setSpacing(0)
        self.tab_bar = QtWidgets.QTabBar()
        self.tab_bar.setExpanding(False)
        self.tab_bar.setDrawBase(False)
        nav_lay.addWidget(self.tab_bar, 0, QtCore.Qt.AlignmentFlag.AlignBottom)
        nav_lay.addStretch(1)
        nav_lay.addWidget(make_label("config", color=THEME.text_muted))
        self.preset_combo = SearchableComboBox()
        self.preset_combo.setMinimumWidth(240)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        nav_lay.addWidget(self.preset_combo)
        nav_lay.addWidget(make_btn("Open", self._open))
        nav_lay.addWidget(make_btn("Save", self._save))
        nav_lay.addWidget(make_btn("Save As", self._save_as))
        nav_lay.addWidget(make_btn("Reload", self._reload))
        nav_lay.addWidget(make_btn("Preferences", self._preferences_dialog))
        main.addWidget(nav, 0)

        self.experimental_chk = QtWidgets.QCheckBox("Enable advanced experimental features")
        self.experimental_chk.setToolTip(
            "Show Self-Flow, dual timestep, RTI, decoded perceptual and eye adversarial controls. "
            "Individual features remain off until enabled. Active features must be disabled "
            "before hiding their controls."
        )
        main.addWidget(self.experimental_chk, 0)

        frame = QtWidgets.QFrame()
        set_role(frame, "mainFrame")
        frame_lay = QtWidgets.QVBoxLayout(frame)
        frame_lay.setContentsMargins(0, 0, 0, 0)
        self.stack = QtWidgets.QStackedWidget()
        frame_lay.addWidget(self.stack, 1)

        experimental_groups = []
        for title, groups in LAYOUT:
            regular_groups = []
            for group_title, keys in groups:
                if group_title.startswith("Experimental "):
                    experimental_groups.append((group_title, keys))
                    continue
                dual = [k for k in keys if k.startswith("flow.dual_timestep")]
                if dual:
                    experimental_groups.append(("Dual timestep noising", dual))
                regular_groups.append((group_title, [k for k in keys if k not in dual]))
            self.tab_bar.addTab(title)
            self.stack.addWidget(self._build_page(regular_groups))

        self.experimental_tab = self.tab_bar.addTab("Experimental")
        self.stack.addWidget(self._build_page(experimental_groups))
        self.tab_bar.setTabVisible(self.experimental_tab, False)
        self.experimental_chk.toggled.connect(self._show_experimental)

        self.run_queue = RunQueueWidget()
        self.run_queue.startRequested.connect(self._start_queue)
        self.queue_tab = self.tab_bar.addTab("Run Queue")
        self.stack.addWidget(self.run_queue)

        self.metrics = LiveMetricsWidget()
        self.tab_bar.addTab("Live Metrics")
        self.stack.addWidget(self.metrics)

        console_page = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(console_page)
        cl.setContentsMargins(12, 12, 12, 12)
        self.console = VirtualConsoleWidget(visible_lines=900)
        cl.addWidget(self.console, 1)
        self.tab_bar.addTab("Console")
        self.stack.addWidget(console_page)

        self.tab_bar.currentChanged.connect(self.stack.setCurrentIndex)
        main.addWidget(frame, 1)
        main.addWidget(self._build_footer(), 0)
        self.tab_bar.currentChanged.connect(self._refresh)

    def _show_experimental(self, enabled):
        if not enabled and self.tab_bar.currentIndex() == self.experimental_tab:
            self.tab_bar.setCurrentIndex(0)
        self.tab_bar.setTabVisible(self.experimental_tab, enabled)

    def _build_page(self, groups):
        page = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        sections = []
        actions = QtWidgets.QHBoxLayout()
        actions.addStretch()
        actions.addWidget(make_btn("Expand all", lambda: [s.setExpanded(True) for s in sections]))
        actions.addWidget(make_btn("Collapse all", lambda: [s.setExpanded(False) for s in sections]))
        outer.addLayout(actions)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        inner = QtWidgets.QWidget()
        # Two columns: config forms are tall and narrow, and a single column wastes half a 1600px
        # window while forcing a scroll.
        cols = QtWidgets.QHBoxLayout(inner)
        cols.setContentsMargins(12, 12, 12, 12)
        cols.setSpacing(12)
        left = QtWidgets.QVBoxLayout()
        right = QtWidgets.QVBoxLayout()
        left.setSpacing(10)
        right.setSpacing(10)

        # Balance by control count rather than by group count, so one 10-key group does not sit
        # beside three 2-key ones.
        def field_form():
            form = QtWidgets.QFormLayout()
            form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                                   | QtCore.Qt.AlignmentFlag.AlignVCenter)
            form.setFieldGrowthPolicy(
                QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            return form

        total = sum(len(keys) for _, keys in groups)
        running, target = 0, total / 2
        for title, keys in groups:
            gb, lay = group_box(title, QtWidgets.QVBoxLayout, collapsible=True)
            sections.append(gb)
            self.sections[title] = gb
            collapsed = self.preferences.values.get("collapsed_sections", [])
            gb.setExpanded(title not in collapsed if isinstance(collapsed, list) else True)
            gb.expandedChanged.connect(lambda expanded, title=title: self._remember_section(title, expanded))
            form = field_form()
            for key in keys:
                if title == "Flow Matching" and key == "flow.use_ot":
                    # Keep the chart beside the controls that change its shape,
                    # before the remaining flow and recipe settings.
                    lay.addLayout(form)
                    self.timestep_plot = TimestepDistributionWidget()
                    lay.addWidget(self.timestep_plot)
                    form = field_form()
                self._add_field(form, key)
            lay.addLayout(form)
            (left if running < target else right).addWidget(gb)
            running += len(keys)

        left.addStretch(1)
        right.addStretch(1)
        cols.addLayout(left, 1)
        cols.addLayout(right, 1)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        return page

    def _add_field(self, form, key):
        spec = SPEC[key]
        editor = spec.make()
        editor.changed.connect(self._on_edit)
        editor.widget.setToolTip(spec.tooltip)
        self.editors[key] = editor
        if key == "dataset.subsets":
            label = QtWidgets.QLabel(spec.label)
            label.setToolTip(spec.tooltip)
            form.addRow(label)
            form.addRow(editor.widget)
            self.rows[key] = (label, editor.widget)
        elif key == "train.run_name":
            row = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(editor.widget, 1)
            self.autosave_chk = QtWidgets.QCheckBox("Autosave")
            self.autosave_chk.setChecked(self.preferences.values.get("autosave", True) is not False)
            self.autosave_chk.setToolTip(
                "Save the current settings when starting training. When off, Save and Save As "
                "still work; Start uses a temporary TOML without changing your saved configs.")
            self.autosave_chk.toggled.connect(lambda on: self._save_preference("autosave", on))
            layout.addWidget(self.autosave_chk)
            label = QtWidgets.QLabel(spec.label)
            form.addRow(label, row)
            self.rows[key] = (label, editor.widget)
        elif spec.inline_label:
            form.addRow(editor.widget)
            self.rows[key] = (editor.widget,)
        else:
            label = QtWidgets.QLabel(spec.label)
            label.setToolTip(spec.tooltip)
            form.addRow(label, editor.widget)
            self.rows[key] = (label, editor.widget)

    def _build_footer(self):
        footer = QtWidgets.QWidget()
        set_role(footer, "footer")
        lay = QtWidgets.QVBoxLayout(footer)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(6)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        # Advisories sit below the status line and above the controls, because several of them are
        # about the GPU checkboxes right underneath -- the two worst traps are only visible once
        # the process count is known, which the config file cannot express.
        self.advice = QtWidgets.QLabel("")
        self.advice.setWordWrap(True)
        self.advice.setVisible(False)
        lay.addWidget(self.advice)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        # Checkboxes, not a text field. This used to be a QLineEdit whose placeholder read "all",
        # which is exactly what someone types into it -- and `CUDA_VISIBLE_DEVICES=all` is not a
        # device list, so torch reported zero GPUs and trained a 2B model on the CPU without one
        # word of complaint. A control that cannot express an invalid value is the fix; the trainer
        # now also refuses to start on CPU, and `training_env` rejects a malformed list.
        gpus = detect_gpus()
        row.addWidget(make_label("GPUs", color=THEME.text_muted))
        self.gpu_boxes: dict[int, QtWidgets.QCheckBox] = {}
        for index, name in gpus:
            box = QtWidgets.QCheckBox(str(index))
            # On Windows only the first is preselected, because only one can be selected at all.
            box.setChecked(index == gpus[0][0] if not MULTI_GPU_SUPPORTED else True)
            box.setToolTip(f"GPU {index}: {name}" + ("" if MULTI_GPU_SUPPORTED else
                           "\n\nWindows: pick which GPU to train on. Multi-GPU is unavailable "
                           "here because DDP needs NCCL, which is Linux-only."))
            box.stateChanged.connect(self._on_gpu_selection)
            # One GPU means there is nothing to choose: locked on, so the control cannot be put
            # into a state that differs from what will actually run. On Windows the boxes stay
            # ENABLED even though only one may be picked -- choosing *which* GPU is still a real
            # choice, and disabling them would take that away to prevent a combination that the
            # radio behaviour below already prevents.
            if len(gpus) == 1:
                box.setEnabled(False)
            row.addWidget(box)
            self.gpu_boxes[index] = box

        # Process count is derived, never entered. Under DDP one process per selected GPU is the
        # only sensible pairing, and a spin box that could disagree with the selection is just a
        # way to launch 2 ranks onto 1 visible device.
        self.proc_label = make_label("", color=THEME.text_muted)
        self.proc_label.setToolTip(
            "One process per selected GPU. Every run goes through `accelerate launch`, single GPU "
            "included, so there is only one code path to reason about.\n\nDDP here is bounded by "
            "PCIe 3.0 x8: a full finetune all-reduces ~3.5GB per optimizer step. Raise gradient "
            "accumulation before adding GPUs.")
        row.addWidget(self.proc_label)

        row.addSpacing(16)
        self.audit_btn = make_btn("Audit dataset", self._audit)
        self.audit_btn.setToolTip(
            "Prints the source-size percentile table and proposes a resolution ladder. Reads image "
            "headers only -- no GPU, no model.")
        row.addWidget(self.audit_btn)
        self.cache_btn = make_btn("Cache latents", self._cache)
        self.cache_btn.setToolTip("Encode the dataset at the configured tier(s).")
        row.addWidget(self.cache_btn)
        self.cache_dry_btn = make_btn("Cache (dry run)", lambda: self._cache(dry=True))
        self.cache_dry_btn.setToolTip("Report the bucket plan and cache size; write nothing.")
        row.addWidget(self.cache_dry_btn)

        row.addStretch(1)
        self.pipeline_chk = QtWidgets.QCheckBox("Paired run + merge")
        self.pipeline_chk.setToolTip(
            "Cache every dataset folder, train two arms, then combine them into one LoRA.\n\n"
            "Arm 1 -- random init + [preserve] regularizer.\n"
            "Arm 2 -- spectral 'mid' init, no regularizer.\n"
            "Then their EXACT sum by factor concatenation (rank 8 + 8 -> 16), written to "
            "<output_dir>/<run_name>-merged.safetensors.\n\n"
            "The two arms come out mutually orthogonal, so their style adds while their individual "
            "failure modes do not. Each arm alone needs ~1.5 weight to express, which is also "
            "where each starts leaking unprompted content; merged, each contributes ~0.5 and the "
            "result is stronger at LESS total weight movement.\n\n"
            "Any step failing cancels the rest -- a merge missing a parent looks fine and is not.")
        self.pipeline_chk.toggled.connect(self._refresh)
        self.pipeline_chk.hide()
        self.start_btn = make_btn("Start Training", self._start, style="accent")
        self.start_btn.setFixedWidth(160)
        row.addWidget(self.start_btn)
        self.stop_btn = make_btn("Stop", self._stop, style="danger")
        self.stop_btn.setFixedWidth(100)
        self.stop_btn.setVisible(False)
        row.addWidget(self.stop_btn)
        lay.addLayout(row)
        return footer

    # ---------------------------------------------------------------- gpu selection

    def selected_gpus(self) -> list[int]:
        return sorted(i for i, b in self.gpu_boxes.items() if b.isChecked())

    def gpu_arg(self) -> str:
        """CUDA_VISIBLE_DEVICES for the child. Empty when every GPU is selected -- setting it
        redundantly only creates another place for the index mapping to be wrong."""
        chosen = self.selected_gpus()
        if not chosen or len(chosen) == len(self.gpu_boxes):
            return ""
        return ",".join(str(i) for i in chosen)

    def num_processes(self) -> int:
        return max(1, len(self.selected_gpus()))

    def _on_gpu_selection(self):
        # Deselecting the last GPU would mean "train on nothing"; keep at least one checked.
        if not self.selected_gpus():
            for box in self.gpu_boxes.values():
                box.blockSignals(True)
                box.setChecked(True)
                box.blockSignals(False)

        # Windows: the boxes behave as radio buttons. DDP needs a NCCL backend and NCCL is
        # Linux-only, so more than one process cannot work here at all -- but *which* GPU to use is
        # still a real choice, so the controls stay live and the newest tick simply wins. Enforcing
        # it here rather than by disabling the boxes keeps that choice available.
        if not MULTI_GPU_SUPPORTED and len(self.selected_gpus()) > 1:
            keep = self.sender()
            chosen = keep if isinstance(keep, QtWidgets.QCheckBox) and keep.isChecked() else None
            for index, box in self.gpu_boxes.items():
                box.blockSignals(True)
                box.setChecked(box is chosen if chosen is not None
                               else index == self.selected_gpus()[0])
                box.blockSignals(False)

        n = self.num_processes()
        self.proc_label.setText(
            f"→ {n} process{'es' if n > 1 else ''}"
            + ("" if MULTI_GPU_SUPPORTED else "  (Windows: single GPU only)"))

    # ---------------------------------------------------------------- config I/O

    def _save_preference(self, key, value):
        try:
            self.preferences.set(key, value)
        except OSError as exc:
            self.log(f"Could not save GUI preference: {exc}")

    def _remember_section(self, title, expanded):
        collapsed = [name for name, section in self.sections.items() if not section.header.isChecked()]
        self._save_preference("collapsed_sections", collapsed)

    def _set_config_dir(self, value):
        import tempfile
        target = Path(value).expanduser().resolve() if value else CONFIG_DIR
        try:
            target.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=target):
                pass
            self.preferences.set("config_dir", str(target) if value else "")
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot use TOML folder", str(exc))
            return False
        self.config_dir = target
        return True

    def _preferences_dialog(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("GUI preferences")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel("Folder for saved TOMLs (blank uses the repository's configs/):"))
        row = QtWidgets.QHBoxLayout()
        folder = QtWidgets.QLineEdit(str(self.config_dir))
        folder.setMinimumWidth(420)
        row.addWidget(folder)
        def browse():
            selected = QtWidgets.QFileDialog.getExistingDirectory(dialog, "TOML folder", folder.text())
            if selected:
                folder.setText(selected)
        row.addWidget(make_btn("Browse", browse))
        row.addWidget(make_btn("Use default", folder.clear))
        layout.addLayout(row)
        info = QtWidgets.QLabel(f"Remembered for future launches in:\n{self.preferences.path}\nExisting TOMLs are not moved.")
        info.setWordWrap(True)
        layout.addWidget(info)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Save |
                                             QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted and self._set_config_dir(folder.text().strip()):
            if self._current_path is not None:
                self._external.append(self._current_path)
            self._load_presets(select=self._current_path, apply=False)
            self.log(f"TOML save folder: {self.config_dir}")

    def _load_presets(self, select: Path | None = None, *, apply=True):
        # Stat before populating: deleted/inaccessible files must not break the GUI.
        candidates = {}
        for path in [*self.config_dir.glob("*.toml"), *self._external]:
            try:
                candidates[path.resolve()] = (path.stat().st_mtime_ns, path)
            except OSError:
                continue
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        for modified, path in sorted(candidates.values(), key=lambda entry: (-entry[0], str(entry[1]).casefold())):
            label = path.stem if path.parent.resolve() == self.config_dir.resolve() else f"{path.stem}  —  {path.parent}"
            self.preset_combo.addItem(label, str(path))
            date = QtCore.QDateTime.fromMSecsSinceEpoch(modified // 1_000_000).toString("yyyy-MM-dd HH:mm:ss")
            self.preset_combo.setItemData(self.preset_combo.count() - 1, f"{path}\nModified: {date}",
                                         QtCore.Qt.ItemDataRole.ToolTipRole)
        index = self.preset_combo.findData(str(select)) if select is not None else 0
        if not apply and select is None:
            index = -1  # The form has unsaved values; do not label it as another file.
        self.preset_combo.setCurrentIndex(index if self.preset_combo.count() else -1)
        self.preset_combo.blockSignals(False)
        # The rebuild happened with signals blocked, so the editable box never got told the
        # selection moved; push the current item's name back into it by hand.
        self.preset_combo.sync_display()

        if not apply:
            return
        if self.preset_combo.count():
            self._on_preset_changed(self.preset_combo.currentIndex())
        else:
            self._apply(bridge.defaults())

    def _open(self):
        """Load a TOML from anywhere; saves make a copy in the managed folder."""
        start = usable_dialog_start(
            str(self._current_path) if self._current_path else None, str(self.config_dir))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open config", start, "TOML configs (*.toml);;All files (*)")
        if not path:
            return
        p = Path(path)
        if p.resolve() not in {q.resolve() for q in self._external}:
            self._external.append(p)
        self._load_presets(select=p)

    def _on_preset_changed(self, index):
        path = self.preset_combo.itemData(index)
        if not path:
            return
        self._current_path = Path(path)
        try:
            flat = bridge.flatten(bridge.read_toml(path))
        except Exception as exc:
            self.log(f"ERROR reading {path}: {exc}")
            return
        merged = bridge.defaults(flat.get("optimizer.kind", "adamw"))
        merged.update(flat)
        if merged.get("adapter.kind") == "lycoris_lora" and "train.compile" not in flat:
            merged["train.compile"] = "default"
        self._apply(merged)
        self.log(f"Loaded {path}")
        base = bridge.defaults()
        advanced = sorted(k for k in bridge.CLI_ONLY_KEYS if merged.get(k) != base.get(k))
        if advanced:
            self.log("Retained TOML-only settings: " + ", ".join(advanced))

    def _reload(self):
        self._on_preset_changed(self.preset_combo.currentIndex())

    def _apply(self, flat):
        self._cli_values = {k: v for k, v in flat.items() if k in bridge.CLI_ONLY_KEYS}
        self._applying = True
        try:
            for key, editor in self.editors.items():
                editor.set(flat.get(key))
        finally:
            self._applying = False
        self._last_optimizer_kind = flat.get("optimizer.kind")
        self._last_adapter_selection = (flat.get("adapter.kind"), flat.get("adapter.lycoris_algo"))
        self._lock_moss_sampling()
        self._refresh()

    def collect(self) -> dict:
        flat = {**getattr(self, "_cli_values", {}),
                **{key: editor.get() for key, editor in self.editors.items()}}
        if self.editors["dataset.subsets"].table.rowCount():
            # Even an unfinished subset row switches away from the single-folder
            # source. Do not accidentally launch on the old, now-disabled folder.
            flat["dataset.path"] = None
            flat["dataset.mask_path"] = None
        return flat

    def _managed_target(self, flat: dict) -> Path:
        """Where a Save/Start should land: `<configured folder>/<run_name>.toml`.

        The config file is *named by* `run_name`. Keeping the two in sync by hand is the thing this
        removes -- and they were never independent anyway, since `run_name` already decides the
        checkpoint stem (`out_dir/<run_name>.safetensors`) and the copy of the TOML the trainer
        writes beside it. One name, three places.
        """
        return self.config_dir / f"{safe_stem(flat.get('train.run_name'))}.toml"

    def _persist(self, flat: dict) -> Path | None:
        """Write `flat` to the managed config and return where it went, or None if cancelled.

        Rules about not destroying a file the user did not mean to touch:

        **A config opened from outside the managed folder is never written back to.** Saving copies it in
        instead, and the original is left exactly as it was. Opening someone's config to read its
        settings, tweaking a knob and pressing Start used to rewrite their file in place -- and
        `write_toml` regenerates the TOML, so their comments went with it.

        **Changing `run_name` writes a NEW config and leaves the old one in place.** The filename
        still follows `run_name`, so the two names never drift -- but following a name is not a
        reason to delete a file. Deriving a second run from an existing config is the ordinary way
        to use this GUI, and the previous behaviour moved the original out from under the user:
        load `modan-ft.toml`, rename to `modan-ft-v2`, Save, and `modan-ft.toml` was gone.

        **A save never lands on a *different* existing config without asking.** `run_name` collides
        with more than this file: two runs sharing it also share `out_dir/<run_name>.safetensors`,
        so the checkpoints overwrite each other too. That is worth a dialog.
        """
        target = self._managed_target(flat)
        src = self._current_path
        src_resolved = src.resolve() if src is not None and src.exists() else None
        external = src is not None and src.parent.resolve() != self.config_dir.resolve()

        if target.exists() and (src_resolved is None or target.resolve() != src_resolved):
            answer = QtWidgets.QMessageBox.question(
                self, "Config already exists",
                f"{target.name} already exists in {self.config_dir}.\n\n"
                f"run_name is {flat.get('train.run_name')!r}, which also names the checkpoints "
                f"(output/{flat.get('train.run_name')}.safetensors). Two runs sharing it overwrite "
                f"each other's output as well as this file.\n\nOverwrite it?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No)
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                self.log(f"Save cancelled -- {target.name} already exists")
                return None

        if external:
            self.log(f"{src.name} was opened from {src.parent} -- saving a copy to "
                     f"{target.name}, original untouched")
        elif src_resolved is not None and target.resolve() != src_resolved:
            self.log(f"run_name changed -- saved as {target.name}, {src.name} left as it was")

        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            bridge.write_toml(target, flat)
        except OSError as exc:
            self.log(f"Could not save {target}: {exc}")
            return None
        self._current_path = target
        return target

    def _training_config(self, flat):
        if self.autosave_chk.isChecked():
            path = self._persist(flat)
            if path is not None:
                self._load_presets(select=path, apply=False)
            return path
        path = self._temporary_config_dir() / f"{safe_stem(flat.get('train.run_name'))}.toml"
        bridge.write_toml(path, flat)
        self.log(f"Autosave off: training uses temporary config {path}")
        return path

    def _temporary_config_dir(self):
        import tempfile
        directory = tempfile.TemporaryDirectory(prefix="mageflow-gui-")
        self._launch_configs.append(directory)
        return Path(directory.name)

    def _save(self):
        path = self._persist(self.collect())
        if path is None:
            return
        self.log(f"Saved {path}")
        self._load_presets(select=path)

    def _save_as(self):
        # A real file dialog rather than a name prompt, so a config can be saved beside the dataset
        # or the run it belongs to. Defaults to `configs/`, which is where it used to be forced.
        start = usable_dialog_start(
            str(self._current_path) if self._current_path else None, str(self.config_dir))
        chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save config as", start, "TOML configs (*.toml)")
        if not chosen:
            return
        path = Path(chosen)
        if path.suffix != ".toml":
            path = path.with_suffix(".toml")
        if path.parent.resolve() != self.config_dir.resolve():
            self._external.append(path)
        bridge.write_toml(path, self.collect())
        self._current_path = path
        self._on_gpu_selection()
        self._load_presets(select=path)
        idx = self.preset_combo.findData(str(path))
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)
        self.log(f"Saved {path}")

    # ---------------------------------------------------------------- reactivity

    def _lock_moss_sampling(self):
        if self.editors["flow.timestep_sample_method"].get() != "moss":
            return
        locked = {"flow.sigmoid_scale": 1.0, "flow.shift": 1.0, "flow.flux_shift": False}
        changed = any(self.editors[key].get() != value for key, value in locked.items())
        self._applying = True
        try:
            for key, value in locked.items():
                self.editors[key].set(value)
        finally:
            self._applying = False
        if changed:
            self.log("Kohaku MOSS sampling: locked sigmoid_scale=1, shift=1, flux_shift=false")

    def _on_edit(self):
        if not self._applying:
            self._lock_moss_sampling()
            kind = self.editors["optimizer.kind"].get()
            if kind != getattr(self, "_last_optimizer_kind", kind) and kind in OPTIMIZERS:
                self._applying = True
                try:
                    for key, value in optimizer_defaults(kind).items():
                        self.editors[f"optimizer.{key}"].set(value)
                finally:
                    self._applying = False
            self._last_optimizer_kind = kind
            selection = (self.editors["adapter.kind"].get(), self.editors["adapter.lycoris_algo"].get())
            if selection != getattr(self, "_last_adapter_selection", None) and selection[0] == "lycoris_lora":
                self._applying = True
                try:
                    self.editors["train.compile"].set("default")
                    self.editors["adapter.lycoris_bypass"].set(selection[1] not in ("dora",))
                    self.editors["adapter.lycoris_wd_on_output"].set(True)
                finally:
                    self._applying = False
            self._last_adapter_selection = selection
            self._refresh()

    def _refresh(self):
        flat = self.collect()
        self.timestep_plot.set_config(flat)
        has_subsets = self.editors["dataset.subsets"].table.rowCount() > 0
        for key in ("dataset.path", "dataset.num_repeats", "dataset.mask_path"):
            self.editors[key].set_enabled(not has_subsets)
            for widget in self.rows[key]:
                widget.setEnabled(not has_subsets)
        # Never hide an enabled experiment loaded from a TOML. Visibility is a UI
        # preference; it must not silently mutate or conceal the training config.
        active_experiment = any(flat.get(k) for k in (
            "self_flow.enabled", "flow.dual_timestep", "rti.enabled",
            "perceptual.enabled", "adversarial.enabled",
        ))
        if active_experiment:
            self.experimental_chk.setChecked(True)
        self.experimental_chk.setEnabled(not active_experiment)
        for key, rule in _RULES.items():
            editor = self.editors.get(key)
            if editor is None:
                continue
            try:
                on = bool(rule(flat))
            except Exception:
                on = True
            editor.set_enabled(on)
            for widget in self.rows.get(key, ()):
                widget.setEnabled(on)

        ok, err = bridge.validate(flat)
        running = self.runner is not None or bool(self._queue) or self._batch_queue_active
        self.run_queue.set_busy(running)
        for box in self.gpu_boxes.values():
            box.setEnabled(not running)

        # Advisories are evaluated against the GPU SELECTION, not just the config -- the two worst
        # ones cannot be seen from the file alone. An "error" advisory blocks Start, because the
        # trainer would refuse the same combination anyway; blocking here turns a 30-second wait
        # and a traceback into an immediate, readable no.
        notes = bridge.advisories(flat, self.num_processes())
        blocking = [m for lvl, m in notes if lvl == "error"]
        self.start_btn.setEnabled(ok and not running and not blocking)
        # The queue has its own Start button; do not offer the unrelated editor's
        # single-run action while viewing queued TOMLs.
        self.start_btn.setVisible(not running and self.tab_bar.currentIndex() != self.queue_tab)
        self.stop_btn.setVisible(running)
        # Pressing Start means something different with the box ticked, so it should not still say
        # "Start Training" -- two trainings and a merge is not what that label promises.
        self.start_btn.setText(
            "Start Pipeline" if self.pipeline_chk.isChecked() else "Start Training")
        for btn in (self.cache_btn, self.cache_dry_btn, self.audit_btn):
            btn.setEnabled(not running)
        self.pipeline_chk.setEnabled(not running)
        if ok:
            self.status.setText(
                f"<span style='color:{SUCCESS}'>&#10003;</span> {bridge.summarize(flat)}")
        else:
            self.status.setText(f"<span style='color:{DANGER}'>{err}</span>")

        colour = {"error": DANGER, "warn": WARN, "note": THEME.text_muted}
        mark = {"error": "&#10007;", "warn": "&#9888;", "note": "&#8226;"}
        self.advice.setText("<br>".join(
            f"<span style='color:{colour[lvl]}'>{mark[lvl]} {msg}</span>" for lvl, msg in notes))
        self.advice.setVisible(bool(notes))

    # ---------------------------------------------------------------- running

    def log(self, text, replace_last=False):
        self.console.append_line(text, replace_last=replace_last)

    def _handle_output(self, line, is_progress):
        self.log(line, replace_last=is_progress and self._last_was_progress)
        self._last_was_progress = is_progress

    _last_was_progress = False

    def _run(self, launch, training=True, on_failure="stop"):
        if self.runner is not None and self.runner.isRunning():
            self.log("A process is already running.")
            return
        self._on_failure = on_failure
        self._running_training = training
        try:
            env = training_env(self.gpu_arg())
        except ValueError as exc:            # unreachable from the checkboxes; a guard, not a path
            self.log(f"CONFIG ERROR: {exc}")
            return
        self.runner = ProcessRunner(launch, str(PROJECT_ROOT), env)
        self.runner.logSignal.connect(self.log)
        self.runner.errorSignal.connect(self.log)
        self.runner.progressSignal.connect(self._handle_output)
        self.runner.metricsSignal.connect(self.metrics.parse_and_update)
        self.runner.finishedSignal.connect(self._finished)
        self.log("\n" + "=" * 60 + f"\n{launch.label}\n" + " ".join(launch.argv) + "\n" + "=" * 60)
        self.tab_bar.setCurrentIndex(self.tab_bar.count() - (2 if training else 1))
        if training:
            self.metrics.start_run()
            prevent_sleep(True)
        self.start_btn.setVisible(False)
        self.stop_btn.setVisible(True)
        self._refresh()
        self.runner.start()

    def _start_queue(self):
        from ..training.config import load_config

        if self.runner is not None or self._queue or self._batch_queue_active:
            return
        jobs = []
        destinations = set()
        try:
            for index, path in enumerate(self.run_queue.paths()):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                cfg = load_config(path)
                flat = bridge.flatten(asdict(cfg))
                errors = [text for level, text in bridge.advisories(flat, self.num_processes()) if level == "error"]
                if errors:
                    raise ValueError("; ".join(errors))
                destination = ((PROJECT_ROOT / cfg.train.output_dir).resolve(), cfg.train.run_name)
                if destination in destinations:
                    raise ValueError(f"Duplicate output directory/run_name: {destination}. Give each queued run a distinct name.")
                destinations.add(destination)
                steps = self._cache_jobs(flat) + [Job(train_launch(path, self.num_processes()), training=True)]
                jobs.extend(replace(job, config_path=path, config_digest=digest, queue_index=index) for job in steps)
        except Exception as exc:
            message = f"Queue validation failed for {path}: {exc}"
            self.log(message)
            self.run_queue.status.setText(message)
            return
        if not jobs:
            return
        self._batch_queue_active = True
        self._queue_cancelled = False
        self.run_queue.reset_status()
        self._queue = jobs[1:]
        self.log(f"Starting queue: {len(self.run_queue.paths())} TOMLs, {len(jobs)} cache/train steps.")
        self._run_job(jobs[0])

    def _run_job(self, job):
        self._active_queue_index = job.queue_index
        if job.config_path is not None:
            try:
                if hashlib.sha256(job.config_path.read_bytes()).hexdigest() != job.config_digest:
                    raise ValueError(f"Queued TOML changed after validation: {job.config_path}. Restart the queue to use the edited file.")
            except (OSError, ValueError) as exc:
                self.log(str(exc))
                self._queue = []
                self._batch_queue_active = False
                self.run_queue.mark(job.queue_index, "Failed")
                self.run_queue.end(str(exc))
                self._active_queue_index = None
                self.start_btn.setVisible(True)
                self.stop_btn.setVisible(False)
                self._refresh()
                return
            self.run_queue.mark(job.queue_index, "Training" if job.training else "Caching")
        self._run(job.launch, job.training, job.on_failure)

    def _finished(self, code):
        if getattr(self, "_running_training", False):
            self.metrics.finish_run()
        prevent_sleep(False)
        self.start_btn.setVisible(True)
        self.stop_btn.setVisible(False)
        self.log(f"Process finished with exit code {code}")
        # `self.runner = None` here would drop the last reference to the QThread *while it is still
        # emitting* `finishedSignal`, and PySide then tears the C++ object down mid-emission: every
        # slot connected after this one is silently skipped. Observed, not theoretical -- it is how
        # this was found. Hand the object to the next event-loop turn instead, by which time the
        # emit has returned.
        self._retired, self.runner = self.runner, None
        QtCore.QTimer.singleShot(0, lambda: setattr(self, "_retired", None))
        if code != 0 and self._on_failure == "stop" and self._queue:
            # Every later step depends on this one having produced its output, so carrying on would
            # not salvage anything -- it would produce a plausible-looking artifact built on a
            # missing input, which is the failure mode worth spending code to prevent.
            self.log(f"Chain stopped: {len(self._queue)} remaining step(s) skipped.")
            self._queue = []

        if self._batch_queue_active:
            status = "Stopped" if self._queue_cancelled else ("Failed" if code else "Completed")
            if code or self._queue_cancelled or self._running_training:
                self.run_queue.mark(self._active_queue_index, status)
            if not self._queue:
                self._batch_queue_active = False
                self.run_queue.end("Queue completed." if status == "Completed" else f"Queue {status.lower()}.")
                self._active_queue_index = None
        self.start_btn.setVisible(not bool(self._queue))
        self.stop_btn.setVisible(bool(self._queue))
        self._refresh()

        if self._queue:
            # `_run` refuses to start while a runner is live, so this waits for the next event-loop
            # turn -- `self.runner` is already None above, but the QThread it referred to is still
            # finishing.
            nxt = self._queue.pop(0)
            generation = self._queue_generation
            QtCore.QTimer.singleShot(
                0, lambda: self._run_job(nxt) if generation == self._queue_generation else None)

    def _start(self):
        if self.pipeline_chk.isChecked():
            self._start_pipeline()
            return
        flat = self.collect()
        ok, err = bridge.validate(flat)
        if not ok:
            self.log(f"CONFIG ERROR: {err}")
            return
        # The trainer reads a file, so what runs is exactly what is on disk -- no hidden in-memory
        # variant that a later "why did it use that value" cannot account for. It goes through
        # `_persist` for the same reason Save does: starting a run must not rewrite a config that
        # was opened from somewhere else on disk.
        path = self._training_config(flat)
        if path is None:
            return

        # Cache first here too. Same reasoning as the pipeline: an uncached folder is silent, and
        # a warm cache makes this a no-op. A cache step that fails cancels the training, because
        # training past it is exactly the silent-wrong-dataset case.
        jobs = self._cache_jobs(flat) + [Job(train_launch(path, self.num_processes()),
                                             training=True)]
        if len(jobs) > 1:
            self.log(f"Caching {len(jobs) - 1} dataset folder(s), then training.")
        self._queue = jobs[1:]
        self._run(jobs[0].launch, jobs[0].training, jobs[0].on_failure)

    # ------------------------------------------------------------------ paired pipeline

    # Two arms that differ ONLY in initialisation and objective, then their exact sum.
    #
    # Measured on Mage-Flow: LoRAs trained on the same data with different inits come out mutually
    # ORTHOGONAL (pairwise cosine 0.000-0.001), so their style contributions add while their
    # individual failure modes stay in disjoint subspaces. Each arm alone needs weight ~1.5 to
    # express, and ~1.5 is also where each starts dragging in unprompted content; combined, each
    # contributes at ~0.5 and the total weight movement is SMALLER than either arm alone while
    # rendering stronger. The overdrive was the damage.
    PIPELINE_ARMS = (
        # (suffix, adapter.init, keeps [preserve])
        ("preserve", "random", True),   # first: it can fail at setup, so it fails in minute one
        ("svdmid", "mid", False),
    )

    def _arm_config(self, flat: dict, suffix: str, init: str, preserve: bool) -> Path | None:
        """Save a generated arm, or use a temporary TOML when autosave is off."""
        arm = dict(flat)
        base = safe_stem(flat.get("train.run_name")) or "run"
        arm["train.run_name"] = f"{base}-{suffix}"
        arm["adapter.init"] = init
        # The concat step needs the final weights at a predictable path.
        arm["train.skip_final_save"] = False
        if not preserve:
            arm["preserve.prompts"] = None

        ok, err = bridge.validate(arm)
        if not ok:
            self.log(f"CONFIG ERROR ({suffix} arm): {err}")
            return None

        out_dir = self.config_dir / "generated" if self.autosave_chk.isChecked() else self._temporary_config_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{arm['train.run_name']}.toml"
        bridge.write_toml(path, arm)
        return path

    def _cache_jobs(self, flat: dict) -> list[Job]:
        """A caching step per dataset folder, to run before training.

        Free when the cache is warm -- `overwrite=False` skips what already exists, so this costs a
        directory scan. Worth doing unconditionally because the failure it prevents is silent: the
        dataset layer reads latents and never images, so an uncached folder does not raise, it just
        contributes nothing. A two-subset config with one folder uncached trains happily on the
        other one and produces a LoRA of the wrong thing.
        """
        tiers = flat.get("dataset.resolutions") or [flat.get("dataset.resolution") or 1024]
        return [Job(cache_launch(
            pth, flat.get("train.model_path") or "", tiers,
            vae_path=flat.get("train.vae_path"),
            flux2_vae=bool(flat.get("train.flux2_vae")),
            model_family=flat.get("train.model_family") or "auto",
            cache_validation=flat.get("dataset.cache_validation") or "warn",
            min_bucket_reso=flat.get("dataset.min_bucket_reso") or 256,
            max_bucket_reso=flat.get("dataset.max_bucket_reso") or 1920,
            bucket_reso_steps=flat.get("dataset.bucket_reso_steps") or 64,
            upscale=not flat.get("dataset.bucket_no_upscale", True),
            multires_training=bool(flat.get("dataset.multires_training")),
            gpus=self.gpu_arg(),
        )) for pth in self._dataset_paths(flat)]

    def _start_pipeline(self):
        """Cache every dataset folder, train both arms, then concatenate them.

        Caching first is free when the cache is warm (`overwrite=False` skips what exists) and is
        the difference between a clear error and a run that trains on a partial dataset.
        """
        flat = self.collect()
        ok, err = bridge.validate(flat)
        if not ok:
            self.log(f"CONFIG ERROR: {err}")
            return
        if flat.get("adapter.kind") != "lora":
            self.log("The paired pipeline needs adapter.kind = 'lora' "
                     "(spectral init and [preserve] both require an adapter).")
            return
        if not flat.get("preserve.prompts"):
            self.log("The paired pipeline needs a [preserve] contrast-pair file "
                     "-- see configs/preserve-general.txt.")
            return

        paths = self._dataset_paths(flat)
        if not paths:
            self.log("Nothing to train on -- set a Dataset path, or give the subset rows a folder.")
            return

        jobs = self._cache_jobs(flat)

        out_dir = Path(flat.get("train.output_dir") or "output")
        base = safe_stem(flat.get("train.run_name")) or "run"
        parents = []
        for suffix, init, preserve in self.PIPELINE_ARMS:
            cfg = self._arm_config(flat, suffix, init, preserve)
            if cfg is None:
                return
            jobs.append(Job(train_launch(cfg, self.num_processes()), training=True))
            name = f"{base}-{suffix}"
            parents.append((str(out_dir / name / f"{name}.safetensors"), 1.0))

        merged = out_dir / f"{base}-merged.safetensors"
        jobs.append(Job(concat_launch(merged, parents)))

        self.log(f"Pipeline: {len(paths)} cache step(s), "
                 f"{len(self.PIPELINE_ARMS)} training arms, then concat -> {merged}")
        self._queue = jobs[1:]
        first = jobs[0]
        self._run(first.launch, first.training, first.on_failure)

    def _stop(self):
        self._queue_generation += 1
        self._queue_cancelled = True
        # Drop the rest of the queue first: Stop means stop, not "skip to the next folder".
        if self._queue:
            self.log(f"Stopped -- {len(self._queue)} queued step(s) skipped.")
            self._queue = []
        if self.runner is not None and self.runner.isRunning():
            self.runner.stop()
        elif self._batch_queue_active:
            self._batch_queue_active = False
            if (self._active_queue_index is not None and
                    self.run_queue.list.item(self._active_queue_index).text().startswith(("Caching —", "Training —"))):
                self.run_queue.mark(self._active_queue_index, "Stopped")
            self.run_queue.end("Queue stopped.")
            self._active_queue_index = None
            self.start_btn.setVisible(True)
            self.stop_btn.setVisible(False)
            self._refresh()

    def _dataset_paths(self, flat: dict) -> list[str]:
        """Every directory this config actually reads images from, in config order.

        `path` and `subsets` are mutually exclusive in the loader, so a config with subset rows has
        no single dataset path -- and both tools below take exactly one directory per invocation.

        Empty entries are dropped rather than passed through, because `Path("")` is `Path(".")` and
        passes `.is_dir()`: an empty box would silently audit or cache the repo root instead of
        failing. That was the live bug here -- with subsets configured, `dataset.path` is unset, so
        both buttons ran against `.`.
        """
        subsets = flat.get("dataset.subsets") or []
        raw = [s.get("path") for s in subsets] if subsets else [flat.get("dataset.path")]
        return [p for p in (str(x or "").strip() for x in raw) if p]

    def _run_each(self, launches: list, what: str) -> None:
        """Run one launch per dataset directory, in turn.

        Sequential rather than concurrent: caching is GPU-bound, and interleaving the output of
        several would make the per-folder reports unreadable.
        """
        if not launches:
            return
        self._queue = [Job(l, training=False, on_failure="continue") for l in launches[1:]]
        if self._queue:
            self.log(f"{what} {len(launches)} dataset folders in turn.")
        self._run(launches[0], training=False, on_failure="continue")

    def _cache(self, dry=False):
        c = self.collect()
        paths = self._dataset_paths(c)
        if not paths:
            self.log("Nothing to cache -- set a Dataset path, or give the subset rows a folder.")
            return
        tiers = c.get("dataset.resolutions") or [c.get("dataset.resolution") or 1024]
        self._run_each([cache_launch(
            p, c.get("train.model_path") or "", tiers,
            vae_path=c.get("train.vae_path"),
            flux2_vae=bool(c.get("train.flux2_vae")),
            model_family=c.get("train.model_family") or "auto",
            cache_validation=c.get("dataset.cache_validation") or "warn",
            min_bucket_reso=c.get("dataset.min_bucket_reso") or 256,
            max_bucket_reso=c.get("dataset.max_bucket_reso") or 1920,
            bucket_reso_steps=c.get("dataset.bucket_reso_steps") or 64,
            upscale=not c.get("dataset.bucket_no_upscale", True),
            multires_training=bool(c.get("dataset.multires_training")),
            dry_run=dry, gpus=self.gpu_arg(),
        ) for p in paths], "Caching")

    def _audit(self):
        c = self.collect()
        paths = self._dataset_paths(c)
        if not paths:
            self.log("Nothing to audit -- set a Dataset path, or give the subset rows a folder.")
            return
        steps = c.get("dataset.bucket_reso_steps") or 64
        self._run_each([audit_launch(p, steps) for p in paths], "Auditing")

    def closeEvent(self, event):
        if self.runner is not None and self.runner.isRunning():
            answer = QtWidgets.QMessageBox.question(
                self, "Training is running",
                "Stop the running process and quit?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No)
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self._stop()
        prevent_sleep(False)
        event.accept()


def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config-dir", help="Remember this folder for GUI TOML saves")
    options, _ = parser.parse_known_args(sys.argv[1:])
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    app.setApplicationName("Mage-Flow Trainer")
    window = TrainingGUI(config_dir=options.config_dir)
    window.show()
    sys.exit(app.exec())
