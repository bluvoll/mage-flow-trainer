"""One editor per config value type.

Ours. Aozora dispatches on widget type inside the main window and reads values back with a big
`isinstance` ladder; that works when every key is a scalar, and ours are not -- `train.batch_size`
is a bucket->size map, `adapter.components` is a set drawn from a fixed vocabulary, and fourteen
keys are `X | None` where None means something specific and is not the same as 0.

So each editor owns its own get/set. An empty box means **unset** for the optional types, which is
the only honest spelling: `component_lr.mlp` unset inherits `optimizer.lr`, while
`component_lr.mlp = 0.0` freezes the component. A spin box cannot express that difference.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .widgets import (
    NoScrollComboBox,
    NoScrollDoubleSpinBox,
    NoScrollSpinBox,
    make_btn,
    make_label,
    usable_dialog_start,
)


class Editor(QtCore.QObject):
    """Base: owns `widget`, converts between the Qt state and the TOML value."""

    changed = QtCore.Signal()

    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def get(self):
        raise NotImplementedError

    def set(self, value):
        raise NotImplementedError

    def set_enabled(self, on: bool):
        self.widget.setEnabled(on)


class _Blockable(Editor):
    """Applying a config must not look like the user typing, or every load marks the config
    dirty and re-validates 96 times."""

    def __init__(self, widget):
        super().__init__(widget)
        self._applying = False

    def _emit(self, *_):
        if not self._applying:
            self.changed.emit()

    def set(self, value):
        self._applying = True
        try:
            self._set(value)
        finally:
            self._applying = False

    def _set(self, value):
        raise NotImplementedError


class TextEditor(_Blockable):
    """`strip=False` matters for exactly one key and matters a lot: `tag_delimiter` defaults to
    `", "`, and stripping it to `","` silently rewrites every caption in the dataset."""

    def __init__(self, placeholder="", strip=True):
        w = QtWidgets.QLineEdit()
        w.setPlaceholderText(placeholder)
        super().__init__(w)
        self.strip = strip
        w.textChanged.connect(self._emit)

    def get(self):
        text = self.widget.text()
        return text.strip() if self.strip else text

    def _set(self, value):
        self.widget.setText("" if value is None else str(value))


class OptTextEditor(TextEditor):
    """Empty string means the key is absent, not `""`."""

    def __init__(self, placeholder="unset"):
        super().__init__(placeholder)

    def get(self):
        return self.widget.text().strip() or None


class PathEditor(_Blockable):
    def __init__(self, kind="folder"):
        container = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.line = QtWidgets.QLineEdit()
        browse = make_btn("Browse", self._browse)
        browse.setFixedWidth(80)
        row.addWidget(self.line, 1)
        row.addWidget(browse)
        super().__init__(container)
        self.kind = kind
        self.line.textChanged.connect(self._emit)

    def _browse(self):
        start = usable_dialog_start(self.line.text())
        if self.kind == "folder":
            path = QtWidgets.QFileDialog.getExistingDirectory(self.widget, "Select folder", start)
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self.widget, "Select file", start)
        if path:
            self.line.setText(path)

    def get(self):
        return self.line.text().strip()

    def _set(self, value):
        self.line.setText("" if value is None else str(value))


class IntEditor(_Blockable):
    def __init__(self, lo=0, hi=1_000_000, step=1):
        w = NoScrollSpinBox()
        w.setLocale(QtCore.QLocale.c())     # no thousands separator on step counts
        w.setRange(lo, hi)
        w.setSingleStep(step)
        super().__init__(w)
        w.valueChanged.connect(self._emit)

    def get(self):
        return int(self.widget.value())

    def _set(self, value):
        self.widget.setValue(int(value or 0))


class OptIntEditor(_Blockable):
    """Empty = unset. `max_steps`, `keep_last_n`, `save_every_steps` all mean "no limit" when
    absent, which no spin-box value can represent."""

    def __init__(self, placeholder="unset"):
        w = QtWidgets.QLineEdit()
        w.setPlaceholderText(placeholder)
        w.setValidator(QtWidgets.QLineEdit().validator())
        super().__init__(w)
        w.textChanged.connect(self._emit)

    def get(self):
        text = self.widget.text().strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None

    def _set(self, value):
        self.widget.setText("" if value is None else str(int(value)))


class FloatEditor(_Blockable):
    """Locale is pinned per widget, not globally. A spin box formats through the *widget's* locale,
    which Qt inherits from QApplication at construction -- so on a Spanish desktop the form shows
    `0,25` while the TOML it writes says `0.25`. `QLocale.setDefault` does not fix that; this does.
    """

    def __init__(self, lo=0.0, hi=1e9, step=0.05, decimals=3):
        w = NoScrollDoubleSpinBox()
        w.setLocale(QtCore.QLocale.c())
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setDecimals(decimals)
        super().__init__(w)
        w.valueChanged.connect(self._emit)

    def get(self):
        return float(self.widget.value())

    def _set(self, value):
        self.widget.setValue(float(value or 0.0))


class SciEditor(_Blockable):
    """Learning rates and epsilons. A double spin box would need 8 decimals to show `1e-8` and
    would still round-trip `5e-5` into `0.00005000`, so this is a text field parsed as a float."""

    def __init__(self, placeholder="", optional=False):
        w = QtWidgets.QLineEdit()
        w.setPlaceholderText(placeholder or ("unset" if optional else "e.g. 3e-5"))
        super().__init__(w)
        self.optional = optional
        w.textChanged.connect(self._emit)

    def get(self):
        text = self.widget.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _set(self, value):
        if value is None:
            self.widget.setText("")
        else:
            self.widget.setText(f"{float(value):g}")


class BoolEditor(_Blockable):
    def __init__(self, label=""):
        w = QtWidgets.QCheckBox(label)
        super().__init__(w)
        w.stateChanged.connect(self._emit)

    def get(self):
        return bool(self.widget.isChecked())

    def _set(self, value):
        self.widget.setChecked(bool(value))


class TrainComponentEditor(_Blockable):
    """Train/freeze toggle with an optional LR override, retained while toggling."""

    def __init__(self, label=""):
        container = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        self.check = QtWidgets.QCheckBox(label)
        self.lr = QtWidgets.QLineEdit()
        self.lr.setPlaceholderText("Blank = main optimizer LR")
        row.addWidget(self.check)
        row.addWidget(QtWidgets.QLabel("AdaLN LR"))
        row.addWidget(self.lr, 1)
        super().__init__(container)
        self.check.toggled.connect(self._toggle)
        self.lr.textChanged.connect(self._emit)
        self.lr.setEnabled(False)

    def _toggle(self, on):
        self.lr.setEnabled(on)
        self._emit()

    def get(self):
        if not self.check.isChecked():
            return 0.0
        text = self.lr.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            # Preserve invalid input for config validation, rather than silently
            # substituting the global LR when an override was intended.
            return text

    def _set(self, value):
        self.lr.setText("" if value is None or value == 0 else str(value))
        self.check.setChecked(value != 0)
        self.lr.setEnabled(value != 0)


class ChoiceEditor(_Blockable):
    """`values` may differ from the labels shown -- `quant.use_quantized_matmul` is
    `"auto" | True | False`, three different Python types behind three words."""

    def __init__(self, options, values=None):
        w = NoScrollComboBox()
        w.addItems([str(o) for o in options])
        super().__init__(w)
        self.values = list(values) if values is not None else list(options)
        w.currentIndexChanged.connect(self._emit)

    def get(self):
        idx = self.widget.currentIndex()
        return self.values[idx] if 0 <= idx < len(self.values) else self.values[0]

    def _set(self, value):
        """An unrecognised value is KEPT, not silently coerced.

        This used to fall back to index 0, which is a data-destroying default: loading a config
        whose value this build does not know about would quietly rewrite it to the first option and
        then save that back over the file. It has bitten twice -- `dataset.source = "encode"` before
        the option existed, and `quant.skip_policy = "mageflow_int8_mm"` in a GUI still running from
        before that policy was added, where index 0 happens to be `"default"`, i.e. no skipping at
        all. Both times the config silently became a different, valid-looking run.

        Appending the value instead keeps the round-trip lossless and follows this GUI's standing
        rule: never reimplement the trainer's validation, show the trainer's error. A genuinely
        bogus value survives the form and `load_config` rejects it by name, which is a far better
        outcome than the GUI deciding it meant something else.
        """
        try:
            self.widget.setCurrentIndex(self.values.index(value))
        except ValueError:
            if value is None:
                self.widget.setCurrentIndex(0)
                return
            self.values.append(value)
            self.widget.addItem(f"{value}  (not in this build)")
            self.widget.setCurrentIndex(len(self.values) - 1)


class OptimizerEditor(ChoiceEditor):
    """Optimizer families grouped without changing TOML kind names."""

    def __init__(self):
        from ..training.optimizer_specs import OPTIMIZERS
        options, values, headers = [], [], []
        for family in ("PyTorch", "SDNQ", "Optimi"):
            headers.append(len(options))
            options.append(f"— {family} —")
            values.append(None)
            for kind, spec in OPTIMIZERS.items():
                if spec.family == family:
                    options.append(f"{spec.name} ({family})")
                    values.append(kind)
        super().__init__(options, values)
        for index in headers:
            self.widget.model().item(index).setEnabled(False)
        self.widget.setCurrentIndex(1)


class NumListEditor(_Blockable):
    """Comma-separated numbers. `dataset.resolutions` (ints, empty = single tier) and
    `optimizer.betas` (floats, fixed length)."""

    def __init__(self, placeholder="e.g. 768, 1024, 1280", cast=int, allow_empty=True,
                 as_tuple=False):
        w = QtWidgets.QLineEdit()
        w.setPlaceholderText(placeholder)
        super().__init__(w)
        self.cast = cast
        self.allow_empty = allow_empty
        # Tuple-typed fields (mask_ratio) must come back as tuples or the drift
        # check sees a change on every load.
        self.as_tuple = as_tuple
        w.textChanged.connect(self._emit)

    def get(self):
        text = self.widget.text().strip()
        if not text:
            return None
        out = []
        for part in text.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(self.cast(float(part)))
            except ValueError:
                return None
        if not out:
            return None
        return tuple(out) if self.as_tuple else out

    def _set(self, value):
        if not value:
            self.widget.setText("")
        else:
            self.widget.setText(", ".join(f"{v:g}" if isinstance(v, float) else str(v)
                                          for v in value))


def IntListEditor(placeholder="e.g. 768, 1024, 1280"):
    return NumListEditor(placeholder, cast=int)


def FloatListEditor(placeholder="0.9, 0.99"):
    return NumListEditor(placeholder, cast=float)


class StrListEditor(_Blockable):
    def __init__(self, placeholder="comma separated", as_set=False, height=None):
        w = QtWidgets.QPlainTextEdit()
        w.setPlaceholderText(placeholder)
        if height:
            w.setFixedHeight(height)
        super().__init__(w)
        self.as_set = as_set
        w.textChanged.connect(self._emit)

    def get(self):
        parts = [p.strip() for p in self.widget.toPlainText().replace("\n", ",").split(",")]
        parts = [p for p in parts if p]
        return sorted(set(parts)) if self.as_set else parts

    def _set(self, value):
        self.widget.setPlainText(", ".join(sorted(value)) if value else "")


class CheckSetEditor(_Blockable):
    """`adapter.components` -- a subset of a fixed vocabulary. Order is normalised to the
    vocabulary's, so two configs selecting the same components compare equal."""

    def __init__(self, options, tooltips=None):
        container = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        self.boxes = {}
        super().__init__(container)
        for opt in options:
            box = QtWidgets.QCheckBox(opt)
            if tooltips and opt in tooltips:
                box.setToolTip(tooltips[opt])
            box.stateChanged.connect(self._emit)
            row.addWidget(box)
            self.boxes[opt] = box
        row.addStretch(1)

    def get(self):
        return [k for k, b in self.boxes.items() if b.isChecked()]

    def _set(self, value):
        chosen = set(value or [])
        for k, b in self.boxes.items():
            b.setChecked(k in chosen)

    def set_enabled(self, on):
        for b in self.boxes.values():
            b.setEnabled(on)


class BucketMapEditor(_Blockable):
    """`train.batch_size`: a plain int, or one entry per resolution tier.

    Text, deliberately. A table widget would be prettier and would also make the common case --
    one number -- take four clicks. `4` stays `4`; `768=16, 1024=12` becomes the map, whose keys
    must be exactly the declared tiers.
    """

    def __init__(self):
        w = QtWidgets.QLineEdit()
        w.setPlaceholderText("12   or one per tier:   768=16, 1024=12, 1280=8")
        super().__init__(w)
        w.textChanged.connect(self._emit)

    def get(self):
        text = self.widget.text().strip()
        if not text:
            return 1
        if "=" not in text:
            try:
                return int(float(text))
            except ValueError:
                return 1
        out = {}
        for part in text.split(","):
            if "=" not in part:
                continue
            k, _, v = part.partition("=")
            try:
                out[int(float(k.strip()))] = int(float(v.strip()))
            except ValueError:
                continue
        return out or 1

    def _set(self, value):
        if isinstance(value, dict):
            items = sorted((int(k), int(v)) for k, v in value.items())
            self.widget.setText(", ".join(f"{k}={v}" for k, v in items))
        else:
            self.widget.setText(str(int(value or 1)))


class WeightsEditor(_Blockable):
    """`caption.mixed_weights` -- relative, not percentages, so no normalisation is imposed."""

    def __init__(self, keys, per_row=4):
        container = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        self.spins = {}
        super().__init__(container)
        for i, k in enumerate(keys):
            row, col = divmod(i, per_row)
            grid.addWidget(make_label(k, bold=False), row, col * 2)
            spin = NoScrollSpinBox()
            spin.setLocale(QtCore.QLocale.c())
            spin.setRange(0, 1000)
            spin.setMinimumWidth(78)
            spin.valueChanged.connect(self._emit)
            grid.addWidget(spin, row, col * 2 + 1)
            self.spins[k] = spin
        grid.setColumnStretch(per_row * 2, 1)

    def get(self):
        return {k: int(s.value()) for k, s in self.spins.items()}

    def _set(self, value):
        value = value or {}
        for k, s in self.spins.items():
            s.setValue(int(value.get(k, 0)))

    def set_enabled(self, on):
        for s in self.spins.values():
            s.setEnabled(on)


class CanvasListEditor(_Blockable):
    """`dataset.texture.canvases` -- a list of [width, height] pairs, edited as `WxH, WxH, ...`.

    Duplicates are preserved rather than collapsed: the reference's preset list contains
    1024x1024 twice on purpose, which is what gives the square a 1/3 chance of being drawn instead
    of 1/5. Sorting or de-duplicating here would silently reweight canvas selection.
    """

    def __init__(self, placeholder="1024x1024, 832x1216, ...", height=60):
        w = QtWidgets.QPlainTextEdit()
        w.setPlaceholderText(placeholder)
        w.setFixedHeight(height)
        super().__init__(w)
        w.textChanged.connect(self._emit)

    def get(self):
        out = []
        for part in self.widget.toPlainText().replace("\n", ",").split(","):
            part = part.strip().lower()
            if not part:
                continue
            if "x" not in part:
                continue
            a, _, b = part.partition("x")
            try:
                out.append((int(a), int(b)))
            except ValueError:
                continue
        return out

    def _set(self, value):
        self.widget.setPlainText(", ".join(f"{int(w)}x{int(h)}" for w, h in (value or [])))


class CurriculumEditor(_Blockable):
    """`[[curriculum]]` -- the one config key that is an array of tables, so it gets a table.

    Every other editor here maps one dotted key to one value. A curriculum is a list of phases,
    each with four fields, and the ordering between them is load-bearing (a phase runs until the
    next one's `at`). A comma-separated line would be unreadable at the nine phases the reference
    schedule actually uses, so this is a grid with add/remove.

    `at` is shown as a **percentage** because that is how the schedule is reasoned about ("texture
    from 15% in"), and stored as the [0,1) fraction the loader wants. The two are not the same
    number and conflating them would silently put every phase in the first 1% of the run.

    No validation lives here. `Phase.__post_init__` already rejects lo >= hi, out-of-range `at`,
    duplicate `at`, discrete 0-1000 timesteps, and a first phase that does not start at 0 -- and
    the GUI's rule is to show the trainer's error, never to reimplement it. So this widget can
    emit an invalid curriculum and the status bar will say exactly why.

    **The enable checkbox is widget state, not config state.** Unchecked, `get()` returns `[]` --
    which prunes the key out of the TOML entirely, i.e. plain full-resolution training, the
    default. The rows stay in the table so re-checking restores the schedule instead of making you
    retype it. Without this the only way back to an ordinary LoRA run from a texture config is
    Clear, which throws the schedule away; a checkbox that turns nine phases off and on again is
    the difference between a switch and a one-way door.
    """

    MODES = ("fullres", "texture")
    _HEADERS = ("Start %", "t min", "t max", "Mode", "LR x")

    def __init__(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.enable = QtWidgets.QCheckBox("Enable curriculum (off = plain full-resolution training)")
        self.enable.setToolTip(
            "Off writes no [[curriculum]] at all, which is the default and is bit-identical to the "
            "feature not existing -- every step trains full-resolution over the whole timestep "
            "range. The phases below are kept so you can switch back."
        )
        lay.addWidget(self.enable)

        self.table = QtWidgets.QTableWidget(0, len(self._HEADERS))
        self.table.setHorizontalHeaderLabels(self._HEADERS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        head = self.table.horizontalHeader()
        for i in range(len(self._HEADERS)):
            head.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(150)
        lay.addWidget(self.table)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        self.add_btn = make_btn("Add phase")
        self.del_btn = make_btn("Remove")
        self.preset_btn = make_btn("TrainTrain schedule")
        self.preset_btn.setToolTip(
            "The nine-phase schedule from the reference's run_1.sh: a full-resolution warm-up, "
            "then texture crops alternating between the full timestep range and t<=0.6."
        )
        self.clear_btn = make_btn("Clear")
        for b in (self.add_btn, self.del_btn, self.preset_btn, self.clear_btn):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        super().__init__(w)
        self.add_btn.clicked.connect(self._add_row)
        self.del_btn.clicked.connect(self._remove_row)
        self.preset_btn.clicked.connect(self._load_preset)
        self.clear_btn.clicked.connect(self._clear)
        self.enable.toggled.connect(self._on_toggle)
        self._on_toggle(False)

    # -- one row's five cell widgets ---------------------------------------------------------
    def _make_row(self, at, t_lo, t_hi, mode, lr_mul, index):
        pct = NoScrollDoubleSpinBox()
        pct.setRange(0.0, 99.9)
        pct.setDecimals(1)
        pct.setSuffix(" %")
        pct.setValue(at * 100.0)
        lo = NoScrollDoubleSpinBox()
        lo.setRange(0.0, 1.0)
        lo.setDecimals(3)
        lo.setSingleStep(0.05)
        lo.setValue(t_lo)
        hi = NoScrollDoubleSpinBox()
        hi.setRange(0.0, 1.0)
        hi.setDecimals(3)
        hi.setSingleStep(0.05)
        hi.setValue(t_hi)
        combo = NoScrollComboBox()
        combo.addItems(self.MODES)
        combo.setCurrentIndex(self.MODES.index(mode) if mode in self.MODES else 0)
        mul = NoScrollDoubleSpinBox()
        mul.setRange(0.001, 100.0)
        mul.setDecimals(3)
        mul.setSingleStep(0.1)
        mul.setValue(lr_mul)

        cells = (pct, lo, hi, combo, mul)
        for col, cell in enumerate(cells):
            self.table.setCellWidget(index, col, cell)
            sig = cell.currentIndexChanged if isinstance(cell, QtWidgets.QComboBox) \
                else cell.valueChanged
            sig.connect(self._emit)
        return cells

    def _insert_row(self):
        i = self.table.rowCount()
        self.table.insertRow(i)
        # A new phase inherits the previous one's settings and starts a little later, so adding a
        # row is a small edit rather than a form to fill in. The first row starts at 0, which is
        # the only value `Curriculum` accepts there.
        if i == 0:
            self._make_row(0.0, 0.0, 1.0, "fullres", 1.0, i)
        else:
            prev = self._read_row(i - 1)
            at = min(0.999, prev["at"] + 0.1)
            self._make_row(at, prev["t_range"][0], prev["t_range"][1],
                           prev["mode"], prev["lr_mul"], i)
        self._emit()

    def _remove_row(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            rows = [self.table.rowCount() - 1] if self.table.rowCount() else []
        for r in rows:
            self.table.removeRow(r)
        self._emit()

    def _clear(self):
        # Clear discards the phases; the checkbox is the reversible way off. Both reach the same
        # config -- no `[[curriculum]]` -- which is why the checkbox has to exist: otherwise the
        # only route back to plain training is the destructive one.
        self.table.setRowCount(0)
        self.enable.setChecked(False)
        self._emit()

    def _load_preset(self):
        self.set(TRAINTRAIN_CURRICULUM)
        self._emit()

    def _read_row(self, i):
        pct, lo, hi, combo, mul = (self.table.cellWidget(i, c) for c in range(5))
        row = {
            "at": round(pct.value() / 100.0, 5),
            "t_range": [round(lo.value(), 5), round(hi.value(), 5)],
            "mode": combo.currentText(),
        }
        # `prune_defaults` works on whole values, so it cannot reach inside an array of tables:
        # an explicit `lr_mul = 1.0` would be written into all nine phases of a schedule that
        # never uses it. Omitting the exact no-op keeps a GUI-written curriculum identical to a
        # hand-written one, which is what makes the round-trip check meaningful.
        if mul.value() != 1.0:
            row["lr_mul"] = round(mul.value(), 5)
        return row

    def _on_toggle(self, on):
        for w in (self.table, self.add_btn, self.del_btn, self.preset_btn, self.clear_btn):
            w.setEnabled(on)
        self._emit()

    def _add_row(self):
        # Adding a phase to a disabled curriculum is unambiguously a request to have one.
        if not self.enable.isChecked():
            self.enable.setChecked(True)
        self._insert_row()

    def get(self):
        if not self.enable.isChecked():
            return []
        return [self._read_row(i) for i in range(self.table.rowCount())]

    def _set(self, value):
        self.table.setRowCount(0)
        for phase in value or []:
            t = phase.get("t_range", (0.0, 1.0))
            i = self.table.rowCount()
            self.table.insertRow(i)
            self._make_row(float(phase.get("at", 0.0)), float(t[0]), float(t[1]),
                           phase.get("mode", "fullres"), float(phase.get("lr_mul", 1.0)), i)
        self.enable.setChecked(bool(value))
        self._on_toggle(bool(value))

    def set_enabled(self, on: bool):
        self.widget.setEnabled(on)


# The reference's `run_1.sh` schedule, transcribed. Its timesteps are 0-1000 integers; this model
# takes t in [0,1], so every bound is divided by 1000 -- see `curriculum.py` for why that
# conversion is enforced rather than guessed.
# `lr_mul` is omitted throughout: the reference does not vary it across these phases, and 1.0 is an
# exact no-op, so writing it would be nine lines of noise in the saved TOML.
TRAINTRAIN_CURRICULUM = [
    {"at": 0.00, "t_range": [0.0, 1.0], "mode": "fullres"},
    {"at": 0.10, "t_range": [0.0, 1.0], "mode": "texture"},
    {"at": 0.15, "t_range": [0.0, 0.6], "mode": "texture"},
    {"at": 0.40, "t_range": [0.0, 1.0], "mode": "texture"},
    {"at": 0.45, "t_range": [0.0, 0.6], "mode": "texture"},
    {"at": 0.60, "t_range": [0.0, 1.0], "mode": "texture"},
    {"at": 0.65, "t_range": [0.0, 0.6], "mode": "texture"},
    {"at": 0.80, "t_range": [0.0, 1.0], "mode": "texture"},
    {"at": 0.85, "t_range": [0.0, 0.6], "mode": "texture"},
]


class SubsetEditor(_Blockable):
    """`[[dataset.subsets]]` -- the second array-of-tables key, and the same shape as
    `CurriculumEditor`: a grid with add/remove rather than one dotted key per value.

    Three columns because three things vary per source directory. `num_repeats` is the weighting
    knob -- a 61-image regularization set against 413 training images is 13% of the epoch at x1 and
    23% at x2 -- and `texture` is the one that changes semantics rather than proportions: an
    unchecked box keeps that subset in fullres for the whole run, captions intact, whatever the
    curriculum is doing.

    Why that matters enough to be a column: texture crops replace the caption with
    `texture.trigger`. For detail crops of a captioned subject that is deliberate (Mage-Flow has no
    size micro-conditioning, so tagging a zoomed crop with the whole image's tags teaches the
    confusion). For a regularization subset it inverts the intent -- flat colour trained with an
    empty caption teaches the UNCONDITIONAL branch that colour, and CFG subtracts the
    unconditional, so the anchors would push generations away from colour.

    Empty table = no `[[dataset.subsets]]` written, i.e. the ordinary single-`dataset.path` config.
    The two are mutually exclusive in the loader, so `prune_defaults` drops `dataset.path` as soon
    as a row exists -- the GUI writes the file, so it resolves the conflict rather than handing
    back an error about a key the user cannot see.

    The fourth column is the subset's painted loss-mask folder (`mask_path`); empty leaves that
    folder unmasked. It is last so the first three keep their indexes.
    """

    _HEADERS = ("Folder", "Repeats", "Texture crops", "Masks")

    def __init__(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        hint = make_label(
            "Add rows to use these folders instead of the single Dataset path above. "
            "Remove all rows to restore the single folder. Each row has its own repeat count. "
            "Click a row number to select it, then Remove selected. Ctrl/Shift selects multiple rows."
        )
        hint.setWordWrap(True)
        lay.addWidget(hint)

        self.table = QtWidgets.QTableWidget(0, len(self._HEADERS))
        self.table.setHorizontalHeaderLabels(self._HEADERS)
        self.table.verticalHeader().setVisible(True)
        self.table.verticalHeader().setSectionsClickable(True)
        self.table.verticalHeader().setMinimumWidth(36)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        head = self.table.horizontalHeader()
        head.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        head.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for i in (1, 2):
            head.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeMode.Fixed)
        head.resizeSection(1, 110)
        head.resizeSection(2, max(130, self.table.fontMetrics().horizontalAdvance("Texture crops") + 28))
        self.table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.table.setMinimumHeight(110)
        lay.addWidget(self.table)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        self.add_btn = make_btn("Add folder")
        self.del_btn = make_btn("Remove selected")
        self.del_btn.setEnabled(False)
        self.clear_btn = make_btn("Clear")
        for b in (self.add_btn, self.del_btn, self.clear_btn):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        super().__init__(w)
        self.add_btn.clicked.connect(self._add_row)
        self.del_btn.clicked.connect(self._remove_row)
        self.clear_btn.clicked.connect(self._clear)
        self.table.itemSelectionChanged.connect(self._update_remove_button)

    @staticmethod
    def _folder_cell(path):
        # A plain QLineEdit + button rather than a nested PathEditor. PathEditor is a QObject
        # wrapper around its widget; putting only the widget in a table cell leaves the wrapper
        # unreferenced, Python collects it, and Qt then owns a half-destroyed object -- which
        # segfaults on the next repaint rather than raising. Everything here is parented to the
        # cell widget, so Qt owns the whole row and there is no Python lifetime to get wrong.
        holder_path = QtWidgets.QWidget()
        hp = QtWidgets.QHBoxLayout(holder_path)
        hp.setContentsMargins(0, 0, 0, 0)
        hp.setSpacing(4)
        line = QtWidgets.QLineEdit(str(path or ""))
        browse = make_btn("...")
        browse.setFixedWidth(max(36, browse.sizeHint().width()))
        line.setMinimumWidth(100)
        hp.addWidget(line, 1)
        hp.addWidget(browse)

        def _pick():
            start = usable_dialog_start(line.text())
            chosen = QtWidgets.QFileDialog.getExistingDirectory(holder_path, "Select folder", start)
            if chosen:
                line.setText(chosen)
        browse.clicked.connect(_pick)
        return holder_path, line, browse

    def _make_row(self, path, num_repeats, texture, index, mask_path=""):
        holder_path, line, browse = self._folder_cell(path)
        holder_mask, mask_line, _ = self._folder_cell(mask_path)
        mask_line.setPlaceholderText("unmasked")
        mask_line.setToolTip(
            "Optional folder of grayscale loss masks for this folder's images, matched by filename "
            "stem. White = learn, black = weight dataset.mask_outside_weight. Empty = unmasked."
        )

        reps = NoScrollSpinBox()
        reps.setLocale(QtCore.QLocale.c())
        reps.setRange(1, 1000)
        reps.setValue(int(num_repeats))
        reps.setMinimumWidth(100)
        self.table.horizontalHeader().resizeSection(1, max(110, reps.sizeHint().width() + 12))
        tex = QtWidgets.QCheckBox()
        tex.setChecked(bool(texture))
        tex.setToolTip(
            "On: this folder can be texture-cropped when the curriculum is in a texture phase, "
            "and those crops carry texture.trigger instead of the image's caption.\n"
            "Off: always full-resolution, captions intact. Use this for regularization sets -- an "
            "uncaptioned flat colour trains the unconditional branch, which CFG then subtracts."
        )
        holder = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(holder)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addStretch(1)
        hl.addWidget(tex)
        hl.addStretch(1)

        self.table.setCellWidget(index, 0, holder_path)
        self.table.setCellWidget(index, 1, reps)
        self.table.setCellWidget(index, 2, holder)
        self.table.setCellWidget(index, 3, holder_mask)
        self.table.setRowHeight(index, max(40, line.sizeHint().height() + 8,
                                           reps.sizeHint().height() + 8,
                                           browse.sizeHint().height() + 8))
        line.textChanged.connect(self._emit)
        mask_line.textChanged.connect(self._emit)
        reps.valueChanged.connect(self._emit)
        tex.toggled.connect(self._emit)
        return line, reps, tex, mask_line

    def _add_row(self):
        i = self.table.rowCount()
        self.table.insertRow(i)
        # Texture OFF for a newly added row, even though the TOML default is `texture = true`.
        # The two defaults answer different questions. A subset written by hand in TOML is usually
        # the training data, so `true` is right there. A subset added in the GUI is a SECOND
        # directory next to one that already exists -- a regularization or anchor set -- and for
        # those texture mode is actively wrong: it replaces the caption with `texture.trigger`, so
        # an uncaptioned flat colour trains the unconditional branch and CFG subtracts it.
        # Defaulting on would make the damaging choice the silent one.
        self._make_row("", 1, False, i)
        self._emit()

    def _update_remove_button(self):
        self.del_btn.setEnabled(bool(self.table.selectionModel().selectedRows()))

    def _remove_row(self):
        rows = sorted((i.row() for i in self.table.selectionModel().selectedRows()), reverse=True)
        for r in rows:
            self.table.removeRow(r)
        self.table.clearSelection()
        self._update_remove_button()
        self._emit()

    def _clear(self):
        self.table.setRowCount(0)
        self._emit()

    def _read_row(self, i):
        line = self.table.cellWidget(i, 0).findChild(QtWidgets.QLineEdit)
        reps = self.table.cellWidget(i, 1)
        tex = self.table.cellWidget(i, 2).findChild(QtWidgets.QCheckBox)
        row = {"path": line.text().strip()}
        # Same reasoning as CurriculumEditor's `lr_mul`: `prune_defaults` cannot reach inside an
        # array of tables, so writing exact defaults would put noise in every row.
        if reps.value() != 1:
            row["num_repeats"] = int(reps.value())
        if not tex.isChecked():
            row["texture"] = False
        masks = self.table.cellWidget(i, 3).findChild(QtWidgets.QLineEdit).text().strip()
        if masks:
            row["mask_path"] = masks
        return row

    def get(self):
        rows = [self._read_row(i) for i in range(self.table.rowCount())]
        # A blank path is a half-filled row, not a request to train from the working directory.
        return [r for r in rows if r["path"]]

    def _set(self, value):
        self.table.setRowCount(0)
        for sub in value or []:
            i = self.table.rowCount()
            self.table.insertRow(i)
            self._make_row(sub.get("path", ""), sub.get("num_repeats", 1),
                           sub.get("texture", True), i, sub.get("mask_path") or "")


class LineListEditor(_Blockable):
    """One entry per line, nothing split on commas.

    For `attribution_patterns`: these are regexes, and `{1,3}` or `(a|b), c` would be cut apart by
    the comma-splitting `StrListEditor`.
    """

    def __init__(self, placeholder="one per line", height=60):
        w = QtWidgets.QPlainTextEdit()
        w.setPlaceholderText(placeholder)
        w.setFixedHeight(height)
        super().__init__(w)
        w.textChanged.connect(self._emit)

    def get(self):
        return [line.strip() for line in self.widget.toPlainText().splitlines() if line.strip()]

    def _set(self, value):
        if isinstance(value, str):
            value = [value]
        self.widget.setPlainText("\n".join(value or []))


def _toml_inline(table: dict) -> str:
    """`{ prompt = "...", seed = 7 }`. Strings go through json.dumps, whose escapes are a subset
    of TOML basic-string escapes, so the line parses back to the same value."""
    import json

    def value(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        return json.dumps(str(v), ensure_ascii=False)

    return "{ " + ", ".join(f"{k} = {value(v)}" for k, v in table.items()) + " }"


class PromptListEditor(_Blockable):
    """`sampling.prompts`: one prompt per line.

    A line may instead be an inline TOML table that overrides settings for that prompt:
    `{ prompt = "a castle at dusk", width = 1216, height = 832, seed = 7 }`. Once one line is a
    table every prompt is written as a table, because TOML cannot hold a mixed list of strings and
    tables as an array of tables.

    An unparseable table line comes back with an `INVALID_TOML` key, so the trainer's loader
    rejects it by name instead of the GUI quietly sampling a prompt that starts with `{`.
    """

    def __init__(self, placeholder="one prompt per line", height=110):
        w = QtWidgets.QPlainTextEdit()
        w.setPlaceholderText(placeholder)
        w.setFixedHeight(height)
        super().__init__(w)
        w.textChanged.connect(self._emit)

    def get(self):
        import tomllib

        out = []
        for line in self.widget.toPlainText().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    out.append(tomllib.loads(f"x = {line}")["x"])
                except tomllib.TOMLDecodeError as exc:
                    out.append({"prompt": line, "INVALID_TOML": str(exc)})
            else:
                out.append(line)
        if any(isinstance(p, dict) for p in out):
            out = [p if isinstance(p, dict) else {"prompt": p} for p in out]
        return out

    def _set(self, value):
        if isinstance(value, str):
            value = [value]
        lines = []
        for p in value or []:
            if isinstance(p, dict) and set(p) != {"prompt"}:
                lines.append(_toml_inline(p))
            else:
                lines.append(p["prompt"] if isinstance(p, dict) else str(p))
        self.widget.setPlainText("\n".join(lines))


class StageListEditor(_Blockable):
    """`schedule.stages` -- StageLR's segments as a grid.

    One LR column holds each stage type's main value: the target `end_lr` of linear/cosine, the
    `lr` of constant, the starting `max_val` of rex. `Min LR` is rex's `min_val`, unused by the
    other types. Percent is shown as a percentage and stored as the fraction StageLR takes; the
    running total is shown because the loader rejects a list that does not sum to 100%.

    Rows survive `schedule.kind` being something else (the grid greys out and `get()` returns []),
    so switching to cosine and back does not throw a hand-built schedule away.
    """

    TYPES = ("linear", "cosine", "constant", "rex")
    _HEADERS = ("Type", "Percent", "LR", "Min LR (rex)")
    _MAIN_KEY = {"linear": "end_lr", "cosine": "end_lr", "constant": "lr", "rex": "max_val"}

    def __init__(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.table = QtWidgets.QTableWidget(0, len(self._HEADERS))
        self.table.setHorizontalHeaderLabels(self._HEADERS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        head = self.table.horizontalHeader()
        for i in range(len(self._HEADERS)):
            head.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(130)
        lay.addWidget(self.table)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        self.add_btn = make_btn("Add stage")
        self.del_btn = make_btn("Remove")
        self.preset_btn = make_btn("Warmup / hold / REX")
        self.preset_btn.setToolTip(
            "A common three-stage shape: linear ramp to 7e-6 over 10%, hold for 50%, REX decay to "
            "0 over the last 40%. Edit the LRs for your run.")
        self.total = make_label("")
        for b in (self.add_btn, self.del_btn, self.preset_btn):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(self.total)
        lay.addLayout(row)
        super().__init__(w)
        self._active = True
        self.add_btn.clicked.connect(self._add_row)
        self.del_btn.clicked.connect(self._remove_row)
        self.preset_btn.clicked.connect(self._load_preset)

    def _make_row(self, kind, percent, lr, min_lr, index):
        combo = NoScrollComboBox()
        combo.addItems(self.TYPES)
        combo.setCurrentIndex(self.TYPES.index(kind) if kind in self.TYPES else 0)
        pct = NoScrollDoubleSpinBox()
        pct.setLocale(QtCore.QLocale.c())
        pct.setRange(0.0, 100.0)
        pct.setDecimals(3)
        pct.setSuffix(" %")
        pct.setValue(float(percent) * 100.0)
        main = QtWidgets.QLineEdit("" if lr is None else f"{lr:g}" if isinstance(lr, (int, float))
                                   else str(lr))
        main.setPlaceholderText("e.g. 7e-6")
        low = QtWidgets.QLineEdit("" if min_lr is None else f"{min_lr:g}"
                                  if isinstance(min_lr, (int, float)) else str(min_lr))
        low.setPlaceholderText("0")
        for col, cell in enumerate((combo, pct, main, low)):
            self.table.setCellWidget(index, col, cell)
        combo.currentIndexChanged.connect(self._changed)
        pct.valueChanged.connect(self._changed)
        main.textChanged.connect(self._changed)
        low.textChanged.connect(self._changed)
        self._sync_row(index)

    def _sync_row(self, index):
        kind = self.table.cellWidget(index, 0).currentText()
        self.table.cellWidget(index, 3).setEnabled(kind == "rex" and self._active)

    def _changed(self, *_):
        for i in range(self.table.rowCount()):
            self._sync_row(i)
        self._update_total()
        self._emit()

    def _update_total(self):
        total = sum(self.table.cellWidget(i, 1).value() for i in range(self.table.rowCount()))
        ok = abs(total - 100.0) < 1e-4 or not self.table.rowCount()
        self.total.setText(f"total {total:.3f}%" + ("" if ok else "  (must be 100%)"))

    def _add_row(self):
        i = self.table.rowCount()
        self.table.insertRow(i)
        prev = self._read_row(i - 1) if i else None
        lr = prev.get(self._MAIN_KEY[prev["type"]]) if prev else None
        self._make_row("constant" if prev else "linear", 0.0, lr, None, i)
        self._changed()

    def _remove_row(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            rows = [self.table.rowCount() - 1] if self.table.rowCount() else []
        for r in rows:
            self.table.removeRow(r)
        self._changed()

    def _load_preset(self):
        self.set([
            {"type": "linear", "end_lr": 7e-6, "percent": 0.1},
            {"type": "constant", "lr": 7e-6, "percent": 0.5},
            {"type": "rex", "max_val": 7e-6, "min_val": 0.0, "percent": 0.4},
        ])
        self._emit()

    @staticmethod
    def _number(text):
        text = text.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            # Kept as text so the loader names the bad value instead of the GUI dropping it.
            return text

    def _read_row(self, i):
        combo, pct, main, low = (self.table.cellWidget(i, c) for c in range(4))
        kind = combo.currentText()
        row = {"type": kind, "percent": round(pct.value() / 100.0, 8)}
        value = self._number(main.text())
        if value is not None:
            row[self._MAIN_KEY[kind]] = value
        if kind == "rex":
            low_value = self._number(low.text())
            row["min_val"] = 0.0 if low_value is None else low_value
        return row

    def get(self):
        if not self._active:
            return []
        return [self._read_row(i) for i in range(self.table.rowCount())]

    def _set(self, value):
        self.table.setRowCount(0)
        for stage in value or []:
            i = self.table.rowCount()
            self.table.insertRow(i)
            kind = stage.get("type", "constant")
            self._make_row(kind, float(stage.get("percent", 0.0)),
                           stage.get(self._MAIN_KEY.get(kind, "lr")), stage.get("min_val"), i)
        self._update_total()

    def set_enabled(self, on: bool):
        self._active = bool(on)
        self.widget.setEnabled(on)
        for i in range(self.table.rowCount()):
            self._sync_row(i)
