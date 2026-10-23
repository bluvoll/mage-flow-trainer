"""Dataset source exclusivity and discoverable experimental controls."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets

from trainer.gui import bridge
from trainer.gui.app import TrainingGUI
from trainer.gui.widgets import STYLESHEET


@pytest.fixture
def gui(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(STYLESHEET)
    monkeypatch.setattr(TrainingGUI, "_load_presets", lambda self: None)
    window = TrainingGUI()
    window._apply(bridge.defaults() | {"dataset.path": "/images"})
    yield window
    window.close()
    window.deleteLater()
    app.processEvents()


def test_subset_rows_disable_single_source_and_restore_it(gui):
    subsets = gui.editors["dataset.subsets"]
    subsets._add_row()
    assert not gui.editors["dataset.path"].widget.isEnabled()
    assert not gui.editors["dataset.num_repeats"].widget.isEnabled()
    assert gui.collect()["dataset.path"] is None
    assert not bridge.validate(gui.collect())[0]
    line = subsets.table.cellWidget(0, 0).findChild(QtWidgets.QLineEdit)
    line.setText("/other")
    assert gui.collect()["dataset.subsets"] == [{"path": "/other", "texture": False}]
    assert "dataset.path" not in bridge.prune_defaults(gui.collect())
    subsets._clear()
    assert gui.editors["dataset.path"].widget.isEnabled()
    assert gui.editors["dataset.num_repeats"].widget.isEnabled()
    assert gui.collect()["dataset.path"] == "/images"


def test_convrot_controls_and_roundtrip(gui, tmp_path):
    import toml
    from trainer.training.config import load_config

    gui._apply(bridge.defaults() | {
        "dataset.path": "/images", "train.model_family": "moss_image",
        "adapter.kind": "lycoris_lora", "adapter.lycoris_algo": "locon",
        "adapter.dtype": "bfloat16",
        "quant.mode": "frozen", "quant.backend": "convrot",
    })
    flat = gui.collect()
    assert flat["quant.backend"] == "convrot"
    for key in ("weights_dtype", "use_quantized_matmul", "group_size", "dynamic_loss_threshold"):
        assert not gui.editors[f"quant.{key}"].widget.isEnabled()
    path = tmp_path / "convrot.toml"
    path.write_text(toml.dumps(bridge.nest(bridge.prune_defaults(flat))))
    assert load_config(path).quant.backend == "convrot"
    gui.editors["quant.quantize_text_encoder"].set(True)
    gui._apply(gui.collect())
    assert gui.editors["quant.text_encoder_weights_dtype"].widget.isEnabled()


def test_moss_sampler_locks_shared_controls_and_roundtrips(gui, tmp_path):
    from trainer.training.config import load_config
    gui._apply(bridge.defaults() | {"dataset.path": "/images", "flow.sigmoid_scale": 3,
                                   "flow.shift": None, "flow.flux_shift": True})
    gui.editors["flow.timestep_sample_method"].set("moss")
    gui._on_edit()
    flat = gui.collect()
    assert flat["flow.shift"] == flat["flow.sigmoid_scale"] == 1
    assert not flat["flow.flux_shift"]
    for name in ("shift", "sigmoid_scale", "flux_shift"):
        assert not gui.editors[f"flow.{name}"].widget.isEnabled()
    assert gui.editors["flow.moss_std"].widget.isEnabled()
    path = tmp_path / "moss.toml"
    bridge.write_toml(path, flat)
    cfg = load_config(path)
    assert cfg.flow.timestep_sample_method == "moss"
    assert cfg.flow.shift == 1 and cfg.flow.moss_std == .95
    gui._apply(bridge.defaults() | bridge.flatten(bridge.read_toml(path)))
    assert gui.collect() == flat
    gui.editors["flow.timestep_sample_method"].set("logit_normal")
    gui._on_edit()
    assert gui.editors["flow.shift"].widget.isEnabled()
    assert gui.editors["flow.sigmoid_scale"].widget.isEnabled()
    assert not gui.editors["flow.moss_std"].widget.isEnabled()


def test_loading_moss_in_gui_locks_old_shared_values(gui):
    gui._apply(bridge.defaults() | {"dataset.path": "/images", "flow.timestep_sample_method": "moss",
                                   "flow.sigmoid_scale": 3, "flow.shift": 3})
    assert gui.collect()["flow.shift"] == gui.collect()["flow.sigmoid_scale"] == 1


def test_live_timestep_plot_reacts_and_renders(gui):
    import torch
    gui._apply(bridge.defaults() | {"dataset.path": "/images", "flow.shift": 1})
    original = gui.timestep_plot.canvas.result
    rng_state = torch.random.get_rng_state().clone()
    gui.editors["flow.sigmoid_scale"].set(3)
    gui._on_edit()
    wider = gui.timestep_plot.canvas.result
    assert wider.high_noise > original.high_noise
    assert wider.median == original.median == .5
    gui.editors["flow.shift"].set(3)
    gui._on_edit()
    assert gui.timestep_plot.canvas.result.median == .75
    assert torch.equal(rng_state, torch.random.get_rng_state())
    gui.timestep_plot.resize(420, 360)
    assert not gui.timestep_plot.grab().isNull()
    gui.editors["flow.shift"].set(0)
    gui._on_edit()
    assert gui.timestep_plot.canvas.result is None
    assert gui.timestep_plot.canvas.error


def test_experimental_visibility_preserves_active_settings(gui):
    assert not gui.tab_bar.isTabVisible(gui.experimental_tab)
    gui.experimental_chk.setChecked(True)
    assert gui.tab_bar.isTabVisible(gui.experimental_tab)
    assert not gui.collect()["self_flow.enabled"]
    gui.experimental_chk.setChecked(False)
    for key in ["self_flow.enabled", "flow.dual_timestep", "rti.enabled",
                "perceptual.enabled", "adversarial.enabled"]:
        gui._apply(bridge.defaults() | {"dataset.path": "/images", key: True})
        assert gui.tab_bar.isTabVisible(gui.experimental_tab)
        assert not gui.experimental_chk.isEnabled()
        assert gui.collect()[key]
    gui._apply(bridge.defaults() | {"dataset.path": "/images"})
    assert gui.experimental_chk.isEnabled()
    gui.experimental_chk.setChecked(False)
    assert not gui.tab_bar.isTabVisible(gui.experimental_tab)


def test_subset_controls_have_room_at_small_window(gui):
    gui.resize(1100, 820)
    gui.editors["dataset.subsets"]._add_row()
    gui.show()
    QtWidgets.QApplication.processEvents()
    table = gui.editors["dataset.subsets"].table
    spin = table.cellWidget(0, 1)
    assert table.columnWidth(1) >= spin.sizeHint().width()
    assert table.rowHeight(0) >= spin.sizeHint().height() + 8
    spin.setValue(999)
    assert gui.collect()["dataset.subsets"] == []
    assert spin.value() == 999


def test_remove_subset_by_row_number(gui):
    from PySide6 import QtCore, QtTest

    subsets = gui.editors["dataset.subsets"]
    subsets.set([{"path": "/first"}, {"path": "/middle"}, {"path": "/last"}])
    gui.tab_bar.setCurrentIndex(next(i for i in range(gui.tab_bar.count())
                                    if gui.tab_bar.tabText(i) == "Dataset"))
    gui.show()
    QtWidgets.QApplication.processEvents()
    assert not subsets.del_btn.isEnabled()
    subsets._remove_row()
    assert len(subsets.get()) == 3  # No implicit removal of the last row.
    header = subsets.table.verticalHeader()
    point = QtCore.QPoint(header.width() // 2,
                         header.sectionViewportPosition(1) + header.sectionSize(1) // 2)
    QtTest.QTest.mouseClick(header.viewport(), QtCore.Qt.MouseButton.LeftButton,
                           pos=point)
    assert subsets.del_btn.isEnabled()
    subsets.del_btn.click()
    assert [row["path"] for row in subsets.get()] == ["/first", "/last"]
    assert not subsets.del_btn.isEnabled()
    subsets.table.selectAll()
    assert subsets.del_btn.isEnabled()
    subsets.del_btn.click()
    assert subsets.get() == []
    assert gui.editors["dataset.path"].widget.isEnabled()


def test_offload_cache_policy_roundtrip_and_gates(gui, tmp_path):
    import toml
    from trainer.training.config import load_config
    gui._apply(bridge.defaults() | {
        'dataset.path': '/images', 'dataset.cache_validation': 'strict',
        'train.activation_offload': True, 'train.activation_offload_budget_mb': 2048,
        'train.activation_offload_pin_memory': True, 'train.activation_offload_prefetch': True,
        'train.nonfinite_patience': 5, 'train.compile': None,
    })
    flat = gui.collect()
    assert gui.editors['train.activation_offload_prefetch'].widget.isEnabled()
    path = tmp_path / 'offload.toml'
    path.write_text(toml.dumps(bridge.nest(bridge.prune_defaults(flat))))
    cfg = load_config(path)
    assert cfg.train.activation_offload
    assert cfg.train.activation_offload_budget_mb == 2048
    assert cfg.train.activation_offload_prefetch
    assert cfg.train.nonfinite_patience == 5
    assert cfg.dataset.cache_validation == 'strict'
    jobs = gui._cache_jobs(flat)
    argv = jobs[0].launch.argv
    # The fork caches through `cache-config`, which reads the policy from the config it is given.
    cache_cfg = load_config(argv[argv.index('cache-config') + 1])
    assert cache_cfg.dataset.cache_validation == 'strict'


@pytest.mark.parametrize('layout', ['auto', 'fused', 'separated'])
def test_moss_qkv_export_gui_roundtrip(gui,tmp_path,layout):
    from trainer.training.config import load_config
    gui._apply(bridge.defaults() | {
        'dataset.path':'/images','train.model_family':'moss_image',
        'adapter.kind':'lycoris_lora','adapter.lycoris_algo':'locon',
        'train.moss_qkv_format':layout,
    })
    assert gui.editors['train.moss_qkv_format'].widget.isEnabled()
    path=tmp_path/'fused.toml'
    bridge.write_toml(path,gui.collect())
    assert load_config(path).train.moss_qkv_format==layout
    gui._apply(gui.collect() | {'adapter.lycoris_algo':'lokr','train.moss_qkv_format':'separated'})
    assert not gui.editors['train.moss_qkv_format'].widget.isEnabled()
