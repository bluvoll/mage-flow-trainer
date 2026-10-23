"""Exercise the real GUI job chain without starting GPU training."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from pathlib import Path
from PySide6 import QtCore, QtWidgets

from trainer.gui import app as gui_module, bridge
from trainer.gui.app import TrainingGUI


class FakeRunner(QtCore.QObject):
    logSignal = QtCore.Signal(str)
    errorSignal = QtCore.Signal(str)
    progressSignal = QtCore.Signal(str, bool)
    metricsSignal = QtCore.Signal(str)
    finishedSignal = QtCore.Signal(int)
    instances = []

    def __init__(self, launch, cwd, env):
        super().__init__()
        self.launch, self.cwd, self.env = launch, cwd, env
        self.active = False
        self.instances.append(self)

    def isRunning(self):
        return self.active

    def start(self):
        self.active = True

    def complete(self, code=0):
        self.active = False
        self.finishedSignal.emit(code)

    def stop(self):
        self.complete(-15)


@pytest.fixture
def queue_gui(monkeypatch, tmp_path):
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(TrainingGUI, "_load_presets", lambda self: None)
    monkeypatch.setattr(gui_module, "ProcessRunner", FakeRunner)
    monkeypatch.setattr(gui_module, "prevent_sleep", lambda *args: None)
    FakeRunner.instances = []
    gui = TrainingGUI()
    gui._apply(bridge.defaults() | {"dataset.path": "/editor-dataset"})
    paths = []
    for i in range(2):
        path = tmp_path / f"run {i}.toml"
        path.write_text(f'[dataset]\npath="/dataset{i}"\n[train]\nrun_name="run{i}"\n')
        paths.append(path)
    gui.run_queue.add_paths(paths)
    yield gui, paths, qapp
    gui._stop()
    qapp.processEvents()
    gui.close()
    gui.deleteLater()
    qapp.processEvents()


def test_queue_add_remove_and_reorder(queue_gui):
    gui, paths, _ = queue_gui
    queue = gui.run_queue
    queue.add_paths(paths)
    assert queue.paths() == paths
    queue.list.item(1).setSelected(True)
    queue.move(-1)
    assert queue.paths() == paths[::-1]
    queue.remove_selected()
    assert queue.paths() == paths[:1]


def test_configs_cache_then_train_in_order_without_rewriting(queue_gui):
    gui, paths, qapp = queue_gui
    originals = [p.read_bytes() for p in paths]
    gui.run_queue.start_btn.click()
    assert gui._batch_queue_active
    assert not gui.run_queue.start_btn.isEnabled()
    expected = [(False, '/dataset0'), (True, str(paths[0])), (False, '/dataset1'), (True, str(paths[1]))]
    for i, (training, argument) in enumerate(expected):
        runner = FakeRunner.instances[i]
        if training:
            assert argument in runner.launch.argv
        else:
            # The fork caches through one `cache-config` call on a config naming the dataset.
            argv = runner.launch.argv
            assert argv[argv.index("cache-config") + 1].endswith(".toml")
            assert argument in Path(argv[argv.index("cache-config") + 1]).read_text(encoding="utf-8")
        assert gui._running_training == training
        if training:
            assert gui.metrics.elapsed_timer.isActive()
        runner.complete()
        qapp.processEvents()
    assert not gui._batch_queue_active and not gui._queue and gui.runner is None
    assert gui.run_queue.status.text() == 'Queue completed.'
    assert not gui.metrics.elapsed_timer.isActive()
    assert len(FakeRunner.instances) == 4
    assert [p.read_bytes() for p in paths] == originals
    assert gui.collect()['dataset.path'] == '/editor-dataset'


@pytest.mark.parametrize('stop', [False, True])
def test_failed_cache_or_stop_cancels_later_training(queue_gui, stop):
    gui, _, qapp = queue_gui
    gui._start_queue()
    if stop:
        gui._stop()
    else:
        gui.runner.complete(1)
    qapp.processEvents()
    assert len(FakeRunner.instances) == 1
    assert not gui._queue and not gui._batch_queue_active
    assert 'Skipped' in gui.run_queue.list.item(1).text()
    assert gui.run_queue.start_btn.isEnabled()


def test_all_configs_validated_before_launch(queue_gui):
    gui, paths, _ = queue_gui
    paths[1].write_text('[train]\nwrong_key=3\n')
    gui._start_queue()
    assert not FakeRunner.instances
    assert 'validation failed' in gui.run_queue.status.text()


def test_duplicate_outputs_rejected(queue_gui):
    gui, paths, _ = queue_gui
    paths[1].write_bytes(paths[0].read_bytes())
    gui._start_queue()
    assert not FakeRunner.instances
    assert 'Duplicate output' in gui.run_queue.status.text()


def test_config_changed_between_cache_and_training_stops_queue(queue_gui):
    gui, paths, qapp = queue_gui
    gui._start_queue()
    paths[0].write_text(paths[0].read_text() + '\n# edited\n')
    gui.runner.complete()
    qapp.processEvents()
    assert len(FakeRunner.instances) == 1
    assert 'changed after validation' in gui.run_queue.status.text()
    assert not gui._batch_queue_active


def test_stop_between_jobs_cancels_scheduled_start(queue_gui):
    gui, _, qapp = queue_gui
    gui._start_queue()
    gui.runner.complete()
    gui._stop()  # next job is scheduled but has not started yet
    qapp.processEvents()
    assert len(FakeRunner.instances) == 1
    assert not gui._batch_queue_active


def test_stop_after_completed_run_preserves_its_status(queue_gui):
    gui, _, qapp = queue_gui
    gui._start_queue()
    gui.runner.complete()  # cache
    qapp.processEvents()
    gui.runner.complete()  # training
    gui._stop()  # before second config's cache callback
    qapp.processEvents()
    assert len(FakeRunner.instances) == 2
    assert gui.run_queue.list.item(0).text().startswith('Completed')
    assert gui.run_queue.list.item(1).text().startswith('Skipped')


def test_queue_tab_has_one_unambiguous_start_action(queue_gui):
    gui, _, _ = queue_gui
    gui.tab_bar.setCurrentIndex(gui.queue_tab)
    assert gui.start_btn.isHidden()
    assert gui.run_queue.start_btn.isEnabled()
