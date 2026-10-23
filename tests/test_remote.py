"""Remote training: server, client and the GUI's runner, against a real local server."""

import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image

import trainer.remote.server as server_mod
from trainer.gui.launch import Launch
from trainer.remote.client import RemoteClient, RemoteError, parse_link
from trainer.remote.server import RemoteServer

# The marker is assembled at run time: the job log echoes the command line, so a literal marker
# would be in the log before the process even exists.
SLEEPER = ("import sys, time\n"
           "print('fake run ' + 'is up', flush=True)\n"
           "for i in range(600):\n"
           "    time.sleep(0.1)\n")


def write_config(root: Path, **extra_train) -> str:
    data = root / "data"
    data.mkdir(exist_ok=True)
    for i in range(2):
        Image.new("RGB", (128, 128), (i * 50, 80, 120)).save(data / f"im{i}.png")
        (data / f"im{i}.txt").write_text("Drawn by emily, 1girl", encoding="utf-8")
    train = "\n".join(f'{k} = "{v}"' for k, v in extra_train.items())
    return (f'[train]\nmodel_path = "{(root / "model").as_posix()}"\n'
            f'output_dir = "{(root / "out").as_posix()}"\nrun_name = "remote-test"\n{train}\n'
            f'[dataset]\npath = "{data.as_posix()}"\nresolution = 128\n')


class RemoteServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)
        self.jobs = patch.object(server_mod, "JOBS_DIR", self.root / "jobs")
        self.jobs.start()

    def tearDown(self):
        for s in getattr(self, "servers", []):
            s.jobs.stop()
            s.stop()
        self.jobs.stop()
        self.tmp.cleanup()

    def start(self, mode="jobs") -> tuple[RemoteServer, RemoteClient]:
        server = RemoteServer(host="127.0.0.1", port=0, mode=mode,
                              receive_dir=self.root / "received").start(tunnel="none")
        self.servers = getattr(self, "servers", []) + [server]
        return server, RemoteClient(server.connect_link())

    def wait_job(self, client, timeout=120):
        deadline = time.time() + timeout
        offset, text = 0, ""
        while time.time() < deadline:
            chunk = client.log(offset)
            text += chunk["text"]
            offset = chunk["offset"]
            if chunk["job"] and not chunk["job"]["running"] and not chunk["text"]:
                return chunk["job"], text
            time.sleep(0.3)
        self.fail("job did not finish")

    def test_link_parsing(self):
        self.assertEqual(parse_link("https://a-b.trycloudflare.com/#token=XYZ"),
                         ("https://a-b.trycloudflare.com", "XYZ"))
        self.assertEqual(parse_link("10.0.0.5:8765", token="T"), ("http://10.0.0.5:8765", "T"))
        with self.assertRaisesRegex(RemoteError, "no #token"):
            parse_link("https://a.trycloudflare.com")

    def test_token_is_required_and_landing_page_reveals_nothing(self):
        server, client = self.start()
        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/") as resp:
            page = resp.read().decode()
        self.assertIn("connect link", page)
        self.assertNotIn(str(server_mod.PROJECT_ROOT), page)
        wrong = RemoteClient(f"http://127.0.0.1:{server.port}#token=nope")
        with self.assertRaisesRegex(RemoteError, "401"):
            wrong.status()
        status = client.status()
        self.assertEqual(status["server"], "mageflow-remote")
        self.assertEqual(status["mode"], "jobs")

    def test_validate_runs_the_real_loader_on_the_server(self):
        _, client = self.start()
        ok = client.validate(write_config(self.root))
        self.assertTrue(ok["ok"], ok)
        self.assertTrue(any("model not found" in w for w in ok["warnings"]))
        bad = client.validate(write_config(self.root) + "\n[schedule]\nkind = \"stage\"\n")
        self.assertFalse(bad["ok"])
        self.assertIn("stages", bad["error"])

    def test_cache_dry_run_job_streams_its_log(self):
        _, client = self.start()
        started = client.run(write_config(self.root), "remote-test", ["cache_dry"])
        self.assertTrue(started["job"]["running"])
        with self.assertRaisesRegex(RemoteError, "still running"):
            client.run(write_config(self.root), "again", ["cache_dry"])
        job, text = self.wait_job(client)
        self.assertEqual(job["exit_code"], 0, text)
        self.assertIn("--dry-run: nothing written", text)
        self.assertIn("cache-config", text)

    def test_stop_kills_the_job_and_signals_reach_the_run_folder(self):
        _, client = self.start()

        def fake(runner, job, step):
            return Launch([sys.executable, "-u", "-c", SLEEPER], f"fake {step}")

        with patch.object(server_mod.JobRunner, "_launch", fake):
            client.run(write_config(self.root), "remote-test", ["train"])
            deadline = time.time() + 30
            while "fake run is up" not in client.log(0)["text"] and time.time() < deadline:
                time.sleep(0.2)
            path = Path(client.signal("save_quit")["path"])
            self.assertEqual(path, self.root / "out" / "remote-test" / "save_quit")
            self.assertTrue(path.exists())
            client.stop()
            job, _ = self.wait_job(client, timeout=60)
        self.assertNotEqual(job["exit_code"], 0)
        with self.assertRaisesRegex(RemoteError, "no training"):
            client.signal("save")

    def test_stop_before_the_process_spawns_still_stops_it(self):
        _, client = self.start()

        def fake(runner, job, step):
            # Stop lands between "step chosen" and "process spawned" -- the window in which a
            # quickly stopped job used to start anyway and run to completion.
            runner.stop()
            return Launch([sys.executable, "-u", "-c", SLEEPER], f"fake {step}")

        with patch.object(server_mod.JobRunner, "_launch", fake):
            client.run(write_config(self.root), "remote-test", ["train"])
            job, text = self.wait_job(client, timeout=30)
        self.assertNotEqual(job["exit_code"], 0)
        self.assertNotIn("fake run is up", text)

    def test_receive_mode_hands_the_config_to_the_waiting_cell(self):
        server, client = self.start(mode="receive")
        result = {}
        waiter = threading.Thread(target=lambda: result.update(path=server.wait_for_config(30)))
        waiter.start()
        with self.assertRaisesRegex(RemoteError, "409"):
            client.run(write_config(self.root), "x", ["train"])
        with self.assertRaisesRegex(RemoteError, "400"):
            client.send_config("[train]\nnot_a_key = 1\n", "bad")
        sent = client.send_config(write_config(self.root), "my run")
        waiter.join(10)
        self.assertEqual(Path(sent["path"]), result["path"])
        self.assertEqual(result["path"].name, "my-run.toml")
        self.assertIn("run_name = \"remote-test\"", result["path"].read_text())

    def test_gpu_list_is_validated_like_locally(self):
        _, client = self.start()
        with self.assertRaisesRegex(RemoteError, "device indices"):
            client.run(write_config(self.root), "x", ["cache_dry"], gpus="all")


class RemoteRunnerTests(unittest.TestCase):
    """The GUI side: a remote job seen through ProcessRunner's signals."""

    @classmethod
    def setUpClass(cls):
        from PySide6 import QtWidgets
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)
        self.jobs = patch.object(server_mod, "JOBS_DIR", self.root / "jobs")
        self.jobs.start()
        self.server = RemoteServer(host="127.0.0.1", port=0).start()
        self.client = RemoteClient(self.server.connect_link())

    def tearDown(self):
        self.server.jobs.stop()
        self.server.stop()
        self.jobs.stop()
        self.tmp.cleanup()

    def follow(self, runner, timeout=120):
        lines, metrics, finished = [], [], []
        runner.progressSignal.connect(lambda line, _p: lines.append(line))
        runner.logSignal.connect(lines.append)
        runner.errorSignal.connect(lines.append)
        runner.metricsSignal.connect(metrics.append)
        runner.finishedSignal.connect(finished.append)
        runner.start()
        deadline = time.time() + timeout
        while not finished and time.time() < deadline:
            self.app.processEvents()
            time.sleep(0.05)
        runner.wait(5000)
        self.app.processEvents()
        return lines, metrics, finished

    def test_runner_replays_the_remote_log_and_exit_code(self):
        from trainer.gui.remote_runner import RemoteRunner

        toml = write_config(self.root)
        runner = RemoteRunner(self.client, "remote cache_dry",
                              start=lambda: self.client.run(toml, "t", ["cache_dry"]))
        lines, metrics, finished = self.follow(runner)
        self.assertEqual(finished, [0], lines[-10:])
        self.assertTrue(any("nothing written" in l for l in lines))
        self.assertTrue(any("started on" in l for l in lines))
        self.assertIn("--dry-run: nothing written", metrics)

    def test_runner_reports_a_rejected_start(self):
        from trainer.gui.remote_runner import RemoteRunner

        def bad():
            raise RemoteError("400: the remote machine rejects this config")

        lines, _, finished = self.follow(RemoteRunner(self.client, "remote train", start=bad))
        self.assertEqual(finished, [-1])
        self.assertTrue(any("rejects" in l for l in lines))

    def test_gui_connects_attaches_and_sends_in_receive_mode(self):
        from trainer.gui.app import TrainingGUI

        gui = TrainingGUI()
        try:
            status = self.client.status()
            gui._on_remote_status(self.client, status)
            self.assertIs(gui.remote, self.client)
            self.assertTrue(gui.remote_gpu_edit.isVisibleTo(gui))
            gui.remote_gpu_edit.setText("0,1,2")
            self.assertEqual(gui.num_processes(), 3)
            self.assertIn("3 processes", gui.remote_gpu_label.text())
            gui._toggle_remote()                      # disconnect
            self.assertIsNone(gui.remote)
            self.assertEqual(gui.remote_label.text(), "local")

            receiver = RemoteServer(host="127.0.0.1", port=0, mode="receive",
                                    receive_dir=self.root / "got").start()
            try:
                client = RemoteClient(receiver.connect_link())
                gui._on_remote_status(client, client.status())
                cfg = self.root / "gui-run.toml"
                cfg.write_text(write_config(self.root), encoding="utf-8")
                gui._remote_start(cfg, ["cache", "train"], training=True)
                path = receiver.wait_for_config(30)
                self.assertEqual(path.name, "gui-run.toml")
                deadline = time.time() + 10
                while gui._remote_calls and time.time() < deadline:
                    self.app.processEvents()
                    time.sleep(0.05)
            finally:
                receiver.stop()
        finally:
            gui._set_remote(None, None)
            gui.close()


if __name__ == "__main__":
    unittest.main()
