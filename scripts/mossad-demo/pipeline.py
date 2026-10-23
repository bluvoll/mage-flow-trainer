"""MossAD-Demo pilot pipeline, unattended.

1. Waits for the speed test, picks each resolution's batch size from its results.
2. LR test: run A (4e-5) and run B (2e-5), 512 px, each until its epoch-10 checkpoint, then save-and-quit.
3. Continues the winner (A unless /root/work/pipeline/winner says B) through 512 -> 768 -> 1024, each stage
   starting from the previous stage's final weights.
Writing B into the winner file at any time stops the current A run (save-and-quit) and continues B instead.
A deadline watchdog sends save-and-quit in time for the last checkpoint to upload before the droplet goes.
Progress: /root/work/pipeline/status.txt. Logs: /root/work/logs/<run>.log.

Later stages' lengths can change while it runs: /root/work/pipeline/plan.json, e.g. {"768": {"epochs": 45}}
(keys epochs / save_every / sample_every / lr, e.g. {"1024": {"lr": "2e-5"}}), read when a stage starts or resumes.
An lr override keeps the run's name (the name says which LR test it continues). Restarting the pipeline is safe:
it first waits for a trainer that is already running (the deadline and switch watchdogs still apply to it).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

WORK = Path(os.environ.get("PIPELINE_WORK", "/root/work"))
TEMPLATE = WORK / "configs/MossAD-Demo-512-lr5e5.toml"
OUT, LOGS, STATE = WORK / "output", WORK / "logs", WORK / "pipeline"
RESULTS = WORK / "speed/results2.txt"
TRAIN = os.environ.get("PIPELINE_TRAIN", f"{sys.executable} -m trainer.training.train").split()
TRAIN_MARKER = os.environ.get("PIPELINE_TRAIN_MARKER", "trainer.training.train")   # how a running trainer is recognised
TRAINER_DIR = os.environ.get("PIPELINE_TRAINER_DIR", "/root/work/mage-flow-trainer")
DEADLINE = datetime.fromisoformat(os.environ.get("PIPELINE_DEADLINE", "2026-10-23T23:00:00+00:00"))
STOP_BEFORE = timedelta(minutes=int(os.environ.get("PIPELINE_STOP_MINUTES", "45")))
POLL = float(os.environ.get("PIPELINE_POLL", "30"))
MAX_PEAK_GB = 150                      # headroom below 192 GB for batches of unusually long captions
LRS = {"A": ("lr4e5", "4e-5"), "B": ("lr2e5", "2e-5")}
STAGES = [("512", 70, 5, 5), ("768", 70, 5, 5), ("1024", 14, 2, 2)]   # resolution, epochs, save every, sample every
LR_TEST_EPOCHS = 10
FALLBACK_BATCH = {"512": 96, "768": 32, "1024": 16}


def now():
    return datetime.now(timezone.utc)


def status(line):
    STATE.mkdir(parents=True, exist_ok=True)
    text = f"{now():%m-%d %H:%M} UTC  {line}"
    print(text, flush=True)
    with open(STATE / "status.txt", "a", encoding="utf-8") as f:
        f.write(text + "\n")


def deadline():
    """The cutoff: /root/work/pipeline/deadline (an ISO time, e.g. 2026-10-24T09:00:00+00:00) when present, so it
    can change while the pipeline runs; otherwise PIPELINE_DEADLINE."""
    try:
        return datetime.fromisoformat((STATE / "deadline").read_text().strip())
    except (OSError, ValueError):
        return DEADLINE


def winner():
    try:
        value = (STATE / "winner").read_text().strip().upper()
    except OSError:
        return "A"
    return value if value in LRS else "A"


def run_name(res, letter):
    return f"MossAD-Demo-{res}-{LRS[letter][0]}"


def stage_plan():
    """STAGES with /root/work/pipeline/plan.json's overrides applied, plus each stage's peak LR (None: the run's)."""
    try:
        overrides = json.loads((STATE / "plan.json").read_text())
    except (OSError, ValueError):
        overrides = {}
    plan = []
    for res, epochs, save_every, sample_every in STAGES:
        o = overrides.get(res, {}) if isinstance(overrides, dict) else {}
        lr = o.get("lr")
        if lr is not None:
            lr = str(lr)
            float(lr)   # a typo fails here, before a trainer starts
        plan.append((res, int(o.get("epochs", epochs)), int(o.get("save_every", save_every)),
                     int(o.get("sample_every", sample_every)), lr))
    return plan


def running_trainers():
    """{pid: run name} of trainer processes already running (data-loader workers included)."""
    import psutil
    found = {}
    for proc in psutil.process_iter(["pid", "cmdline"]):
        args = proc.info.get("cmdline") or []
        if proc.pid != os.getpid() and any(TRAIN_MARKER in a for a in args):
            config = next((a for a in args if a.endswith(".toml")), None)
            if config:
                found[proc.pid] = Path(config).stem
    return found


def adopt():
    """Wait for a trainer started by an earlier pipeline, watching it like our own. False when it was stopped
    for the deadline."""
    told, reason = set(), None
    while procs := running_trainers():
        run = next(iter(procs.values()))
        if run not in told:
            status(f"waiting for the already running {run}")
            told.add(run)
        if reason is None:
            letter = next((k for k, (name, _) in LRS.items() if run.endswith(name)), None)
            if now() >= deadline() - STOP_BEFORE:
                reason = "deadline"
            elif letter and (STATE / "lr_test_B.done").exists() and winner() != letter:
                reason = "switching to " + winner()
            if reason:
                (OUT / run / "save_quit").touch()
                status(f"{run}: save-and-quit ({reason})")
        time.sleep(POLL)
    return reason != "deadline"


def wait_for_speed_test():
    while not (RESULTS.is_file() and "ALL DONE" in RESULTS.read_text()):
        time.sleep(POLL)


def pick_batches():
    picked = {}
    if RESULTS.is_file():
        for line in RESULTS.read_text().splitlines():
            m = re.search(r"(\d+) px batch (\d+): .*?peak ([\d.]+)GB", line)
            if m and "OUT OF MEMORY" not in line and float(m.group(3)) <= MAX_PEAK_GB:
                res, batch = m.group(1), int(m.group(2))
                picked[res] = max(picked.get(res, 0), batch)
    return {res: picked.get(res, FALLBACK_BATCH[res]) for res, *_ in STAGES}


def make_config(res, letter, batch, epochs, save_every, sample_every, transformer_path=None, resume_from=None,
                fresh=False, lr=None):
    text = TEMPLATE.read_text(encoding="utf-8")
    lrname, run_lr = LRS[letter]
    lr = lr or run_lr

    def sub(pattern, new, count=1):
        nonlocal text
        text, n = re.subn(pattern, new, text, flags=re.M)
        if n != count:
            raise ValueError(f"config template: {pattern!r} matched {n} times, expected {count}")

    # varlen + packing always (no padding, mixed resolutions in one batch): refuse a template without them
    sub(r'^attention_backend = "torch_varlen"$', 'attention_backend = "torch_varlen"')
    sub(r"^pack_resolutions = true$", "pack_resolutions = true")
    sub(r"^resolution = 512$", f"resolution = {res}")
    sub(r"^run_name = .*$", f'run_name = "{run_name(res, letter)}"')
    sub(r"^epochs = 70$", f"epochs = {epochs}")
    sub(r"^batch_size = \d+$", f"batch_size = {batch}")
    sub(r"^save_every_epochs = 5$", f"save_every_epochs = {save_every}")
    sub(r"^every_n_epochs = 5$", f"every_n_epochs = {sample_every}")
    sub(r"^at_start = true$", f"at_start = {'true' if fresh else 'false'}")
    sub(r"^wandb_tags = .*$", f'wandb_tags = ["moss-pilot", "{res}", "{lrname}", "peak-{lr}"]')
    sub(r"^(end_lr|lr|max_val) = 5e-5$", rf"\1 = {lr}", count=3)
    extra = ""
    if transformer_path:
        extra += f'transformer_path = "{Path(transformer_path).as_posix()}"\n'
    if resume_from:
        extra += f'resume_from = "{Path(resume_from).as_posix()}"\n'
    if extra:
        text = re.sub(r"^(model_path = .*)$", lambda m: m.group(1) + "\n" + extra.rstrip("\n"), text, count=1, flags=re.M)
    path = WORK / "configs" / f"{run_name(res, letter)}.toml"
    path.write_text(text, encoding="utf-8")
    return path


def latest_state(run):
    best = None
    for state_json in (OUT / run).glob(f"{run}-*-state/state.json"):
        try:
            step = int(json.loads(state_json.read_text())["global_step"])
        except (OSError, ValueError, KeyError):
            continue
        if best is None or step > best[0]:
            best = (step, state_json.parent)
    return best[1] if best else None


def train(config, run, stop_when=None):
    """Run the trainer; save-and-quit it when `stop_when()` names a reason. Returns (exit code, reason or None)."""
    LOGS.mkdir(parents=True, exist_ok=True)
    run_dir = OUT / run
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(LOGS / f"{run}.log", "a", encoding="utf-8") as log:
        log.write(f"\n===== {now():%m-%d %H:%M} UTC  {' '.join(TRAIN)} {config}\n")
        log.flush()
        proc = subprocess.Popen(TRAIN + [str(config)], cwd=TRAINER_DIR, stdout=log, stderr=subprocess.STDOUT)
        reason = None
        while proc.poll() is None:
            time.sleep(POLL)
            if reason is None:
                if now() >= deadline() - STOP_BEFORE:
                    reason = "deadline"
                elif stop_when:
                    reason = stop_when()
                if reason:
                    (run_dir / "save_quit").touch()
                    status(f"{run}: save-and-quit ({reason})")
        return proc.returncode, reason


def lr_test(letter, batches):
    res, epochs, save_every, sample_every = STAGES[0]
    run = run_name(res, letter)
    marker = STATE / f"lr_test_{letter}.done"
    if marker.exists():
        return True
    checkpoint = OUT / run / f"{run}-epoch{LR_TEST_EPOCHS:03d}.safetensors"
    state = latest_state(run)
    config = make_config(res, letter, batches[res], epochs, save_every, sample_every, resume_from=state, fresh=state is None)
    status(f"LR test {letter} ({LRS[letter][1]}): {run}, batch {batches[res]}" + (f", resuming {state.name}" if state else ""))
    code, reason = train(config, run, stop_when=lambda: "lr test done" if checkpoint.exists() else None)
    if reason == "deadline":
        return False
    if code != 0 and reason is None:
        status(f"LR test {letter} FAILED (exit {code}); see {LOGS / (run + '.log')}")
        return False
    marker.touch()
    status(f"LR test {letter} done")
    return True


def chain(letter, batches):
    """The full stage plan for one LR. Returns 'done', 'switch', 'deadline' or 'failed'."""
    previous = None
    for index in range(len(STAGES)):
        res, epochs, save_every, sample_every, lr = stage_plan()[index]
        run = run_name(res, letter)
        final = OUT / run / f"{run}.safetensors"
        if final.exists():
            # Finished, possibly by a trainer an earlier pipeline started: its optimizer states can go.
            for state_dir in (OUT / run).glob(f"{run}-*-state"):
                shutil.rmtree(state_dir, ignore_errors=True)
            previous = final
            continue
        state = latest_state(run)
        config = make_config(res, letter, batches[res], epochs, save_every, sample_every,
                             transformer_path=previous, resume_from=state, fresh=state is None and res == "512", lr=lr)
        status(f"stage {res} ({lr or LRS[letter][1]}): {run}, batch {batches[res]}"
               + (f", resuming {state.name}" if state else "")
               + (f", from {previous.name}" if previous else ""))
        code, reason = train(config, run, stop_when=lambda: "switching to " + winner() if winner() != letter else None)
        if reason == "deadline":
            return "deadline"
        if reason:
            return "switch"
        if code != 0 or not final.exists():
            status(f"stage {res} FAILED (exit {code}); see {LOGS / (run + '.log')}")
            return "failed"
        # A finished stage is never resumed: its optimizer states (~30 GB each) go, its weights stay.
        for state_dir in (OUT / run).glob(f"{run}-*-state"):
            shutil.rmtree(state_dir, ignore_errors=True)
        status(f"stage {res} done: {final.name} (optimizer states removed)")
        previous = final
    return "done"


def chain_check():
    """Stage 2 starts from stage 1's exported checkpoint: load the smoke test's export the same way, once."""
    smoke = WORK / "smoke/out/smoke/smoke.safetensors"
    if os.environ.get("PIPELINE_SKIP_CHAIN_CHECK") or not smoke.exists() or (STATE / "chain_check.done").exists():
        return
    try:
        import torch
        sys.path.insert(0, TRAINER_DIR)
        from trainer.modeling.moss_loader import load_transformer
        load_transformer(WORK / "moss-image-base", smoke, torch.bfloat16)
        status("chain check: an exported checkpoint loads as transformer_path")
        (STATE / "chain_check.done").touch()
    except Exception as exc:   # stage 2 would fail later; stage 1 can still run
        status(f"chain check FAILED: {type(exc).__name__}: {exc}")


def main():
    status(f"pipeline started; deadline {deadline():%m-%d %H:%M} UTC (change: {STATE / 'deadline'}), "
           f"stops {int(STOP_BEFORE.total_seconds() // 60)} min before")
    if not adopt():
        status("stopped for the deadline; last checkpoint saved")
        return
    wait_for_speed_test()
    batches = pick_batches()
    status("batch sizes: " + ", ".join(f"{r} px {b}" for r, b in batches.items()) + " (grad accum 4)")
    chain_check()
    for letter in ("A", "B"):
        if not lr_test(letter, batches):
            status("pipeline stopped during the LR test")
            return
    status(f"LR test complete: compare A ({LRS['A'][1]}) and B ({LRS['B'][1]}) in wandb; "
           f"continuing {winner()} (echo B > {STATE / 'winner'} to switch)")
    while True:
        letter = winner()
        result = chain(letter, batches)
        if result == "switch":
            continue
        status({"done": "ALL STAGES DONE", "deadline": "stopped for the deadline; last checkpoint saved",
                "failed": "pipeline stopped after a failure"}[result])
        return


if __name__ == "__main__":
    main()
