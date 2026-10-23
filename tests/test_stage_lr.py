"""StageLR: parity with the stage-lr package, per-group scheduling, DDP scaling and resume."""

import math
import unittest

import torch

from trainer.training.config import ScheduleConfig
from trainer.training.optim import build_scheduler
from trainer.training.stage_lr import StagePlan, fit_stage_lengths, validate_stages


class ReferenceStageLR:
    """stage_lr.StageLR's arithmetic (https://github.com/nruaif/stage-lr, MIT), single group.

    Embedded rather than imported so parity is checked on every run, not only where the package
    happens to be installed. `counts` lets a test pin the stage lengths, because the package
    rounds `percent * total` per stage while the port fits lengths to the exact run.
    """

    def __init__(self, base_lr, stages, total_iters, warmup_steps=0, counts=None):
        self.base_lr, self.total_iters, self.warmup_steps = base_lr, total_iters, warmup_steps
        self.info, cumulative = [], 0
        for i, s in enumerate(stages):
            n = counts[i] if counts else max(1, round(s["percent"] * total_iters))
            n = min(n, total_iters - cumulative)
            self.info.append(dict(type=s["type"], params=s, start=cumulative,
                                  end=cumulative + n, n_steps=n))
            cumulative += n

    def lr(self, step):
        if self.warmup_steps > 0 and step < self.warmup_steps:
            if self.warmup_steps <= 1:
                return self.base_lr
            w = self.warmup_steps
            return self.base_lr * (1.0 / w + (1.0 - 1.0 / w) * step / w)
        step = min(max(step - self.warmup_steps, 0), self.total_iters - 1)
        for idx, stage in enumerate(self.info):
            if stage["start"] <= step < stage["end"]:
                return self._compute(stage, step, self._start(idx))
        last = self.info[-1]
        return self._compute(last, last["end"] - 1, self._start(len(self.info) - 1))

    def _start(self, idx):
        if idx == 0:
            return self.base_lr
        prev = self.info[idx - 1]
        return self._compute(prev, prev["end"] - 1, self._start(idx - 1))

    @staticmethod
    def _compute(stage, step, start_lr):
        p, n = stage["params"], stage["n_steps"]
        local = step - stage["start"]
        progress = local / max(n - 1, 1)
        if stage["type"] == "linear":
            return start_lr + (p["end_lr"] - start_lr) * progress
        if stage["type"] == "cosine":
            return start_lr + (p["end_lr"] - start_lr) * (1 - math.cos(math.pi * progress)) / 2
        if stage["type"] == "constant":
            return p["lr"]
        z = float(n - local) / n
        return p["min_val"] + (p["max_val"] - p["min_val"]) * (z / (1 - 0.9 + 0.9 * z))


STAGES = [
    {"type": "linear", "end_lr": 7e-6, "percent": 0.1},
    {"type": "cosine", "end_lr": 5e-6, "percent": 0.2},
    {"type": "constant", "lr": 6e-6, "percent": 0.3},
    {"type": "rex", "max_val": 6e-6, "min_val": 1e-7, "percent": 0.4},
]


def scheduled_lrs(scheduler, optimizer, updates, per_update=1):
    """LR of every group used by each optimizer update, stepping the scheduler like Accelerate."""
    out = []
    for _ in range(updates):
        out.append([g["lr"] for g in optimizer.param_groups])
        optimizer.step()
        for _ in range(per_update):
            scheduler.step()
    return out


class StageLRTests(unittest.TestCase):
    def optimizer(self, lrs=(1e-7,)):
        params = [torch.nn.Parameter(torch.zeros(1)) for _ in lrs]
        return torch.optim.SGD([{"params": [p], "lr": lr} for p, lr in zip(params, lrs)], lr=lrs[0])

    def test_matches_reference_formulas_exactly(self):
        for total, warmup in ((1000, 0), (1000, 50), (850, 0), (37, 3)):
            with self.subTest(total=total, warmup=warmup):
                plan = StagePlan(STAGES, total, warmup, 1e-7)
                ref = ReferenceStageLR(1e-7, STAGES, total - warmup, warmup, counts=plan.counts)
                for step in range(total + 5):
                    self.assertAlmostEqual(plan.lr(step), ref.lr(step), delta=1e-18,
                                           msg=f"step {step}")

    def test_lengths_match_package_when_percentages_divide_evenly(self):
        # 850 updates is the fork's documented example: 85 / 425 / 340.
        stages = [{"type": "linear", "end_lr": 7e-6, "percent": 0.1},
                  {"type": "constant", "lr": 7e-6, "percent": 0.5},
                  {"type": "rex", "max_val": 7e-6, "min_val": 0, "percent": 0.4}]
        self.assertEqual(fit_stage_lengths(stages, 850), [85, 425, 340])
        plan = StagePlan(stages, 850, 0, 1e-7)
        ref = ReferenceStageLR(1e-7, stages, 850)
        self.assertTrue(all(plan.lr(s) == ref.lr(s) for s in range(850)))
        self.assertEqual(plan.describe()[0].split()[:3], ["linear", "updates", "1..85"])

    def test_fitted_lengths_always_sum_to_the_run(self):
        thirds = [{"type": "constant", "lr": 1e-5, "percent": 1 / 3}] * 3
        for budget in (3, 4, 10, 100, 101, 1001):
            counts = fit_stage_lengths(thirds, budget)
            self.assertEqual(sum(counts), budget)
            self.assertTrue(all(c >= 1 for c in counts))
        with self.assertRaisesRegex(ValueError, "at least one"):
            fit_stage_lengths(thirds, 2)

    def test_every_param_group_follows_the_curve(self):
        # The package returns one LR (group 0's); the port scales every group by its ratio.
        opt = self.optimizer((1e-7, 2e-7, 0.5e-7))
        sched = build_scheduler(opt, ScheduleConfig(kind="stage", stages=STAGES), 100,
                                base_lr=1e-7)
        plan = StagePlan(STAGES, 100, 0, 1e-7)
        for step, lrs in enumerate(scheduled_lrs(sched, opt, 100)):
            expected = plan.lr(step)
            self.assertAlmostEqual(lrs[0], expected, delta=1e-18)
            self.assertAlmostEqual(lrs[1], 2 * expected, delta=1e-18)
            self.assertAlmostEqual(lrs[2], 0.5 * expected, delta=1e-18)

    def test_world_size_scaling_keeps_boundaries_in_update_units(self):
        cfg = ScheduleConfig(kind="stage", stages=STAGES, warmup_steps=5)
        single_opt = self.optimizer()
        single = scheduled_lrs(build_scheduler(single_opt, cfg, 60, base_lr=1e-7), single_opt, 60)
        # What Trainer._build_optimizer passes on 4 processes, and how Accelerate then steps it.
        quad_cfg = ScheduleConfig(kind="stage", stages=STAGES, warmup_steps=20)
        quad_opt = self.optimizer()
        quad = scheduled_lrs(build_scheduler(quad_opt, quad_cfg, 240, base_lr=1e-7,
                                             process_scale=4), quad_opt, 60, per_update=4)
        self.assertEqual(single, quad)

    def test_resume_uses_the_new_run_length(self):
        opt = self.optimizer()
        sched = build_scheduler(opt, ScheduleConfig(kind="stage", stages=STAGES), 100,
                                base_lr=1e-7)
        scheduled_lrs(sched, opt, 40)
        # Accelerate's save_state goes through torch.save, and resume loads with
        # weights_only=True: nothing of ours may be pickled into the scheduler state.
        import io
        buf = io.BytesIO()
        torch.save(sched.state_dict(), buf)
        buf.seek(0)
        state = torch.load(buf, weights_only=True)
        from trainer.training.stage_lr import stage_plan_of
        self.assertEqual(stage_plan_of(sched).counts, StagePlan(STAGES, 100, 0, 1e-7).counts)
        # The resumed run is longer (dataset grew): its plan, not the saved one, must apply.
        opt2 = self.optimizer()
        sched2 = build_scheduler(opt2, ScheduleConfig(kind="stage", stages=STAGES), 200,
                                 base_lr=1e-7)
        sched2.load_state_dict(state)
        self.assertEqual(sched2.last_epoch, 40)
        sched2.step()
        self.assertAlmostEqual(opt2.param_groups[0]["lr"],
                               StagePlan(STAGES, 200, 0, 1e-7).lr(41), delta=1e-18)

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            ScheduleConfig(kind="stage")
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            validate_stages([{"type": "constant", "lr": 1e-5, "percent": 0.5}])
        with self.assertRaisesRegex(ValueError, "unknown key"):
            validate_stages([{"type": "constant", "lr": 1e-5, "end_lr": 1, "percent": 1.0}])
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_stages([{"type": "rex", "max_val": 1e-5, "percent": 1.0}])
        with self.assertRaisesRegex(ValueError, "min_val"):
            validate_stages([{"type": "rex", "max_val": 1e-6, "min_val": 1e-5, "percent": 1.0}])
        with self.assertRaisesRegex(ValueError, "type"):
            validate_stages([{"type": "step", "lr": 1, "percent": 1.0}])
        with self.assertRaisesRegex(ValueError, "only apply"):
            ScheduleConfig(kind="cosine", stages=STAGES)
        with self.assertRaisesRegex(ValueError, "optimizer.lr > 0"):
            StagePlan(STAGES, 100, 0, 0.0)

    def test_installed_package_agrees_when_present(self):
        try:
            from stage_lr import StageLR
        except ImportError:
            self.skipTest("stage-lr not installed")
        stages = [{"type": "linear", "end_lr": 7e-6, "percent": 0.1},
                  {"type": "constant", "lr": 7e-6, "percent": 0.5},
                  {"type": "rex", "max_val": 7e-6, "min_val": 0, "percent": 0.4}]
        opt = self.optimizer()
        pkg = StageLR(opt, stages=stages, total_iters=850, warmup_steps=0)
        plan = StagePlan(stages, 850, 0, 1e-7)
        for step in range(850):
            self.assertAlmostEqual(opt.param_groups[0]["lr"], plan.lr(step), delta=1e-18)
            opt.step()
            pkg.step()


if __name__ == "__main__":
    unittest.main()
