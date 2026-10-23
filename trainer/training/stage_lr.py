"""StageLR: chained linear / cosine / constant / REX segments, as one LambdaLR multiplier.

Port of the `stage-lr` package (https://github.com/nruaif/stage-lr) used by the
diffusion-pipe-mageflow-ft fork, with two deliberate differences:

* **Every param group follows the schedule.** The package returns a one-element LR list computed
  from group 0, and `_LRScheduler.step` zips that against the groups -- so with `[component_lr]`
  overrides only the first group was scheduled and every other group sat at its initial LR for the
  whole run. Here the curve is expressed as a multiplier on `optimizer.lr` and applied to each
  group's own initial LR, which is how every other schedule in `optim.py` already preserves
  per-component ratios.
* **Stage lengths are fitted to the exact run.** The package rounds `percent * total` per stage,
  so the stage budgets can sum to one step more or less than the run; the remainder was either
  clamped onto the last stage's final value or silently cut. Here the run's optimizer-step count
  is known exactly (`Trainer._build_data`), so stage lengths are integers that sum to it, each
  stage gets at least one step, and the rounding remainder goes to the final stage -- the same
  fitting the diffusion-pipe fork's `stage_schedule_spec` applied before handing stages over.

The curve formulas themselves are the package's, verbatim -- including its REX endpoint convention
(`z = (n - i) / n`, so a REX stage never quite reaches `min_val`) and its warmup factor
(`1/w + (1 - 1/w) * step / w`, which starts at `lr / w`). `tests/test_stage_lr.py` checks this
against the package when it is installed.
"""

from __future__ import annotations

import math

STAGE_TYPES = ("linear", "cosine", "constant", "rex")

# Keys each stage type needs besides `type` and `percent`. Anything else in a stage table is an
# error: a typo'd `end_Lr` that silently fell back to something would cost a whole run to notice.
_REQUIRED = {
    "linear": ("end_lr",),
    "cosine": ("end_lr",),
    "constant": ("lr",),
    "rex": ("max_val", "min_val"),
}


def _number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{where} must be a finite number, got {value!r}")
    return float(value)


def validate_stages(stages) -> list[dict]:
    """Check a stage list and return it as plain dicts. Raised at config load, not at step 1."""
    if not isinstance(stages, (list, tuple)) or not stages:
        raise ValueError(
            "schedule.kind = \"stage\" needs a non-empty schedule.stages list, e.g. "
            "stages = [{ type = \"linear\", end_lr = 1e-5, percent = 0.1 }, "
            "{ type = \"rex\", max_val = 1e-5, min_val = 0, percent = 0.9 }]")
    out = []
    for i, raw in enumerate(stages):
        where = f"schedule.stages[{i}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{where} must be a table, got {type(raw).__name__}")
        stage = dict(raw)
        kind = stage.get("type")
        if kind not in STAGE_TYPES:
            raise ValueError(f"{where}.type must be one of {list(STAGE_TYPES)}, got {kind!r}")
        allowed = {"type", "percent", *_REQUIRED[kind]}
        unknown = sorted(set(stage) - allowed)
        if unknown:
            raise ValueError(f"{where} ({kind}) has unknown key(s) {unknown}; "
                             f"valid: {sorted(allowed)}")
        missing = [k for k in ("percent", *_REQUIRED[kind]) if k not in stage]
        if missing:
            raise ValueError(f"{where} ({kind}) is missing {missing}")
        if _number(stage["percent"], f"{where}.percent") <= 0:
            raise ValueError(f"{where}.percent must be > 0, got {stage['percent']}")
        for key in _REQUIRED[kind]:
            if _number(stage[key], f"{where}.{key}") < 0:
                raise ValueError(f"{where}.{key} must be >= 0, got {stage[key]}")
        if kind == "rex" and stage["min_val"] > stage["max_val"]:
            raise ValueError(f"{where}: min_val ({stage['min_val']}) must be <= max_val "
                             f"({stage['max_val']})")
        out.append(stage)
    total = sum(s["percent"] for s in out)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"schedule.stages percentages must sum to 1, got {total:g}")
    return out


def fit_stage_lengths(stages: list[dict], budget: int) -> list[int]:
    """Integer step counts per stage that sum to exactly `budget`, at least one each."""
    if budget < len(stages):
        raise ValueError(
            f"StageLR has {len(stages)} stages but only {budget} optimizer step(s) after warmup; "
            f"every stage needs at least one. Lengthen the run, shorten warmup, or drop stages.")
    counts, remaining = [], budget
    for i, stage in enumerate(stages):
        if i == len(stages) - 1:
            counts.append(remaining)
            break
        later = len(stages) - i - 1
        n = min(max(1, round(stage["percent"] * budget)), remaining - later)
        counts.append(n)
        remaining -= n
    return counts


def _stage_value(stage: dict, n_steps: int, local: int, start_lr: float) -> float:
    """One stage's LR at `local` steps into it. Formulas verbatim from stage_lr.StageLR."""
    kind = stage["type"]
    progress = local / max(n_steps - 1, 1)
    if kind == "linear":
        return start_lr + (stage["end_lr"] - start_lr) * progress
    if kind == "cosine":
        return start_lr + (stage["end_lr"] - start_lr) * (1 - math.cos(math.pi * progress)) / 2
    if kind == "constant":
        return float(stage["lr"])
    z = (n_steps - local) / n_steps
    return stage["min_val"] + (stage["max_val"] - stage["min_val"]) * (z / (1 - 0.9 + 0.9 * z))


class StagePlan:
    """The fitted schedule for one run: absolute LR per optimizer update, plus a printable table.

    Not handed to the LR scheduler as an object. LambdaLR serializes a callable *object's*
    `__dict__` into `state_dict()` and restores it on `load_state_dict()`, which would resurrect
    the old run's stage boundaries on a resume whose length changed. A closure is not serialized,
    so the boundaries always come from the run being resumed into.
    """

    def __init__(self, stages: list[dict], total_steps: int, warmup_steps: int, base_lr: float):
        if base_lr <= 0:
            raise ValueError(f"StageLR needs optimizer.lr > 0 as its starting LR, got {base_lr}")
        if warmup_steps < 0:
            raise ValueError(f"schedule.warmup_steps must be >= 0, got {warmup_steps}")
        self.stages = stages
        self.base_lr = float(base_lr)
        self.warmup = int(warmup_steps)
        self.budget = int(total_steps) - self.warmup
        self.counts = fit_stage_lengths(stages, self.budget)
        self.bounds = [0]
        for n in self.counts:
            self.bounds.append(self.bounds[-1] + n)
        # Each stage starts where the previous one ended, at its final step.
        self.starts = [self.base_lr]
        for i in range(len(stages) - 1):
            self.starts.append(_stage_value(stages[i], self.counts[i], self.counts[i] - 1,
                                            self.starts[i]))

    def lr(self, step: int) -> float:
        """Absolute LR (for a group at `optimizer.lr`) used by optimizer update `step`, 0-based."""
        if self.warmup > 1 and step < self.warmup:
            w = self.warmup
            return self.base_lr * (1.0 / w + (1.0 - 1.0 / w) * step / w)
        s = min(max(step - self.warmup, 0), self.budget - 1)
        for i, stage in enumerate(self.stages):
            if s < self.bounds[i + 1]:
                return _stage_value(stage, self.counts[i], s - self.bounds[i], self.starts[i])
        last = len(self.stages) - 1
        return _stage_value(self.stages[last], self.counts[last], self.counts[last] - 1,
                            self.starts[last])

    def multiplier(self, step: int) -> float:
        return self.lr(step) / self.base_lr

    def describe(self) -> list[str]:
        """One line per phase in 1-based optimizer-update numbers, for the startup report."""
        lines = []
        if self.warmup > 1:
            lines.append(f"warmup    updates 1..{self.warmup} ({self.warmup}), "
                         f"lr {self.lr(0):.3g} -> {self.lr(self.warmup - 1):.3g}")
        for i, stage in enumerate(self.stages):
            first = self.warmup + self.bounds[i]
            last = self.warmup + self.bounds[i + 1] - 1
            lines.append(f"{stage['type']:<9} updates {first + 1}..{last + 1} ({self.counts[i]}), "
                         f"lr {self.lr(first):.3g} -> {self.lr(last):.3g}")
        return lines


def stage_lambda(plan: StagePlan, process_scale: int = 1):
    """LambdaLR multiplier over *scheduler* steps.

    Accelerate's scheduler wrapper advances the inner scheduler `num_processes` times per optimizer
    step (see `Trainer._build_optimizer`), so scheduler step `k` is optimizer update
    `k // process_scale`. Evaluating the plan in update units keeps the stage boundaries exactly
    where they were fitted rather than where `percent * total * world_size` happens to round.
    """
    scale = max(1, int(process_scale))

    def multiplier(step: int) -> float:
        return plan.multiplier(step // scale)

    # Reachable for the startup report, and safe there: LambdaLR saves a plain function's state as
    # None, whereas an attribute on the scheduler itself would be pickled into the resume state
    # (and then refused by `torch.load(weights_only=True)`).
    multiplier.plan = plan
    return multiplier


def stage_plan_of(scheduler) -> StagePlan | None:
    """The StagePlan behind a (possibly Accelerate-wrapped) scheduler, or None."""
    inner = getattr(scheduler, "scheduler", scheduler)
    lambdas = getattr(inner, "lr_lambdas", None) or [None]
    return getattr(lambdas[0], "plan", None)
