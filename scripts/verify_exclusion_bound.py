#!/usr/bin/env python3
"""Check Theorem 2 numerically: is q* the real boundary, and does the bound hold?

An attacker alternating honest and corrupted rounds keeps its reputation above the
floor, so a constant-q model does not settle the question. The boundary is stated
instead for an arbitrary adaptive schedule and a sustainable *detected* rate

    q* = alpha*Delta_max / (beta + alpha*Delta_max).

Both the boundary and the hitting-time bound are properties of the reputation
recursion alone, so they can be checked exactly without a federated run. This
script does three things and writes what it found to a manifest:

  1. Bisects the largest sustainable detected rate and compares it with q*.
  2. Checks candidate hitting-time bounds against the realised exclusion round
     over a grid of rates in (q*, 1].
  3. Checks that no *ordering* of a fixed detection count survives above q* and
     that some ordering survives below it, which is the order-independence the
     theorem's proof sketch asserts.

Two earlier versions of the schedule failed for opposite reasons, and both
failures are informative enough to record here. Corrupting only when reputation
can absorb the penalty is the greedy attacker of clause (ii): it is never
excluded at any rate, so the test is vacuous. Imposing the rate with
``detections < rate*t`` corrupts at t=1 from r_0, where a single penalty already
zeroes the agent, so everything is excluded at every rate. The deficit form
``detections + 1 <= rate*t`` imposes the rate while deferring the first detection
to t = 1/rate, which is the banking interval a real alternating attacker uses.

Note beta = r_0 = 0.5 in the shipped configuration, so one detection at the
initial reputation zeroes an agent: it must bank to r_min + beta before it can
afford even one detection. That is why the attacker in the schedules below is
silent for its first 1/rate rounds.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xfedagent.config import load_config  # noqa: E402


def parameters(config_path: Path) -> dict[str, float]:
    """Read the recursion's constants from the config the experiments used.

    Delta_max is the largest margin the predicate can award, so it follows from
    the thresholds rather than being a free parameter: 1 - tau under raw accuracy,
    and 1 - max(tau_se, tau_sp) under the class-aware predicate, whose margin is
    the smaller of the two per-class surpluses.
    """
    cfg = load_config(config_path)
    if cfg.pov.predicate == "class_aware":
        delta_max = 1.0 - max(cfg.pov.threshold_sensitivity, cfg.pov.threshold_specificity)
    else:
        delta_max = 1.0 - cfg.pov.threshold
    return {
        "alpha": cfg.federation.reputation_reward_alpha,
        "beta": cfg.federation.reputation_penalty_beta,
        "r0": cfg.federation.reputation_initial,
        "r_min": cfg.federation.reputation_min,
        "r_max": 1.0,
        "delta_max": delta_max,
        "predicate": cfg.pov.predicate,
    }


class Recursion:
    """The reputation update the runner implements, isolated from everything else."""

    def __init__(self, p: dict[str, float]) -> None:
        self.beta = p["beta"]
        self.reward = p["alpha"] * p["delta_max"]
        self.r0, self.r_min, self.r_max = p["r0"], p["r_min"], p["r_max"]
        self.q_star = self.reward / (self.beta + self.reward)

    def drift(self, s: float) -> float:
        """Mean reputation loss per round at a sustained detected rate s."""
        return self.beta * s - self.reward * (1.0 - s)

    def step(self, r: float, detected: bool) -> float:
        return max(0.0, r - self.beta) if detected else min(self.r_max, r + self.reward)

    def simulate(self, rate: float, rounds: int) -> tuple[bool, int, float]:
        """Impose a detected rate; report exclusion, hitting time and realised rate."""
        r, detections = self.r0, 0
        for t in range(1, rounds + 1):
            detected = detections + 1 <= rate * t
            r = self.step(r, detected)
            detections += int(detected)
            if r < self.r_min:
                return True, t, detections / t
        return False, rounds, detections / rounds

    def replay(self, schedule: list[int]) -> int | None:
        r = self.r0
        for t, detected in enumerate(schedule, 1):
            r = self.step(r, bool(detected))
            if r < self.r_min:
                return t
        return None

    # The three candidate bounds. ``published`` is the smooth form as printed in
    # Theorem 2(i) with its max{T_0, .} dropped. ``slack`` is the form that
    # needs no hypothesis on T_0: held only to D(T) >= s*T - 1, the weakest
    # condition a lattice schedule of rate s can satisfy, the same additive
    # accounting closes because the cap r <= r_max only ever discards credit, and
    # one round of lattice slack costs at most one forgone penalty plus one reward.
    def published(self, s: float) -> float:
        d = self.drift(s)
        return math.ceil((self.r0 - self.r_min) / d) if d > 0 else math.inf

    def slack(self, s: float) -> float:
        d = self.drift(s)
        if d <= 0:
            return math.inf
        return math.ceil((self.r_max - self.r_min + self.beta + self.reward) / d)


def bisect_boundary(rec: Recursion, rounds: int, iterations: int = 60) -> float:
    """Largest imposed rate that survives. Should land on q* from above and below."""
    low, high = 0.0, 1.0
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        excluded, _, _ = rec.simulate(mid, rounds)
        if excluded:
            high = mid
        else:
            low = mid
    return high


def check_bounds(rec: Recursion, grid: int, rounds: int) -> dict:
    """Compare realised exclusion rounds with each candidate bound over (q*, 1]."""
    tallies = {"published": [], "slack": []}
    violations = {"published": 0, "slack": 0}
    excluded_at = 0
    for k in range(1, grid + 1):
        s = rec.q_star + (1.0 - rec.q_star) * k / grid
        excluded, t_ex, _ = rec.simulate(s, rounds)
        if not excluded:
            continue
        excluded_at += 1
        for name, bound in (("published", rec.published(s)), ("slack", rec.slack(s))):
            if t_ex > bound:
                violations[name] += 1
            else:
                tallies[name].append(bound / t_ex)
    return {
        "rates_tested": grid,
        "rates_excluded": excluded_at,
        "published_violations": violations["published"],
        "slack_violations": violations["slack"],
        "slack_mean_looseness": (
            sum(tallies["slack"]) / len(tallies["slack"]) if tallies["slack"] else None
        ),
        "slack_max_looseness": max(tallies["slack"]) if tallies["slack"] else None,
    }


def check_order_independence(rec: Recursion, trials: int, seed: int = 0) -> dict:
    """No ordering of a fixed detection count should survive above q*.

    The proof sketch claims exclusion is governed by the time-averaged detected
    rate and that no rescheduling of a fixed number of detections lowers that
    average. The observable consequence is asymmetric: above q* every permutation
    must be excluded, while below it at least one permutation must survive. Both
    directions are checked, since only testing the first would pass for a bound
    that was merely conservative.
    """
    rng = random.Random(seed)
    horizon = 400
    above_survivors = below_survivors = 0
    above_tested = below_tested = 0
    for _ in range(trials):
        detections = rng.randint(1, horizon - 1)
        rate = detections / horizon
        schedule = [1] * detections + [0] * (horizon - detections)
        rng.shuffle(schedule)
        survived = rec.replay(schedule) is None
        if rate > rec.q_star:
            above_tested += 1
            above_survivors += int(survived)
        else:
            below_tested += 1
            below_survivors += int(survived)
    return {
        "orderings_above_qstar": above_tested,
        "survivors_above_qstar": above_survivors,
        "orderings_below_qstar": below_tested,
        "survivors_below_qstar": below_survivors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/predicate-class-aware.json")
    parser.add_argument("--rounds", type=int, default=200000)
    parser.add_argument("--grid", type=int, default=2500)
    parser.add_argument("--order-trials", type=int, default=20000)
    parser.add_argument("--output", default="results/exclusion-bound/bound_manifest.json")
    args = parser.parse_args()

    p = parameters(Path(args.config))
    rec = Recursion(p)
    boundary = bisect_boundary(rec, args.rounds)
    manifest = {
        "config": str(args.config),
        "parameters": p,
        "q_star": rec.q_star,
        "banking_target": rec.r_min + rec.beta,
        "bisected_boundary": boundary,
        "boundary_relative_error": abs(boundary - rec.q_star) / rec.q_star,
        "rounds": args.rounds,
        "bounds": check_bounds(rec, args.grid, args.rounds),
        "order_independence": check_order_independence(rec, args.order_trials),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    b = manifest["bounds"]
    print(f"q*                  = {rec.q_star:.6f}")
    print(f"bisected boundary   = {boundary:.6f}  ({manifest['boundary_relative_error']*100:.3f}% off)")
    print(f"banking target      = {manifest['banking_target']:.2f} (r_min + beta)")
    print(f"published form      : {b['published_violations']}/{b['rates_excluded']} violations")
    print(f"hypothesis-free form: {b['slack_violations']}/{b['rates_excluded']} violations, "
          f"mean {b['slack_mean_looseness']:.2f}x loose, max {b['slack_max_looseness']:.2f}x")
    o = manifest["order_independence"]
    print(f"orderings above q*  : {o['survivors_above_qstar']}/{o['orderings_above_qstar']} survive")
    print(f"orderings below q*  : {o['survivors_below_qstar']}/{o['orderings_below_qstar']} survive")
    print(f"manifest={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
