"""Tests for the p_det estimator and its Wilson intervals."""
from __future__ import annotations

import math

from xfedagent.detection import AttackDetection, wilson_interval


def test_wilson_interval_contains_the_point_estimate():
    lo, hi = wilson_interval(7, 10)
    assert lo < 0.7 < hi


def test_wilson_interval_stays_inside_the_unit_interval_at_the_boundary():
    # The Wald interval leaves [0, 1] here; Wilson must not.
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and 0.0 < hi < 1.0
    lo, hi = wilson_interval(20, 20)
    assert 0.0 < lo < 1.0 and hi == 1.0


def test_wilson_interval_narrows_with_more_trials():
    narrow = wilson_interval(200, 400)
    wide = wilson_interval(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_empty_sample_reports_the_full_interval():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_p_det_is_submission_weighted_not_a_mean_of_rates():
    # Two rounds with unequal denominators: 1/1 admitted and 1/9 admitted.
    # Submission-weighted p_det = 1 - 2/10 = 0.8, whereas averaging the two
    # per-round rates would give 1 - (1.0 + 1/9)/2 = 0.444.
    pooled = AttackDetection(attack="alie", seed=0, submitted=10, admitted=2)
    assert math.isclose(pooled.p_det, 0.8)
    naive = 1.0 - (1.0 / 1 + 1.0 / 9) / 2
    assert not math.isclose(pooled.p_det, naive, abs_tol=1e-3)


def test_detected_and_p_det_agree_with_the_counts():
    d = AttackDetection(attack="backdoor", seed=3, submitted=40, admitted=38)
    assert d.detected == 2
    assert math.isclose(d.p_det, 0.05)
    lo, hi = d.interval()
    assert lo < 0.05 < hi


def test_zero_submissions_yields_zero_rate_and_full_interval():
    d = AttackDetection(attack="minmax", seed=1, submitted=0, admitted=0)
    assert d.p_det == 0.0
    assert d.interval() == (0.0, 1.0)


def test_a_replayed_global_model_is_a_free_rider_the_copy_detector_refuses():
    """The canonical free-rider vector, and the one claim about it the paper quantifies.

    A replay submits the current global model unchanged: its utility is whatever the
    global model already attains, so a utility predicate alone admits it. That is why
    ``R_PoV`` carries the copy-detector bound as a fourth condition rather than leaving
    it to an external check. The property the measurement rests on is that a replay
    lands at ``copy_distance == 0`` exactly, so any positive ``copy_epsilon`` refuses
    it while leaving an honestly trained update untouched.
    """
    from collections import OrderedDict

    import numpy as np
    import torch

    from xfedagent.attacks import apply_update_attacks
    from xfedagent.model import state_to_vector

    rng = np.random.default_rng(0)
    global_state = OrderedDict(
        (name, torch.randn(4, 3)) for name in ("encoder.weight", "head.weight")
    )
    trained = {
        client: OrderedDict((n, t + 0.5) for n, t in global_state.items())
        for client in range(4)
    }
    malicious = {0, 2}

    result = apply_update_attacks(global_state, trained, malicious, ("replay",), rng)
    reference = state_to_vector(global_state)
    for client, state in sorted(result.items()):
        distance = float(np.linalg.norm(state_to_vector(state) - reference))
        if client in malicious:
            assert distance == 0.0, (client, distance)
        else:
            # Honest submissions are returned untouched, so the detector never fires
            # on them: the arm measures the detector, not a side effect of the attack.
            assert distance > 0.0, (client, distance)
            assert state is trained[client]

    # No malicious clients means no rewriting at all, which keeps an ablation arm that
    # disables the attack schedule from silently differing from its control.
    assert apply_update_attacks(global_state, trained, set(), ("replay",), rng) is trained


def test_minmax_obeys_the_honest_diameter_and_maximizes_the_fixed_direction():
    import numpy as np
    from xfedagent.attacks import minmax_update

    honest = np.array([[1.0, 2.0], [1.5, 3.5], [2.5, 2.2], [3.0, 4.0]])
    candidate = minmax_update(honest)
    diameter = max(np.linalg.norm(a - b) for a in honest for b in honest)
    assert max(np.linalg.norm(candidate - row) for row in honest) <= diameter + 1e-12
    direction = honest.std(axis=0)
    direction /= np.linalg.norm(direction)
    beyond = candidate - 1e-6 * direction
    assert max(np.linalg.norm(beyond - row) for row in honest) > diameter
    assert not np.allclose(candidate, -2 * honest.mean(axis=0))


def test_minmax_has_a_closed_form_one_dimensional_boundary_and_degenerate_case():
    import numpy as np
    from xfedagent.attacks import minmax_update

    assert np.allclose(minmax_update(np.array([[1.0], [3.0]])), [1.0])
    assert np.array_equal(minmax_update(np.array([[2.0, 3.0]])), [2.0, 3.0])
    assert np.array_equal(minmax_update(np.zeros((3, 4))), np.zeros(4))


def test_minmax_runner_path_shares_the_crafted_update_and_preserves_honest_states():
    from collections import OrderedDict
    import numpy as np
    import torch
    from xfedagent.attacks import apply_update_attacks
    from xfedagent.model import vector_update

    base = OrderedDict(weight=torch.tensor([0.0], dtype=torch.float64))
    states = {cid: OrderedDict(weight=torch.tensor([value], dtype=torch.float64))
              for cid, value in enumerate([1.0, 3.0, 10.0, 20.0])}
    output = apply_update_attacks(base, states, {2, 3}, ("minmax",), np.random.default_rng(0))
    assert output[0] is states[0] and output[1] is states[1]
    assert np.allclose(vector_update(base, output[2]), [1.0])
    assert np.array_equal(vector_update(base, output[2]), vector_update(base, output[3]))
