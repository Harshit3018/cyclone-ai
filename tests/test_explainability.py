"""Tests for ml/explainability/gradcam.py.

Saliency and feature-importance code is where a system is most tempting to fabricate: the
output is a picture or a ranked table, both of which look authoritative whatever produced
them. A uniform heatmap and a table of zeros are indistinguishable at a glance from a real
attribution, so the checks here are mostly about what the module *refuses* to emit.

Three live defects these pin down, all measured before the fix:
  - hooks were never removed, leaking one pair per /api/predict request onto the shared
    detector (64 KiB retained per request, unbounded, stale hooks firing on unrelated calls)
  - a missing gradient returned a uniform 0.5 map that the UI drew as a real heatmap
  - permutation importance differenced batch means, which permutation cannot move: exactly
    0.0 for every feature on an affine readout
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.explainability.gradcam import (GradCAM, GradCAMUnavailable,
                                       compute_feature_importance)


def _hooks(model: torch.nn.Module) -> int:
    """Count forward and backward hooks across a model.

    register_full_backward_hook stores into `_backward_hooks` on torch 2.10 (flagged by
    `_is_full_backward_hook`), not `_full_backward_hooks`. Both are counted so this keeps
    working if that moves -- an undercount here would make the leak test pass vacuously,
    which is how the leak went unnoticed in the first place.
    """
    return sum(len(getattr(m, name, ()) or ())
               for m in model.modules()
               for name in ("_forward_hooks", "_backward_hooks", "_full_backward_hooks"))


class TinyConv(torch.nn.Module):
    """Smallest model with a Conv2d, so a target layer exists to attribute to."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 3, padding=1)
        self.head = torch.nn.Linear(8, 1)

    def forward(self, x):
        f = torch.relu(self.conv(x))
        return {"detection_logit": self.head(f.mean(dim=[2, 3]))}


class Exploding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 2, 3)

    def forward(self, x):
        raise ValueError("forward blew up")


@pytest.fixture
def img() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, 4, 16, 16)


# --- hook lifetime --------------------------------------------------------------------

def test_hooks_are_removed_on_context_exit(img):
    """The leak. Hooks attach to a model that outlives the GradCAM object.

    In the API the model is a process-lifetime singleton, so a hook left behind is a hook
    left behind forever -- and it keeps firing on every unrelated inference.
    """
    model = TinyConv()
    assert _hooks(model) == 0

    with GradCAM(model) as cam:
        assert _hooks(model) == 2, "expected one forward and one backward hook while in use"
        cam.generate_overlay(img, task="detection")

    assert _hooks(model) == 0, "hooks survived the context manager"


def test_hooks_are_removed_even_when_the_forward_pass_raises():
    """A failing model must not leak. try/finally, not a happy-path cleanup."""
    model = Exploding()
    with pytest.raises(ValueError):
        with GradCAM(model) as cam:
            cam.generate(torch.randn(1, 4, 8, 8))
    assert _hooks(model) == 0


def test_repeated_explanations_do_not_accumulate_hooks(img):
    """The measured failure was linear growth: 50 calls -> 50 hook pairs.

    One iteration cannot distinguish "removed" from "overwritten", so this runs enough to
    make accumulation obvious.
    """
    model = TinyConv()
    for _ in range(25):
        with GradCAM(model) as cam:
            cam.generate_overlay(img, task="detection")
    assert _hooks(model) == 0


def test_a_removed_gradcam_holds_no_tensors(img):
    """While a hook was registered it kept a slice of every forward pass alive.

    That is what turned a hook leak into a memory leak, so remove() must drop the captures
    as well as the handles.
    """
    model = TinyConv()
    cam = GradCAM(model)
    cam.generate(img, task="detection")
    assert cam.activations is not None, "the forward hook never fired; this test proves nothing"

    cam.remove()
    assert cam.activations is None and cam.gradients is None
    cam.remove()  # idempotent: a double release must not raise


def test_a_reused_gradcam_refuses_rather_than_returning_a_stale_map(img):
    """After remove() the hooks are gone, so any map produced would be from stale captures."""
    cam = GradCAM(TinyConv())
    cam.generate(img, task="detection")
    cam.remove()
    with pytest.raises(GradCAMUnavailable):
        cam.generate(img, task="detection")


# --- not fabricating an explanation ---------------------------------------------------

def test_a_model_with_no_conv_layer_reports_unavailable(img):
    """There is no spatial layer to attribute to, so there is no heatmap. Say so."""
    with GradCAM(torch.nn.Sequential(torch.nn.Linear(4, 1))) as cam:
        out = cam.generate_overlay(torch.randn(2, 4))
    assert out["available"] is False
    assert out["reason"]
    # The absence matters: a caller doing `out["heatmap"]` must fail loudly, not draw a
    # square. The old code returned a uniform 0.5 map here.
    assert "heatmap" not in out
    assert "attention_area_fraction" not in out and "peak_row" not in out


def test_an_available_heatmap_is_not_uniform(img):
    """A flat map is what the old fallback produced; it must not be what success looks like."""
    with GradCAM(TinyConv()) as cam:
        out = cam.generate_overlay(img, task="detection")

    assert out["available"] is True
    hm = np.array(out["heatmap"])
    assert hm.shape == (16, 16), "heatmap must match the input's spatial size"
    assert len(np.unique(hm)) > 1, "a single-valued heatmap is not an attribution"
    assert not np.allclose(hm, 0.5), "this is the fabricated fallback, not a real map"
    assert 0.0 <= hm.min() and hm.max() <= 1.0


def test_reported_statistics_describe_the_returned_heatmap(img):
    """The summary numbers are what the UI shows when the picture is too small to read."""
    with GradCAM(TinyConv()) as cam:
        out = cam.generate_overlay(img, task="detection")
    hm = np.array(out["heatmap"])

    assert out["heatmap_min"] == pytest.approx(float(hm.min()), abs=1e-6)
    assert out["heatmap_max"] == pytest.approx(float(hm.max()), abs=1e-6)
    assert out["heatmap_mean"] == pytest.approx(float(hm.mean()), abs=1e-6)
    assert out["attention_area_fraction"] == pytest.approx(
        float((hm > out["attention_threshold"]).mean()), abs=1e-4)
    assert hm[out["peak_row"], out["peak_col"]] == pytest.approx(float(hm.max()), abs=1e-6)


def test_explaining_does_not_leave_grad_on_the_callers_tensor(img):
    """The satellite tensor is reused by the plain inference paths.

    Calling requires_grad_ on it in place added gradient bookkeeping to every later
    prediction made from the same array.
    """
    assert img.requires_grad is False
    with GradCAM(TinyConv()) as cam:
        cam.generate(img, task="detection")
    assert img.requires_grad is False, "explain() mutated the caller's tensor"


def test_heatmap_keeps_two_dimensions_for_a_single_pixel_map():
    """A bare .squeeze() also collapses a 1-pixel spatial axis.

    The caller and the canvas both index [row][col]; a 1-D return would throw there.
    """
    model = TinyConv()
    with GradCAM(model) as cam:
        hm = cam.generate(torch.randn(1, 4, 1, 1), task="detection")
    assert hm.ndim == 2 and hm.shape == (1, 1)


class ConvBypassed(torch.nn.Module):
    """A Conv2d that runs but does not feed the score.

    So the forward hook fires and activations are captured, while the backward pass produces
    no gradient at that layer. This is the shape of a real misconfiguration -- an
    auto-detected "last conv" that sits on a branch the chosen head does not use -- and it
    is the one path that reached the uniform-0.5 fallback.
    """

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 3, padding=1)
        self.head = torch.nn.Linear(4 * 16 * 16, 1)

    def forward(self, x):
        self.conv(x)                                        # activations, but off-path
        return {"detection_logit": self.head(x.flatten(1))}  # score bypasses the conv


def test_a_layer_with_no_gradient_reports_unavailable(img):
    """The fallback that mattered most: it produced a heatmap the UI drew as real.

    A uniform 0.5 map reads as "the model attends everywhere equally". The truth is that no
    attribution exists, and the two must not look alike.
    """
    model = ConvBypassed()
    with GradCAM(model) as cam:
        out = cam.generate_overlay(img, task="detection")

    assert out["available"] is False
    assert "no gradients" in out["reason"]
    assert "heatmap" not in out, "a fabricated map is being returned for a gradient-free layer"
    assert _hooks(model) == 0

    # And the raw call raises rather than quietly substituting a map.
    with GradCAM(model) as cam:
        with pytest.raises(GradCAMUnavailable):
            cam.generate(img, task="detection")


def test_the_gradient_free_fixture_really_captures_activations(img):
    """Guards the test above: if the conv stopped running, it would pass for the wrong reason.

    The point is activations present, gradients absent -- not simply an inert layer.
    """
    cam = GradCAM(ConvBypassed())
    with pytest.raises(GradCAMUnavailable):
        cam.generate(img, task="detection")
    assert cam.activations is not None, "the conv never ran; this fixture proves nothing"
    assert cam.gradients is None, "gradients reached the layer; the fixture is not off-path"
    cam.remove()


# --- permutation sensitivity ----------------------------------------------------------

class KnownWeights(torch.nn.Module):
    """Affine readout with dependence fixed at 3 : 1 : 0, so the truth is known.

    Affine is the case that exposed the bug: permuting a column cannot change its mean, and
    for an affine readout the batch-mean prediction depends only on the column means, so the
    old batch-mean difference was identically zero.
    """

    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(3, 1)
        with torch.no_grad():
            self.l.weight.copy_(torch.tensor([[3.0, 1.0, 0.0]]))
            self.l.bias.zero_()

    def forward(self, x):
        return {"wind_kph": self.l(x)}


@pytest.fixture
def samples() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(400, 3)


def test_sensitivity_recovers_a_known_dependence(samples):
    """The regression test for the batch-mean bug.

    Weights are 3 : 1 : 0. The measured movement must reproduce that ratio and score the
    unused feature at zero. The old implementation returned 0.0 for all three.
    """
    out = compute_feature_importance(KnownWeights(), samples,
                                     ["strong", "weak", "unused"], n_repeats=20)
    assert out["available"] is True
    f = out["features"]

    strong = f["strong"]["mean_abs_prediction_change"]
    weak = f["weak"]["mean_abs_prediction_change"]
    unused = f["unused"]["mean_abs_prediction_change"]

    assert strong > 0 and weak > 0, (
        "an affine model shows zero movement only if batch means are being differenced")
    assert unused == pytest.approx(0.0, abs=1e-6), "a feature with weight 0 must not move it"
    assert strong / weak == pytest.approx(3.0, rel=0.15), (
        f"expected the 3:1 weight ratio, measured {strong / weak:.2f}")
    assert f["strong"]["share"] > f["weak"]["share"] > 0


def test_a_single_sample_refuses_instead_of_scoring_everything_zero():
    """randperm(1) is the identity, so every score is 0.0 -- which reads as a real finding.

    The single-observation inference path supplies exactly one row.
    """
    out = compute_feature_importance(KnownWeights(), torch.randn(1, 3), ["a", "b", "c"])
    assert out["available"] is False
    assert "identity" in out["reason"]
    assert "features" not in out


def test_mismatched_feature_names_refuse_rather_than_mislabel(samples):
    """Scores attributed to the wrong names are worse than no scores."""
    out = compute_feature_importance(KnownWeights(), samples, ["a", "b"])
    assert out["available"] is False and "feature columns" in out["reason"]


def test_a_model_that_ignores_every_feature_reports_no_share(samples):
    """Zero total movement makes a share undefined; 0.0 would assert 'unimportant'.

    A saturated or untrained model lands here, and it is a fact about the model, not a
    ranking of the features.
    """
    class Saturated(torch.nn.Module):
        def forward(self, x):
            return {"wind_kph": torch.zeros(x.size(0), 1)}

    out = compute_feature_importance(Saturated(), samples, ["a", "b", "c"])
    assert out["available"] is True and out["total_movement"] == 0.0
    assert all(v["share"] is None for v in out["features"].values())


def test_sensitivity_is_reproducible_by_default(samples):
    """A ranking that changes between two identical calls cannot be quoted anywhere."""
    a = compute_feature_importance(KnownWeights(), samples, ["a", "b", "c"])
    b = compute_feature_importance(KnownWeights(), samples, ["a", "b", "c"])
    assert a["features"] == b["features"]


def test_sensitivity_reports_its_spread_and_declines_to_claim_accuracy(samples):
    """Without labels this measures output movement, not error growth.

    The distinction decides whether the number may be called "feature importance", so it
    travels with the payload rather than living only in a docstring.
    """
    out = compute_feature_importance(KnownWeights(), samples, ["a", "b", "c"], n_repeats=8)
    assert out["n_repeats"] == 8 and out["n_samples"] == 400
    assert "sensitivity" in out["measure"]
    assert out["not_accuracy_based"], "the payload must say what it is not measuring"
    for v in out["features"].values():
        assert "std" in v, "a mean over repeats without its spread hides an unstable estimate"


def test_sensitivity_does_not_leak_hooks_or_grad(samples):
    """It runs under no_grad and registers nothing; a leak here would be silent.

    Gradients matter beyond memory: this is called on the same model instance the inference
    routes use, so a stray .grad left on its parameters would ride along into training or
    into any later gradient-based call.
    """
    model = KnownWeights()
    compute_feature_importance(model, samples, ["a", "b", "c"])
    assert _hooks(model) == 0
    assert all(p.grad is None for p in model.parameters())
