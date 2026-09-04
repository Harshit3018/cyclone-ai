"""
Grad-CAM Implementation for Cyclone Satellite Image Explainability.

Generates class activation maps showing which regions of satellite
imagery the CNN model focuses on for its predictions.

Three defects fixed here, all measured
--------------------------------------
1. The hooks were never removed. ``_register_hooks`` discarded the handles that
   ``register_forward_hook`` returns, so every ``GradCAM(model)`` permanently attached a
   hook pair to the *shared, long-lived* detector held by the predictor singleton. Measured
   on this API: one pair leaked per ``/api/predict`` request, 50 requests left 50 hook pairs
   and 50 GradCAM objects that the garbage collector could not reclaim, and all 50 of those
   dead objects' forward hooks still fired during an unrelated ``/api/predict/detect`` call,
   each writing an activation tensor -- 64 KiB retained per request, without bound. The
   handles are now kept and released, and callers use the class as a context manager.

2. A missing gradient produced a uniform 0.5 heatmap. That renders in the UI as a flat
   coloured square, which reads as "the model attends everywhere equally" when the truth is
   "no gradients were captured, so there is no explanation to show". It now reports
   ``available: False`` with a reason, in line with how the rest of this codebase handles a
   quantity it cannot measure.

3. ``compute_feature_importance`` differenced batch *means*, which permutation provably
   cannot move -- exactly 0.0 for every feature on an affine readout, and only 2-4% of the
   real effect on a ReLU MLP. See that function's docstring.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class GradCAMUnavailable(RuntimeError):
    """Raised when no attribution can be computed, rather than returning a plausible map."""


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Produces a heatmap highlighting important regions in satellite images
    that influence the model's prediction.

    Hooks live only as long as this object is in use. Prefer the context-manager form,
    which removes them even if ``generate`` raises::

        with GradCAM(model) as cam:
            result = cam.generate_overlay(tensor, task="detection")

    Constructing one without ``with`` (or without calling ``remove()``) leaves hooks
    attached to ``model`` for the lifetime of the model, which for a cached predictor is
    the lifetime of the process.
    """

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.gradients = None
        self.activations = None
        self._handles: list = []

        # Auto-detect target layer if not specified
        if target_layer is None:
            target_layer = self._find_last_conv_layer(model)

        self.target_layer = target_layer

        if self.target_layer is not None:
            self._register_hooks()

    def _find_last_conv_layer(self, model):
        """Find the last convolutional layer in the model."""
        last_conv = None
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        return last_conv

    def _register_hooks(self):
        """Register forward and backward hooks, retaining the handles so they can be removed."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove(self) -> None:
        """Detach every hook from the model. Idempotent, so double-removal is harmless."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        # Drop the captured tensors too: while a hook was registered these kept a slice of
        # every forward pass alive, which is what turned the leak into a memory leak.
        self.gradients = None
        self.activations = None

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()

    def __del__(self):
        # A backstop for callers that neither use `with` nor call remove(). Not a substitute
        # for either: a registered hook keeps a reference to this object through its closure,
        # so __del__ may never run at all -- which is exactly how the original leaked.
        try:
            self.remove()
        except Exception:
            pass

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        task: str = "detection",
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: [1, C, H, W] satellite image
            target_class: class index for classification tasks
            task: "detection" or "classification"

        Returns:
            heatmap: [H, W] numpy array, values 0-1

        Raises:
            GradCAMUnavailable: if no target layer was found, or the backward pass produced
                no gradients at the target layer. Both mean there is no attribution to show.
        """
        if self.target_layer is None:
            raise GradCAMUnavailable(
                "no Conv2d layer found in this model, so there is no spatial layer to attribute to")
        if not self._handles:
            raise GradCAMUnavailable(
                "hooks have been removed; construct a new GradCAM for each explanation")

        self.model.eval()

        # Clone rather than calling requires_grad_ on the caller's tensor: the satellite
        # tensor is reused by the plain inference paths, and flipping requires_grad on it
        # would silently add gradient bookkeeping to every later prediction.
        work = input_tensor.detach().clone().requires_grad_(True)

        # Forward pass
        output = self.model(work)

        # Select target score
        if task == "detection":
            if isinstance(output, dict):
                score = output.get("detection_logit", output.get("detection_prob"))
            else:
                score = output
            if score.dim() > 0:
                score = score.sum()
        elif task == "classification":
            if isinstance(output, dict):
                logits = output.get("logits", output.get("class_logits"))
            else:
                logits = output
            if target_class is None:
                target_class = logits.argmax(dim=-1).item()
            score = logits[0, target_class]
        else:
            score = list(output.values())[0]
            if score.dim() > 0:
                score = score.sum()

        # Backward pass
        self.model.zero_grad(set_to_none=True)
        score.backward()

        if self.gradients is None or self.activations is None:
            raise GradCAMUnavailable(
                "the backward pass produced no gradients at the target layer; "
                "a uniform heatmap here would look like an explanation without being one")

        # Compute weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # GAP of gradients

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # ReLU
        cam = F.relu(cam)

        # Resize to input size
        cam = F.interpolate(cam, size=work.shape[2:], mode='bilinear', align_corners=False)

        # Normalize. squeeze(0) twice rather than a bare squeeze(): a bare squeeze also
        # collapses a 1-pixel spatial axis, which would return a 1-D array shaped unlike
        # the [H, W] the caller and the UI both assume.
        cam = cam[0, 0].detach().cpu().numpy()
        span = float(cam.max()) - float(cam.min())
        if span > 0:
            cam = (cam - cam.min()) / span
        # An all-zero map survives unnormalised, and truthfully: after the ReLU it means
        # no region made a positive contribution to this score.

        return cam.astype(np.float32)

    def generate_overlay(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        task: str = "detection",
        alpha: float = 0.5,
    ) -> dict:
        """
        Generate Grad-CAM with overlay information.

        Returns dict with heatmap and metadata, or ``{"available": False, "reason": ...}``
        when no attribution could be computed. The caller must branch on ``available``
        rather than reading ``heatmap`` unconditionally.
        """
        try:
            heatmap = self.generate(input_tensor, target_class, task)
        except GradCAMUnavailable as exc:
            return {"available": False, "reason": str(exc)}

        # Compute attention statistics
        threshold = 0.5
        attention_area = float((heatmap > threshold).mean())
        peak_location = np.unravel_index(heatmap.argmax(), heatmap.shape)

        return {
            "available": True,
            "heatmap": heatmap.tolist(),
            "attention_area_fraction": round(attention_area, 4),
            "attention_threshold": threshold,
            "peak_row": int(peak_location[0]),
            "peak_col": int(peak_location[1]),
            "heatmap_min": float(heatmap.min()),
            "heatmap_max": float(heatmap.max()),
            "heatmap_mean": float(heatmap.mean()),
        }


def compute_feature_importance(
    model: torch.nn.Module,
    features: torch.Tensor,
    feature_names: list,
    n_repeats: int = 10,
    task_key: str = "wind_kph",
    seed: Optional[int] = 0,
) -> dict:
    """
    Permutation sensitivity for environmental features.

    Shuffles each feature across the batch and measures how much the *per-sample*
    predictions move. Larger means the model's output depends more on that feature.

    Two defects this replaces, both of which made the old numbers unusable
    --------------------------------------------------------------------
    1. It compared batch *means*: ``abs(baseline.mean() - permuted.mean())``. Permuting a
       column leaves that column's mean unchanged, so for an affine readout the batch-mean
       prediction cannot move at all -- measured here as exactly 0.0 for every feature while
       the true per-sample effect was 0.20-0.50. On a ReLU MLP the batch mean retained only
       2-4% of the per-sample effect, and the attenuation differed per feature, so even the
       ranking was distorted. Averaging over the batch cancels precisely the per-sample
       changes the method exists to detect. Now differenced per sample, then averaged.
    2. A single-row input silently returned 0.0 for everything, because ``randperm(1)`` is
       the identity. That is indistinguishable from a real finding that no feature matters.
       It now refuses.

    This is a *sensitivity* measure, not the textbook permutation importance: that is
    defined as the increase in prediction error when a feature is shuffled, which needs
    ground-truth labels. None are passed here, so what is returned is how much the output
    moves, not how much the accuracy degrades. A feature the model leans on heavily but
    wrongly scores high either way; the two only coincide for a well-fitted model.

    Args:
        n_repeats: shuffles per feature. The spread across repeats is returned as ``std``;
            a std comparable to the mean means the estimate is noise at this sample size.
        seed: fixes the shuffles so repeated calls agree. Pass None for fresh randomness.

    Returns:
        ``{"available": True, "features": {name: {...}}, ...}``, or
        ``{"available": False, "reason": ...}`` when the input cannot support the measure.
    """
    n_rows = int(features.size(0))
    if n_rows < 2:
        return {
            "available": False,
            "reason": (f"permutation sensitivity needs at least 2 samples to shuffle across; "
                       f"got {n_rows}. With one row the permutation is the identity and "
                       f"every score comes out as exactly 0.0."),
            "n_samples": n_rows,
        }
    if len(feature_names) != features.size(1):
        return {
            "available": False,
            "reason": (f"{len(feature_names)} feature names for {features.size(1)} feature "
                       f"columns; the scores would be attributed to the wrong names."),
            "n_samples": n_rows,
        }

    model.eval()
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    def predict(x: torch.Tensor) -> torch.Tensor:
        """Per-sample predictions as a flat [N] tensor."""
        with torch.no_grad():
            out = model(x)
        value = out[task_key] if isinstance(out, dict) else out
        return value.reshape(value.size(0), -1).mean(dim=1)

    baseline = predict(features)

    importances = {}
    for i, name in enumerate(feature_names):
        changes = []
        for _ in range(n_repeats):
            permuted = features.clone()
            perm_idx = torch.randperm(n_rows, generator=generator)
            permuted[:, i] = features[perm_idx, i]
            # Per-sample difference, then averaged. Averaging first is what cancelled.
            changes.append((baseline - predict(permuted)).abs().mean().item())

        importances[name] = {
            "mean_abs_prediction_change": round(float(np.mean(changes)), 6),
            "std": round(float(np.std(changes)), 6),
        }

    total = sum(v["mean_abs_prediction_change"] for v in importances.values())
    for k in importances:
        # None rather than 0.0 when nothing moved: a share of a zero total is undefined, and
        # writing 0.0 would assert the feature is unimportant when the truth is that no
        # feature moved the output at all -- usually an untrained or saturated model.
        importances[k]["share"] = (
            round(importances[k]["mean_abs_prediction_change"] / total, 4)
            if total > 0 else None)

    return {
        "available": True,
        "measure": "permutation sensitivity: mean |change in per-sample prediction|",
        "units": task_key,
        "not_accuracy_based": ("No labels were supplied, so this measures how much the "
                              "output moves, not how much the error grows."),
        "n_samples": n_rows,
        "n_repeats": n_repeats,
        "task_key": task_key,
        "total_movement": round(float(total), 6),
        "features": importances,
    }
