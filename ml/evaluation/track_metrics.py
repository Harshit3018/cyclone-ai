"""
Track-forecast verification, in the form operational centres actually report.

A single mean-error number hides what a forecast gets wrong. IMD and NHC verification
decomposes position error into two components that have different causes and different
consequences:

* **Cross-track error** -- perpendicular to the observed direction of motion. This is
  the "will it hit Odisha or West Bengal" error. It drives evacuation-zone decisions and
  is the component that matters for landfall.
* **Along-track error** -- parallel to the motion. This is a timing error: right place,
  wrong hour. It drives when to close ports, not where.

A model can post a respectable total error while being systematically slow (large
negative along-track bias), and only the decomposition reveals it.

Skill is always reported against a baseline, never in isolation:

    skill = (error_baseline - error_model) / error_baseline

Positive means the model beat the reference. A negative skill score against persistence
means the network is worse than assuming the storm keeps doing what it is doing, which is
the honest verdict on most under-trained track models and must be reported when it is
the outcome.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..baselines.physics import haversine_km, initial_bearing_deg


def great_circle_error_km(pred_lat, pred_lon, true_lat, true_lon) -> np.ndarray:
    """Element-wise great-circle distance between forecast and observed positions."""
    pred_lat = np.atleast_1d(np.asarray(pred_lat, dtype=np.float64))
    pred_lon = np.atleast_1d(np.asarray(pred_lon, dtype=np.float64))
    true_lat = np.atleast_1d(np.asarray(true_lat, dtype=np.float64))
    true_lon = np.atleast_1d(np.asarray(true_lon, dtype=np.float64))
    return np.array([
        haversine_km(a, b, c, d)
        for a, b, c, d in zip(pred_lat, pred_lon, true_lat, true_lon)
    ])


def along_cross_track_km(origin_lat: float, origin_lon: float,
                         pred_lat: float, pred_lon: float,
                         true_lat: float, true_lon: float) -> tuple[float, float]:
    """Decompose one position error into (along-track, cross-track) km.

    The reference direction is the *observed* motion from the analysis position to the
    verifying position -- not the forecast motion. Using the forecast's own heading would
    make the decomposition depend on the error being measured.

    Sign conventions:
        along  > 0  error points along the observed motion (forecast beyond the storm)
        cross  > 0  forecast is to the right of the observed track

    IMPORTANT -- along-track error is not a clean speed diagnostic when errors are
    large. A forecast with a perfect speed but a 90 degree direction error projects onto
    the reference axis as a large *negative* along-track value: with the truth 500 km due
    north and the forecast 500 km due east, this function returns along = -500 km, which
    reads as "500 km too slow" for a forecast that travelled exactly the right distance.
    The decomposition is only interpretable as timing-versus-position when the error is
    small compared with the displacement. In this basin that holds at +6 and +12 h, where
    mean error is 35-70 km against 79-154 km of displacement, but not at +48 h, where
    290 km of error sits against 544 km of displacement.

    Use ``HorizonScore.speed_bias_km`` for an unambiguous speed bias; it compares
    displacement magnitudes directly and is immune to this effect.
    """
    ref_bearing = initial_bearing_deg(origin_lat, origin_lon, true_lat, true_lon)
    err_km = haversine_km(true_lat, true_lon, pred_lat, pred_lon)
    if err_km == 0.0:
        return 0.0, 0.0
    # Bearing from the verifying position to the forecast position.
    err_bearing = initial_bearing_deg(true_lat, true_lon, pred_lat, pred_lon)
    theta = math.radians(err_bearing - ref_bearing)
    return err_km * math.cos(theta), err_km * math.sin(theta)


@dataclass
class HorizonScore:
    """Verification statistics at one forecast lead time."""
    horizon_h: int
    n: int
    mean_km: float
    median_km: float
    p90_km: float
    along_bias_km: float      # signed mean; negative = error points back toward origin
    cross_bias_km: float      # signed mean; positive = forecast right of observed track
    along_mae_km: float
    cross_mae_km: float
    # Unambiguous speed diagnostic: mean(|forecast displacement| - |observed
    # displacement|). Positive means the forecast moved the storm too far. Unlike
    # along_bias_km this is a pure magnitude comparison and cannot be contaminated by
    # direction error -- see the note on along_bias_km in along_cross_track_km.
    speed_bias_km: float = 0.0
    speed_mae_km: float = 0.0

    def __repr__(self) -> str:
        return (f"+{self.horizon_h}h n={self.n} mean={self.mean_km:.1f} km "
                f"median={self.median_km:.1f} speed_bias={self.speed_bias_km:+.1f} "
                f"cross_bias={self.cross_bias_km:+.1f}")


@dataclass
class Verification:
    """Full verification of one forecast method over a sample set."""
    name: str
    horizons_h: tuple[int, ...]
    scores: list[HorizonScore] = field(default_factory=list)
    n_samples: int = 0
    n_failed: int = 0          # samples the method could not produce a forecast for

    def mean_by_horizon(self) -> dict[int, float]:
        return {s.horizon_h: s.mean_km for s in self.scores}

    def table(self) -> str:
        lines = [f"{self.name}  ({self.n_samples} samples"
                 + (f", {self.n_failed} not forecastable)" if self.n_failed else ")")]
        lines.append(f"  {'lead':>5} {'n':>5} {'mean':>8} {'median':>8} {'p90':>8} "
                     f"{'spd_bias':>9} {'cross':>8}")
        for s in self.scores:
            lines.append(f"  {'+' + str(s.horizon_h) + 'h':>5} {s.n:>5} "
                         f"{s.mean_km:>7.1f}  {s.median_km:>7.1f}  {s.p90_km:>7.1f}  "
                         f"{s.speed_bias_km:>+8.1f}  {s.cross_bias_km:>+7.1f}")
        return "\n".join(lines)


def verify(name: str,
           origins: list[tuple[float, float]],
           predictions: list[dict[int, tuple[float, float]] | None],
           truths: list[dict[int, tuple[float, float]]],
           horizons_h: tuple[int, ...]) -> Verification:
    """Score a set of forecasts against observed positions.

    Args:
        origins: (lat, lon) analysis position per sample.
        predictions: per sample, {horizon_h: (lat, lon)}; None if the method declined
            to forecast that sample (too little history, for example). Declining is
            recorded rather than silently skipped, because a method that only forecasts
            the easy half of the cases would otherwise post a flattering mean.
        truths: per sample, {horizon_h: (lat, lon)} for horizons that verify. A horizon
            absent from a sample's truth dict is not scored for any method.
        horizons_h: lead times to score.

    A sample is scored for a horizon only when the truth exists *and* the method
    produced a forecast, so every method in a comparison is scored on the same cases
    provided none of them declined -- check ``n`` per horizon across methods to confirm.
    """
    v = Verification(name=name, horizons_h=tuple(horizons_h),
                     n_samples=len(origins))
    per_h: dict[int, list[tuple[float, float, float, float]]] = {h: [] for h in horizons_h}

    for (olat, olon), pred, truth in zip(origins, predictions, truths):
        if pred is None:
            v.n_failed += 1
            continue
        for h in horizons_h:
            if h not in truth or h not in pred:
                continue
            tlat, tlon = truth[h]
            plat, plon = pred[h]
            err = haversine_km(plat, plon, tlat, tlon)
            along, cross = along_cross_track_km(olat, olon, plat, plon, tlat, tlon)
            # Displacement magnitudes from the same origin: a direct speed comparison.
            speed_bias = (haversine_km(olat, olon, plat, plon)
                          - haversine_km(olat, olon, tlat, tlon))
            per_h[h].append((err, along, cross, speed_bias))

    for h in horizons_h:
        rows = per_h[h]
        if not rows:
            continue
        e = np.array([r[0] for r in rows])
        a = np.array([r[1] for r in rows])
        c = np.array([r[2] for r in rows])
        sb = np.array([r[3] for r in rows])
        v.scores.append(HorizonScore(
            horizon_h=h, n=len(rows),
            mean_km=float(e.mean()), median_km=float(np.median(e)),
            p90_km=float(np.percentile(e, 90)),
            along_bias_km=float(a.mean()), cross_bias_km=float(c.mean()),
            along_mae_km=float(np.abs(a).mean()), cross_mae_km=float(np.abs(c).mean()),
            speed_bias_km=float(sb.mean()), speed_mae_km=float(np.abs(sb).mean()),
        ))
    return v


def skill_score(model: Verification, reference: Verification) -> dict[int, float]:
    """Fractional error reduction of `model` relative to `reference`, per horizon.

    Positive is better than the reference. Returns a fraction, not a percentage.
    Horizons where either side has no scored samples are omitted rather than reported
    as zero skill.
    """
    m, r = model.mean_by_horizon(), reference.mean_by_horizon()
    out = {}
    for h in sorted(set(m) & set(r)):
        if r[h] <= 0:
            continue
        out[h] = (r[h] - m[h]) / r[h]
    return out


def paired_significance(model_errors: dict[int, np.ndarray],
                        reference_errors: dict[int, np.ndarray],
                        n_resamples: int = 10000,
                        seed: int = 0) -> dict[int, dict]:
    """Is a difference in mean error real, or is it the sample size?

    Track-forecast sample sets in this basin are small -- 113 verifying cases at +48 h in
    the 2024-2025 test split -- and a 4% mean-error difference over 113 paired cases is
    routinely produced by chance. Reporting such a difference as "our model beats CLIPER"
    without a significance test is the most common way track-forecast results are
    oversold, so the test lives in the pipeline rather than in a notebook someone ran once.

    Two complementary procedures on the paired per-case error differences
    (model - reference, so negative favours the model):

    * **Paired bootstrap** of the mean difference, giving a 95% interval. If the interval
      contains zero, the sign of the difference is not established.
    * **Sign-flip permutation test**, which is exact under the null that the two methods
      are exchangeable on each case. Distribution-free, so the long tail of track errors
      does not invalidate it the way it would a t-test.

    Both are paired: each case contributes one difference, which removes the
    case-difficulty variance that dominates the raw error spread. An unpaired comparison
    of these two methods would have far less power and would be the wrong test.

    Note on multiplicity: with four horizons and two splits, eight comparisons are made,
    so an isolated p just under 0.05 is unremarkable. ``significant`` is reported per
    horizon without a multiplicity correction; interpret a single hit accordingly.
    """
    rng = np.random.default_rng(seed)
    out: dict[int, dict] = {}
    for h in sorted(set(model_errors) & set(reference_errors)):
        m = np.asarray(model_errors[h], dtype=np.float64)
        r = np.asarray(reference_errors[h], dtype=np.float64)
        if m.shape != r.shape or m.size == 0:
            continue
        diff = m - r
        n = diff.size
        boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_resamples)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        flips = np.array([(diff * rng.choice([-1.0, 1.0], n)).mean()
                          for _ in range(n_resamples)])
        p = float((np.abs(flips) >= abs(diff.mean())).mean())
        out[h] = {
            "n": int(n),
            "mean_difference_km": float(diff.mean()),
            "ci95_km": [float(lo), float(hi)],
            "p_value": p,
            "significant": bool(p < 0.05 and lo * hi > 0),
            "favours": ("model" if diff.mean() < 0 else "reference"),
        }
    return out


def significance_table(sig: dict[int, dict], model_name: str, reference_name: str) -> str:
    """Render :func:`paired_significance` output, with an explicit verdict line."""
    if not sig:
        return "no paired comparisons available"
    lines = [f"{model_name} vs {reference_name} -- paired, per verifying case",
             f"  {'lead':>5} {'n':>5} {'delta_km':>10} {'95% CI':>20} {'p':>7}  verdict"]
    for h, d in sig.items():
        lo, hi = d["ci95_km"]
        verdict = ("significant, favours " + d["favours"]) if d["significant"] \
            else "not distinguishable"
        lines.append(f"  {'+' + str(h) + 'h':>5} {d['n']:>5} "
                     f"{d['mean_difference_km']:>+10.1f} "
                     f"[{lo:>+7.1f},{hi:>+7.1f}] {d['p_value']:>7.3f}  {verdict}")
    n_sig = sum(1 for d in sig.values() if d["significant"])
    lines.append(f"  => {n_sig} of {len(sig)} horizons show a statistically "
                 f"distinguishable difference (negative delta favours {model_name})")
    return "\n".join(lines)


def per_case_errors(origins: list[tuple[float, float]],
                    predictions: list[dict[int, tuple[float, float]] | None],
                    truths: list[dict[int, tuple[float, float]]],
                    horizons_h) -> dict[int, np.ndarray]:
    """Per-case great-circle errors keyed by horizon, aligned across methods.

    Only cases where the truth exists are included, in sample order, so two methods
    scored through this function produce arrays that can be paired element-wise. A method
    that declines to forecast a case would break that alignment, so such cases are
    excluded from every horizon by returning NaN-free arrays only for cases both the
    truth and the prediction cover -- callers comparing two methods must therefore pass
    methods that forecast the same cases (persistence, CLIPER and the network all do).
    """
    out: dict[int, list[float]] = {h: [] for h in horizons_h}
    for (olat, olon), pred, truth in zip(origins, predictions, truths):
        for h in horizons_h:
            if h not in truth or pred is None or h not in pred:
                continue
            tlat, tlon = truth[h]
            plat, plon = pred[h]
            out[h].append(haversine_km(plat, plon, tlat, tlon))
    return {h: np.asarray(v, dtype=np.float64) for h, v in out.items() if v}


def comparison_table(verifications: list[Verification],
                     reference: Verification | None = None) -> str:
    """Side-by-side mean error per horizon, with skill against `reference`.

    This is the table that belongs in an SIH submission: several methods, the same
    held-out samples, one honest number each, and the sample count visible so nobody has
    to wonder whether the best row was scored on fewer cases.
    """
    if not verifications:
        return "no verifications to compare"
    horizons = sorted({h for v in verifications for h in v.mean_by_horizon()})
    width = max(len(v.name) for v in verifications) + 2

    head = f"{'method':<{width}}" + "".join(f"{'+' + str(h) + 'h':>11}" for h in horizons)
    lines = ["Mean great-circle track error, km (lower is better)", head,
             "-" * len(head)]
    for v in verifications:
        m = v.mean_by_horizon()
        row = f"{v.name:<{width}}" + "".join(
            (f"{m[h]:>11.1f}" if h in m else f"{'--':>11}") for h in horizons)
        lines.append(row)

    counts = {}
    for v in verifications:
        for s in v.scores:
            counts.setdefault(s.horizon_h, set()).add(s.n)
    n_row = f"{'(samples)':<{width}}" + "".join(
        (f"{min(counts[h]):>11}" if h in counts else f"{'--':>11}") for h in horizons)
    lines.append(n_row)

    if reference is not None:
        lines += ["", f"Skill vs {reference.name} (positive = better)", head,
                  "-" * len(head)]
        for v in verifications:
            if v.name == reference.name:
                continue
            sk = skill_score(v, reference)
            row = f"{v.name:<{width}}" + "".join(
                (f"{100 * sk[h]:>10.1f}%" if h in sk else f"{'--':>11}")
                for h in horizons)
            lines.append(row)
    return "\n".join(lines)
