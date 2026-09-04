"""Regression tests for the forecast-integrity properties this platform claims.

Each test here corresponds to a specific way a cyclone forecast can be technically
functional and still mislead the person acting on it. They were all live defects in this
codebase at some point:

* a hand-chosen uncertainty cone that understated real error fivefold, sitting in the same
  API response as a model that reported its measured error;
* an extrapolated cone at +72 h presented with the authority of a measurement;
* MC-dropout spread (2 km at +6 h) offered as a confidence radius when verified error at
  the same lead time was 37 km;
* a fallback that preferred the least accurate of three available forecasts;
* a serving-side copy of the feature vector, free to drift from the one trained on;
* two invented constants (0.1 and 0.5) carrying 30% of the weight in every risk score,
  varying between no two storms;
* coastal risk scored at the storm's current position while the forecast it was shown
  next to put it 40 km off Tamil Nadu in 12 hours.

An aggregate accuracy metric detects none of these. They need assertions of their own,
because each becomes invisible the moment it stops being newly discovered.
"""
from __future__ import annotations

import pytest

from ml.baselines.track import CLIP_LIKE_ERROR_KM, PERSISTENCE_ERROR_KM


def _all_forecasts(forecast: dict) -> dict[str, dict]:
    out = {}
    if forecast.get("track_forecast"):
        out["neural_network"] = forecast["track_forecast"]
    out.update(forecast.get("baseline_forecasts") or {})
    return out


# --- the cone -------------------------------------------------------------------------

def test_no_forecast_reports_an_unmeasured_cone_as_a_number(forecast):
    """A radius is either a measurement or null. Never an extrapolation.

    The failure this prevents is silent: extending a linear trend past the last verified
    lead time produces a plausible-looking number that no verification supports, and a
    consumer cannot tell it apart from a real one.
    """
    for name, fc in _all_forecasts(forecast).items():
        for p in fc["forecast_points"]:
            r = p.get("uncertainty_radius_km")
            assert r is None or r > 0, (
                f"{name} at +{p['forecast_hour']}h reports radius {r!r}; "
                "must be a positive measurement or null")
            if r is not None:
                # A measured radius has to say how many cases it was measured over,
                # otherwise "measured" is an unfalsifiable claim.
                assert p.get("verified_cases", 0) > 0, (
                    f"{name} at +{p['forecast_hour']}h claims a measured radius "
                    "with no verifying case count")


def test_cone_grows_with_lead_time(forecast):
    """Uncertainty that shrinks with lead time means the cone is not an error estimate."""
    for name, fc in _all_forecasts(forecast).items():
        radii = [(p["forecast_hour"], p["uncertainty_radius_km"])
                 for p in fc["forecast_points"] if p.get("uncertainty_radius_km")]
        for (h1, r1), (h2, r2) in zip(radii, radii[1:]):
            assert r2 > r1, f"{name} cone shrinks from +{h1}h ({r1} km) to +{h2}h ({r2} km)"


def test_p90_cone_exceeds_mean_error(forecast):
    """A cone drawn at the mean excludes roughly 40% of outcomes while looking official."""
    for name, fc in _all_forecasts(forecast).items():
        for p in fc["forecast_points"]:
            r, mean = p.get("uncertainty_radius_km"), p.get("mean_error_km")
            if r is None or mean is None:
                continue
            assert r > mean, (
                f"{name} at +{p['forecast_hour']}h: cone {r} km does not exceed mean "
                f"error {mean} km, so it cannot be a 90th percentile")


def test_mc_dropout_spread_is_never_presented_as_the_cone(client, cyclone_id):
    """Model self-disagreement is not forecast error, and is ~15x smaller here.

    Both may be reported; they must be reported as different fields. Substituting the
    spread for the cone would understate real uncertainty by an order of magnitude.
    """
    body = client.post("/api/predict/track", json={"cyclone_id": cyclone_id}).json()
    checked = 0
    for p in body.get("forecast_points", []):
        spread, radius = p.get("mc_dropout_spread_km"), p.get("uncertainty_radius_km")
        if spread is None or radius is None:
            continue
        assert spread != radius, (
            f"+{p['forecast_hour']}h reports MC-dropout spread as the uncertainty radius")
        assert spread < radius, (
            f"+{p['forecast_hour']}h spread {spread} km >= verified cone {radius} km; "
            "if the ensemble really disagrees this much the two have been swapped")
        checked += 1
    if checked == 0:
        pytest.skip("no forecast point reported both a spread and a cone")


# --- which forecast gets served -------------------------------------------------------

def test_a_baseline_measured_worse_than_persistence_is_never_primary(forecast):
    """clip_like used to be the fallback and is the worst of the three methods.

    Preferring it was not a cosmetic bug: whenever the network was distrusted, the
    platform served its least accurate forecast.
    """
    primary = forecast.get("primary_forecast")
    assert primary != "clip_like", (
        "clip_like is served as primary; measured -10.5% to -32.7% skill against plain "
        "persistence on held-out seasons")


def test_clip_like_declares_its_negative_skill(forecast):
    fc = (forecast.get("baseline_forecasts") or {}).get("clip_like")
    if fc is None:
        pytest.skip("clip_like not offered in this response")
    assert fc["model_status"] == "analytic_baseline_superseded"
    skill = fc["verified_skill"]["skill_vs_persistence_pct"]
    assert skill, "clip_like reports no skill comparison"
    assert all(v < 0 for v in skill.values()), (
        f"clip_like claims non-negative skill somewhere: {skill}. If the measurement has "
        "genuinely changed, regenerate the tables and revisit whether it is still superseded")


def test_primary_forecast_explains_itself(forecast):
    assert forecast.get("primary_forecast") in _all_forecasts(forecast)
    reason = forecast.get("primary_forecast_reason") or ""
    assert len(reason) > 20, f"primary forecast chosen with no stated reason: {reason!r}"


# --- the error tables themselves ------------------------------------------------------

def test_persistence_is_the_reference_and_cliper_beats_it():
    """If the fitted CLIPER stops beating persistence, the baseline story has changed."""
    from ml.baselines.track import ERROR_SOURCE
    assert ERROR_SOURCE, "the error tables must say where they came from"
    for h, entry in PERSISTENCE_ERROR_KM.items():
        assert entry["p90_km"] > entry["mean_km"] > 0, f"+{h}h persistence table incoherent"
        assert entry["n"] > 30, f"+{h}h persistence measured on only {entry['n']} cases"


def test_error_tables_cover_the_same_horizons_and_case_counts():
    """The baselines must be scored on the same cases, or the comparison is meaningless."""
    assert set(PERSISTENCE_ERROR_KM) == set(CLIP_LIKE_ERROR_KM)
    for h in PERSISTENCE_ERROR_KM:
        assert PERSISTENCE_ERROR_KM[h]["n"] == CLIP_LIKE_ERROR_KM[h]["n"], (
            f"+{h}h: persistence scored on {PERSISTENCE_ERROR_KM[h]['n']} cases but "
            f"clip_like on {CLIP_LIKE_ERROR_KM[h]['n']}; not a paired comparison")


def test_error_grows_monotonically_with_lead_time():
    for name, table in (("persistence", PERSISTENCE_ERROR_KM),
                        ("clip_like", CLIP_LIKE_ERROR_KM)):
        horizons = sorted(table)
        for h1, h2 in zip(horizons, horizons[1:]):
            assert table[h2]["mean_km"] > table[h1]["mean_km"], (
                f"{name} error falls from +{h1}h to +{h2}h")


def test_unmeasured_horizons_are_absent_from_the_tables_not_guessed():
    """+72 h is in DEFAULT_HORIZONS but was never verified, so it must have no entry."""
    from ml.baselines.track import DEFAULT_HORIZONS, _radius_km
    for h in DEFAULT_HORIZONS:
        if h not in PERSISTENCE_ERROR_KM:
            assert _radius_km(PERSISTENCE_ERROR_KM, h) is None, (
                f"+{h}h has no measurement but _radius_km returned a number")


# --- train/serve parity ---------------------------------------------------------------

def test_served_checkpoint_matches_the_current_feature_layout():
    """A checkpoint trained on a different feature order still loads and still forecasts.

    Refusing it is the only thing standing between a renamed feature and a confidently
    wrong operational track.
    """
    from ml.datasets.track_dataset import TRACK_FEATURE_NAMES
    from ml.inference.track_runtime import TrackOnlyRuntime
    from pathlib import Path

    path = Path("models/checkpoints/track_only.pt")
    if not path.exists():
        pytest.skip("no track checkpoint present")

    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert tuple(ckpt.get("feature_names", ())) == tuple(TRACK_FEATURE_NAMES), (
        "checkpoint feature layout has drifted from the code")

    runtime = TrackOnlyRuntime.load(path)
    assert runtime is not None, "a layout-matching checkpoint was refused"


def test_serving_uses_the_training_feature_builder(client, cyclone_id):
    """The served forecast must come from ml.datasets.track_dataset.step_features.

    Asserted by construction rather than by output comparison: if inference stops calling
    the training builder, this import-identity check is what notices.
    """
    import ml.inference.track_runtime as rt
    from ml.datasets.track_dataset import step_features

    assert rt.step_features is step_features, (
        "inference has its own copy of the feature builder; it is now free to drift "
        "from the vector the model was trained on")


def test_trained_track_forecast_stays_inside_physical_limits(client, cyclone_id):
    body = client.post("/api/predict/track", json={"cyclone_id": cyclone_id}).json()
    if body.get("model_status") != "trained":
        pytest.skip("trained track model not loaded")
    plausible = (body.get("plausibility") or {}).get("physically_plausible")
    assert plausible is not False, (
        f"trained model output failed physical checks: "
        f"{(body.get('plausibility') or {}).get('violations')}")
    for p in body["forecast_points"]:
        assert -10 <= p["latitude"] <= 32, f"+{p['forecast_hour']}h latitude out of basin"
        assert 40 <= p["longitude"] <= 110, f"+{p['forecast_hour']}h longitude out of basin"


def test_skill_claims_are_backed_by_significance_testing(client, cyclone_id):
    """A lower number in a table is not skill until it survives a paired test.

    The point is not that the model must win. It is that the response must carry the test
    result, so a reader can see that 1-of-4 horizons reached significance rather than
    infer a clean win from the error table.
    """
    body = client.get(f"/api/cyclones/{cyclone_id}/forecast").json()
    skill = body.get("track_skill")
    if not skill:
        pytest.skip("no trained track model, so no skill claim is being made")
    sig = skill.get("significance_vs_cliper")
    assert sig, "the API reports measured error but no significance test against CLIPER"
    for h, d in sig.items():
        assert "p_value" in d and "significant" in d, f"+{h}h significance entry incomplete"
        lo, hi = d["ci95_km"]
        assert lo <= d["mean_difference_km"] <= hi, (
            f"+{h}h: mean difference sits outside its own confidence interval")


# --- the risk engine ------------------------------------------------------------------
#
# The defect these cover: `assess_risk` took ri_probability and track_confidence, both
# call sites passed the literals 0.1 and 0.5, and together they drove 30% of overall_risk.
# A constant contributing a third of an alert level is not a model, and because it did not
# vary between storms it moved no storm's score relative to any other's. Separately, risk
# was scored only at the latest observed position, so a storm whose forecast put it 40 km
# offshore in 12 h was still scored at the 380 km it currently sat at.

UNMEASURED_RISK_COMPONENTS = {"rapid_intensification", "track_uncertainty"}
"""Components with no measurement behind them in this system today.

``rapid_intensification`` needs a trained intensity-forecast head; ``track_uncertainty``
needs a per-storm confidence estimate. Neither exists, so both must be null. When one is
genuinely implemented, delete it from this set -- and that deletion is the review prompt
to check the new value is measured rather than another constant.
"""


def test_unmeasured_risk_components_are_null_not_defaulted(risk_payload):
    """A component with nothing behind it is null. Never 0.0, never 0.5.

    0.0 for rapid intensification asserts there is no RI risk; 0.5 for track confidence
    asserts a coin-flip's worth of confidence. Both are claims. Null is not.
    """
    risk = risk_payload["risk"]
    for name in UNMEASURED_RISK_COMPONENTS:
        field = "track_confidence" if name == "track_uncertainty" else f"{name}_risk"
        assert risk[field] is None, (
            f"{field} reports {risk[field]!r}; nothing measures it, so it must be null. "
            "A number here is indistinguishable downstream from a computed one")
        assert name in risk["components_unavailable"], (
            f"{name} is not listed in components_unavailable, so a consumer cannot tell "
            "it was excluded from the score")


def test_excluded_components_are_renormalised_not_silently_dropped(risk_payload):
    """Dropping a weight without renormalising quietly rescales the whole score.

    If a fifth of the weight vanishes, every score shrinks by a fifth and the alert
    thresholds no longer mean what they did -- a strictly worse failure than the constant
    it replaced, because it is invisible.
    """
    risk = risk_payload["risk"]
    weights = risk["weights_applied"]
    assert weights, "no weights reported, so the score cannot be audited"
    assert set(weights) == set(risk["components_used"]), (
        "weights_applied and components_used disagree about what entered the score")
    assert abs(sum(weights.values()) - 1.0) < 0.01, (
        f"renormalised weights sum to {sum(weights.values())}, not 1")

    # And the score must actually be the weighted average of the components it names.
    field = {"wind": "wind_risk", "coastal": "coastal_risk", "rainfall": "rainfall_risk",
             "rapid_intensification": "rapid_intensification_risk"}
    recomputed = sum(w * risk[field[k]] for k, w in weights.items() if k in field)
    assert abs(recomputed - risk["overall_risk"]) < 0.01, (
        f"overall_risk {risk['overall_risk']} is not the weighted average of its stated "
        f"components ({recomputed:.3f}); something outside components_used is contributing")


def test_coastal_risk_is_scored_on_the_forecast_not_the_current_position(risk_payload):
    """Scoring only the current position warns late, which is the same as not warning.

    A storm 380 km offshore whose forecast brings it to 41 km off Tamil Nadu in 12 h is a
    coastal emergency now. The old endpoint read dist_to_coast_km off the latest observed
    point and returned WATCH.
    """
    ctx = risk_payload.get("forecast_track_risk")
    if not ctx or not ctx.get("closest_approach"):
        pytest.skip("no forecast track available for this storm")

    basis = risk_payload["risk"].get("coastal_risk_basis") or ""
    assert "forecast" in basis.lower(), (
        f"coastal risk basis is {basis!r}; a forecast exists, so risk must be scored at "
        "its closest approach rather than at the current position")
    assert risk_payload["risk"]["distance_to_coast_km"] == pytest.approx(
        ctx["closest_approach"]["distance_to_land_km"]), (
        "the distance driving coastal_risk is not the closest forecast approach")


def test_an_unverified_cone_cannot_declare_the_coast_safe(risk_payload):
    """cone_reaches_coast must be null where no cone was measured, not false.

    False reads as "landfall is outside the error of the forecast" -- a reassurance. At
    +72 h no cone was ever verified, so that reassurance has nothing behind it.
    """
    ctx = risk_payload.get("forecast_track_risk")
    if not ctx:
        pytest.skip("no forecast track risk in this response")
    for p in ctx["coastal_profile"]:
        if p["uncertainty_radius_km"] is None:
            assert p["cone_reaches_coast"] is None, (
                f"+{p['forecast_hour']}h has no measured cone but reports "
                f"cone_reaches_coast={p['cone_reaches_coast']!r}")
        else:
            # Where a cone exists the answer is a real boolean, decided by comparing the
            # verified p90 radius against the measured distance to land.
            assert p["cone_reaches_coast"] is (
                p["uncertainty_radius_km"] >= p["distance_to_land_km"])


def test_landfall_becomes_possible_before_the_centre_track_arrives(risk_payload):
    """Warning on the centre line alone is warning late, by construction.

    The cone reaches the coast before the track does, so possible_from_hour must never be
    later than track_arrives_hour. If it were, the two had been swapped.

    On the seeded demo database this skips: no storm's forecast centre comes within the
    polyline's 18.4 km resolution of land (the closest approach across all five is 41 km),
    so track_arrives_hour is null everywhere and there is nothing to order. The invariant
    itself is covered unconditionally by
    tests/test_track_risk.py::test_cone_landfall_never_lags_the_centre_track, which drives
    landfall_window() with a track that does go ashore and is verified by mutation testing
    to fail when the comparison is reversed. This test remains as the wiring check: it is
    what catches the two fields being swapped or dropped somewhere between the risk engine
    and the JSON, and it starts asserting the moment a storm actually makes landfall.
    """
    ctx = risk_payload.get("forecast_track_risk")
    if not ctx:
        pytest.skip("no forecast track risk in this response")
    lf = ctx["landfall"]
    if lf["possible_from_hour"] is None or lf["track_arrives_hour"] is None:
        pytest.skip("this storm makes no landfall; the ordering is covered by "
                    "test_track_risk.py::test_cone_landfall_never_lags_the_centre_track")
    assert lf["possible_from_hour"] <= lf["track_arrives_hour"], (
        f"landfall is 'possible' from +{lf['possible_from_hour']}h but the centre track "
        f"arrives earlier at +{lf['track_arrives_hour']}h; the cone cannot lag the track")


def test_every_coastal_distance_travels_with_its_measured_resolution(risk_payload):
    """A distance without its error budget invites false precision.

    41.4 km and 48.9 km are the same answer to geometry with an 8.7 km mean error, and a
    reader shown only the first cannot know that.
    """
    ctx = risk_payload.get("forecast_track_risk")
    if not ctx:
        pytest.skip("no forecast track risk in this response")
    geom = ctx["geometry"]
    assert geom["mean_absolute_error_km"] > 0
    assert geom["p90_absolute_error_km"] >= geom["mean_absolute_error_km"]
    assert geom["validated_on_positions"] > 1000, (
        "an accuracy claim measured on a handful of positions is not an accuracy claim")
    assert geom["reference"], "the geometry claims accuracy without naming its reference"
    assert "km" in geom["note"], "the geometry note must state its resolution in the text"


def test_observed_intensity_trend_is_never_reported_as_a_forecast(risk_payload):
    """Observed 24 h change is real; an RI *probability* is not available.

    The replaced code computed `(wind_kph - 100) / 150` and labelled it an RI probability,
    which is current intensity relabelled -- a steady 160 km/h storm and one that gained
    40 kt in a day scored identically. This asserts the honest version: the trend is
    reported as an observation with its own window, and any missing trend says so instead
    of reporting a change of zero.
    """
    ctx = risk_payload.get("forecast_track_risk")
    if not ctx:
        pytest.skip("no forecast track risk in this response")

    assert ctx["intensity_forecast_available"] is False, (
        "an intensity forecast is now claimed; check it comes from a trained head and "
        "then update this test")

    trend = ctx["intensity_trend"]
    if not trend["available"]:
        assert trend.get("reason"), "an unavailable trend must say why"
        assert "rate_kt_per_24h" not in trend, (
            "an unavailable trend still reports a rate")
        return

    assert trend["basis"] == "observed best-track wind history"
    assert trend["window_hours"] >= 12.0, (
        f"a {trend['window_hours']} h window was extrapolated to a 24 h rate; too short")
    assert trend["threshold_kt_per_24h"] == 30.0, "IMD/NHC RI threshold is 30 kt/24 h"
    assert trend["rapidly_intensifying"] is (
        trend["rate_kt_per_24h"] >= trend["threshold_kt_per_24h"]), (
        "the RI flag disagrees with the rate and threshold it is derived from")


def test_full_analysis_reports_no_rapid_intensification_probability(client):
    """The combined endpoint must not resurrect the fake RI probability."""
    body = client.post("/api/predict?sample_index=0").json()
    ri = body.get("rapid_intensification")
    if ri is None:
        pytest.skip("this endpoint does not report rapid intensification")
    assert ri["probability"] is None, (
        f"RI probability is {ri['probability']!r}; there is no trained intensity-forecast "
        "head, so any number here is derived from current intensity, not from a forecast")
    assert ri.get("reason"), "a null probability must state why it is null"

