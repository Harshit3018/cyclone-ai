"""Unit tests for ml/inference/track_risk.py.

Why these exist alongside the API-level tests in test_forecast_integrity.py
--------------------------------------------------------------------------
The API tests guard the wiring: that the risk route really scores the forecast track, that
nulls survive serialisation, that the geometry note travels with the numbers. But they can
only assert what the demo data happens to exercise, and it does not exercise much. No demo
storm's forecast centre gets within 18.4 km of land -- the closest approach across all five
is 41 km -- so the API-level check that landfall becomes possible before the centre arrives
skips every single run. A test that can never execute is not a test.

These drive the pure functions directly with synthetic tracks and wind histories, which is
the only way to reach the branches that matter: a track that actually crosses the coast, a
wind record too short to support a 24 h rate, a lead time with no verified cone, an
out-of-order forecast list.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from ml.geo.coastline import MEASURED_ACCURACY
from ml.inference import track_risk as tr

T0 = datetime(2020, 5, 18, 0, 0, 0)
RESOLUTION_KM = float(MEASURED_ACCURACY["p90_abs_error_km"])


def _pt(hour: float, lat: float, lon: float, radius: float | None):
    return {"forecast_hour": hour, "latitude": lat, "longitude": lon,
            "uncertainty_radius_km": radius}


def _hist(n: int, step_h: float, w0: float, dw: float):
    return [{"timestamp": T0 + timedelta(hours=step_h * i), "wind_kph": w0 + dw * i}
            for i in range(n)]


# --- landfall window ------------------------------------------------------------------

def test_cone_landfall_never_lags_the_centre_track():
    """The invariant the API test can only skip on this demo data.

    A track running northwest into the Odisha coast, with cones from the verified error
    table. The cone must reach land at or before the hour the centre does -- if it did not,
    the cone and the centre distance had been compared the wrong way round, and the system
    would warn only once the centre line hit the coast, which is warning late.
    """
    profile = tr.coastal_profile([
        _pt(6, 17.0, 88.0, 37.1),     # 389 km offshore, cone falls well short
        _pt(12, 18.0, 87.0, 74.0),    # 237 km, still short
        _pt(24, 19.0, 86.3, 150.0),   # 103 km, cone now overlaps the coast
        _pt(48, 19.75, 85.70, 300.0),  # 0.5 km -- ashore near Puri
    ])
    lf = tr.landfall_window(profile)

    assert lf["track_arrives_hour"] is not None, (
        "this track was built to reach the coast; if it no longer does, the geometry moved")
    assert lf["possible_from_hour"] is not None
    assert lf["possible_from_hour"] <= lf["track_arrives_hour"]
    # Not merely <=: the whole reason to test the cone rather than the centre line is the
    # lead time it buys. Here the cone touches Odisha a full day before the centre does.
    assert lf["possible_from_hour"] == 24 and lf["track_arrives_hour"] == 48
    # `possible_coast` carries the human-readable label, not the internal key -- it is
    # rendered straight into the verdict a forecaster reads.
    assert lf["possible_coast"] == "mainland coast"
    assert lf["track_arrives_coast"] == "mainland coast"
    assert lf["geometry_resolution_km"] == pytest.approx(RESOLUTION_KM)


def test_track_arrival_uses_the_geometry_resolution_not_zero():
    """Requiring distance == 0 would mean the centre never "arrives".

    The polyline's p90 error is ~18 km, so a centre 8 km offshore is indistinguishable from
    one on the beach. Demanding an exact zero would report no landfall for a storm visibly
    over land. The pair below straddles the measured resolution -- 8.5 km in, 34 km out --
    rather than testing 0 against the open ocean, which any threshold at all would pass.
    """
    inside = tr.coastal_profile([_pt(6, 19.6, 85.6, 40.0)])     # 8.5 km off Odisha
    outside = tr.coastal_profile([_pt(6, 19.5, 85.9, 40.0)])    # 34.1 km off

    assert 0 < inside[0]["distance_to_land_km"] < RESOLUTION_KM
    assert outside[0]["distance_to_land_km"] > RESOLUTION_KM

    assert tr.landfall_window(inside)["track_arrives_hour"] == 6
    assert tr.landfall_window(outside)["track_arrives_hour"] is None


def test_unverified_horizon_does_not_declare_the_coast_out_of_reach():
    """+72 h has no measured cone, so it must not answer the question either way."""
    profile = tr.coastal_profile([_pt(72, 18.0, 87.0, None)])
    assert profile[0]["cone_reaches_coast"] is None

    lf = tr.landfall_window(profile)
    assert lf["possible_from_hour"] is None
    assert "does not reach any coast" in lf["verdict"] or lf["track_arrives_hour"] is None


def test_landfall_window_is_computed_in_lead_time_order():
    """"The first hour at which" is only meaningful on a sorted sequence.

    Handed the same positions in a different order, the answer must not change. Reversed is
    the ordering that actually distinguishes: scanning it unsorted returns +48 h as the
    first cone landfall when the true answer is +24 h -- a day of warning thrown away. An
    arbitrary shuffle is a weaker probe, since many shuffles happen to return the right
    row by luck; mutation testing showed one such shuffle passing against an unsorted
    implementation, so both are checked here.
    """
    pts = [_pt(6, 17.0, 88.0, 37.1), _pt(12, 18.0, 87.0, 74.0),
           _pt(24, 19.0, 86.3, 150.0), _pt(48, 19.75, 85.70, 300.0)]
    expected = tr.landfall_window(tr.coastal_profile(pts))

    assert expected["possible_from_hour"] == 24 and expected["track_arrives_hour"] == 48, (
        "the fixture no longer distinguishes cone landfall from centre arrival")

    for label, order in (("reversed", pts[::-1]),
                         ("shuffled", [pts[2], pts[0], pts[3], pts[1]]),
                         ("last first", [pts[3], pts[1], pts[0], pts[2]])):
        assert tr.landfall_window(tr.coastal_profile(order)) == expected, (
            f"{label} input produced a different landfall window; lead times are being "
            f"read in the order supplied rather than in time order")


def test_no_landfall_anywhere_says_so_explicitly():
    """Mid-Arabian-Sea, small cones. The verdict must be a statement, not an empty dict."""
    lf = tr.landfall_window(tr.coastal_profile([_pt(6, 14.0, 64.0, 37.1),
                                               _pt(12, 14.5, 63.5, 74.0)]))
    assert lf["possible_from_hour"] is None and lf["track_arrives_hour"] is None
    assert "No landfall within the forecast horizon" in lf["verdict"]


# --- closest approach -----------------------------------------------------------------

def test_closest_approach_picks_the_minimum_not_the_last_point():
    """A storm that grazes the coast and moves back out peaks in the middle of the track.

    Reading the final position -- or the current one -- would miss the approach entirely,
    which is the specific failure the old risk endpoint had.
    """
    profile = tr.coastal_profile([
        _pt(6, 16.0, 84.0, 37.1),
        _pt(12, 17.2, 83.0, 74.0),    # closest to the Andhra coast
        _pt(24, 16.0, 80.0, 150.0),
        _pt(48, 13.0, 76.0, 300.0),
    ])
    closest = tr.closest_approach(profile)
    assert closest["forecast_hour"] == 12
    assert closest["distance_to_land_km"] == min(
        r["distance_to_land_km"] for r in profile)


def test_closest_approach_names_the_coast_it_refers_to():
    """"180 km from land" means something different for Odisha than for an islet."""
    sri = tr.closest_approach(tr.coastal_profile([_pt(6, 7.5, 81.0, 37.1)]))
    assert sri["nearest_coast"] == "sri_lanka"
    assert sri["nearest_coast_label"] == "Sri Lanka"

    andaman = tr.closest_approach(tr.coastal_profile([_pt(6, 11.5, 92.9, 37.1)]))
    assert andaman["nearest_coast"] == "andaman_nicobar"


def test_empty_and_malformed_forecasts_do_not_raise():
    assert tr.closest_approach([]) is None
    assert tr.coastal_profile([]) == []
    # A point missing coordinates is skipped rather than crashing the risk endpoint.
    assert tr.coastal_profile([{"forecast_hour": 6, "latitude": None,
                               "longitude": 87.0, "uncertainty_radius_km": 37.1}]) == []


# --- intensity trend ------------------------------------------------------------------

def test_missing_wind_history_is_unavailable_not_a_change_of_zero():
    """"No data" and "no change" are opposite conclusions for a forecaster.

    Reporting 0.0 kt for a storm with no wind record asserts it is holding steady.
    """
    for history, label in (
        ([], "empty"),
        (_hist(1, 6, 60, 0), "single observation"),
        ([{"timestamp": T0, "wind_kph": None}], "wind is null"),
        ([{"timestamp": None, "wind_kph": 60}, {"timestamp": None, "wind_kph": 90}],
         "no timestamps"),
        # Two observations stamped at the same instant. A naive rate divides by a zero
        # span; found by mutation testing, which showed this branch had no test at all.
        ([{"timestamp": T0, "wind_kph": 60}, {"timestamp": T0, "wind_kph": 130}],
         "zero span"),
    ):
        out = tr.intensity_trend(history)
        assert out["available"] is False, f"{label} produced an available trend"
        assert out.get("reason"), f"{label} gave no reason"
        assert "rate_kt_per_24h" not in out, f"{label} still reported a rate"
        assert "rapidly_intensifying" not in out, f"{label} still flagged RI"


def test_a_short_window_is_not_extrapolated_to_a_24_hour_rate():
    """A 3-hour trend multiplied by eight is an artefact, not a rate.

    +20 km/h over 3 h scales to ~86 kt/24 h, which would trip the RI flag on two
    observations six hours apart. Refusing is the honest answer.
    """
    out = tr.intensity_trend(_hist(2, 3, 60, 20))
    assert out["available"] is False
    assert "too short" in out["reason"]
    # The observed change is still reported -- it is a real measurement, just not a rate.
    assert out["observed_change_kt"] == pytest.approx(20 * tr.KT_PER_KPH, abs=0.1)
    assert "rate_kt_per_24h" not in out


def test_rapid_intensification_flag_matches_the_imd_threshold():
    """RI is >= 30 kt in 24 h. The flag must be exactly that comparison."""
    assert tr.RI_THRESHOLD_KT_24H == 30.0

    # +25 km/h per 6 h over 18 h = 75 km/h = 40.5 kt, scaled to 54 kt/24 h.
    fast = tr.intensity_trend(_hist(4, 6, 80, 25))
    assert fast["available"] and fast["rapidly_intensifying"] is True
    assert fast["rate_kt_per_24h"] >= tr.RI_THRESHOLD_KT_24H

    # A steady storm at high intensity is NOT rapidly intensifying. This is the case the
    # replaced code got wrong: `(wind_kph - 100) / 150` scored a constant 200 km/h storm
    # as 0.67 "RI probability" purely because it was strong.
    steady = tr.intensity_trend(_hist(5, 6, 200, 0))
    assert steady["available"] and steady["rapidly_intensifying"] is False
    assert steady["observed_change_kt"] == pytest.approx(0.0)


def test_weakening_is_reported_as_a_negative_rate_not_clipped():
    """A decaying storm is a different decision from a steady one; clipping hides that."""
    out = tr.intensity_trend(_hist(5, 6, 200, -20))
    assert out["available"] and out["rate_kt_per_24h"] < 0
    assert out["rapidly_intensifying"] is False


def test_trend_accepts_both_datetime_and_iso_string_timestamps():
    """Routes pass DB datetimes; the eval scripts pass ISO strings."""
    as_dt = tr.intensity_trend([{"timestamp": T0, "wind_kph": 60},
                                {"timestamp": T0 + timedelta(hours=24), "wind_kph": 150}])
    as_iso = tr.intensity_trend([{"timestamp": "2020-05-18T00:00:00Z", "wind_kph": 60},
                                 {"timestamp": "2020-05-19T00:00:00Z", "wind_kph": 150}])
    assert as_dt["available"] and as_dt == as_iso


def test_trend_uses_the_trailing_window_not_the_whole_record():
    """A storm's whole life is not its current trend.

    Six days of history that peaked and decayed must report the last 24 h, not the
    first-to-last difference, which would call a weakening storm steady.
    """
    history = _hist(13, 6, 60, 20)                      # 72 h of steady intensification
    history += [{"timestamp": T0 + timedelta(hours=72 + 6 * i), "wind_kph": 320 - 30 * i}
                for i in range(1, 5)]                    # then 24 h of decay
    out = tr.intensity_trend(history)
    assert out["available"]
    assert out["window_hours"] == pytest.approx(24.0)
    assert out["rate_kt_per_24h"] < 0, (
        "the trailing window is decaying but the trend came out positive; the whole "
        "record is being differenced instead of the last 24 h")


# --- assembled payload ----------------------------------------------------------------

def test_build_reports_the_geometry_resolution_with_the_distances():
    out = tr.build([_pt(6, 17.0, 88.0, 37.1)], _hist(5, 6, 60, 10))
    assert out["geometry"]["mean_absolute_error_km"] == MEASURED_ACCURACY["mae_km"]
    assert out["geometry"]["reference"]
    assert "km" in out["geometry"]["note"]


def test_build_declares_that_intensity_is_persisted_not_forecast():
    """Carrying wind forward unchanged is a null hypothesis; it must be labelled as one."""
    out = tr.build([_pt(6, 17.0, 88.0, 37.1)], _hist(5, 6, 60, 10))
    assert out["intensity_forecast_available"] is False
    assert out["intensity_at_forecast_positions"] == "persisted"
    assert "invented" in out["intensity_forecast_note"]


def test_build_survives_an_empty_forecast():
    """The risk endpoint must degrade rather than 500 when there is no track."""
    out = tr.build([], [])
    assert out["closest_approach"] is None
    assert out["landfall"]["possible_from_hour"] is None
    assert out["intensity_trend"]["available"] is False
