"""API endpoint tests.

Replaces a smoke script that called every endpoint and counted any JSON response as a
pass. That script could not fail for the reasons that matter: an endpoint returning 200
with a fabricated metric, a missing disclaimer, or a forecast pointing into Kazakhstan
would all have been recorded as PASS. These tests assert on content.

The honesty invariants -- that no unmeasured cone is reported as a number, that a
measurably-worse baseline is never served as primary -- live in
tests/test_forecast_integrity.py.
"""
from __future__ import annotations

import pytest

PREDICT_ENDPOINTS = [
    "/api/predict/detect",
    "/api/predict/classification",
    "/api/predict/intensity",
    "/api/predict/track",
    "/api/predict",
]


def test_health_reports_live_state(client):
    body = client.get("/api/health").json()
    assert body["status"] in {"ok", "degraded"}
    assert body["database"] == "connected"
    # Must reflect what is actually loaded. A hardcoded "untrained" or "baseline" string
    # here was the original bug: it kept its value after models were trained.
    assert body["model_status"] != "unavailable"
    assert body["model_status"] == "untrained" or "trained" in body["model_status"]


def test_system_status_lists_loaded_models(client):
    body = client.get("/api/system/status").json()
    loaded = body["models_loaded"]
    assert isinstance(loaded, dict) and loaded, "models_loaded must not be empty"
    assert all(isinstance(v, bool) for v in loaded.values())
    assert body["total_cyclones"] >= body["active_cyclones"] >= 0


def test_cyclone_list_and_detail(client, cyclone_id):
    body = client.get("/api/cyclones").json()
    assert body["count"] == len(body["cyclones"])

    # The detail endpoint returns storm metadata, not a position -- positions live in
    # /track. Asserting on the metadata that downstream pages actually key off.
    detail = client.get(f"/api/cyclones/{cyclone_id}").json()
    assert detail["id"] == cyclone_id
    assert detail["basin"], "cyclone has no basin"
    assert detail["num_observations"] > 0, "cyclone has no observations"
    if detail.get("peak_wind_kph") is not None:
        # 400 km/h exceeds the strongest tropical cyclone ever observed anywhere.
        assert 0 < detail["peak_wind_kph"] < 400, detail["peak_wind_kph"]
    if detail.get("min_pressure_hpa") is not None:
        assert 850 < detail["min_pressure_hpa"] < 1030, detail["min_pressure_hpa"]


def test_unknown_cyclone_is_404_not_500(client):
    r = client.get("/api/cyclones/NO_SUCH_STORM_12345")
    assert r.status_code == 404


def test_track_points_are_in_the_north_indian_ocean(client, cyclone_id):
    body = client.get(f"/api/cyclones/{cyclone_id}/track").json()
    points = body["track_points"]
    assert points, "cyclone has no track points"
    assert body["count"] == len(points)
    for p in points:
        # Generous basin bounds. Catches a swapped lat/lon or a degrees/radians slip,
        # which is the failure mode that produces a confident forecast over Siberia.
        assert -10 <= p["latitude"] <= 32, f"latitude {p['latitude']} outside NIO"
        assert 40 <= p["longitude"] <= 110, f"longitude {p['longitude']} outside NIO"


@pytest.mark.parametrize("path", PREDICT_ENDPOINTS)
def test_prediction_endpoints_respond_with_a_disclaimer(client, path):
    r = client.post(f"{path}?sample_index=0")
    assert r.status_code == 200, r.text
    body = r.json()
    # Every model output leaving this API must carry the not-an-official-warning notice.
    # This is a requirement of presenting forecasts alongside IMD's, not a nicety.
    assert body.get("disclaimer"), f"{path} returned no disclaimer"
    assert "India Meteorological Department" in body["disclaimer"]


@pytest.mark.parametrize("path", PREDICT_ENDPOINTS)
def test_prediction_endpoints_declare_model_status(client, path):
    body = client.post(f"{path}?sample_index=0").json()
    # Either at the top level or per-task for the combined endpoint.
    statuses = [body["model_status"]] if "model_status" in body else [
        v["model_status"] for v in body.values()
        if isinstance(v, dict) and "model_status" in v
    ]
    assert statuses, f"{path} declares no model_status anywhere"
    for s in statuses:
        assert s, f"{path} returned an empty model_status"


ALERT_LEVELS = {"NORMAL", "WATCH", "ADVISORY", "HIGH RISK", "EXTREME RISK", "UNAVAILABLE"}
"""Mirrors the ladder in ml/inference/predictor.py assess_risk(), plus UNAVAILABLE.

Duplicated deliberately: if someone adds or renames a level there, this test fails and
forces a look at every consumer that branches on the string -- the frontend colours alert
banners by exact match, so a new level would render unstyled rather than loudly break.

UNAVAILABLE is returned by the risk route when no predictor is loaded. That path used to
return a fixed 0.3 / "WATCH", which downstream was indistinguishable from a real
assessment of a real storm.
"""


def test_risk_assessment_is_bounded(client, cyclone_id):
    body = client.get(f"/api/cyclones/{cyclone_id}/risk").json()
    risk = body["risk"]
    for key in ("overall_risk", "wind_risk", "coastal_risk"):
        if key in risk and risk[key] is not None:
            assert 0.0 <= risk[key] <= 1.0, f"{key}={risk[key]} outside [0, 1]"
    assert risk["alert_level"] in ALERT_LEVELS, risk["alert_level"]


def test_models_endpoint_never_reports_a_metric_for_an_untrained_model(client):
    body = client.get("/api/models").json()
    assert body["count"] == len(body["models"])
    for m in body["models"]:
        if m["status"] != "untrained":
            continue
        metrics = m.get("metrics") or {}
        numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        # An untrained model with a number attached to it is a fabricated metric,
        # regardless of how it got there.
        assert not numeric, f"untrained model {m['id']} reports numbers: {numeric}"


def test_datasets_alerts_and_observations_respond(client):
    for path, key in (("/api/datasets", "datasets"),
                      ("/api/alerts", "alerts"),
                      ("/api/observations", "observations")):
        body = client.get(path).json()
        assert isinstance(body[key], list), f"{path} did not return a list of {key}"
