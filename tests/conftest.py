"""Shared pytest fixtures.

The suite talks to the app in-process through FastAPI's TestClient rather than to a
server on localhost:8000. The previous smoke script required a developer to remember to
start uvicorn first, which meant "tests fail" and "I forgot to start the server" were the
same observation -- and it silently tested whatever code that server happened to be
running, which during this project was repeatedly stale.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from backend.app.main import app

    # Context-manager form so startup/shutdown events run: the predictor and the track
    # checkpoint are loaded there, and a suite that skipped them would test a stripped
    # app that can never produce a trained forecast.
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def cyclone_id(client) -> str:
    r = client.get("/api/cyclones")
    assert r.status_code == 200, r.text
    cyclones = r.json()["cyclones"]
    assert cyclones, "no cyclones seeded; the demo database is empty"
    return cyclones[0]["id"]


@pytest.fixture(scope="session")
def forecast(client, cyclone_id) -> dict:
    r = client.get(f"/api/cyclones/{cyclone_id}/forecast")
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="session")
def risk_payload(client, cyclone_id) -> dict:
    r = client.get(f"/api/cyclones/{cyclone_id}/risk")
    assert r.status_code == 200, r.text
    return r.json()
