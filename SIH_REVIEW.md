# CYCLONE-AI — Review for SIH 2026 (PS 26070, MoES / IMD)

Method: built the frontend (`npm run build` — clean, `tsc -b` passes), ran the FastAPI
backend on :8000 serving the production SPA, screenshotted all 8 top-level pages plus
`/cyclone/:id` in headless Chrome, and probed every ML endpoint with curl. Everything
below is observed behaviour, not code reading alone.

---

## 0. What is genuinely good — keep it

- **Architecture.** Clean `ml/` ↔ `backend/` ↔ `frontend/` separation, Pydantic schemas,
  a typed Axios client, SQLAlchemy models. This is above the median SIH codebase.
- **Honesty layer.** The amber "not an official warning system" banner and the
  DEMO/LIVE/HISTORICAL badge. Do **not** remove these. IMD judges respond well to a
  team that knows the boundary between decision support and an official bulletin.
- **The right feature list.** MC-dropout uncertainty, Grad-CAM, permutation importance,
  a 9-stage IMD scale, multi-horizon track. The *shape* of the project is correct.
- **Ops.** Multi-stage Dockerfile with a healthcheck. Sensible `manualChunks`.
- **Secrets hygiene.** `.env` and `*.db` gitignored, `git ls-files` confirms neither is
  tracked, and the API-key entries in `.env` are commented-out placeholders. Clean.

The problem is not the plan. It is that nothing behind the UI is real yet, and three
things will visibly break on stage.

---

## 1. Demo-day blockers (fix these first — hours of work, they decide the round)

### 1.1 `/cyclone/:id` renders a blank white page in the production build

The most-clicked path in your app is dead. Click any cyclone on the dashboard → white
screen.

Cause: [`frontend/vite.config.ts`](frontend/vite.config.ts) sets `base: './'`, so
`index.html` emits `./assets/index-XXX.js`. From the URL `/cyclone/DEMO_NI_2023_001` the
browser resolves that to `/cyclone/assets/index-XXX.js`. Your SPA catch-all in
[`backend/app/main.py`](backend/app/main.py) doesn't find that file, so it returns
`index.html` with **HTTP 200**. The browser receives HTML where it asked for JavaScript,
the module never executes, nothing renders. Verified:

```
curl /cyclone/assets/index-DAl74dQ1.js  →  200, body starts <!DOCTYPE html>
```

Fix: `base: '/'`. Also make the catch-all return a real 404 for anything under
`/assets/` or with a file extension, so this class of bug is loud instead of silent.

### 1.2 The dashboard map is covered in "API KEY REQUIRED" watermarks

[`frontend/src/maps/CycloneMap.tsx:56`](frontend/src/maps/CycloneMap.tsx) uses
`basemaps.cartocdn.com/dark_all/...`, which now requires a key. Your hero visual is
tiled with the words "API KEY REQUIRED".

Fix: switch to a keyless provider, and for the venue **pre-download the North Indian
Ocean tiles for zoom 3–7 and serve them from your own backend**. A cyclone dashboard
that works with the wifi unplugged is a talking point, not a compromise.

### 1.3 The demo depends on the internet in two more places

`dist/index.html` pulls Google Fonts and `unpkg.com/leaflet@1.9.4/dist/leaflet.css`.
Venue wifi at SIH is unreliable. Vendor both into the bundle (`npm i leaflet` already
ships the CSS; import it, and self-host the font).

### 1.4 Contradictory numbers on the same screen

The dashboard shows **"High Risk Systems: 0"** directly above a list containing AMPHAN at
258 km/h, Super Cyclonic Storm. [`Dashboard.tsx:27`](frontend/src/pages/Dashboard.tsx)
counts only `alerts` with level `HIGH RISK`/`EXTREME RISK`; the seeded alerts don't use
those strings. A judge reads this as "the dashboard doesn't know what it's showing."

Related: `/api/cyclones/{id}/risk` returns `alert_level: "WATCH"`, `overall_risk: 0.27`
for a system whose own intensity forecast on the next tab says 244 km/h. The risk engine
reads the current DB track point and never looks at the forecast. Same storm, two
screens, opposite conclusions.

### 1.5 Identical inputs give different answers each click

Three consecutive identical calls to `POST /api/predict/intensity?sample_index=0`:

```
call 1  wind 1.3 km/h   pressure 898.1 hPa
call 2  wind 2.3 km/h   pressure 898.7 hPa
call 3  wind 1.7 km/h   pressure 898.4 hPa
```

Root cause in §3.1. If a judge clicks twice and gets two answers, credibility is gone.
Make the headline numbers deterministic (fixed seed, `eval()` mode) and show uncertainty
as an explicit interval instead of as jitter.

---

## 2. Scientific credibility — what an IMD judge will catch

This is the section that decides whether you win. Your judges forecast cyclones for a
living. Every number on screen is being sanity-checked against their intuition, and right
now most of them fail.

### 2.1 The models are untrained. The Models page admits it.

All four models: **"BASELINE", "Not yet evaluated — baseline weights", zero metrics.**
So the outputs are random-weight noise, which produces:

| Endpoint | Output | Why it's fatal |
|---|---|---|
| `/predict/classification` | "Severe Cyclonic Storm", **confidence 0.9999** | 99.99% certainty from random weights. Overconfidence is worse than being wrong. |
| `/predict/detect` | confidence **0.5165** | A coin flip presented as a detection. |
| `/predict/intensity` | **1.3 km/h wind @ 898.6 hPa**, labelled "Low Pressure Area" | 898 hPa is a Super Cyclonic Storm's core pressure. Paired with dead-calm wind. Physically impossible. |

**This is the single highest-leverage fix in the whole project.** See §4.1.

### 2.2 Wind and pressure are predicted independently, with no physical constraint

Two separate heads, no coupling. Hence 898 hPa at 1.3 km/h. Tropical cyclones obey a
wind–pressure relationship; RSMC New Delhi uses a Koba-type relation for the North
Indian Ocean. Predict one, derive the other through that relation, or add a consistency
penalty to the loss. Then clamp pressure to a physical range (`ml/models/intensity_model.py`
does `pressure = pressure + 900.0` with **no clamp** — the output is unbounded).

### 2.3 The track forecast is physically impossible

`GET /api/cyclones/DEMO_NI_2023_001/forecast`, actual values returned:

- **+6 h latitude 27.7°** from a current 18.43° → ~1030 km in 6 hours → a translation
  speed of **~172 km/h**. Real North Indian Ocean systems move at 10–25 km/h.
- **`uncertainty_radius_km: 1745.1` at +6 h.** IMD's operational 24-hour track error is
  on the order of 60–100 km. You are claiming an error bar 20× larger than IMD's, six
  hours out.
- **+48 h longitude 58.91° is *west* of +24 h longitude 71.01°.** The track zigzags back
  on itself.

Add hard physical guards on the output: max translation speed, monotonic time
consistency, and uncertainty radii that grow with lead time from a realistic base.

### 2.4 The intensity forecast is fabricated from positional uncertainty

[`backend/app/api/routes.py:243`](backend/app/api/routes.py):

```python
for i, h in enumerate([6, 12, 24, 48]):
    wind_delta = (track_forecast["forecast_points"][i]["lat_uncertainty"] * 20) - 5
    pres_delta = -wind_delta * 0.3
```

Your intensity forecast is a linear function of *how unsure you are about latitude*.
It is not a model output at all. The result, from a Depression at 51.1 km/h / 1002.3 hPa:

> **+6 h: 244.6 km/h at 827.4 hPa**

51 → 245 km/h in six hours (rapid intensification is defined as 30 kt per **24** h), and
827 hPa is **below the lowest sea-level pressure ever measured in a tropical cyclone**
(870 hPa, Typhoon Tip, 1979). An IMD scientist will spot 827 hPa instantly.

If a judge opens the repo and finds this loop, the project is over. Either train a real
intensity-forecast head or state plainly in the UI that intensity forecast is not yet
implemented. **The second option costs you far less than being caught.**

### 2.5 The IMD classification thresholds are wrong

[`ml/utils/__init__.py`](ml/utils/__init__.py) has 9 categories with "Well Marked Low"
inserted as a wind class, which shifts every threshold above it:

| Stage | Your threshold | Actual IMD (kt) |
|---|---|---|
| Depression | 22 | **17–27** |
| Deep Depression | 28 | 28–33 |
| Cyclonic Storm | 34 | 34–47 |
| Severe CS | 48 | 48–63 |
| Very Severe CS | 64 | 64–89 |
| Extremely Severe CS | 90 | 90–119 |
| Super Cyclonic Storm | 120 | ≥120 |

Consequences visible on your Historical page right now:

- **TAUKTAE, 223 km/h → labelled "Very Severe Cyclonic Storm"** (≥222 km/h is Super)
- **FANI, 236 km/h → labelled "Extremely Severe"** (should be Super Cyclonic Storm)

You are mislabelling two of the most well-known recent Indian cyclones, in front of the
agency that named them. "Well Marked Low" is a pressure-pattern descriptor, not a wind
category — drop it from the wind scale and keep 7 classes (or 8 including "no system").

### 2.6 The dashboard map plots invented coordinates

[`CycloneMap.tsx:179`](frontend/src/maps/CycloneMap.tsx):

```ts
// Use approximate positions based on basin (NI = North Indian Ocean)
const baseLat = 12 + idx * 2.5;
const baseLon = 80 + idx * 3;
```

Your cyclones sit on a synthetic diagonal line across the Bay of Bengal, ordered by array
index. Meanwhile **real track data already exists in your database** and is served by
`/api/cyclones/{id}/track` — the map just never fetches it. Fix this and the map becomes
your strongest visual instead of your most obvious fake.

### 2.7 No baseline to compare against

Nothing in the project answers "better than what?" IMD judges will ask this in the first
two minutes. See §4.2 — this is the most winnable point on the list.

### 2.8 `/api/imd/live-tracks` calls an endpoint you haven't verified

It hits `https://api.imd.gov.in/api/v1/cyclone_track` with a Bearer token, and
`imd_service.py` itself comments that the auth scheme "may need to be updated" if IMD uses
`x-api-key`. If a judge says "show me live mode," you get an error. Either confirm the real
integration (MOSDAC/IMD data access is requestable) or remove the LIVE badge and be
explicit that the demo is hindcast-driven.

---

## 3. Code correctness bugs

### 3.1 MC dropout corrupts BatchNorm running statistics on every request

`ml/models/intensity_model.py:108` and `ml/models/track_model.py:132`:

```python
def predict_with_uncertainty(self, satellite, env_features, n_samples=20):
    self.train()   # intended: enable dropout
```

`self.train()` also switches every `nn.BatchNorm2d` in `SatelliteCNNBackbone` (7 of them)
into batch-statistics mode, so each inference call **overwrites the running mean/var using
a single sample**. That is the measured drift in §1.5 — the model degrades a little with
every prediction a user makes. Fix:

```python
self.eval()
for m in self.modules():
    if isinstance(m, (nn.Dropout, nn.Dropout2d)):
        m.train()
```

### 3.2 Grad-CAM leaks hooks on the singleton model — FIXED, and it was worse than this

`ml/explainability/gradcam.py::_register_hooks` calls `register_forward_hook` and
`register_full_backward_hook` and **never retains or removes the handles**, while
`predictor.explain()` constructs a fresh `GradCAM(model)` on *every* request against the
same long-lived model.

Measured on the running API before the fix, one pair per `/api/predict` request:

| requests | hook pairs on detector | live `GradCAM` objects after gc | retained |
|---|---|---|---|
| 0 | 0 | 0 | — |
| 50 | 50 | 50 | 3200 KiB |

That is **64 KiB retained per request, without bound**. Two details make it worse than a
plain leak:

- The hooks keep the dead `GradCAM` objects reachable, so the garbage collector cannot
  reclaim them — 50/50 survived a `gc.collect()`.
- All 50 stale forward hooks **still fired during an unrelated `/api/predict/detect` call**,
  each writing a fresh activation tensor onto a dead object. The leak grows even when
  nobody asks for an explanation.

Latency was *not* measurably affected at this scale (0.85–1.15× baseline across 50 leaked
pairs, i.e. within measurement noise) — worth stating plainly rather than claiming a speed
win the numbers don't support. The cost is memory and unbounded object retention.

Fixed: handles retained and released, `GradCAM` is a context manager, `explain()` uses
`with`. Verified at **0 hooks and 0 retained objects after 201 requests**. Covered by
`tests/test_explainability.py`, whose hook counter deliberately counts `_backward_hooks`,
`_full_backward_hooks` *and* `_forward_hooks` — on torch 2.10
`register_full_backward_hook` writes to `_backward_hooks`, and a counter looking only at
`_full_backward_hooks` reports zero backward hooks and makes the leak test pass vacuously.

**Two fabrications found in the same file while fixing this** — both more damaging than the
leak, because a leak degrades the service while these misinform the user:

1. **A missing gradient returned a uniform 0.5 heatmap.** `AIAnalysis.tsx` drew it as a real
   attribution: a flat coloured square that reads as "the model attends everywhere equally"
   when the truth is "no gradients were captured, there is nothing to show". Now
   `{available: false, reason: ...}` with no `heatmap` key, so a caller that ignores the flag
   fails loudly instead of drawing. Reached in practice when the auto-detected "last conv"
   sits on a branch the chosen head does not use.

2. **`compute_feature_importance` measured a quantity permutation provably cannot move.** It
   differenced batch *means*: `abs(baseline.mean() - permuted.mean())`. Shuffling a column
   leaves that column's mean unchanged, and for an affine readout the batch-mean prediction
   depends only on the column means — so the result is **identically zero**. Measured:

   | model | \|Δ batch-mean\| (what it measured) | mean \|Δ per-sample\| (the real effect) | signal retained |
   |---|---|---|---|
   | affine readout | 0.0 | 0.20 – 0.50 | **0%** |
   | ReLU MLP | 1.7e-3 – 3.2e-3 | 0.055 – 0.111 | 2 – 4% |

   Averaging over the batch cancels precisely the per-sample changes the method exists to
   detect. On the MLP the attenuation also *differs per feature* (2.2%, 3.0%, 4.4%), so even
   the ranking was distorted. Now differenced per sample, then averaged — validated against
   a model with dependence fixed at 3 : 1 : 0, recovering a **3.05 : 1** ratio and exactly
   0.0 for the unused feature, where the old code returned 0.0 for all three.

   Also: a single-row input made `randperm(1)` the identity, so every score came out 0.0 —
   indistinguishable from a real finding that no feature matters. It now refuses. And when a
   model moves nothing at all, each feature's `share` is `None`, not `0.0`: a share of a zero
   total is undefined, and writing 0.0 would assert "unimportant" when the fact is about the
   model, not the feature.

Both were reachable from `/api/predict`, which the UI calls.

### 3.3 The wind head ends in ReLU — it can die

`nn.Linear(64,1), nn.ReLU()`. Once the pre-activation goes negative the gradient is zero
and that output is stuck at 0 forever. Use `softplus`, or predict in normalised space and
denormalise. This is likely part of why wind reads 1.3 km/h.

### 3.4 Leaflet layers are mutated during iteration

```ts
map.eachLayer(layer => { if (!(layer instanceof L.TileLayer)) map.removeLayer(layer); });
```

Removing during `eachLayer` skips entries. Collect into an array first, then remove.
Symptom: stale markers/tracks accumulating on re-render.

### 3.5 There is effectively no test suite — FIXED (76 passing, 1 skipped)

`pytest tests/` → `ERROR at setup of test / fixture 'name' not found`. `tests/test_api.py:20`
defines `def test(name, fn)`, which pytest collects as a test and tries to inject `name` as
a fixture. Rename it to `check(...)` and add real `test_*` functions. Even 15 honest tests —
IMD threshold boundaries, wind–pressure consistency, forecast monotonicity, endpoint smoke
tests — is a slide worth having, and it would have caught §2.3 and §2.5.

Now `76 passed, 1 skipped`, across four files:

| file | what it defends |
|---|---|
| `test_api.py` | endpoints return content, not just 200s; no metric on an untrained model |
| `test_forecast_integrity.py` | the honesty invariants: no unmeasured cone reported as a number, no measurably-worse baseline served as primary, risk scored on the forecast |
| `test_track_risk.py` | the risk engine's pure functions, driven with synthetic tracks |
| `test_explainability.py` | hook lifetime, and that saliency refuses rather than fabricates |

Two lessons from building it, both worth a sentence on the slide:

**A skipped test is not a test.** `test_landfall_becomes_possible_before_the_centre_track_arrives`
was written at the API level and skipped on **all five** demo storms — no forecast centre
comes within the coastline polyline's 18.4 km resolution of land (closest approach: 41 km),
so `track_arrives_hour` is null everywhere and there was nothing to order. The invariant had
zero coverage while looking covered. It is now asserted unconditionally in
`test_track_risk.py` against a synthetic track that does go ashore, where the cone reaches
Odisha at **+24 h and the centre only at +48 h** — the day of warning that testing the cone
rather than the centre line actually buys. The API-level test stays as the wiring check and
its skip message points at the unit test.

**The tests were themselves mutation-tested.** Every honesty invariant was verified by
reintroducing the bug it guards and confirming the suite goes red — 17 mutations, 17 caught,
but only after two rounds:

- *The lead-time sort.* An arbitrary shuffle passed against a deliberately unsorted
  implementation, because that particular shuffle happened to put the right row first. A
  *reversed* list is what distinguishes: unsorted it returns +48 h as the first cone landfall
  when the answer is +24 h. Both orderings are now checked.
- *The uniform-0.5 fallback.* Unreachable with a normal fixture, because gradients always
  flow. Needed a model whose conv layer runs but does not feed the score — plus a second test
  asserting that fixture really does capture activations while producing no gradients, so it
  cannot start passing for the wrong reason.
- *The hook counter.* An earlier version counted `_full_backward_hooks`, which torch 2.10
  leaves empty (`register_full_backward_hook` writes to `_backward_hooks`). It reported zero
  backward hooks whether or not they leaked.

All three were tests that passed while asserting nothing. Worth stating to a judge: the
suite's value is what it catches, and the only way to know that is to break the code on
purpose.

### 3.6 A fresh clone behaves differently from your machine

`frontend/dist` is gitignored, but `main.py` auto-detects and serves it even without
`SERVE_FRONTEND=true`. On your laptop the SPA appears; on a clean clone it 404s. Make the
behaviour explicit and document the build step (or wire it into `start.sh`).

---

## 4. How to actually win SIH 2026

Ranked by judge-impact per hour of work.

### 4.1 Train on real IBTrACS data — this is the whole game

**IBTrACS is a plain CSV from NOAA NCEI. No authentication, no application, ~free.** It
contains every North Indian Ocean cyclone with 3-hourly position, wind, and pressure going
back over a century. You can train the LSTM track model and the intensity model on it
**on CPU, overnight.**

The moment you do, the Models page stops saying "Not yet evaluated" and starts showing
mean track error in km at +12/24/48 h and intensity MAE in knots. That single change
converts the project from "a nicely-built UI with random numbers behind it" into "a system
with measured skill." Nothing else you can do in the remaining time comes close.

Hold out recent seasons (say 2019–2023) as a test set so your numbers are honest, and be
ready to say which storms were held out.

### 4.2 Benchmark against CLIPER and persistence

IMD judges know these baselines cold. Implement two trivial ones:

- **Persistence** — extrapolate current motion linearly.
- **CLIPER** — climatology + persistence regression, the standard skill floor.

Then the sentence you say on stage becomes:

> *"Our LSTM reduces 24-hour track error by 31% against CLIPER on held-out 2019–2023
> North Indian Ocean storms."*

instead of

> *"Our LSTM predicts cyclone tracks."*

The first wins. The second is what every other team says. **A modest model with an honest
skill score beats an impressive-sounding model with no number, every single time, in front
of domain judges.**

### 4.3 Add a physical-consistency layer

One module, high credibility return: enforce the wind–pressure relation, clamp pressure to
870–1020 hPa, cap translation speed, force forecast-time monotonicity, and cap
intensification rate at physically plausible limits. Then say so on a slide: *"model
outputs pass a physical-plausibility filter before display."* That's the language of
operational meteorology and it signals you understand the domain.

### 4.4 Differentiate downstream, not upstream — this is your real winning angle

Be strategic about PS 26070. **IMD does not need another track model.** They run
GFS/ECMWF/HWRF ensembles and their own multi-model consensus, with skill you will not beat
on a laptop in a semester. If you frame the project as "our AI forecasts cyclones better
than IMD," you invite the one comparison you lose.

What operational disaster management genuinely lacks is the **last mile**. Build one or two
of these and you stop competing on forecast accuracy entirely:

- **Exposure & impact.** Overlay the forecast cone on gridded population (WorldPop/GHSL)
  and coastal district boundaries → *"3.4 lakh people inside the 100 km cone, across 7
  districts of Odisha; 2 ports, 14 PHCs, 1 thermal plant."* This is what a District
  Collector actually needs, and no forecast model gives it to them.
- **Auto-generated bulletins.** Emit a draft advisory in IMD's own bulletin format, plus a
  short SMS/WhatsApp text in the relevant regional languages (Odia, Bengali, Tamil,
  Telugu). Translating a forecast into an actionable warning in the language of the person
  at risk is a demonstrable public-good contribution.
- **Sector-specific advisories.** Fishermen, ports, power utilities, and railways each need
  different thresholds from the same forecast. Trivial to implement, very legible to judges.
- **INSAT-3D/3DR nowcasting via MOSDAC.** Half-hourly Indian satellite imagery instead of
  NOAA-only data. For an MoES problem statement, using ISRO/MoES data sources is a real
  credibility signal — it says you understand whose infrastructure you're building on.
- **Position as augmentation, not replacement.** Blend your output *with* the official IMD
  bulletin and show both. This matches the disclaimer you already ship and reframes the
  whole project as decision support, which is exactly what the PS asks for.

### 4.5 Build one hero case study

Pick **AMPHAN or FANI**. Hindcast it end to end with models that never saw it in training.
Show, on the map: your predicted track, the actual track, and the error in km at each lead
time. Show the Grad-CAM attending to the eye region. One storm, told properly, with
verifiable numbers, beats eight tabs of placeholders. Make it the spine of your demo.

### 4.6 Demo-day discipline

- **Rehearse a fixed 6-minute click path.** Every page you show must be populated. Right
  now Predictions and AI Analysis are near-empty until a button is pressed — pre-load them,
  or open on a result.
- **Seed determinism.** Fixed seeds, `eval()` mode for headline numbers. Never let a
  repeated click change the answer (§1.5).
- **Run fully offline.** Vendor every asset, self-host tiles. Assume the wifi dies.
- **Record a 3-minute video fallback.** Laptops fail at SIH.
- **Have the code ready to open.** Judges do read it. Before that happens, `routes.py:243`
  (§2.4) and `CycloneMap.tsx:179` (§2.6) must be gone.

---

## 5. Suggested order of work

**Days 1–2 — stop the bleeding.** `base: '/'` (§1.1). Keyless/self-hosted map tiles (§1.2).
Vendor CDN assets (§1.3). Fix IMD thresholds (§2.5). Real coordinates on the map (§2.6).
Delete the fabricated intensity-forecast loop (§2.4). Fix the BatchNorm/MC-dropout bug
(§3.1) and the Grad-CAM hook leak (§3.2).

**Days 3–7 — earn the "AI" claim.** Download IBTrACS. Train track + intensity with a held-out
test set. Get real metrics onto the Models page (§4.1). Implement persistence + CLIPER and
publish the comparison (§4.2).

**Week 2 — earn the domain vote.** Physical-consistency layer (§4.3). Pick one downstream
feature from §4.4 — I'd choose exposure/impact, it demos beautifully on the map you already
have. Build the AMPHAN/FANI hero case study (§4.5).

**Week 3 — polish.** Real pytest suite (§3.5). Fix the dashboard contradictions (§1.4, and
make risk read the forecast). Rehearse. Record the fallback video.

---

## Bottom line

The engineering is solid and the feature list is the right one — that's a real asset, and
most teams don't have it. What's missing is that **nothing behind the UI is trained, and
several numbers on screen are physically impossible in ways an IMD judge will catch in
seconds**: 898 hPa at 1.3 km/h wind, an 827 hPa forecast, a storm moving at 172 km/h,
TAUKTAE mislabelled, and an intensity forecast computed from positional uncertainty.

The fix is not more features. It is: train on IBTrACS so you have honest metrics, beat a
named baseline, enforce physical plausibility, and differentiate on the last mile rather
than on forecast accuracy. Do that and you arrive with something IMD can actually use —
which is what wins this problem statement.
