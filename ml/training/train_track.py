"""
Train the track-only displacement forecaster on IBTrACS North Indian Ocean data.

Run from the project root:

    python -m ml.training.train_track --epochs 300

What this script guarantees
---------------------------
* **Season-held-out evaluation.** Training uses 1990-2018, validation 2019-2023, test
  2024-2025. No storm contributes points to more than one split, and no future season is
  ever used to forecast a past one. A random split over track points would let the model
  interpolate a storm it has already seen and inflate apparent skill several-fold.
* **The test split is touched exactly once**, at the end, after early stopping has
  already selected the checkpoint on validation. Peeking at test to choose an epoch turns
  it into a second validation set and the reported number stops meaning anything.
* **Baselines on identical samples.** Persistence and fitted CLIPER are scored on the
  same held-out samples with the same metric, so the comparison cannot flatter the
  network by scoring it on an easier subset.
* **Selection on great-circle error, not on loss.** The training loss is Huber on scaled
  displacement, which is not the quantity anyone cares about. The checkpoint is chosen by
  mean positional error in kilometres at the horizons being reported.

The result is written to models/checkpoints/track_only.pt with the architecture, the
feature layout, the training seasons and the verification table embedded, so a checkpoint
can never be loaded under a different feature convention than it was trained with.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..baselines.cliper import fit_cliper
from ..datasets.ibtracs import load_ibtracs, split_by_season
from ..datasets.track_dataset import (
    DEFAULT_HISTORY_STEPS,
    DEFAULT_HORIZONS_H,
    DISPLACEMENT_SCALE_KM,
    MOTION_SCALE_KMH,
    TRACK_FEATURE_DIM,
    TRACK_FEATURE_NAMES,
    TrackSample,
    build_samples,
    describe_samples,
    offset_from_km,
    stack_samples,
)
from ..evaluation.track_metrics import (
    Verification,
    comparison_table,
    paired_significance,
    per_case_errors,
    significance_table,
    verify,
)
from ..models.track_only import TrackOnlyForecaster

logger = logging.getLogger("cyclone_ai.train_track")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "data" / "raw" / "ibtracs.NI.csv"
DEFAULT_OUT = PROJECT_ROOT / "models" / "checkpoints" / "track_only.pt"
DEFAULT_CLIPER_OUT = PROJECT_ROOT / "models" / "checkpoints" / "cliper_ni.json"
DEFAULT_REPORT = PROJECT_ROOT / "models" / "checkpoints" / "track_evaluation.json"


# --- helpers --------------------------------------------------------------------

def truths_for(samples: list[TrackSample], horizons_h) -> list[dict]:
    """Observed verifying positions per sample, as {horizon: (lat, lon)}."""
    out = []
    for s in samples:
        d = {}
        for i, h in enumerate(horizons_h):
            if s.target_mask[i] > 0:
                d[h] = offset_from_km(s.origin_lat, s.origin_lon, *s.target_km[i])
        out.append(d)
    return out


def persistence_prediction(s: TrackSample, horizons_h) -> dict:
    """Constant-motion forecast built from the same history the network sees."""
    u = float(s.history[-1][3]) * MOTION_SCALE_KMH
    v = float(s.history[-1][4]) * MOTION_SCALE_KMH
    return {h: offset_from_km(s.origin_lat, s.origin_lon, u * h, v * h)
            for h in horizons_h}


def model_predictions(model: TrackOnlyForecaster, samples: list[TrackSample],
                      horizons_h, device) -> list[dict]:
    """Batched deterministic forecast for a sample list."""
    model.eval()
    hist = torch.from_numpy(np.stack([s.history for s in samples])).to(device)
    with torch.no_grad():
        disp = model(hist)["displacement_scaled"].cpu().numpy() * DISPLACEMENT_SCALE_KM
    return [
        {h: offset_from_km(s.origin_lat, s.origin_lon, disp[i, j, 0], disp[i, j, 1])
         for j, h in enumerate(horizons_h)}
        for i, s in enumerate(samples)
    ]


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                 delta: float = 0.2) -> torch.Tensor:
    """Huber loss over horizons that have a verifying observation.

    Huber rather than MSE because the displacement distribution has a long tail -- the
    largest +48 h target in the training set is 1410 km against a mean of 544 km -- and
    squared error lets a handful of fast recurving storms dominate the gradient.

    ``delta`` is in scaled units: 0.2 * 500 km = 100 km, so errors under 100 km are
    treated quadratically and larger ones linearly.

    The mask is applied per (sample, horizon) and the sum is divided by the number of
    *unmasked* elements, not by the total. Dividing by the total would silently scale the
    loss down for batches full of dissipated storms and make the effective learning rate
    depend on data availability.
    """
    diff = pred - target                             # [B, H, 2]
    abs_d = diff.abs()
    quad = 0.5 * diff.pow(2)
    lin = delta * (abs_d - 0.5 * delta)
    elem = torch.where(abs_d <= delta, quad, lin).sum(dim=-1)   # [B, H]
    denom = mask.sum().clamp(min=1.0)
    return (elem * mask).sum() / denom


def mean_error_km(v: Verification, horizons_h) -> float:
    """Mean positional error averaged over horizons, for checkpoint selection.

    Averaging the per-horizon means weights every lead time equally rather than letting
    +48 h -- whose errors are an order of magnitude larger -- decide the checkpoint by
    itself. A model that is excellent at +48 h and poor at +6 h is not the one to ship,
    because +6 h is what a landfall bulletin depends on.
    """
    m = v.mean_by_horizon()
    vals = [m[h] for h in horizons_h if h in m]
    return float(np.mean(vals)) if vals else float("inf")


# --- training -------------------------------------------------------------------

def train(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    horizons_h = tuple(DEFAULT_HORIZONS_H)

    print("=" * 74)
    print("Track forecaster training -- IBTrACS North Indian Ocean")
    print("=" * 74)

    storms, report = load_ibtracs(args.csv, min_season=args.min_season)
    print(report.summary())
    train_st, val_st, test_st = split_by_season(storms)

    tr = build_samples(train_st, history_steps=args.history_steps, horizons_h=horizons_h)
    va = build_samples(val_st, history_steps=args.history_steps, horizons_h=horizons_h)
    te = build_samples(test_st, history_steps=args.history_steps, horizons_h=horizons_h)
    if not tr or not va:
        raise SystemExit("not enough samples to train; check the CSV and season filters")

    print("\nTRAIN\n" + describe_samples(tr, horizons_h))
    print("\nVAL (early stopping)\n" + describe_samples(va, horizons_h))
    print("\nTEST (scored once, at the end)\n" + describe_samples(te, horizons_h))

    # --- baselines, on the same samples ----------------------------------------
    cliper = fit_cliper(tr, horizons_h=horizons_h)
    cliper.save(args.cliper_out)
    print("\n" + cliper.summary())
    print(f"  saved -> {args.cliper_out}")

    def baseline_verifications(samples, tag):
        origins = [(s.origin_lat, s.origin_lon) for s in samples]
        truths = truths_for(samples, horizons_h)
        vp = verify("Persistence (PERS)", origins,
                    [persistence_prediction(s, horizons_h) for s in samples],
                    truths, horizons_h)
        vc = verify("CLIPER (fitted)", origins,
                    [cliper.predict_latlon(s) for s in samples], truths, horizons_h)
        return vp, vc, origins, truths

    val_pers, val_clip, val_origins, val_truths = baseline_verifications(va, "val")

    # --- model -----------------------------------------------------------------
    model = TrackOnlyForecaster(
        input_dim=TRACK_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        n_horizons=len(horizons_h),
        dropout=args.dropout,
        residual_persistence=not args.no_residual,
        motion_scale_kmh=MOTION_SCALE_KMH,
        horizons_h=horizons_h,
        displacement_scale_km=DISPLACEMENT_SCALE_KM,
    ).to(device)
    print(f"\nmodel: {type(model).__name__}, {model.n_parameters()} trainable parameters, "
          f"residual_persistence={model.residual_persistence}")

    packed = stack_samples(tr)
    X = torch.from_numpy(packed["history"]).to(device)
    Y = torch.from_numpy(packed["target_scaled"]).to(device)
    M = torch.from_numpy(packed["mask"]).to(device)
    n = X.shape[0]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=args.plateau_patience)

    best = {"score": float("inf"), "epoch": -1, "state": None}
    history: list[dict] = []
    patience_left = args.patience
    t0 = time.time()

    print(f"\n{'epoch':>6} {'train_loss':>11} {'val_mean_km':>12} "
          f"{'+6h':>7} {'+48h':>8} {'lr':>9}  note")
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        nb = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            out = model(X[idx])
            loss = masked_huber(out["displacement_scaled"], Y[idx], M[idx],
                                delta=args.huber_delta)
            loss.backward()
            # Clip because an LSTM on short sequences with a long-tailed target can take
            # a single very large step early and never recover on a dataset this size.
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            total += float(loss)
            nb += 1
        train_loss = total / max(nb, 1)

        vm = verify("model", val_origins,
                    model_predictions(model, va, horizons_h, device),
                    val_truths, horizons_h)
        score = mean_error_km(vm, horizons_h)
        by_h = vm.mean_by_horizon()
        sched.step(score)

        note = ""
        if score < best["score"] - 1e-6:
            best = {"score": score, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
            patience_left = args.patience
            note = "* best"
        else:
            patience_left -= 1

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_mean_km": score,
                        "val_by_horizon": {str(k): v for k, v in by_h.items()}})

        if epoch % args.log_every == 0 or note or epoch == 1:
            print(f"{epoch:>6} {train_loss:>11.5f} {score:>12.2f} "
                  f"{by_h.get(6, float('nan')):>7.1f} {by_h.get(48, float('nan')):>8.1f} "
                  f"{opt.param_groups[0]['lr']:>9.2e}  {note}")

        if patience_left <= 0:
            print(f"\nearly stop at epoch {epoch}: no validation improvement in "
                  f"{args.patience} epochs")
            break

    elapsed = time.time() - t0
    if best["state"] is None:
        raise SystemExit("training produced no improving checkpoint")
    model.load_state_dict(best["state"])
    print(f"\nrestored best epoch {best['epoch']} "
          f"(val mean {best['score']:.2f} km) after {elapsed:.1f}s")

    # --- final verification ----------------------------------------------------
    val_model = verify("CycloneAI TrackNet", val_origins,
                       model_predictions(model, va, horizons_h, device),
                       val_truths, horizons_h)

    print("\n" + "=" * 74)
    print("VALIDATION 2019-2023 (used for early stopping)")
    print("=" * 74)
    print(comparison_table([val_pers, val_clip, val_model], reference=val_clip))
    print("\n" + val_model.table())

    val_pred = model_predictions(model, va, horizons_h, device)
    val_sig = paired_significance(
        per_case_errors(val_origins, val_pred, val_truths, horizons_h),
        per_case_errors(val_origins, [cliper.predict_latlon(s) for s in va],
                        val_truths, horizons_h))
    print("\n" + significance_table(val_sig, "TrackNet", "CLIPER (fitted)"))

    results = {
        "val": {v.name: {"mean_by_horizon": v.mean_by_horizon(),
                         "scores": [vars(s) for s in v.scores]}
                for v in (val_pers, val_clip, val_model)},
        "val_significance_vs_cliper": {str(h): d for h, d in val_sig.items()},
    }

    if te:
        te_pers, te_clip, te_origins, te_truths = baseline_verifications(te, "test")
        te_pred = model_predictions(model, te, horizons_h, device)
        te_model = verify("CycloneAI TrackNet", te_origins, te_pred, te_truths, horizons_h)
        print("\n" + "=" * 74)
        print("TEST 2024-2025 (scored once, never used for any decision)")
        print("=" * 74)
        print(comparison_table([te_pers, te_clip, te_model], reference=te_clip))
        print("\n" + te_model.table())
        te_sig = paired_significance(
            per_case_errors(te_origins, te_pred, te_truths, horizons_h),
            per_case_errors(te_origins, [cliper.predict_latlon(s) for s in te],
                            te_truths, horizons_h))
        print("\n" + significance_table(te_sig, "TrackNet", "CLIPER (fitted)"))
        results["test"] = {v.name: {"mean_by_horizon": v.mean_by_horizon(),
                                    "scores": [vars(s) for s in v.scores]}
                           for v in (te_pers, te_clip, te_model)}
        results["test_significance_vs_cliper"] = {str(h): d for h, d in te_sig.items()}

        n_sig = sum(1 for d in te_sig.values() if d["significant"])
        print(f"\nVERDICT: on the untouched test split the network is distinguishable "
              f"from a fitted CLIPER at {n_sig} of {len(te_sig)} horizons. "
              + ("Report it as comparable to CLIPER, not better."
                 if n_sig <= 1 else
                 "The improvement is statistically supported at those horizons."))

    # --- persist ---------------------------------------------------------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": model.architecture(),
        # The feature layout travels with the weights. Loading a checkpoint under a
        # different feature order produces confident, wrong forecasts and no error.
        "feature_names": list(TRACK_FEATURE_NAMES),
        "history_steps": args.history_steps,
        "horizons_h": list(horizons_h),
        "displacement_scale_km": DISPLACEMENT_SCALE_KM,
        "motion_scale_kmh": MOTION_SCALE_KMH,
        "train_seasons": sorted({s.season for s in tr}),
        "val_seasons": sorted({s.season for s in va}),
        "test_seasons": sorted({s.season for s in te}),
        "n_train_samples": len(tr),
        "best_epoch": best["epoch"],
        "val_mean_km": best["score"],
        "evaluation": results,
        "dataset": "IBTrACS v04r01 North Indian Ocean (NOAA NCEI)",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, out)
    print(f"\ncheckpoint -> {out}")

    Path(args.report).write_text(json.dumps(
        {"evaluation": results, "training_history": history,
         "best_epoch": best["epoch"], "n_train_samples": len(tr),
         "feature_names": list(TRACK_FEATURE_NAMES)}, indent=2, default=float))
    print(f"report     -> {args.report}")
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--cliper-out", default=str(DEFAULT_CLIPER_OUT))
    p.add_argument("--report", default=str(DEFAULT_REPORT))
    p.add_argument("--min-season", type=int, default=1990)
    p.add_argument("--history-steps", type=int, default=DEFAULT_HISTORY_STEPS)
    p.add_argument("--hidden-dim", type=int, default=48)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--no-residual", action="store_true",
                   help="predict displacement directly instead of a correction to "
                        "persistence (expect markedly worse results)")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--huber-delta", type=float, default=0.2)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--plateau-patience", type=int, default=12)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cpu")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train(build_parser().parse_args())
