"""Eval gate: GLPC fit quality + optimization accuracy on the synthetic fleet.

Metrics
-------
r2_mean         Average GLPC fit R² across all wells (gate ≥ 0.90)
opt_accuracy    Fraction of wells where recommended injection is within 10% of
                true optimum (gate ≥ 0.80)
econ_capture    Net revenue at recommended injection / max possible (gate ≥ 0.95)

Writes evals/optimization_report.json with per-well results + aggregate summary.

Run: python evals/optimization_accuracy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.glpc import GLPCParams, fit_glpc, glpc_rate, optimal_injection, net_revenue_daily

FLEET_DIR = ROOT / "data" / "synthetic" / "fleet"
GT_PATH = ROOT / "data" / "synthetic" / "ground_truth.csv"
REPORT_PATH = ROOT / "evals" / "optimization_report.json"

# Default economics (same as app defaults)
OIL_PRICE = 70.0
GAS_COST = 1.50
NRI = 0.80

# CI gates
R2_MIN = 0.90
OPT_ACC_MIN = 0.80
ECON_CAP_MIN = 0.95
OPT_TOL = 0.10   # within 10% of true optimal injection


def _compute_water_cut(df: pd.DataFrame) -> float:
    q_liq = df["bopd"] + df["bwpd"]
    return float(df["bwpd"].sum() / q_liq.sum()) if q_liq.sum() > 0 else 0.5


def main() -> int:
    if not GT_PATH.exists():
        print("ERROR: ground_truth.csv not found — run data/synthetic/generate_fleet.py first")
        return 1

    gt = pd.read_csv(GT_PATH).set_index("well_id")
    well_ids = sorted(gt.index.tolist())

    if not FLEET_DIR.exists() or not any(FLEET_DIR.glob("well_*.csv")):
        print("ERROR: fleet CSVs missing — run data/synthetic/generate_fleet.py first")
        return 1

    rows = []
    r2_vals, acc_flags, econ_caps = [], [], []

    for well_id in well_ids:
        csv = FLEET_DIR / f"{well_id}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        q_inj = df["injection_gas_mcfd"].values
        q_liq = df["bopd"].values + df["bwpd"].values
        water_cut = _compute_water_cut(df)

        mask = (q_inj > 0.05)
        params = fit_glpc(q_inj[mask], q_liq[mask])
        opt = optimal_injection(params, water_cut, OIL_PRICE, GAS_COST, NRI)

        true_row = gt.loc[well_id]
        true_opt = float(true_row["true_opt_inj"])
        true_wc = float(true_row["water_cut"])
        true_params = GLPCParams(
            q_sl=float(true_row["q_sl"]),
            q_max=float(true_row["q_max"]),
            a=float(true_row["a"]),
        )

        # opt_accuracy: recommended injection within OPT_TOL of true optimum
        denom = max(true_opt, 0.1)
        on_target = abs(opt.q_inj_opt - true_opt) / denom <= OPT_TOL

        # economic capture: net revenue at RECOMMENDED injection evaluated against the
        # TRUE GLPC (what does operating at that rate actually earn?) vs. the theoretical
        # maximum (true params, true optimum). Uses true water_cut throughout.
        rev_at_rec = float(net_revenue_daily(
            opt.q_inj_opt, true_params, true_wc, OIL_PRICE, GAS_COST, NRI))
        rev_at_true = float(net_revenue_daily(
            true_opt, true_params, true_wc, OIL_PRICE, GAS_COST, NRI))
        cap = min(rev_at_rec / rev_at_true, 1.0) if rev_at_true > 1.0 else 1.0

        r2_vals.append(params.r2)
        acc_flags.append(int(on_target))
        econ_caps.append(cap)

        rows.append({
            "well_id": well_id,
            "r2": round(params.r2, 4),
            "true_opt_inj": round(true_opt, 3),
            "recommended_inj": round(opt.q_inj_opt, 3),
            "on_target": on_target,
            "econ_capture": round(cap, 4),
        })

    r2_mean = float(np.mean(r2_vals))
    opt_acc = float(np.mean(acc_flags))
    econ_cap = float(np.mean(econ_caps))

    summary = {
        "n_wells": len(rows),
        "r2_mean": round(r2_mean, 4),
        "opt_accuracy": round(opt_acc, 4),
        "econ_capture": round(econ_cap, 4),
        "gates": {
            "r2_mean": {"value": round(r2_mean, 4), "min": R2_MIN, "pass": r2_mean >= R2_MIN},
            "opt_accuracy": {"value": round(opt_acc, 4), "min": OPT_ACC_MIN, "pass": opt_acc >= OPT_ACC_MIN},
            "econ_capture": {"value": round(econ_cap, 4), "min": ECON_CAP_MIN, "pass": econ_cap >= ECON_CAP_MIN},
        },
        "per_well": rows,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(summary, indent=2))

    print(f"GLPC fit R² (mean):      {r2_mean:.4f}  (gate ≥ {R2_MIN})")
    print(f"Optimization accuracy:   {opt_acc:.4f}  (gate ≥ {OPT_ACC_MIN})")
    print(f"Economic capture (mean): {econ_cap:.4f}  (gate ≥ {ECON_CAP_MIN})")

    failures = [
        k for k, g in summary["gates"].items() if not g["pass"]
    ]
    if failures:
        print(f"\nFAIL: eval gate(s) below threshold: {failures}")
        return 1

    print("\nPASS: all eval gates green ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
