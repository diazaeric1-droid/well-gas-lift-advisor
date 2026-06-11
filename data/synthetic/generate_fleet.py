"""Generate synthetic gas-lift fleet data for the Gas-Lift Advisor demo.

Produces:
  data/synthetic/fleet/well_0NN.csv  — 120-day injection history with embedded survey
  data/synthetic/ground_truth.csv    — true GLPC parameters + true optimum per well

Data structure (120 days per well)
-----------------------------------
* Days 1–20   : pre-survey baseline at current operating injection rate
* Days 21–96  : formal injection survey (19 levels × 4 days, low→high)
* Days 97–120 : post-survey operation back at current rate

The full dataset is used for GLPC fitting; the survey period provides the variation
needed for curve identification, and the tail gives a clean "current injection" signal
(app uses `tail(7).mean()` to estimate the current operating rate).

GLPC model: q_liq = q_sl + (q_max − q_sl) × (1 − exp(−a × Qinj))

Run: python data/synthetic/generate_fleet.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FLEET_DIR = ROOT / "data" / "synthetic" / "fleet"
GT_PATH = ROOT / "data" / "synthetic" / "ground_truth.csv"

_REF_PRICE = 70.0
_REF_NRI = 0.80
_REF_GAS_COST = 1.50

N_WELLS = 20
PRE_DAYS = 20    # baseline before survey
SURVEY_LEVELS = 19
DAYS_PER_LEVEL = 4  # survey = 76 days
POST_DAYS = 24   # back to current after survey
TOTAL_DAYS = PRE_DAYS + SURVEY_LEVELS * DAYS_PER_LEVEL + POST_DAYS  # = 120
SEED = 42


def _true_opt(q_sl, q_max, a, wc,
              price=_REF_PRICE, nri=_REF_NRI, gas=_REF_GAS_COST):
    dq = q_max - q_sl
    rev_slope = dq * a * (1.0 - wc) * price * nri
    if rev_slope <= gas:
        return 0.0
    return max(0.0, float(np.log(rev_slope / gas) / a))


def main():
    FLEET_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    end_date = pd.Timestamp("2024-06-30")
    dates = pd.date_range(end=end_date, periods=TOTAL_DAYS, freq="D")

    gt_rows = []

    for n in range(1, N_WELLS + 1):
        well_id = f"well_{n:03d}"

        # GLPC parameters
        q_sl = float(rng.uniform(120, 380))
        q_max = q_sl * float(rng.uniform(1.6, 3.2))
        a = float(rng.uniform(0.7, 2.8))
        water_cut = float(rng.uniform(0.28, 0.68))
        opt_inj = _true_opt(q_sl, q_max, a, water_cut)

        # current injection: scattered ±40% around optimum
        if rng.random() < 0.50:
            bias = rng.uniform(1.12, 1.55)   # over-injected
        else:
            bias = rng.uniform(0.50, 0.90)   # under-injected
        q_inj_current = max(0.10, opt_inj * bias if opt_inj > 0.05 else rng.uniform(0.2, 1.5))

        # Injection rate array (120 days)
        # Pre-survey: at current rate (± small daily noise)
        pre = np.full(PRE_DAYS, q_inj_current) + rng.normal(0, 0.05, PRE_DAYS)

        # Survey: log-spaced from near-zero to 2.2× optimum so the rising portion
        # of the GLPC (where 'a' is identifiable) gets denser coverage.
        survey_max = max(opt_inj * 2.2, 1.5)
        survey_levels = np.exp(
            np.linspace(np.log(0.05), np.log(survey_max), SURVEY_LEVELS)
        )
        survey = np.repeat(survey_levels, DAYS_PER_LEVEL)
        survey += rng.normal(0, 0.02, len(survey))

        # Post-survey: back at current rate
        post = np.full(POST_DAYS, q_inj_current) + rng.normal(0, 0.05, POST_DAYS)

        q_inj = np.concatenate([pre, survey, post])
        q_inj = np.maximum(0.02, q_inj)

        # Production: GLPC + noise
        q_liq_true = q_sl + (q_max - q_sl) * (1.0 - np.exp(-a * q_inj))
        noise = rng.normal(1.0, 0.04, TOTAL_DAYS)
        q_liq = np.maximum(5.0, q_liq_true * noise)
        bwpd = np.maximum(0.0, q_liq * water_cut * rng.normal(1.0, 0.015, TOTAL_DAYS))
        bopd = np.maximum(0.0, q_liq - bwpd)

        df = pd.DataFrame({
            "date": dates.date,
            "injection_gas_mcfd": np.round(q_inj, 3),
            "bopd": np.round(bopd, 1),
            "bwpd": np.round(bwpd, 1),
        })
        df.to_csv(FLEET_DIR / f"{well_id}.csv", index=False)

        gt_rows.append({
            "well_id": well_id,
            "q_sl": round(q_sl, 2),
            "q_max": round(q_max, 2),
            "a": round(a, 5),
            "water_cut": round(water_cut, 5),
            "true_opt_inj": round(opt_inj, 5),
            "current_inj": round(q_inj_current, 3),
            "over_injected": int(q_inj_current > opt_inj),
        })

    pd.DataFrame(gt_rows).to_csv(GT_PATH, index=False)
    n_over = sum(r["over_injected"] for r in gt_rows)
    print(f"Generated {N_WELLS} wells → {FLEET_DIR}")
    print(f"  {n_over}/{N_WELLS} over-injected · {N_WELLS - n_over}/{N_WELLS} under-injected")
    print(f"  {TOTAL_DAYS} rows/well: {PRE_DAYS}d baseline + {SURVEY_LEVELS * DAYS_PER_LEVEL}d survey ({SURVEY_LEVELS} levels) + {POST_DAYS}d post")
    print(f"  Ground truth → {GT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
