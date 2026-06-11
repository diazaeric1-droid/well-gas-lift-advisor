"""econ_core — single source of truth for upstream project & intervention economics.

Vendored byte-identical into every economics-bearing app in the Upstream Copilot Suite
(Production Engineer Copilot, AFE Copilot, Capital Program Optimizer, PE Pipeline) so all
of them discount, risk, and roll up cash flows with ONE convention.

Why this exists
---------------
Before this module each app reimplemented discounted cash flow, and they had quietly
diverged: PE Copilot discounted at ``(1 + r/12)**m`` — a 10% input compounded monthly to a
**10.47% effective annual rate** — while AFE Copilot and Capital Optimizer used the correct
``(1 + r)**(m/12)`` (a true effective-annual 10%). Same "10% NPV", three engines, two
answers. econ_core gives everyone one tested kernel and eliminates that drift.

Conventions
-----------
* Rates are oil bopd; volumes bbl; cash USD. Discount and decline are **annual** (1/yr).
* Month index ``m`` runs 1..N with end-of-month cash; ``DAYS_PER_MONTH = 365.25/12``.
* "net margin" / "net revenue" is already net of LOE and any water-disposal drag ($/bbl).
* Discounting is **effective-annual**: ``DF(m) = (1 + r)**(m/12)``. A 10% input means
  10% per *year*, full stop.
* Risking is one convention everywhere:
      ``risked_NPV = pc * PV(net revenue) - cost``
  Cost is certain (you spend it whether or not the upside lands); only the revenue PV is
  chance-weighted. This is algebraically identical to the dry-hole framing
  ``pc * NPV_success + (1 - pc) * (-cost)`` — see ``risked_npv`` and its test.

Sources: SPE-PRMS (reserves & P-naming); standard upstream DCF (Mian, *Project Economics
and Decision Analysis*; SPE 10% corporate hurdle convention). Arps (1945) decline curves.
"""
from __future__ import annotations

import numpy as np

DAYS_PER_MONTH = 365.25 / 12.0
DEFAULT_DISCOUNT = 0.10


def month_index(horizon_years: int) -> np.ndarray:
    """1-based month indices 1..12*horizon_years (end-of-month cash convention)."""
    return np.arange(1, int(horizon_years) * 12 + 1)


def discount_factors(months, annual_rate: float = DEFAULT_DISCOUNT) -> np.ndarray:
    """Effective-annual discount factors ``DF(m) = (1 + r)**(m/12)``.

    A 10% input is 10% *per year*. The monthly-compounded form ``(1 + r/12)**m`` implies a
    10.47% effective annual rate and is exactly the bug this core removes — do not use it.
    """
    months = np.asarray(months, dtype=float)
    return (1.0 + annual_rate) ** (months / 12.0)


def arps_monthly_rate(qi: float, di: float, b: float, months) -> np.ndarray:
    """Arps (1945) rate (bopd) at each month. ``b≈0`` → exponential, else hyperbolic.

    qi  initial rate (bopd); di nominal annual decline (1/yr); b hyperbolic exponent.
    """
    t = np.asarray(months, dtype=float) / 12.0
    if b < 1e-6:
        return qi * np.exp(-di * t)
    return qi / np.power(1.0 + b * di * t, 1.0 / b)


def exp_uplift_rate(qi: float, decline_per_yr: float, months) -> np.ndarray:
    """Exponential decline of an intervention *uplift* (bopd): ``qi * exp(-D t)``.

    ``qi`` is the initial incremental rate; ``decline_per_yr`` the uplift's annual decline.
    Accepts scalar or array ``qi``/``decline_per_yr`` (broadcast against ``months``) so the
    same kernel serves the point estimate and a Monte-Carlo batch.
    """
    qi = np.asarray(qi, dtype=float)
    decline = np.asarray(decline_per_yr, dtype=float)
    t = np.asarray(months, dtype=float) / 12.0
    if qi.ndim or decline.ndim:                 # batched: -> (n_draws, n_months)
        return qi[..., None] * np.exp(-decline[..., None] * t[None, ...])
    return qi * np.exp(-decline * t)


def discounted_pv(monthly_net_revenue, annual_rate: float = DEFAULT_DISCOUNT):
    """PV of an end-of-month net-revenue stream; time is the last axis.

    Shape ``(T,)`` → ``float``; shape ``(n, T)`` → ``(n,)`` array (Monte-Carlo batch).
    """
    mr = np.asarray(monthly_net_revenue, dtype=float)
    months = np.arange(1, mr.shape[-1] + 1)
    pv = np.sum(mr / discount_factors(months, annual_rate), axis=-1)
    return float(pv) if np.ndim(pv) == 0 else pv


def npv(pv_net_revenue, cost):
    """NPV = PV(net revenue) − cost. Scalar or array in ``pv_net_revenue``."""
    return np.asarray(pv_net_revenue, dtype=float) - cost


def risked_npv(pv_net_revenue, cost, pc: float = 1.0):
    """Risked NPV = ``pc * PV(net revenue) - cost``.

    Cost is certain; only the revenue PV is chance-weighted. Identical to the dry-hole
    framing ``pc * (PV - cost) + (1 - pc) * (-cost)`` (proven in test_econ_core). Scalar or
    array in ``pv_net_revenue``; returns the matching shape.
    """
    out = pc * np.asarray(pv_net_revenue, dtype=float) - cost
    return float(out) if np.ndim(out) == 0 else out


def payout_months(monthly_net_revenue, cost) -> float:
    """First 1-based month where cumulative *undiscounted* net revenue ≥ cost (else inf)."""
    mr = np.asarray(monthly_net_revenue, dtype=float)
    cum = np.cumsum(mr)
    idx = int(np.searchsorted(cum, cost))
    return float(idx + 1) if idx < mr.shape[0] else float("inf")


def irr_annual(monthly_cf, capex: float) -> float | None:
    """Annual IRR via bisection on the rate zeroing NPV of ``[-capex, monthly_cf...]``.

    Uses the same effective-annual discounting as everything else. Returns None if NPV is
    negative even at a deep discount, or 500.0 if still positive at 500%.
    """
    monthly_cf = np.asarray(monthly_cf, dtype=float)
    months = np.arange(1, len(monthly_cf) + 1)

    def _npv(rate_annual: float) -> float:
        if rate_annual <= -0.999:
            return float("inf")
        df = (1.0 + rate_annual) ** (months / 12.0)
        return float(-capex + np.sum(monthly_cf / df))

    lo, hi = -0.9, 5.0
    if _npv(lo) < 0:
        return None
    if _npv(hi) > 0:
        return 500.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _npv(mid) > 0:
            lo = mid
        else:
            hi = mid
    return round(0.5 * (lo + hi) * 100.0, 1)
