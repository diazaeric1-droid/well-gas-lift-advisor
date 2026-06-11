"""glpc — Gas-Lift Performance Curve fitting and injection optimization.

GLPC model (empirical exponential plateau):
    q_liq(Qinj) = q_sl + (q_max − q_sl) × (1 − exp(−a × Qinj))

    q_sl   = static liquid rate at zero injection [bopd]
    q_max  = maximum liquid rate (plateau as Qinj → ∞) [bopd]
    a      = efficiency coefficient [Mscfd⁻¹]
    Qinj   = gas injection rate [Mscfd]

Economic optimum (dNet/dQinj = 0):
    Qinj_opt = ln[(q_max − q_sl) · a · (1 − wc) · price · nri / gas_cost] / a

When gas_cost ≥ marginal revenue at Qinj = 0, optimum is 0 (injection never pays).

Fleet allocation under a compressor capacity limit uses the equal-marginal-revenue
principle: at the constrained optimum, dNet_i/dQinj_i = λ (same shadow price for
all wells). This is solved by bisecting on λ — exact, not a greedy approximation.

Sources
-------
Brown, K.E. (1984). "The Technology of Artificial Lift Methods," Vol. 4.
Takács, G. (2005). "Gas Lift Manual." PennWell.
Golan, M. & Whitson, C.H. (1991). "Well Performance," 2nd ed.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

try:
    from scipy.optimize import curve_fit, brentq
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


# ---- data structures -------------------------------------------------------

@dataclass
class GLPCParams:
    """Fitted parameters for the exponential GLPC model."""
    q_sl: float    # static (zero-injection) liquid rate [bopd]
    q_max: float   # plateau liquid rate [bopd]
    a: float       # efficiency coefficient [1/Mscfd]
    r2: float = 1.0  # goodness of fit (1.0 if manually specified)


@dataclass
class WellOptimum:
    """Economic optimum for a single gas-lift well."""
    q_inj_opt: float            # optimal injection rate [Mscfd]
    q_liq_opt: float            # gross liquid at optimum [bopd]
    q_oil_opt: float            # oil rate at optimum [bopd]
    net_revenue_per_day: float  # $/day at optimum
    q_inj_display_max: float    # injection ceiling for display (2× opt or 0.5 Mscfd min)


# ---- core functions --------------------------------------------------------

def glpc_rate(q_inj, params: GLPCParams) -> np.ndarray:
    """Evaluate the fitted GLPC: q_liq = q_sl + (q_max − q_sl) × (1 − exp(−a × Qinj))."""
    q = np.asarray(q_inj, dtype=float)
    return params.q_sl + (params.q_max - params.q_sl) * (1.0 - np.exp(-params.a * q))


def fit_glpc(q_inj: np.ndarray, q_liq: np.ndarray) -> GLPCParams:
    """Fit GLPC parameters from injection-rate / liquid-rate data pairs.

    Requires scipy.  Returns a fallback estimate (r2=0) if scipy is absent or
    the fit fails to converge.  Caller should check r2 for quality.
    """
    q_inj = np.asarray(q_inj, dtype=float)
    q_liq = np.asarray(q_liq, dtype=float)

    # --- initial guesses ---
    order = np.argsort(q_inj)
    # q_sl: median liquid rate at the lowest 10% of injection rates
    low_mask = q_inj <= np.percentile(q_inj, 10)
    q_sl0 = float(np.median(q_liq[low_mask])) if low_mask.sum() >= 2 else float(q_liq.min())
    q_max0 = float(np.median(q_liq[q_inj >= np.percentile(q_inj, 90)])) * 1.05
    q_max0 = max(q_max0, q_sl0 * 1.5 + 1.0)
    # a: half-saturation estimate: at q_half, production ≈ 0.5×(q_max-q_sl) above q_sl
    q_target = q_sl0 + 0.5 * (q_max0 - q_sl0)
    above = q_liq >= q_target
    if above.any():
        q_half = float(q_inj[above].min())
        a0 = float(np.log(2.0) / max(q_half, 0.1))  # = 0.693 / q_half
    else:
        a0 = 1.0

    def _model(x, q_sl, q_max, a):
        return q_sl + (q_max - q_sl) * (1.0 - np.exp(-a * x))

    if _HAS_SCIPY and len(q_inj) >= 4:
        try:
            popt, _ = curve_fit(
                _model, q_inj, q_liq,
                p0=[q_sl0, q_max0, a0],
                bounds=([0.0, q_sl0 * 0.5 + 1.0, 1e-3], [q_max0 * 2.0, q_max0 * 5.0, 20.0]),
                maxfev=10_000,
            )
            q_pred = _model(q_inj, *popt)
            ss_res = float(np.sum((q_liq - q_pred) ** 2))
            ss_tot = float(np.sum((q_liq - q_liq.mean()) ** 2))
            r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0
            return GLPCParams(
                q_sl=float(popt[0]),
                q_max=float(popt[1]),
                a=float(popt[2]),
                r2=round(r2, 4),
            )
        except Exception:  # noqa: BLE001
            pass

    # Fallback: crude estimate (q_sl = 10th-pctile, q_max = max, a=1)
    return GLPCParams(q_sl=q_sl0, q_max=q_max0, a=a0, r2=0.0)


def net_revenue_daily(
    q_inj,
    params: GLPCParams,
    water_cut: float,
    oil_price: float,
    gas_cost_per_mscf: float,
    nri: float,
) -> np.ndarray:
    """Net revenue per day at each injection rate [$/day].

    net_revenue = q_oil × oil_price × nri − Qinj × gas_cost
    """
    q = np.asarray(q_inj, dtype=float)
    q_oil = glpc_rate(q, params) * (1.0 - water_cut)
    return q_oil * oil_price * nri - q * gas_cost_per_mscf


def optimal_injection(
    params: GLPCParams,
    water_cut: float,
    oil_price: float,
    gas_cost_per_mscf: float,
    nri: float,
) -> WellOptimum:
    """Compute the economic optimum injection rate analytically.

    Derived from dNet/dQinj = 0:
        (q_max − q_sl) · a · exp(−a·Qinj) · (1−wc) · price · nri = gas_cost
        → Qinj_opt = ln[revenue_slope / gas_cost] / a

    ``revenue_slope`` = (q_max − q_sl) · a · (1−wc) · price · nri is the marginal
    revenue ($/Mscfd) at Qinj = 0.  If gas_cost ≥ revenue_slope, injection never
    makes money (optimum = 0).
    """
    dq = params.q_max - params.q_sl          # incremental capacity [bopd]
    revenue_slope = dq * params.a * (1.0 - water_cut) * oil_price * nri  # $/Mscfd at Qinj=0

    if revenue_slope <= gas_cost_per_mscf or dq <= 0:
        q_opt = 0.0
    else:
        q_opt = max(0.0, float(np.log(revenue_slope / gas_cost_per_mscf) / params.a))

    q_liq_opt = float(glpc_rate(q_opt, params))
    q_oil_opt = q_liq_opt * (1.0 - water_cut)
    net_rev = q_oil_opt * oil_price * nri - q_opt * gas_cost_per_mscf

    display_max = max(q_opt * 2.5, 0.5)  # sensible x-axis ceiling for charts

    return WellOptimum(
        q_inj_opt=round(q_opt, 3),
        q_liq_opt=round(q_liq_opt, 1),
        q_oil_opt=round(q_oil_opt, 1),
        net_revenue_per_day=round(net_rev, 2),
        q_inj_display_max=round(display_max, 2),
    )


# ---- fleet allocation -------------------------------------------------------

def _q_at_shadow(shadow_price: float, w: dict, oil_price: float,
                 gas_cost_per_mscf: float, nri: float) -> float:
    """Optimal injection for well ``w`` when effective gas cost = gas_cost + shadow_price."""
    p: GLPCParams = w["params"]
    wc: float = w["water_cut"]
    dq = p.q_max - p.q_sl
    revenue_slope = dq * p.a * (1.0 - wc) * oil_price * nri
    eff_cost = gas_cost_per_mscf + shadow_price
    if revenue_slope <= eff_cost or dq <= 0:
        return 0.0
    return max(0.0, float(np.log(revenue_slope / eff_cost) / p.a))


def allocate_fleet(
    wells: list[dict],
    total_injection_mscfd: float,
    oil_price: float,
    gas_cost_per_mscf: float,
    nri: float,
) -> list[dict]:
    """Optimal fleet injection allocation under a compressor capacity limit.

    Uses the equal-marginal-revenue principle: at the constrained optimum,
    dNet_i/dQinj_i = λ for all wells.  λ is found by bisecting on the shadow
    price until sum of well allocations equals ``total_injection_mscfd``.

    Each dict in ``wells`` must have::

        {well_id: str, params: GLPCParams, water_cut: float, current_q_inj: float}

    Returns a list of dicts::

        {well_id, allocated_q_inj, expected_q_liq, expected_q_oil, expected_net_rev_day}
    """
    if not wells:
        return []

    # Unconstrained optima
    unconstrained = {
        w["well_id"]: _q_at_shadow(0.0, w, oil_price, gas_cost_per_mscf, nri)
        for w in wells
    }
    total_unconstrained = sum(unconstrained.values())

    if total_unconstrained <= total_injection_mscfd + 1e-6:
        # Constraint not binding
        allocations = unconstrained
    elif not _HAS_SCIPY:
        # Manual bisection fallback
        lo, hi = 0.0, 1e6
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if sum(_q_at_shadow(mid, w, oil_price, gas_cost_per_mscf, nri)
                   for w in wells) > total_injection_mscfd:
                lo = mid
            else:
                hi = mid
        sp = 0.5 * (lo + hi)
        allocations = {
            w["well_id"]: _q_at_shadow(sp, w, oil_price, gas_cost_per_mscf, nri)
            for w in wells
        }
    else:
        def _total(sp):
            return sum(_q_at_shadow(sp, w, oil_price, gas_cost_per_mscf, nri)
                       for w in wells)
        sp_star = brentq(lambda sp: _total(sp) - total_injection_mscfd, 0.0, 1e8)
        allocations = {
            w["well_id"]: _q_at_shadow(sp_star, w, oil_price, gas_cost_per_mscf, nri)
            for w in wells
        }

    result = []
    for w in wells:
        q = allocations[w["well_id"]]
        q_liq = float(glpc_rate(q, w["params"]))
        q_oil = q_liq * (1.0 - w["water_cut"])
        net_rev = q_oil * oil_price * nri - q * gas_cost_per_mscf
        result.append({
            "well_id": w["well_id"],
            "allocated_q_inj": round(q, 3),
            "expected_q_liq": round(q_liq, 1),
            "expected_q_oil": round(q_oil, 1),
            "expected_net_rev_day": round(net_rev, 2),
        })

    return result
