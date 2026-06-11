"""Physics + economics tests for src/glpc.py.

Each test pins a specific numerical result so regressions are caught immediately.
"""
from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.glpc import (
    GLPCParams,
    allocate_fleet,
    fit_glpc,
    glpc_rate,
    net_revenue_daily,
    optimal_injection,
)

# Canonical test well
PARAMS = GLPCParams(q_sl=200.0, q_max=600.0, a=1.0, r2=1.0)
WC = 0.40
PRICE = 70.0
GAS = 1.50
NRI = 0.80


# ---- glpc_rate ---------------------------------------------------------------

def test_glpc_zero_injection_returns_q_sl():
    assert abs(float(glpc_rate(0.0, PARAMS)) - PARAMS.q_sl) < 0.01


def test_glpc_large_injection_approaches_q_max():
    q = float(glpc_rate(20.0, PARAMS))   # 20 Mscfd → virtually at plateau
    assert abs(q - PARAMS.q_max) < 0.5


def test_glpc_monotone_increasing():
    q_inj = np.linspace(0, 5, 50)
    q_liq = glpc_rate(q_inj, PARAMS)
    assert np.all(np.diff(q_liq) >= 0)


# ---- optimal_injection -------------------------------------------------------

def test_optimal_injection_analytical():
    """Qinj_opt = ln(revenue_slope / gas_cost) / a — pin the number."""
    dq = PARAMS.q_max - PARAMS.q_sl     # 400
    rev_slope = dq * PARAMS.a * (1 - WC) * PRICE * NRI  # 400*1*0.6*70*0.8 = 13440
    expected_opt = np.log(rev_slope / GAS) / PARAMS.a   # ln(8960) ≈ 9.10
    opt = optimal_injection(PARAMS, WC, PRICE, GAS, NRI)
    assert abs(opt.q_inj_opt - expected_opt) < 0.001


def test_optimal_injection_is_net_revenue_maximum():
    """Net revenue at Qinj_opt ≥ net revenue at any other injection rate."""
    opt = optimal_injection(PARAMS, WC, PRICE, GAS, NRI)
    q_test = np.linspace(0, opt.q_inj_opt * 2, 500)
    rev_test = net_revenue_daily(q_test, PARAMS, WC, PRICE, GAS, NRI)
    assert float(net_revenue_daily(opt.q_inj_opt, PARAMS, WC, PRICE, GAS, NRI)) >= rev_test.max() - 0.1


def test_no_injection_when_gas_too_expensive():
    """When gas cost exceeds the marginal revenue at Qinj=0, optimal injection = 0."""
    # marginal revenue at 0 = dq * a * (1-wc) * price * nri = 400*1*0.6*70*0.8 = 13440
    # set gas_cost > 13440 → no injection
    very_expensive_gas = 20_000.0
    opt = optimal_injection(PARAMS, WC, PRICE, very_expensive_gas, NRI)
    assert opt.q_inj_opt == 0.0


# ---- net_revenue_daily -------------------------------------------------------

def test_net_revenue_at_zero_injection():
    """At Qinj=0: rev = q_sl × (1−wc) × price × nri (no gas cost)."""
    expected = PARAMS.q_sl * (1 - WC) * PRICE * NRI
    got = float(net_revenue_daily(0.0, PARAMS, WC, PRICE, GAS, NRI))
    assert abs(got - expected) < 0.01


# ---- fit_glpc ----------------------------------------------------------------

def test_fit_glpc_recovers_true_params():
    """Fit on noise-free data should recover params within 1%."""
    q_inj = np.linspace(0.05, 5.0, 30)
    q_liq = glpc_rate(q_inj, PARAMS)
    fitted = fit_glpc(q_inj, q_liq)
    assert abs(fitted.q_sl - PARAMS.q_sl) / PARAMS.q_sl < 0.01
    assert abs(fitted.q_max - PARAMS.q_max) / PARAMS.q_max < 0.01
    assert abs(fitted.a - PARAMS.a) / PARAMS.a < 0.01
    assert fitted.r2 >= 0.999


def test_fit_glpc_r2_high_for_clean_data():
    rng = np.random.default_rng(7)
    q_inj = np.linspace(0.1, 4.0, 25)
    q_liq = glpc_rate(q_inj, PARAMS) * rng.normal(1.0, 0.02, 25)
    fitted = fit_glpc(q_inj, q_liq)
    assert fitted.r2 >= 0.97


# ---- allocate_fleet ----------------------------------------------------------

def test_allocation_unconstrained_each_gets_optimum():
    """If cap ≥ sum of optima, every well gets its unconstrained optimum."""
    wells = [
        {"well_id": "w1", "params": GLPCParams(150, 400, 1.2), "water_cut": 0.35, "current_q_inj": 1.0},
        {"well_id": "w2", "params": GLPCParams(200, 600, 0.8), "water_cut": 0.45, "current_q_inj": 2.0},
    ]
    opt_1 = optimal_injection(wells[0]["params"], wells[0]["water_cut"], PRICE, GAS, NRI).q_inj_opt
    opt_2 = optimal_injection(wells[1]["params"], wells[1]["water_cut"], PRICE, GAS, NRI).q_inj_opt
    big_cap = (opt_1 + opt_2) * 2
    result = allocate_fleet(wells, big_cap, PRICE, GAS, NRI)
    alloc = {r["well_id"]: r["allocated_q_inj"] for r in result}
    assert abs(alloc["w1"] - opt_1) < 0.01
    assert abs(alloc["w2"] - opt_2) < 0.01


def test_allocation_constrained_respects_cap():
    """Total allocated injection must be ≤ total_injection_mscfd."""
    wells = [
        {"well_id": f"w{i}", "params": GLPCParams(100 + i * 50, 400 + i * 50, 1.0 + i * 0.2),
         "water_cut": 0.40, "current_q_inj": 2.0}
        for i in range(5)
    ]
    cap = 5.0
    result = allocate_fleet(wells, cap, PRICE, GAS, NRI)
    total_alloc = sum(r["allocated_q_inj"] for r in result)
    assert total_alloc <= cap + 0.01


def test_allocation_equal_marginal_revenue():
    """At constrained optimum, marginal revenue should be equal (within tolerance) for all wells."""
    wells = [
        {"well_id": "w1", "params": GLPCParams(200, 600, 1.5), "water_cut": 0.40, "current_q_inj": 2.0},
        {"well_id": "w2", "params": GLPCParams(150, 500, 1.0), "water_cut": 0.50, "current_q_inj": 1.5},
        {"well_id": "w3", "params": GLPCParams(100, 400, 2.0), "water_cut": 0.35, "current_q_inj": 1.0},
    ]
    # Set cap tight so constraint binds
    sum_opt = sum(
        optimal_injection(w["params"], w["water_cut"], PRICE, GAS, NRI).q_inj_opt
        for w in wells
    )
    cap = sum_opt * 0.5  # only 50% of unconstrained total
    result = allocate_fleet(wells, cap, PRICE, GAS, NRI)

    # Marginal revenue at each well's allocation
    def marginal(w_data, q):
        p = w_data["params"]
        wc = w_data["water_cut"]
        dq = p.q_max - p.q_sl
        return dq * p.a * np.exp(-p.a * q) * (1 - wc) * PRICE * NRI

    mrs = []
    alloc_map = {r["well_id"]: r["allocated_q_inj"] for r in result}
    for w in wells:
        q = alloc_map[w["well_id"]]
        if q > 1e-3:
            mrs.append(marginal(w, q))

    if len(mrs) >= 2:
        # All non-zero-allocated wells should have equal marginal revenue within 1%
        assert max(mrs) - min(mrs) < 0.01 * max(mrs) + 0.5
