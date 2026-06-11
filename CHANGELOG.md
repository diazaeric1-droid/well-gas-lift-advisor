# Changelog — Gas-Lift Advisor

All notable changes to this project will be documented in this file.

## [0.1.0] — 2026-06-11

### Added
- **Gas-Lift Performance Curve (GLPC)** fitting via nonlinear least squares on injection test data:
  `q_liq(Qinj) = q_sl + (q_max − q_sl) × (1 − exp(−a × Qinj))`
- **Analytical injection optimum**: `Qinj_opt = ln[(q_max−q_sl)·a·(1−wc)·price·nri / gas_cost] / a`
- **Fleet allocation** under compressor capacity limit via equal-marginal-revenue principle
  (shadow price bisection — exact, not greedy approximation)
- **Streamlit app** with three tabs: Fleet Dashboard · Per-Well Analysis · Fleet Allocation
- **BYOD CSV upload**: `well_id, date, injection_gas_mcfd, bopd, bwpd`; column validation + template download
- **Synthetic fleet**: 20 Permian-flavored wells with realistic GLPC parameters;
  injection intentionally scattered around (not at) the true optimum so optimization matters
- **Eval gate**: GLPC fit R² ≥ 0.90, optimization accuracy ≥ 0.80, economic capture ≥ 0.95
- **CI**: AppTest render-smoke on Python 3.14; eval gate; pytest
- Vendored `econ_core.py` + `theme.py` + `fleet_registry.py` (byte-identical to suite)
