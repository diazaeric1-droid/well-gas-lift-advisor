# Gas-Lift Advisor

**Upstream Copilot Suite · Optimize stage**

Fits a Gas-Lift Performance Curve (GLPC) from injection survey data, finds each well's
economic injection optimum analytically, and allocates a compressor-limited total injection
budget across the fleet using the equal-marginal-revenue principle.

**Live demo → [gas-lift-advisor.streamlit.app](https://gas-lift-advisor.streamlit.app)**
&nbsp;&nbsp;|&nbsp;&nbsp;
**Suite → [pe-pipeline.streamlit.app](https://pe-pipeline.streamlit.app)**

---

## What it does

| Tab | Content |
|---|---|
| **Fleet Dashboard** | Ranked table of all wells by $/day value at stake; KPIs; bar chart |
| **Per-Well Analysis** | GLPC scatter + fitted curve; economic curve; recommendation card |
| **Fleet Allocation** | Optimal injection split under compressor capacity limit |

**BYOD CSV**: upload `well_id, date, injection_gas_mcfd, bopd, bwpd` — column validation, template download, nothing stored server-side.

---

## Physics

**GLPC model** (Brown 1984, Takács 2005):

```
q_liq(Qinj) = q_sl + (q_max − q_sl) × (1 − exp(−a × Qinj))
```

| Symbol | Meaning | Units |
|---|---|---|
| `q_sl` | Static liquid rate (zero injection) | bopd |
| `q_max` | Plateau liquid rate (large injection) | bopd |
| `a` | Efficiency coefficient | Mscfd⁻¹ |
| `Qinj` | Gas injection rate | Mscfd |

**Economic optimum** (set dNet/dQinj = 0):

```
Qinj_opt = ln[(q_max − q_sl) · a · (1 − wc) · price · NRI / gas_cost] / a
```

Returns `Qinj_opt = 0` when gas cost ≥ marginal revenue at zero injection.

**Fleet allocation** uses the **equal-marginal-revenue principle**: at the constrained
optimum, `dNet_i/dQinj_i = λ` (same shadow price for all wells). Solved by bisecting on
`λ` — exact, not a greedy approximation.

---

## Eval gates (CI)

| Metric | Gate | Meaning |
|---|---|---|
| GLPC fit R² (mean) | ≥ 0.90 | Survey data is identifiable |
| Optimization accuracy | ≥ 0.80 | Recommended injection within 10% of true optimum |
| Economic capture | ≥ 0.95 | Net revenue at recommended rate ≥ 95% of true maximum |

On the 20-well synthetic fleet: **R² 0.956, opt accuracy 0.95, econ capture 1.00**.

---

## Tech stack

Python · NumPy · SciPy (curve_fit, brentq) · pandas · Plotly · Streamlit

No LLM required — pure petroleum engineering math.  
Vendored `econ_core.py` (suite-wide discounting convention) + `theme.py` + `fleet_registry.py`.

---

## Suite position

```
Design → Monitor → Diagnose → Predict → Quantify → Optimize → Authorize → Allocate → Orchestrate
  WPS     Digest    PECopilot    ESP      Deferment   Gas-Lift   AFE        Capital    Pipeline
```

---

## Run locally

```bash
git clone https://github.com/diazaeric1-droid/well-gas-lift-advisor
cd well-gas-lift-advisor
pip install -e ".[demo,dev]"
python data/synthetic/generate_fleet.py
streamlit run demo/app.py
```

## Run tests

```bash
python data/synthetic/generate_fleet.py
python evals/optimization_accuracy.py
pytest -q
```

---

*Eric A. Diaz II · Houston, TX · 9 years upstream PE (OXY/Shell Permian + GoM)*
