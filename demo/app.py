"""Gas-Lift Advisor — Streamlit demo app.

Stage: Optimize (between Quantify / Deferment IQ and Authorize / AFE Copilot)

Tabs
----
1. Fleet Dashboard — ranked table + KPIs showing value at stake per well
2. Per-Well Analysis — GLPC plot, economic curve, recommendation card
3. Fleet Allocation — optimal injection split under a compressor capacity limit
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- path bootstrap: repo root for `src.*`, demo dir for vendored theme/registry
# (Streamlit adds the entrypoint dir at runtime; AppTest / other contexts may not).
DEMO_DIR = Path(__file__).resolve().parent
ROOT = DEMO_DIR.parent
for _p in (str(ROOT), str(DEMO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import theme
import fleet_registry
from src.glpc import (
    GLPCParams,
    WellOptimum,
    fit_glpc,
    glpc_rate,
    net_revenue_daily,
    optimal_injection,
    allocate_fleet,
)

# ---- page config ------------------------------------------------------------
theme.setup_page("Gas-Lift Advisor", icon="⛽")
theme.suite_nav("gas-lift")

theme.header(
    "Gas-Lift Advisor",
    subtitle="Optimize injection rates · fleet allocation under compressor limits · BYOD",
    chips=[("v0.1.0", "ver"), ("Optimize stage", "info"),
           ("Permian synthetic fleet", "eval")],
)
theme.data_badge("synthetic", "20-well Permian-flavored synthetic fleet · BYOD CSV supported")

# ---- sidebar ----------------------------------------------------------------
with st.sidebar:
    st.markdown("### Data source")
    data_src = st.radio(
        "Source",
        ["Synthetic fleet (20 wells)", "Upload your own CSV"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Economics")
    oil_price = st.slider("Oil price ($/bbl)", 30.0, 150.0, 70.0, 5.0)
    gas_cost = st.slider("Injection gas cost ($/Mscf)", 0.25, 6.0, 1.50, 0.25)
    nri = st.slider("Net revenue interest (%)", 50.0, 100.0, 80.0, 1.0) / 100.0

    st.markdown("---")
    st.markdown("### Fleet allocation")
    comp_cap = st.slider(
        "Compressor capacity (Mscfd total)",
        1.0, 60.0, 20.0, 1.0,
        help="Total gas available for injection across the fleet.",
    )

# ---- constants --------------------------------------------------------------
FLEET_DIR = ROOT / "data" / "synthetic" / "fleet"
REQUIRED_COLS = {"well_id", "date", "injection_gas_mcfd", "bopd", "bwpd"}

TEMPLATE_CSV = "well_id,date,injection_gas_mcfd,bopd,bwpd\n" \
               "well_001,2024-01-01,1.20,320.5,185.2\n" \
               "well_001,2024-01-02,0.80,275.3,159.4\n"


# ---- data loading -----------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_synthetic_fleet() -> dict[str, pd.DataFrame]:
    if not FLEET_DIR.exists() or not any(FLEET_DIR.glob("well_*.csv")):
        return {}
    fleet = {}
    for p in sorted(FLEET_DIR.glob("well_*.csv")):
        df = pd.read_csv(p, parse_dates=["date"])
        fleet[p.stem] = df
    return fleet


def load_byod_fleet(path: str) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(path, parse_dates=["date"])
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return {wid: grp.reset_index(drop=True) for wid, grp in df.groupby("well_id")}


# ---- data acquisition -------------------------------------------------------
fleet: dict[str, pd.DataFrame] = {}

if data_src.startswith("Upload"):
    st.sidebar.download_button(
        "⬇ Download template CSV",
        data=TEMPLATE_CSV,
        file_name="gas_lift_template.csv",
        mime="text/csv",
    )
    st.sidebar.caption("Nothing stored server-side.")
    uploaded = st.file_uploader(
        "Upload fleet CSV (`well_id, date, injection_gas_mcfd, bopd, bwpd`)",
        type=["csv"],
    )
    if uploaded is None:
        st.info("Upload a fleet CSV to get started, or switch to the Synthetic fleet.")
        st.stop()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name
    try:
        fleet = load_byod_fleet(tmp_path)
        st.success(f"Loaded {len(fleet)} wells from uploaded CSV.")
    except ValueError as exc:
        st.error(f"Column validation failed: {exc}")
        st.stop()
else:
    with st.spinner("Loading synthetic fleet…"):
        fleet = load_synthetic_fleet()
    if not fleet:
        st.error(
            "Synthetic fleet not found. Run `python data/synthetic/generate_fleet.py` first."
        )
        st.stop()

if not fleet:
    st.warning("No wells loaded.")
    st.stop()


# ---- per-well analysis -------------------------------------------------------

def _analyze_well(
    well_id: str, df: pd.DataFrame
) -> tuple[GLPCParams, float, float, WellOptimum]:
    """Fit GLPC + compute optimum for one well. Returns (params, water_cut, current_inj, opt)."""
    q_inj = df["injection_gas_mcfd"].values.astype(float)
    bopd = df["bopd"].values.astype(float)
    bwpd = df["bwpd"].values.astype(float)
    q_liq = bopd + bwpd

    mask = q_inj > 0.05
    n_pts = int(mask.sum())
    if n_pts >= 4:
        params = fit_glpc(q_inj[mask], q_liq[mask])
    else:
        # Not enough variation; placeholder
        q_sl = float(np.percentile(q_liq, 10))
        q_max = float(q_liq.max()) * 1.1
        params = GLPCParams(q_sl=q_sl, q_max=q_max, a=1.0, r2=0.0)

    liq_sum = q_liq.sum()
    water_cut = float(bwpd.sum() / liq_sum) if liq_sum > 0 else 0.5
    current_inj = float(df["injection_gas_mcfd"].tail(7).mean())

    opt = optimal_injection(params, water_cut, oil_price, gas_cost, nri)
    return params, water_cut, current_inj, opt


@st.cache_data(show_spinner=False)
def build_fleet_table(
    oil_price: float, gas_cost: float, nri: float
) -> pd.DataFrame:
    """Cached fleet summary DataFrame. Keyed on economic assumptions."""
    rows = []
    for well_id, df in fleet.items():
        params, wc, cur_inj, opt = _analyze_well(well_id, df)
        cur_liq = float(glpc_rate(cur_inj, params))
        cur_oil = cur_liq * (1.0 - wc)
        cur_rev = cur_oil * oil_price * nri - cur_inj * gas_cost

        delta_oil = opt.q_oil_opt - cur_oil
        daily_gain = opt.net_revenue_per_day - cur_rev

        if cur_inj > opt.q_inj_opt + 0.05:
            status = "Over-injected"
        elif opt.q_inj_opt - cur_inj > 0.05:
            status = "Under-injected"
        else:
            status = "At optimum"

        meta = fleet_registry.get(well_id)
        rows.append({
            "well_id": well_id,
            "formation": meta.formation,
            "lift": meta.lift,
            "current_inj_mscfd": round(cur_inj, 2),
            "optimal_inj_mscfd": round(opt.q_inj_opt, 2),
            "current_bopd": round(cur_oil, 1),
            "optimal_bopd": round(opt.q_oil_opt, 1),
            "delta_bopd": round(delta_oil, 1),
            "daily_gain_usd": round(daily_gain, 2),
            "status": status,
            "glpc_r2": round(params.r2, 3),
            "water_cut": round(wc, 3),
        })

    df_out = pd.DataFrame(rows).sort_values("daily_gain_usd", ascending=False)
    return df_out.reset_index(drop=True)


fleet_df = build_fleet_table(oil_price, gas_cost, nri)

# ---- tabs -------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["Fleet Dashboard", "Per-Well Analysis", "Fleet Allocation"]
)

# =============================================================================
# TAB 1 — Fleet Dashboard
# =============================================================================
with tab1:
    n_over = int((fleet_df["status"] == "Over-injected").sum())
    n_under = int((fleet_df["status"] == "Under-injected").sum())
    n_opt = int((fleet_df["status"] == "At optimum").sum())
    total_gain_day = float(fleet_df["daily_gain_usd"].clip(lower=0).sum())
    annual_opp = total_gain_day * 365.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Over-Injected Wells", f"{n_over}",
              delta="wasting gas", delta_color="inverse")
    c2.metric("Under-Injected Wells", f"{n_under}",
              delta="leaving oil", delta_color="inverse")
    c3.metric("Daily Value at Stake", f"${total_gain_day:,.0f}/day",
              delta=f"${annual_opp/1e6:.1f}MM/yr potential")
    c4.metric("At Optimum", f"{n_opt}")

    theme.source_note(
        "Daily value at stake = net revenue at recommended injection − net revenue at current injection "
        "(oil_price × NRI − gas_cost; downside-clipped to 0 for fleet total). "
        "Source: Brown (1984); Takács (2005)."
    )

    st.markdown("#### Fleet Optimization Summary")

    # Color map for status
    STATUS_COLOR = {
        "Over-injected": theme.RED,
        "Under-injected": theme.AMBER,
        "At optimum": theme.GREEN,
    }

    # Bar chart: daily gain per well
    chart_df = fleet_df[fleet_df["daily_gain_usd"].abs() > 0].copy()
    chart_df["color"] = chart_df["status"].map(STATUS_COLOR).fillna(theme.GREY)
    fig_bar = go.Figure(go.Bar(
        x=chart_df["well_id"],
        y=chart_df["daily_gain_usd"],
        marker_color=chart_df["color"].tolist(),
        text=chart_df["daily_gain_usd"].map(lambda v: f"${v:,.0f}"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Daily gain: $%{y:,.0f}<extra></extra>",
    ))
    fig_bar.update_layout(
        title="Daily Value at Stake by Well ($/day)",
        xaxis_title="Well",
        yaxis_title="$/day",
        showlegend=False,
    )
    st.plotly_chart(theme.style_fig(fig_bar, height=340), use_container_width=True)

    # Fleet table
    display_cols = {
        "well_id": "Well",
        "current_inj_mscfd": "Current Inj (Mscfd)",
        "optimal_inj_mscfd": "Optimal Inj (Mscfd)",
        "current_bopd": "Current BOPD",
        "optimal_bopd": "Optimal BOPD",
        "delta_bopd": "Δ BOPD",
        "daily_gain_usd": "Daily Gain ($/day)",
        "status": "Status",
        "glpc_r2": "GLPC R²",
    }
    styled = fleet_df[list(display_cols.keys())].rename(columns=display_cols)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Export fleet table (CSV)",
        data=fleet_df.to_csv(index=False),
        file_name="gas_lift_fleet_optimization.csv",
        mime="text/csv",
    )

    theme.references(["gas_lift", "npv"])

# =============================================================================
# TAB 2 — Per-Well Analysis
# =============================================================================
with tab2:
    well_ids_sorted = fleet_df["well_id"].tolist()
    selected = st.selectbox("Select well", well_ids_sorted, key="well_select")

    if selected not in fleet:
        st.warning("Well data not found.")
        st.stop()

    df_w = fleet[selected]
    params_w, wc_w, cur_inj_w, opt_w = _analyze_well(selected, df_w)
    meta_w = fleet_registry.get(selected)

    # well metadata row
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.markdown(f"**Basin:** {meta_w.basin}")
    col_b.markdown(f"**Formation:** {meta_w.formation}")
    col_c.markdown(f"**Lift:** {meta_w.lift}")
    row = fleet_df[fleet_df["well_id"] == selected].iloc[0]
    status = row["status"]
    kind = {"Over-injected": "high", "Under-injected": "warn", "At optimum": "ok"}.get(status, "ok")
    theme.flag(status, kind)
    theme.well_cross_links("gas-lift", selected)

    # injection rate history chart
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=df_w["date"], y=df_w["injection_gas_mcfd"],
        mode="lines", name="Injection rate",
        line=dict(color=theme.BLUE, width=1.5),
    ))
    fig_hist.add_hline(
        y=opt_w.q_inj_opt, line=dict(color=theme.GREEN, width=1.5, dash="dot"),
        annotation_text=f"Opt {opt_w.q_inj_opt:.2f} Mscfd",
        annotation_position="right",
    )
    fig_hist.add_hline(
        y=cur_inj_w, line=dict(color=theme.AMBER, width=1.2, dash="dash"),
        annotation_text=f"Avg {cur_inj_w:.2f} Mscfd",
        annotation_position="right",
    )
    fig_hist.update_layout(
        title=f"{selected} — Injection rate history",
        xaxis_title="Date", yaxis_title="Injection gas (Mscfd)",
    )
    st.plotly_chart(theme.style_fig(fig_hist, height=240), use_container_width=True)

    col_left, col_right = st.columns(2)

    # GLPC chart
    with col_left:
        q_range = np.linspace(0, max(opt_w.q_inj_display_max, cur_inj_w * 1.5), 200)
        q_liq_curve = glpc_rate(q_range, params_w)

        # scatter: actual data points
        q_inj_actual = df_w["injection_gas_mcfd"].values
        q_liq_actual = (df_w["bopd"] + df_w["bwpd"]).values

        fig_glpc = go.Figure()
        fig_glpc.add_trace(go.Scatter(
            x=q_inj_actual, y=q_liq_actual,
            mode="markers", name="Field data",
            marker=dict(color=theme.GREY, size=5, opacity=0.6),
        ))
        fig_glpc.add_trace(go.Scatter(
            x=q_range, y=q_liq_curve,
            mode="lines", name=f"GLPC fit (R²={params_w.r2:.3f})",
            line=dict(color=theme.BLUE, width=2),
        ))
        fig_glpc.add_trace(go.Scatter(
            x=[opt_w.q_inj_opt], y=[opt_w.q_liq_opt],
            mode="markers", name=f"Optimal ({opt_w.q_inj_opt:.2f} Mscfd)",
            marker=dict(color=theme.GREEN, size=14, symbol="star"),
        ))
        fig_glpc.add_trace(go.Scatter(
            x=[cur_inj_w], y=[float(glpc_rate(cur_inj_w, params_w))],
            mode="markers", name=f"Current ({cur_inj_w:.2f} Mscfd)",
            marker=dict(color=theme.AMBER, size=10, symbol="diamond"),
        ))
        fig_glpc.update_layout(
            title="Gas-Lift Performance Curve",
            xaxis_title="Injection gas (Mscfd)",
            yaxis_title="Gross liquid (bopd)",
        )
        st.plotly_chart(theme.style_fig(fig_glpc, height=340), use_container_width=True)
        theme.source_note(f"q_sl={params_w.q_sl:.0f} bopd · q_max={params_w.q_max:.0f} bopd · a={params_w.a:.3f} Mscfd⁻¹")

    # Economic curve chart
    with col_right:
        net_rev_curve = net_revenue_daily(q_range, params_w, wc_w, oil_price, gas_cost, nri)
        cur_rev = float(net_revenue_daily(cur_inj_w, params_w, wc_w, oil_price, gas_cost, nri))

        fig_econ = go.Figure()
        fig_econ.add_trace(go.Scatter(
            x=q_range, y=net_rev_curve,
            mode="lines", name="Net revenue/day",
            line=dict(color=theme.BLUE, width=2),
            fill="tozeroy", fillcolor=f"rgba(79,129,189,0.08)",
        ))
        fig_econ.add_trace(go.Scatter(
            x=[opt_w.q_inj_opt], y=[opt_w.net_revenue_per_day],
            mode="markers", name=f"Optimal ${opt_w.net_revenue_per_day:,.0f}/day",
            marker=dict(color=theme.GREEN, size=14, symbol="star"),
        ))
        fig_econ.add_trace(go.Scatter(
            x=[cur_inj_w], y=[cur_rev],
            mode="markers", name=f"Current ${cur_rev:,.0f}/day",
            marker=dict(color=theme.AMBER, size=10, symbol="diamond"),
        ))
        fig_econ.add_hline(y=0, line=dict(color=theme.RED, width=1, dash="dot"))
        fig_econ.update_layout(
            title="Net Revenue vs. Injection Rate",
            xaxis_title="Injection gas (Mscfd)",
            yaxis_title="Net revenue ($/day)",
        )
        st.plotly_chart(theme.style_fig(fig_econ, height=340), use_container_width=True)
        theme.source_note("Net revenue = BOPD × (1 − WC) × price × NRI − Qinj × gas_cost")

    # Recommendation card
    daily_gain = opt_w.net_revenue_per_day - cur_rev
    delta_inj = opt_w.q_inj_opt - cur_inj_w
    direction = "Reduce" if delta_inj < -0.05 else ("Increase" if delta_inj > 0.05 else "Maintain")

    st.markdown("#### Recommendation")
    rec_bg = theme.RED if direction == "Reduce" else (theme.AMBER if direction == "Increase" else theme.GREEN)
    st.markdown(
        f"""
        <div style="background:{rec_bg}15; border-left:4px solid {rec_bg};
                    border-radius:8px; padding:1rem 1.2rem; margin-bottom:0.8rem;">
        <b style="font-size:1.05rem">{direction} injection: {cur_inj_w:.2f} → {opt_w.q_inj_opt:.2f} Mscfd</b>
        <br><br>
        Expected result: <b>{opt_w.q_oil_opt:.0f} BOPD</b> (from {float(glpc_rate(cur_inj_w, params_w))*(1-wc_w):.0f} BOPD) &nbsp;|&nbsp;
        Daily net revenue: <b>${opt_w.net_revenue_per_day:,.0f}/day</b> (from ${cur_rev:,.0f}/day) &nbsp;|&nbsp;
        Daily gain: <b>${daily_gain:,.0f}/day</b> · <b>${daily_gain*365/1e6:.2f}MM/year</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    theme.how_to(
        "- The **GLPC chart** shows how liquid production responds to injection gas. "
        "The curve is fit via nonlinear least squares from historical data with injection variation.\n"
        "- The **economic curve** shows net revenue (oil value net of gas injection cost) vs. injection rate. "
        "The green star marks the maximum.\n"
        "- The **recommendation** is derived analytically: set marginal revenue = marginal cost (gas injection cost). "
        "No LLM is used — pure petroleum engineering math.\n"
        "- Source: Brown (1984) *Artificial Lift Methods*; Takács (2005) *Gas Lift Manual*."
    )
    theme.references(["gas_lift", "npv"])

# =============================================================================
# TAB 3 — Fleet Allocation
# =============================================================================
with tab3:
    st.markdown(
        f"**Total injection budget:** {comp_cap:.1f} Mscfd &nbsp;|&nbsp; "
        f"Economics: ${oil_price:.0f}/bbl oil · ${gas_cost:.2f}/Mscf gas · {nri*100:.0f}% NRI"
    )

    # Build per-well allocation input
    well_inputs = []
    for well_id, df_i in fleet.items():
        params_i, wc_i, cur_i, opt_i = _analyze_well(well_id, df_i)
        well_inputs.append({
            "well_id": well_id,
            "params": params_i,
            "water_cut": wc_i,
            "current_q_inj": cur_i,
        })

    alloc_result = allocate_fleet(well_inputs, comp_cap, oil_price, gas_cost, nri)
    alloc_by_id = {r["well_id"]: r for r in alloc_result}

    # Build comparison table
    comp_rows = []
    for w in well_inputs:
        wid = w["well_id"]
        cur = w["current_q_inj"]
        alloc = alloc_by_id[wid]["allocated_q_inj"]
        opt_row = fleet_df[fleet_df["well_id"] == wid]
        unc_opt = float(opt_row["optimal_inj_mscfd"].iloc[0]) if not opt_row.empty else 0.0

        cur_oil = float(glpc_rate(cur, w["params"])) * (1.0 - w["water_cut"])
        alloc_oil = alloc_by_id[wid]["expected_q_oil"]
        alloc_rev = alloc_by_id[wid]["expected_net_rev_day"]

        comp_rows.append({
            "well_id": wid,
            "current_inj": round(cur, 2),
            "allocated_inj": round(alloc, 2),
            "unconstrained_opt": round(unc_opt, 2),
            "current_bopd": round(cur_oil, 1),
            "allocated_bopd": alloc_by_id[wid]["expected_q_oil"],
            "allocated_rev_day": alloc_rev,
        })

    comp_df = pd.DataFrame(comp_rows).sort_values("allocated_rev_day", ascending=False)

    # Fleet-level KPIs
    total_cur_inj = comp_df["current_inj"].sum()
    total_alloc_inj = comp_df["allocated_inj"].sum()
    total_cur_oil = comp_df["current_bopd"].sum()
    total_alloc_oil = comp_df["allocated_bopd"].sum()
    total_alloc_rev = comp_df["allocated_rev_day"].sum()

    # Current revenue
    total_cur_rev = sum(
        float(net_revenue_daily(w["current_q_inj"], w["params"], w["water_cut"], oil_price, gas_cost, nri))
        for w in well_inputs
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current total injection", f"{total_cur_inj:.1f} Mscfd")
    k2.metric("Allocated injection", f"{total_alloc_inj:.1f} Mscfd",
              delta=f"cap: {comp_cap:.0f} Mscfd")
    k3.metric("Current total BOPD", f"{total_cur_oil:,.0f}")
    k4.metric("Allocated total BOPD", f"{total_alloc_oil:,.0f}",
              delta=f"+{total_alloc_oil - total_cur_oil:,.0f}")

    rev_gain = total_alloc_rev - total_cur_rev
    st.metric(
        "Fleet daily revenue gain from reallocation",
        f"${rev_gain:,.0f}/day",
        delta=f"${rev_gain * 365 / 1e6:.2f}MM/year",
    )

    # Grouped bar chart: current vs allocated vs unconstrained
    fig_alloc = go.Figure()
    fig_alloc.add_trace(go.Bar(
        x=comp_df["well_id"], y=comp_df["current_inj"],
        name="Current", marker_color=theme.GREY,
    ))
    fig_alloc.add_trace(go.Bar(
        x=comp_df["well_id"], y=comp_df["allocated_inj"],
        name="Allocated (constrained)", marker_color=theme.BLUE,
    ))
    fig_alloc.add_trace(go.Bar(
        x=comp_df["well_id"], y=comp_df["unconstrained_opt"],
        name="Unconstrained optimum", marker_color=theme.GREEN,
        opacity=0.5,
    ))
    fig_alloc.update_layout(
        title=f"Gas Injection Allocation (cap = {comp_cap:.0f} Mscfd)",
        barmode="group",
        xaxis_title="Well", yaxis_title="Injection gas (Mscfd)",
    )
    st.plotly_chart(theme.style_fig(fig_alloc, height=360), use_container_width=True)

    # Table
    disp_alloc = comp_df.rename(columns={
        "well_id": "Well",
        "current_inj": "Current Inj (Mscfd)",
        "allocated_inj": "Allocated (Mscfd)",
        "unconstrained_opt": "Unconstrained Opt (Mscfd)",
        "current_bopd": "Current BOPD",
        "allocated_bopd": "Allocated BOPD",
        "allocated_rev_day": "Allocated Rev ($/day)",
    })
    st.dataframe(disp_alloc, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Export allocation (CSV)",
        data=comp_df.to_csv(index=False),
        file_name="gas_lift_allocation.csv",
        mime="text/csv",
    )

    theme.how_to(
        "- **Fleet allocation** uses the **equal-marginal-revenue principle**: at the constrained "
        "optimum, the marginal value of the last Mscfd injected is the same for every well (shadow price λ). "
        "This is solved exactly by bisecting on λ — no greedy approximation.\n"
        "- If the sum of unconstrained optima ≤ compressor cap, every well gets its optimum "
        "and the constraint is not binding.\n"
        "- Drag the **Compressor capacity** slider in the sidebar to see how tighter / looser "
        "injection budgets redistribute across the fleet."
    )
    theme.references(["gas_lift", "npv"])
