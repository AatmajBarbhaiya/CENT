#!/usr/bin/env python3
"""app.py — DDA explorer: replay alpha(t) through the controller and inspect every split.

Run:  cd experiments/dda_full_sweep/app && streamlit run app.py

Design in results/APP_TRACE_VIEW_PLAN.md. Every performance number comes from explorer/dse_metrics.py
-- the same module the CSV writers and figures import -- so the app and the paper data cannot drift.
"""
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "explorer")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "simulation")))
import dse_metrics as M          # noqa: E402
import dda_controller as DC      # noqa: E402

import traces as TR             # noqa: E402  -- shared with simulation/controller_eval.py
import tput_latency_frontier as F   # noqa: E402  -- same code as the figure

st.set_page_config(page_title="DDA explorer", layout="wide")

# Theme. Streamlit cannot reliably report the active theme to the script, so it is an explicit
# toggle: the plot ink/grid/paper must invert or half the labels vanish against the background.
_theme = st.sidebar.radio("theme", ["light", "dark"], horizontal=True, key="_thm")
DARK = _theme == "dark"
# Series hues are held constant across themes -- colour identifies the series, not the mode.
# Only ink / grid / paper move, which is what actually needs to invert for legibility.
C = dict(L="#4c9ae8" if DARK else "#2a78d6",
         T="#f5854f" if DARK else "#eb6834",
         spill="#f5c542" if DARK else "#f2b705",
         unmet="#e5697b" if DARK else "#d1495b",
         ok="#35c98e" if DARK else "#1baf7a",
         ink="#e8eaed" if DARK else "#1f2328",
         muted="#9aa3ad" if DARK else "#6b7280",
         grid="#3a4048" if DARK else "#e5e7eb",
         paper="rgba(0,0,0,0)",
         legend_bg="rgba(24,28,33,.75)" if DARK else "rgba(255,255,255,.72)",
         legend_border="#4a5158" if DARK else "#e5e7eb")
PLOT_BG = C["paper"]
ACTION_ORDER = ["PROVISION", "HOLD", "SPILL_HOLD", "SHRINK_WAIT", "DRAINING",
                "GROW_B", "GROW_C", "SHRINK", "GREEDY", "INFEASIBLE"]


@st.cache_data
def load_table():
    return pd.read_csv(os.path.join(M.DDA, "dse_full_table.csv"))


@st.cache_data
def load_ladder(model):
    t = load_table()
    return pd.DataFrame([dict(N_L=r.N_L, N_T=r.N_T, k=r.j, TP=r.m,
                              Tput_L=round(r.Tput_L, 1), Tput_T=round(r.Tput_T, 1),
                              TTFT_L_s=r.TTFT_L_s, req_lat_T_s=r.req_lat_T_s,
                              kv_slack_L=r.kv_slack_L, kv_slack_T=r.kv_slack_T)
                         for r in DC.ladder(model, t)])


# ---------------------------------------------------------------- sidebar
st.sidebar.title("DDA explorer")
table = load_table()
model = st.sidebar.selectbox("Model", M.MODELS)

base = DC.make_controller(model, table)
if base is None:
    st.error(f"No feasible ladder for {model}")
    st.stop()
cap = max(r.Tput_L + r.Tput_T for r in base.rungs)

st.sidebar.markdown("**Workload**")
lam = st.sidebar.slider("λ  (offered tokens/s, fixed)", 0.0, float(round(1.05 * cap)),
                        float(round(0.45 * cap)), step=float(max(1, round(cap / 200))))
motion = st.sidebar.selectbox("α motion", TR.MOTIONS)
a_mean = st.sidebar.slider("α mean", 0.02, 0.97, 0.30, 0.01)
a_amp = st.sidebar.slider("α amplitude", 0.0, 0.5, 0.25, 0.01)
period_h = st.sidebar.slider("period (h)", 0.5, 24.0, 24.0, 0.5)
noise = st.sidebar.slider("α noise (σ)", 0.0, 0.10, 0.0, 0.005)
# 'brief' is only meaningful RELATIVE TO T_grow. The n/100 default is 14 min over a 24 h horizon,
# while 7B's T_grow is 47 s -- so the default spike is ~18x too long and the controller is right
# to grow. Size it as a multiple of T_grow; below 1.0 the controller must ignore it.
spike_mult = st.sidebar.slider("spike length (× T_grow)", 0.1, 20.0, 0.5, 0.1,
                               help=f"T_grow = {base.t_grow():.0f}s for {model}. "
                                    "<1 means the spike is shorter than the commit threshold, "
                                    "so the controller should NOT reconfigure.")

st.sidebar.markdown("**Horizon**")
hours = st.sidebar.slider("horizon (h)", 1, 48, 24)
dt = st.sidebar.select_slider("dt (s)", [10, 15, 30, 60, 120], value=30)

st.sidebar.markdown("**Policy**")
policy = st.sidebar.radio("policy", ["hysteresis", "greedy", "static"], horizontal=True)
max_spill = st.sidebar.slider(
    "max_spill_frac ⚠️", 0.0, 0.5, 0.10, 0.01,
    help="Chosen, not measured. Sets how much of the latency class may be served by Pool T "
         "before the controller starts its grow timer. Needs a sensitivity sweep before the paper.")
grow_f = st.sidebar.slider("T_grow / T_drain", 0.5, 10.0, 2.0, 0.5)
half = st.sidebar.slider("EWMA half-life (s)", 30, 900, 120, 30)
min_shrink = st.sidebar.slider("min shrink devices", 1, 8, 2)
seed = st.sidebar.number_input("noise seed", value=0, step=1)

n = int(hours * 3600 / dt)
spike_ticks = max(1, int(spike_mult * base.t_grow() / dt))
al = TR.alpha_motion(motion, n, a_mean, a_amp, period_h, hours, noise,
                     np.random.default_rng(int(seed)), spike_ticks=spike_ticks)
ctrl, log = DC.run_trace(model, al, np.full(n, lam), dt=float(dt), table=table, policy=policy,
                         max_spill_frac=max_spill, grow_factor=grow_f,
                         ewma_halflife=float(half), min_shrink_devices=int(min_shrink))
if log.empty:
    st.error("No log produced.")
    st.stop()
log["hour"] = log.t / 3600.0
ev = pd.DataFrame(ctrl.events) if ctrl.events else pd.DataFrame()

# ---------------------------------------------------------------- headline counters
lat_tok = (log.load_L.fillna(0) * dt).sum()
spill_tok = (log.spilled.fillna(0) * dt).sum()
unmet_tok = (log.unmet.fillna(0) * dt).sum()
w = (log.served_by_L.fillna(0) + log.spilled.fillna(0)) * dt
eff = (log.eff_ttft_s.fillna(0) * w).sum() / w.sum() if w.sum() > 0 else np.nan
n_be = int(ev.breakeven_t.notna().sum()) if len(ev) else 0

st.markdown(f"### {model} · {motion} · λ = {lam:,.0f} tok/s · {hours} h @ {dt}s ticks "
            f"({n:,} ticks) · policy **{policy}**")

# The four that decide whether the deployment is acceptable. Everything else is diagnostic and
# lives in the expander below.
c = st.columns(4)
c[0].metric("effective TTFT", f"{eff:.3f} s" if eff == eff else "—",
            help="What the latency class ACTUALLY experienced: Pool L's TTFT for routed requests "
                 "blended with Pool T's much slower completion latency for spilled ones. This is "
                 "the QoS number, not the brochure number.")
c[1].metric("unmet fraction", f"{unmet_tok/lat_tok:.1%}" if lat_tok else "—",
            help="Share of latency-class demand neither pool could take. Anything above 0 is "
                 "dropped or queued load — a hard failure, unlike spill which is merely slow.")
c[2].metric("spill fraction", f"{spill_tok/lat_tok:.1%}" if lat_tok else "—",
            help="Share of latency-class demand served by Pool T instead of Pool L. Free in "
                 "capacity, expensive in latency. The controller's max_spill_frac gates this.")
c[3].metric("reconfigurations", len(ev),
            help="Split changes committed. Each costs a drain of the devices whose role changed.")

with st.expander("Diagnostics — drain cost, ledgers, tick timing", expanded=False):
    d = st.columns(5)
    d[0].metric("break-even reached", f"{n_be} of {len(ev)}" if len(ev) else "—",
                help="Reconfigurations that eventually repaid their drain, in EITHER currency. "
                     "'0 of n' is a real result: the move was not worth making.")
    d[1].metric("tokens lost to drain", f"{log.tokens_lost.sum():,.0f}",
                help="Tokens forfeited while draining devices were out of service.")
    d[2].metric("ledger (tokens)", f"{ctrl.ledger:,.0f}",
                help="∫(served_now − served_by_the_split_we_left). Only turns positive when a "
                     "move relieves genuinely UNMET demand, since devices merely move between "
                     "pools and total throughput is roughly conserved.")
    d[3].metric("ledger (request-s)", f"{ctrl.ledger_lat:,.0f}",
                help="∫(latency_before − latency_now) × served. Where shrinks and most grows "
                     "actually pay: request-seconds saved across both classes.")
    d[4].metric("tick", f"{dt} s",
                help=f"Control period. One tick = {dt}s of simulated time; "
                     f"{n:,} ticks span {hours} h. T_drain={base.t_drain:.0f}s, "
                     f"T_grow={base.t_grow():.0f}s — both must be several ticks to be resolved.")

tab_trace, tab_front, tab_ladder, tab_splits = st.tabs(
    ["Trace replay", "Throughput vs latency", "Ladder", "All splits"])

# ---------------------------------------------------------------- trace replay
with tab_trace:
    # (title, y-axis label) per row. Titles sit ABOVE each panel and the y label names the unit,
    # so no panel has to be identified from its traces alone.
    PANELS = [
        ("1 · Interactive fraction α — raw vs the EWMA the controller acts on", "α"),
        ("2 · Offered load λ", "tokens/s"),
        ("3 · Deployed split — devices in each pool", "devices"),
        ("4 · Pool L factorization — instances k and TP per instance", "count"),
        ("5 · ROUTING: where latency-class requests actually went", "tokens/s"),
        ("6 · Pool headroom — offered load vs capacity", "tokens/s"),
        ("7 · Latency (log scale)", "seconds"),
        ("8 · Throughput delivered vs offered", "tokens/s"),
        ("9 · LEDGER vs the split we left behind", "tokens  /  k request-s"),
        ("10 · KV slack — capacity gate", "× C_need"),
    ]
    titles, ylabels = zip(*PANELS)
    fig = make_subplots(rows=10, cols=1, shared_xaxes=True, vertical_spacing=0.042,
                        row_heights=[.9, .6, 1, .8, 1.2, 1, 1, 1, 1.1, .8],
                        subplot_titles=[f"<b>{t}</b>" for t in titles])
    h = log.hour

    # Each panel gets its OWN legend, anchored to that panel's y-domain, so a legend entry is
    # always adjacent to the traces it names. Plotly needs one `legend{i}` per group and each
    # trace routed to it explicitly.
    def add(tr, row):
        tr.update(legend=("legend" if row == 1 else f"legend{row}"))
        fig.add_trace(tr, row=row, col=1)

    add(go.Scatter(x=h, y=log.alpha, name="α", line=dict(color=C["ink"], width=1)), 1)
    add(go.Scatter(x=h, y=log.alpha_s, name="α EWMA", line=dict(color=C["L"], width=2)), 1)
    add(go.Scatter(x=h, y=log.lam, name="λ", line=dict(color=C["muted"], width=1.5)), 2)

    add(go.Scatter(x=h, y=log.N_L, name="N_L", line=dict(color=C["L"], width=2, shape="hv"),
                   fill="tozeroy", fillcolor="rgba(42,120,214,.12)"), 3)
    add(go.Scatter(x=h, y=log.N_T, name="N_T",
                   line=dict(color=C["T"], width=2, shape="hv", dash="dash")), 3)
    add(go.Scatter(x=h, y=log.j, name="k (instances)",
                   line=dict(color=C["ok"], width=2, shape="hv")), 4)
    add(go.Scatter(x=h, y=log.m, name="TP per instance",
                   line=dict(color="#8a5cd6", width=2, shape="hv", dash="dash")), 4)

    # the panel that answers "where did requests go"
    add(go.Scatter(x=h, y=log.served_by_L, name="served by Pool L", stackgroup="r",
                   line=dict(width=0), fillcolor="rgba(42,120,214,.75)"), 5)
    add(go.Scatter(x=h, y=log.spilled, name="spilled → Pool T", stackgroup="r",
                   line=dict(width=0), fillcolor="rgba(242,183,5,.85)"), 5)
    add(go.Scatter(x=h, y=log.unmet, name="unmet", stackgroup="r",
                   line=dict(width=0), fillcolor="rgba(209,73,91,.85)"), 5)

    add(go.Scatter(x=h, y=log.load_L, name="load_L", line=dict(color=C["L"], width=1.5)), 6)
    add(go.Scatter(x=h, y=log.cap_L, name="Tput_L", line=dict(color=C["L"], width=1, dash="dot")), 6)
    add(go.Scatter(x=h, y=log.load_T, name="load_T", line=dict(color=C["T"], width=1.5)), 6)
    add(go.Scatter(x=h, y=log.cap_T, name="Tput_T", line=dict(color=C["T"], width=1, dash="dot")), 6)

    add(go.Scatter(x=h, y=log.eff_ttft_s, name="effective TTFT",
                   line=dict(color=C["unmet"], width=2)), 7)
    add(go.Scatter(x=h, y=log.TTFT_L_s, name="Pool L TTFT",
                   line=dict(color=C["L"], width=1, dash="dot")), 7)
    add(go.Scatter(x=h, y=log.req_lat_T_s, name="Pool T request latency",
                   line=dict(color=C["T"], width=1, dash="dot")), 7)

    add(go.Scatter(x=h, y=log.served_total, name="delivered",
                   line=dict(color=C["ok"], width=2)), 8)
    add(go.Scatter(x=h, y=log.lam, name="offered λ",
                   line=dict(color=C["muted"], width=1, dash="dot")), 8)

    add(go.Scatter(x=h, y=log.ledger, name="ledger (tokens)", line=dict(color=C["L"], width=2),
                   fill="tozeroy", fillcolor="rgba(42,120,214,.15)"), 9)
    add(go.Scatter(x=h, y=log.ledger_lat / 1000, name="ledger (k request-s)",
                   line=dict(color=C["ok"], width=1.5, dash="dot")), 9)
    fig.add_hline(y=0, row=9, col=1, line=dict(color=C["ink"], width=1))

    add(go.Scatter(x=h, y=log.kv_slack_L, name="KV slack L", line=dict(color=C["L"], width=1.5)), 10)
    add(go.Scatter(x=h, y=log.kv_slack_T, name="KV slack T", line=dict(color=C["T"], width=1.5)), 10)
    fig.add_hline(y=1.0, row=10, col=1, line=dict(color=C["unmet"], width=1, dash="dash"))

    # Reconfiguration events: a rule on every panel, but NO text labels -- rotated annotations
    # collided into an unreadable smear as soon as two events landed near each other. The detail
    # lives on an invisible HOVER marker instead, and in the events table below.
    for e in ctrl.events:
        fig.add_vline(x=e["t"] / 3600, line=dict(color=C["ink"], width=1, dash="dot"),
                      opacity=0.35)
        if e.get("breakeven_t"):
            fig.add_vline(x=e["breakeven_t"] / 3600, row=9, col=1,
                          line=dict(color=C["ok"], width=1.5))
    if ctrl.events:
        ex = [e["t"] / 3600 for e in ctrl.events]
        top = float(np.nanmax(log.N_T.values)) if log.N_T.notna().any() else 32
        txt = [(f"<b>{e['action']}</b> at {e['t']/3600:.2f} h<br>"
                f"{e['frm']} → {e['to']}<br>"
                f"drained {e['devices_draining']} dev · kept {e['devices_kept']}<br>"
                f"tokens lost {e['tokens_lost']:,.0f}<br>"
                f"break-even: " + (f"{e['breakeven_after_s']/60:.1f} min"
                                   if e.get("breakeven_after_s") else "NEVER"))
               for e in ctrl.events]
        fig.add_trace(go.Scatter(
            x=ex, y=[top * 1.05] * len(ex), mode="markers",
            marker=dict(size=13, symbol="triangle-down", color=C["ink"]),
            name="reconfiguration", hoverinfo="text", hovertext=txt,
            legend="legend3"), row=3, col=1)

    fig.update_layout(height=2500, hovermode="x unified", showlegend=True,
                      margin=dict(l=96, r=215, t=70, b=60), plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                      font=dict(color=C["ink"]))

    # one legend per row, pinned to the top of that row's y-domain
    for i in range(1, 11):
        dom = fig.layout[f"yaxis{'' if i == 1 else i}"].domain
        fig.update_layout(**{("legend" if i == 1 else f"legend{i}"): dict(
            x=1.005, xanchor="left", y=dom[1], yanchor="top",
            font=dict(size=9), bgcolor=C["legend_bg"],
            bordercolor=C["legend_border"], borderwidth=1)})
        fig.update_yaxes(title_text=f"<b>{ylabels[i - 1]}</b>", row=i, col=1,
                         title_font=dict(size=13, color=C["ink"]),
                         tickfont=dict(size=11, color=C["ink"]),
                         showgrid=True, gridcolor=C["grid"], zeroline=False)

    fig.update_xaxes(showgrid=True, gridcolor=C["grid"],
                     tickfont=dict(size=11, color=C["ink"]))
    fig.update_yaxes(type="log", row=7, col=1)
    fig.update_xaxes(title_text="<b>time (hours)</b>", row=10, col=1,
                     title_font=dict(size=14, color=C["ink"]),
                     rangeslider=dict(visible=True, thickness=0.03))
    # big left-aligned panel titles with clearance above each panel
    for i in range(10):
        fig.layout.annotations[i].update(font=dict(size=17, color=C["ink"]),
                                         x=0, xanchor="left", yshift=14)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📖  What every metric, panel and legend entry means", expanded=False):
        st.markdown(r"""
### Why α EWMA differs from α

`α` is the **instantaneous** latency-tagged fraction — what the workload is doing *right now*,
including noise. `α EWMA` is an exponentially weighted moving average of it:

$$\alpha_s \leftarrow \alpha_s + w\,(\alpha - \alpha_s), \qquad
  w = 1 - e^{-\ln 2 \cdot \Delta t / t_{1/2}}$$

with `t½` = the EWMA half-life slider. **The controller acts on `α EWMA`, never on raw `α`.**
That is deliberate: reconfiguring costs a drain of seconds to minutes, so an estimate that moves
faster than the actuator can respond is how thrash starts. The visible gap between the two lines
is the controller's blind spot — it is *supposed* to lag, and a brief spike that lives entirely
inside that lag is one the controller correctly ignores.

---

### Headline metrics

| metric | formula | reading |
|---|---|---|
| **effective TTFT** | $\dfrac{S_L \cdot \text{TTFT}_L + S_{spill}\cdot \text{reqlat}_T}{S_L + S_{spill}}$ | What the latency class *actually* experienced. Pool L's TTFT for what it carried, blended with Pool T's much slower completion latency for what spilled. Since Pool T is 15–25× slower, even 10% spill roughly triples this. **The QoS number, not the brochure number.** |
| **unmet fraction** | $\dfrac{\int \text{unmet}\,dt}{\int \alpha\lambda\,dt}$ | Demand *neither* pool could take → dropped or queued. **Hard failure.** 0% healthy · <1% usually a drain transient · >5% real overload · 100% infeasible. |
| **spill fraction** | $\dfrac{\int \text{spilled}\,dt}{\int \alpha\lambda\,dt}$ | Latency traffic served by Pool T. Free in capacity, expensive in latency. Gated by `max_spill_frac`. 0–10% is the target band; 80%+ means the controller refused to chase a fast oscillation. |
| **reconfigurations** | count of `GROW_B/GROW_C/SHRINK` | Each costs a drain of the devices whose role changed. |

### Diagnostics

| metric | formula | reading |
|---|---|---|
| **break-even reached** | first $t$ after a commit where either ledger $\geq 0$ | `n of m`. **`0 of n` is a real verdict**, not a bug — that move never repaid its drain. |
| **tokens lost to drain** | $(\text{Tput}_L+\text{Tput}_T)\cdot T_{drain}\cdot \frac{\text{draining}}{32}$ | Forfeited while devices were out of service. |
| **ledger (tokens)** | $\int (\text{served}_{now} - \text{served}_{shadow})\,dt$ | vs the split we *left behind* (chained, resets each commit). Usually **negative** — devices only move *between* pools, so total throughput is roughly conserved. Only turns positive when a move relieves genuinely **unmet** demand. |
| **ledger (request-s)** | $\int (\text{lat}_{shadow} - \text{lat}_{now})\cdot \text{served}\,dt$ | Request-seconds saved. **Where shrinks and most grows actually pay.** |
| **tick** | $\Delta t$ | Control period *and* the resolution of every integral. `T_drain` and `T_grow` must span several ticks to be resolved at all. |

### Panels and their legend entries

| # | panel | entries |
|---|---|---|
| 1 | **α** | `α` raw (thin) · `α EWMA` (thick) — the gap is the controller's blind spot |
| 2 | **λ offered** | total token demand, prompt + generated together |
| 3 | **Split** | `N_L` filled · `N_T` dashed — always sum to 32. ▼ markers hover with the full transition |
| 4 | **Factorization** | `k` instances · `TP per instance`; $k \times TP = N_L$ |
| 5 | **Routing** ⭐ | stacked: `served by Pool L` (fast) + `spilled → Pool T` (slow but served) + `unmet` (lost). Stack height = total latency-class demand |
| 6 | **Pool headroom** | `load_L`/`load_T` solid = demand; `Tput_L`/`Tput_T` dotted = capacity. Crossings are where spill begins |
| 7 | **Latency** (log) | `effective TTFT` solid rides between the two dotted bounds `Pool L TTFT` and `Pool T req latency`, depending on spill |
| 8 | **Throughput** | `delivered` vs `offered λ` — gaps are drain dips or overload |
| 9 | **Ledger** | tokens (filled) + k request-s (dotted); zero line; **green vline = break-even** |
| 10 | **KV slack** | `kv_slack = C_max / C_need` at S=4096; red line at 1.0 is the OOM wall |

### Actions

`PROVISION` cold start · `HOLD` nothing to do · `SPILL_HOLD` spilling, within tolerance ·
`SHRINK_WAIT` over-provisioned, waiting out `T_shrink` · `DRAINING` reconfiguration in flight,
capacity derated · `GROW_B` added a replica at fixed TP (**no Pool L drain**) · `GROW_C` full
re-shard · `SHRINK` bulk return of devices to Pool T · `INFEASIBLE` no rung serves this load.
""")

    st.caption("Drag the range slider under the bottom panel — every panel is locked to the same "
               "time axis, and hovering reads all ten at one timestamp.")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("##### Time spent in each action")
        counts = log.action.value_counts().reindex(ACTION_ORDER).dropna()
        st.dataframe(pd.DataFrame({"action": counts.index, "ticks": counts.values.astype(int),
                                   "share": counts.values / len(log)})
                     .style.format({"share": "{:.1%}"}),
                     use_container_width=True, hide_index=True)
    with col2:
        st.markdown("##### Reconfiguration events")
        if len(ev):
            e2 = ev.copy()
            e2["at (h)"] = (e2.t / 3600).round(2)
            e2["break-even"] = e2.breakeven_after_s.apply(
                lambda v: f"{v/60:.1f} min" if pd.notna(v) else "NEVER")
            cols = ["at (h)", "action", "frm", "to", "devices_draining", "devices_kept",
                    "tokens_lost", "break-even"]
            if "breakeven_currency" in e2:
                cols.append("breakeven_currency")
            st.dataframe(e2[cols], use_container_width=True, hide_index=True)
        else:
            st.info("No reconfiguration fired — spillover absorbed every α excursion.")

# ---------------------------------------------------------------- ladder
with tab_ladder:
    lad = load_ladder(model)
    st.markdown(f"**{len(lad)} rungs** — Pool T tier endpoints × Pareto-efficient Pool L "
                f"factorizations.  T_drain **{base.t_drain:.0f}s** · T_grow **{base.t_grow():.0f}s** "
                f"· Δ_lat **{base.delta_lat:.3f}s** per device returned to Pool T.")
    used = (set(ev.frm) | set(ev.to)) if len(ev) else set()
    lad["used in this trace"] = [f"{r.k}xTP{r.TP}/N_T{r.N_T}" in used for r in lad.itertuples()]
    st.dataframe(lad, use_container_width=True, hide_index=True)

    f2 = go.Figure()
    # markers only -- the printed labels were unreadable against the light markers and the
    # same detail is available on hover
    f2.add_trace(go.Scatter(x=lad.TTFT_L_s, y=lad.Tput_L, mode="markers",
                            marker=dict(size=13, color=lad.N_L, colorscale="Blues",
                                        showscale=True, colorbar=dict(title="N_L"),
                                        line=dict(width=1, color=C["ink"])),
                            name="rung", hoverinfo="text",
                            hovertext=[f"<b>{r.k}×TP{r.TP}</b> · N_L={r.N_L} / N_T={r.N_T}<br>"
                                       f"Tput_L {r.Tput_L:,.0f} tok/s · TTFT {r.TTFT_L_s:.3f}s<br>"
                                       f"Tput_T {r.Tput_T:,.0f} · batch {r.req_lat_T_s:.1f}s<br>"
                                       f"KV slack L {r.kv_slack_L:.1f} / T {r.kv_slack_T:.2f}"
                                       for r in lad.itertuples()]))
    f2.update_layout(height=430, plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                     font=dict(color=C["ink"]),
                     xaxis_title="Pool L TTFT (s)", yaxis_title="Pool L throughput (tok/s)",
                     title="Every rung: the k ↔ TP trade at constant device count")
    st.plotly_chart(f2, use_container_width=True)

# ---------------------------------------------------------------- all splits
with tab_splits:
    g = table[(table.Model == model) & (table.N_L > 0)]
    st.markdown(f"**{len(g)} splits** from the full DSE — every N_L × every k×TP factorization. "
                "`saturated` = TP past TP_sat, kept as evidence but never selected.")
    st.dataframe(g[["N_L", "N_T", "n_instances", "TP_per_instance", "Tput_L", "Tput_T",
                    "system_tput", "TTFT_L_s", "req_lat_T_packed_s", "kv_slack_L", "kv_slack_T",
                    "saturated"]].sort_values(["N_L", "TP_per_instance"]),
                 use_container_width=True, hide_index=True, height=460)


# ---------------------------------------------------------------- throughput vs latency
with tab_front:
    st.markdown("#### At equal latency for the latency class, how much TOTAL fabric throughput?")
    st.caption(
        "x = latency the latency-tagged class actually experiences · y = total throughput served "
        "by all 32 devices (Pool L + Pool T). Both deployments scored on the same quantity.  \n"
        "**Where α and λ enter:** on total throughput the points are otherwise pure deployment "
        "properties. What the workload controls is **spill** — an undersized Pool L is not a "
        "failure, its excess is served by Pool T at Pool T's much slower latency, so the DDA "
        "points *slide right* as α·λ grows. Only demand neither pool can take actually queues.")

    fa, fl = st.columns(2)
    f_alpha = fa.slider("α  (latency-tagged fraction)", 0.0, 1.0, 0.30, 0.05, key="f_a")
    f_lamf = fl.slider("λ / fabric capacity", 0.05, 1.20, 0.45, 0.05, key="f_l")
    f_lam = f_lamf * cap
    st.caption(f"λ = **{f_lam:,.0f} tok/s** · latency class α·λ = **{f_alpha*f_lam:,.0f}** · "
               f"batch (1−α)·λ = **{(1-f_alpha)*f_lam:,.0f}** · "
               f"max Pool L on the ladder = **{max(r.Tput_L for r in base.rungs):,.0f} tok/s**")

    NAT = {"Llama2-7B": 8, "Llama2-13B": 20, "Llama2-70B": 32}
    sp = F.static_points(model, 32, f_lam, f_alpha, "static32")
    npt = F.static_points(model, NAT[model], f_lam, f_alpha, "native") if NAT[model] != 32 else []
    dp = F.dda_points(model, table, f_lam, f_alpha)
    s32, natd, ddd = pd.DataFrame(sp), pd.DataFrame(npt), pd.DataFrame(dp)

    def envelope(df):
        """Best total throughput reachable at or below each effective latency. Monotone by
        construction — raw rungs sorted by x zig-zag, since many share a latency with very
        different Pool T throughput."""
        d = df[df.feasible]
        if d.empty:
            return np.array([]), np.array([])
        d = d.sort_values("eff_lat_s")
        return d.eff_lat_s.values, np.maximum.accumulate(d.total_tput.values)

    ex, ey = envelope(ddd)
    sx, sy = envelope(s32)

    m1 = st.columns(4)
    m1[0].metric("DDA rungs feasible", f"{int(ddd.feasible.sum())} of {len(ddd)}")
    m1[1].metric("static cfgs feasible", f"{int(s32.feasible.sum())} of {len(s32)}")
    m1[2].metric("best total tput · DDA", f"{ey.max():,.0f}" if len(ey) else "—")
    m1[3].metric("best total tput · static", f"{sy.max():,.0f}" if len(sy) else "—")

    ff = go.Figure()
    if len(ex) and len(sx):
        grid = np.unique(np.concatenate([ex, sx]))
        di = np.interp(grid, ex, ey, left=np.nan, right=ey[-1])
        si = np.interp(grid, sx, sy, left=np.nan, right=sy[-1])
        ff.add_trace(go.Scatter(x=grid, y=si, mode="lines", line=dict(width=0),
                                showlegend=False, hoverinfo="skip"))
        ff.add_trace(go.Scatter(x=grid, y=np.maximum(di, si), mode="lines", line=dict(width=0),
                                fill="tonexty", fillcolor="rgba(27,175,122,.18)",
                                name="reachable only by DDA", hoverinfo="skip"))

    def hover(df, dda):
        if dda:
            return [f"<b>{r.cfg}</b> — {'feasible' if r.feasible else 'OVERLOADED (queues)'}<br>"
                    f"N_L={r.N_L} · N_T={r.N_T} · k={r.k} · TP={r.TP}<br>"
                    f"Pool L TTFT {r.lat_TTFT_s:.3f}s → <b>effective {r.eff_lat_s:.3f}s</b><br>"
                    f"spilled to Pool T <b>{r.spill_frac:.1%}</b> of latency class<br>"
                    f"total tput <b>{r.total_tput:,.0f}</b> · served {r.served_total:,.0f}<br>"
                    f"batch request latency {r.batch_lat_s:.1f}s · queued {r.queued:,.0f}"
                    for r in df.itertuples()]
        return [f"<b>{r.cfg}</b> — {'feasible' if r.feasible else 'OVERLOADED (queues)'}<br>"
                f"{r.devices} devices, ONE pool for both classes<br>"
                f"TTFT for EVERY request <b>{r.eff_lat_s:.3f}s</b><br>"
                f"total tput <b>{r.total_tput:,.0f}</b> · served {r.served_total:,.0f}<br>"
                f"batch request latency {r.batch_lat_s:.1f}s · queued {r.queued:,.0f}"
                for r in df.itertuples()]

    for df, col, sym, nm, isd in ((s32, C["T"], "square", "static 32-dev (one pool)", False),
                                  (natd, C["L"], "circle", f"native {NAT[model]}-dev", False),
                                  (ddd, C["ok"], "star", "DDA rungs", True)):
        if df is None or df.empty:
            continue
        for ok, tag in ((True, ""), (False, " · overloaded")):
            g = df[df.feasible == ok]
            if g.empty:
                continue
            ff.add_trace(go.Scatter(
                x=g.eff_lat_s, y=g.total_tput, mode="markers", name=nm + tag,
                marker=dict(size=15 if isd else 11, symbol=sym,
                            color=col if ok else "rgba(0,0,0,0)",
                            line=dict(width=1.6, color=col)),
                hoverinfo="text", hovertext=hover(g, isd)))
    if len(sx):
        ff.add_trace(go.Scatter(x=sx, y=sy, mode="lines", name="static frontier",
                                line=dict(color=C["T"], width=2.5), hoverinfo="skip"))
    if len(ex):
        ff.add_trace(go.Scatter(x=ex, y=ey, mode="lines", name="DDA frontier",
                                line=dict(color=C["ok"], width=3), hoverinfo="skip"))

    ff.update_layout(
        height=620, plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, font=dict(color=C["ink"]),
        xaxis=dict(title="<b>effective latency-class latency (s)  ↓ better</b>", type="log",
                   gridcolor=C["grid"], tickfont=dict(size=12)),
        yaxis=dict(title="<b>TOTAL throughput, Pool L + Pool T (tok/s)  ↑ better</b>",
                   gridcolor=C["grid"], tickfont=dict(size=12)),
        legend=dict(x=1.01, y=1, bgcolor=C["legend_bg"], bordercolor=C["legend_border"],
                    borderwidth=1, font=dict(size=10)),
        margin=dict(l=80, r=250, t=40, b=60), hovermode="closest")
    st.plotly_chart(ff, use_container_width=True)
    st.caption("Hover any marker for its full configuration. Hollow = the deployment cannot carry "
               "this (α, λ) even with spill, so demand queues and the backlog grows.")

    st.markdown("##### At each static latency bar, the best DDA split meeting it")
    mt = pd.DataFrame(F.matched(sp, dp, True))
    if mt.empty:
        st.info("No matched pairs at this (α, λ).")
    else:
        st.dataframe(mt[["ttft_bar_s", "static_cfg", "static_total_tput", "dda_cfg",
                         "dda_total_tput", "tput_ratio", "dda_spill_frac",
                         "static_batch_lat_s", "dda_batch_lat_s", "batch_lat_ratio"]],
                     use_container_width=True, hide_index=True)

    st.markdown("##### α sweep at this λ — where each deployment wins, and by how much")
    rows = []
    for aa in np.round(np.arange(0.0, 1.01, 0.1), 2):
        sp2 = F.static_points(model, 32, f_lam, aa, "static32")
        dp2 = F.dda_points(model, table, f_lam, aa)
        sf = [x for x in sp2 if x["feasible"]]
        df2 = [x for x in dp2 if x["feasible"]]
        bs = max(sf, key=lambda r: r["total_tput"]) if sf else None
        bd = max(df2, key=lambda r: r["total_tput"]) if df2 else None
        mm = F.matched(sp2, dp2, True)
        best = max(mm, key=lambda r: r["tput_ratio"]) if mm else None
        rows.append({
            "α": aa, "α·λ": round(aa * f_lam),
            "static best": bs["cfg"] if bs else "overloaded",
            "static tput": round(bs["total_tput"]) if bs else np.nan,
            "static lat_s": bs["eff_lat_s"] if bs else np.nan,
            "DDA best": bd["cfg"] if bd else "overloaded",
            "DDA tput": round(bd["total_tput"]) if bd else np.nan,
            "DDA eff_lat_s": bd["eff_lat_s"] if bd else np.nan,
            "DDA spill": bd["spill_frac"] if bd else np.nan,
            "tput ×": round(bd["total_tput"] / bs["total_tput"], 2) if (bs and bd) else np.nan,
            "latency ×": round(bs["eff_lat_s"] / bd["eff_lat_s"], 2) if (bs and bd) else np.nan,
            "MATCHED tput ×": best["tput_ratio"] if best else np.nan,
            "matched at (s)": best["ttft_bar_s"] if best else np.nan})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "**`tput ×` and `latency ×` compare the two best-throughput configs** — unmatched, so a "
        "gain there can come from merely sitting at a different latency. **`MATCHED tput ×` is "
        "the defensible number**: the largest total-throughput advantage DDA holds at an "
        "*identical* latency bar, with `matched at (s)` giving where. `DDA spill` shows how much "
        "of the latency class is being carried by Pool T — as it rises, `DDA eff_lat_s` "
        "degrades toward Pool T's latency and the advantage erodes.")
