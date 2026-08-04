#!/usr/bin/env python3
"""dda_controller.py — split selection + reconfiguration controller for a 32-device DDA fabric.

Implements results/DDA_CONTROLLER_PLAN.md. Every performance number comes from dse_metrics.py;
this module owns POLICY only and reimplements no formula.

Three measured facts force the design:

  F1  Pool T throughput is a STAIRCASE. 7B: every N_T in 16..31 is one tier at 6239 tok/s
      (5 tiers over 26 device counts); 13B 3 tiers; 70B 2. Alpha drift inside a tier must not
      move anything, so the ladder only carries tier endpoints.

  F2  Only spillover is free. N_L + N_T = 32, so ANY N_L change perturbs Pool T too: Pool L
      re-shards when m changes (weight row i -> bank i % 512m) and Pool T re-packs when N_T
      changes. Measured: 0 of 1284 transitions were zero-drain.

  F3  Replica freedom collapses with model size. j = N_L/m is bounded by cap_floor <= m and
      N_L <= 32 - N_T_min: 7B j in 1..26, 13B 1..11, 70B j == 1 (k is not a variable at all).

Policy, cheapest response first, never skipping a tier:
  Tier A  spillover -- route excess latency-class load to Pool T. Free and instant; the spilled
          requests simply experience Pool T latency instead of Pool L's.
  Tier B  grow k at fixed m -- existing Pool L instances keep modulus 512m so they do NOT drain;
          only arriving devices load weights (plus Pool T's repack, unavoidable under F2).
  Tier C  full reconfiguration -- change m or jump rungs. Both pools drain.

Hysteresis is ASYMMETRIC, which is the whole point:
  grow    once spillover has been exhausted for T_grow ~ 2*T_drain, take the MINIMUM rung that
          clears the load. The latency class is being missed, so act.
  shrink  only after T_shrink, and then in ONE LARGE STEP:
              T_shrink = T_drain^2 / (2 * (1-alpha) * delta_lat * devices_freed)
          quadratic in T_drain and inversely proportional to how many devices come back, so
          freeing 8 devices repays ~4x faster than freeing 2. WAIT and move in bulk -- the exact
          opposite of the grow path.

Monotone-ish N_L emerges from that asymmetry instead of being imposed; imposing it would force a
reshard on every alpha tick, which is the cost we are avoiding.
"""
import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import dse_metrics as M

FABRIC = M.FABRIC


# ---------------------------------------------------------------- rungs
@dataclass(frozen=True)
class Rung:
    """One deployable split; immutable so it can be compared and used as a dict key."""
    model: str
    N_L: int
    N_T: int
    j: int                      # instances
    m: int                      # TP per instance
    Tput_L: float
    Tput_T: float
    TTFT_L_s: float
    req_lat_T_s: float
    kv_slack_L: float
    kv_slack_T: float

    @property
    def key(self):
        return (self.N_L, self.j, self.m)

    def __str__(self):
        return f"{self.j}xTP{self.m}/N_T{self.N_T}"


def _pareto(rows):
    """Keep factorizations not dominated on (Tput_L up, TTFT_L down). This is where k is chosen;
    for 70B the survivor set is a singleton because j == 1."""
    out = []
    for r in rows:
        dominated = any(o.Tput_L >= r.Tput_L and o.TTFT_L_s <= r.TTFT_L_s and o.key != r.key
                        for o in rows)
        if not dominated:
            out.append(r)
    return out


def ladder(model, table=None, kv_margin=M.KV_MARGIN_DEFAULT):
    """Phase 0 (offline): the short list of rungs the controller may ever deploy.

    Collapses Pool T to throughput-distinct tiers (F1), keeping BOTH endpoints of each tier --
    the smallest N_T frees the most devices for Pool L at no throughput cost, the largest gives
    the best batch latency. Then keeps the Pareto-efficient Pool L factorizations at each.

    Measured sizes: 7B 21 rungs, 13B 16, 70B 4 (vs 92/53/9 raw splits).
    """
    if table is None:
        table = pd.read_csv(os.path.join(M.DDA, "dse_full_table.csv"))
    g = table[(table.Model == model) & (table.N_L > 0) & (~table.saturated)].copy()
    g = g[(g.kv_slack_L >= kv_margin) & (g.kv_slack_T >= kv_margin)]
    if g.empty:
        return []

    rung_NT = set()
    for _, tier in g.groupby("Tput_T"):
        rung_NT.add(int(tier.N_T.min()))
        rung_NT.add(int(tier.N_T.max()))
    rung_NT.add(int(g.N_T.max()))          # guarantees the cap_floor cold start is reachable

    rungs = []
    for N_T in sorted(rung_NT):
        cands = [Rung(model, int(r.N_L), int(r.N_T), int(r.n_instances), int(r.TP_per_instance),
                      float(r.Tput_L), float(r.Tput_T), float(r.TTFT_L_s),
                      float(r.req_lat_T_packed_s), float(r.kv_slack_L), float(r.kv_slack_T))
                 for r in g[g.N_T == N_T].itertuples()]
        rungs.extend(_pareto(cands))
    return sorted(rungs, key=lambda r: (r.N_L, r.m))


def delta_lat_per_device(model, table=None):
    """Batch-latency won per device handed to Pool T, inside a throughput tier.

    Mean slope of req_lat_T_packed vs N_T within each tier (finding-7 packing). Measured:
    7B 0.566 s/device, 13B 1.045, 70B 9.393 -- 70B's is 17x 7B's, which is exactly why it must
    move in bulk or not at all.
    """
    if table is None:
        table = pd.read_csv(os.path.join(M.DDA, "dse_full_table.csv"))
    g = (table[(table.Model == model) & (table.N_L > 0)]
         .drop_duplicates(subset=["N_T"]).sort_values("N_T"))
    slopes = []
    for _, tier in g.groupby("Tput_T"):
        if len(tier) < 2:
            continue
        t = tier.sort_values("N_T")
        slopes.extend(-np.diff(t.req_lat_T_packed_s.values) / np.diff(t.N_T.values))
    return float(np.mean(slopes)) if slopes else 0.0


# ---------------------------------------------------------------- phase 1: cold start
def provision(rungs, lam, alpha):
    """Minimum-N_L rung clearing both pools; ties broken by alpha-weighted normalised latency.

    Minimum N_L is right because devices left in Pool T are not idle -- inside a tier each one
    buys delta_lat of batch latency at zero throughput cost.
    """
    load_L, load_T = alpha * lam, (1 - alpha) * lam
    feas = [r for r in rungs if r.Tput_L >= load_L and r.Tput_T >= load_T]
    if not feas:
        return None
    n_min = min(r.N_L for r in feas)
    tied = [r for r in feas if r.N_L == n_min]
    if len(tied) == 1:
        return tied[0]
    bl = min(r.TTFT_L_s for r in tied)
    bt = min(r.req_lat_T_s for r in tied)
    return min(tied, key=lambda r: alpha * r.TTFT_L_s / bl + (1 - alpha) * r.req_lat_T_s / bt)


# ---------------------------------------------------------------- routing
def route(split, lam, alpha, derate=1.0):
    """How this tick's demand is actually served.

    Spillover is the free release valve: latency-class overflow goes to Pool T and experiences
    Pool T latency instead of Pool L's.

    `derate` < 1 models a reconfiguration in progress -- the draining fraction of the fabric
    serves nothing until in-flight requests finish, so capacity is scaled for T_drain seconds
    after a commit. Booking the drain this way (rather than as a lump charge) makes the dip
    visible in throughput and lets the token ledger fall out of ordinary accounting.
    """
    cap_L, cap_T = split.Tput_L * derate, split.Tput_T * derate
    load_L, load_T = alpha * lam, (1 - alpha) * lam
    served_L = min(load_L, cap_L)
    excess_L = load_L - served_L
    spare_T = max(0.0, cap_T - load_T)
    spilled = min(excess_L, spare_T)
    unmet = excess_L - spilled
    denom = served_L + spilled
    eff_ttft = ((served_L * split.TTFT_L_s + spilled * split.req_lat_T_s) / denom
                if denom > 0 else split.TTFT_L_s)
    served_T = min(load_T, cap_T) + spilled
    return dict(load_L=load_L, load_T=load_T, served_by_L=served_L,
                spare_T=cap_T - load_T, excess_L=excess_L,
                spilled=spilled, unmet=unmet, eff_ttft_s=eff_ttft,
                served_by_T=served_T, served_total=served_L + served_T,
                cap_L=cap_L, cap_T=cap_T, derate=derate)


# ---------------------------------------------------------------- the controller
@dataclass
class Controller:
    model: str
    rungs: list
    t_drain: float                     # s, worst-case in-flight completion on the current split
    delta_lat: float                   # batch seconds won per device returned to Pool T
    ewma_halflife: float = 120.0
    grow_factor: float = 2.0           # T_grow = grow_factor * t_drain
    min_shrink_devices: int = 2        # never trickle devices back one at a time
    policy: str = "hysteresis"         # hysteresis | greedy | static
    # Spillover is only "absorbing" while it does not wreck the latency class. Judging it on
    # CAPACITY alone is wrong: Pool T has vast spare throughput, so a purely capacity-based test
    # reports "absorbed" while quietly routing most latency-critical traffic to a pool that is
    # 15x slower (measured: trace 3 spilled 63% of the latency class at 7.0 s effective TTFT vs
    # 0.458 s, and never grew). Exceeding this fraction starts the T_grow timer.
    max_spill_frac: float = 0.10

    split: Rung = None
    alpha_s: float = None
    t: float = 0.0
    _over_since: float = None
    _under_since: float = None
    log: list = field(default_factory=list)

    # --- drain transient + counterfactual ledger -------------------------------------------
    # A reconfiguration is only worth making if it earns back what the drain costs. Proving that
    # needs the split we LEFT BEHIND simulated alongside the deployed one (a chained shadow,
    # reset at each commit), and a running integral of the difference. `breakeven = never` is a
    # legitimate outcome -- it means the move should not have been made.
    _drain_until: float = 0.0
    _drain_frac: float = 0.0
    _shadow: Rung = None
    # TWO ledgers, because the two directions pay in different currencies. A GROW buys throughput
    # and repays in tokens. A SHRINK hands devices back to Pool T for batch latency and can NEVER
    # repay in tokens -- a token-only ledger reports "breakeven NEVER" for every shrink, which is
    # an artifact of the metric, not a verdict on the move.
    ledger: float = 0.0            # tokens:        integral of (served_actual - served_shadow)
    ledger_lat: float = 0.0        # request-sec:   integral of (lat_shadow - lat_actual)*served
    events: list = field(default_factory=list)

    # ---- smoothing ----
    def _ewma(self, alpha, dt):
        if self.alpha_s is None:
            self.alpha_s = alpha
        else:
            w = 1 - math.exp(-math.log(2) * dt / self.ewma_halflife)
            self.alpha_s += w * (alpha - self.alpha_s)
        return self.alpha_s

    # ---- thresholds ----
    def t_grow(self):
        return self.grow_factor * self.t_drain

    def t_shrink(self, alpha, devices_freed):
        """Quadratic repayment. More devices freed -> repays faster -> wait and move in bulk."""
        gain = max(1e-9, (1 - alpha) * self.delta_lat * max(1, devices_freed))
        return self.t_drain ** 2 / (2 * gain)

    # ---- tiers ----
    def tier_b(self, lam, alpha):
        """Grow k at fixed m: existing instances keep modulus 512m, so none of them drain."""
        want_L, want_T = alpha * lam, (1 - alpha) * lam
        c = [r for r in self.rungs
             if r.m == self.split.m and r.j > self.split.j
             and r.Tput_L >= want_L and r.Tput_T >= want_T]
        return min(c, key=lambda r: r.N_L) if c else None

    def _diff(self, target):
        return M.reconfig_diff(self.model,
                               (self.split.j, self.split.m, self.split.N_T),
                               (target.j, target.m, target.N_T))

    # ---- drain / ledger bookkeeping, run every tick before any policy decision ----
    def _derate(self):
        """Capacity multiplier while a reconfiguration is still draining."""
        return (1.0 - self._drain_frac) if self.t < self._drain_until else 1.0

    def _accrue(self, lam, alpha, dt, r):
        """Integrate deployed-vs-shadow in BOTH currencies and mark each break-even crossing.

        A grow is judged on tokens, a shrink on latency; `_relevant_ledger` picks which one
        decides `breakeven_t` so the verdict matches what the move was actually for.
        """
        if self._shadow is None or not self.events:
            return
        rs = route(self._shadow, lam, alpha, 1.0)
        self.ledger += (r["served_total"] - rs["served_total"]) * dt

        # request-seconds saved across BOTH classes: latency-tagged traffic sees eff_ttft,
        # batch traffic sees the Pool T completion latency
        lat_now = (r["served_by_L"] + r["spilled"]) * r["eff_ttft_s"] + r["served_by_T"] * self.split.req_lat_T_s
        lat_shd = ((rs["served_by_L"] + rs["spilled"]) * rs["eff_ttft_s"]
                   + rs["served_by_T"] * self._shadow.req_lat_T_s)
        self.ledger_lat += (lat_shd - lat_now) * dt

        # Break even on EITHER currency, and record which one paid.
        # Devices only move BETWEEN pools, so total throughput is roughly conserved and most
        # moves -- grows included -- are really QoS plays: a grow pulls traffic off Pool T's slow
        # spill path onto Pool L's fast one without changing tokens/s much. The token ledger
        # therefore only turns positive when a move relieves genuinely UNMET demand. Judging
        # every move on tokens alone would mark almost all of them "never repaid", which measures
        # the metric rather than the controller.
        ev = self.events[-1]
        if ev.get("breakeven_t") is None and (self.ledger >= 0 or self.ledger_lat >= 0):
            ev["breakeven_t"] = self.t
            ev["breakeven_after_s"] = self.t - ev["t"]
            ev["breakeven_currency"] = "tokens" if self.ledger >= 0 else "request-s"

    # ---- one control tick ----
    def step(self, lam, alpha, dt):
        self.t += dt
        a_s = self._ewma(alpha, dt)

        if self.split is None:
            self.split = provision(self.rungs, lam, a_s)
            if self.split is None:
                self._record(lam, alpha, "INFEASIBLE", None)
                return "INFEASIBLE"
            self._record(lam, alpha, "PROVISION", None)
            return "PROVISION"

        if self.policy == "static":
            self._record(lam, alpha, "HOLD", None)
            return "HOLD"

        if self.policy == "greedy":
            tgt = provision(self.rungs, lam, alpha)
            if tgt is not None and tgt.key != self.split.key:
                self._commit(tgt, lam, alpha, "GREEDY")
                return "GREEDY"
            self._record(lam, alpha, "HOLD", None)
            return "HOLD"

        r = route(self.split, lam, alpha, self._derate())
        self._accrue(lam, alpha, dt, r)
        # QoS-aware "absorbed": nothing unmet AND the spilled share of the latency class is
        # within tolerance. Capacity alone would let the controller spill indefinitely.
        spill_frac = r["spilled"] / r["load_L"] if r["load_L"] > 1e-9 else 0.0
        absorbed = (r["unmet"] <= 1e-9) and (spill_frac <= self.max_spill_frac)

        # never start a new reconfiguration while the previous one is still draining
        if self.t < self._drain_until:
            self._record(lam, alpha, "DRAINING", r)
            return "DRAINING"

        # ---- grow: spillover exhausted, and it has stayed exhausted ----
        if not absorbed:
            self._under_since = None
            if self._over_since is None:
                self._over_since = self.t
            if self.t - self._over_since >= self.t_grow():
                tgt = self.tier_b(lam, a_s) or provision(self.rungs, lam, a_s)
                # Accept any target with MORE Pool L throughput -- not only a larger N_L. The
                # useful move is often same-N_L, different-k: at 7B trace 3 the fix is
                # 2xTP8 -> 16xTP1 (Tput_L 1198 -> 5366) at N_L=16 either way. Requiring N_L to
                # grow blocked exactly the k-adjustment this design exists for.
                if (tgt is not None and tgt.key != self.split.key
                        and tgt.Tput_L > self.split.Tput_L):
                    tier = "B" if (tgt.m == self.split.m and tgt.j > self.split.j) else "C"
                    self._commit(tgt, lam, alpha, f"GROW_{tier}")
                    self._over_since = None
                    return f"GROW_{tier}"
                if tgt is None:
                    self._record(lam, alpha, "INFEASIBLE", r)
                    return "INFEASIBLE"
            self._record(lam, alpha, "SPILL_HOLD", r)
            return "SPILL_HOLD"

        self._over_since = None

        # ---- shrink: over-provisioned. Never urgent; must repay the drain, and in bulk ----
        tgt = provision(self.rungs, lam, a_s)
        if tgt is not None and tgt.N_L < self.split.N_L:
            freed = self.split.N_L - tgt.N_L
            if freed < self.min_shrink_devices:
                self._under_since = None
                self._record(lam, alpha, "HOLD", r)
                return "HOLD"
            if self._under_since is None:
                self._under_since = self.t
            if self.t - self._under_since >= self.t_shrink(a_s, freed):
                self._commit(tgt, lam, alpha, "SHRINK")
                self._under_since = None
                return "SHRINK"
            self._record(lam, alpha, "SHRINK_WAIT", r)
            return "SHRINK_WAIT"

        self._under_since = None
        self._record(lam, alpha, "HOLD", r)
        return "HOLD"

    # ---- bookkeeping ----
    def _commit(self, target, lam, alpha, action):
        d = self._diff(target)
        lost = (self.split.Tput_L + self.split.Tput_T) * self.t_drain * (d["draining"] / FABRIC)
        r = route(self.split, lam, alpha, self._derate())
        self._record(lam, alpha, action, r, target=target,
                     drained=d["draining"], kept=d["kept"], tokens_lost=lost)

        # the split we are leaving becomes the counterfactual; the ledger restarts from zero so
        # each move is judged against the one it replaced, not against the cold start
        self._shadow = self.split
        self.ledger = 0.0
        self.ledger_lat = 0.0
        self.events.append(dict(t=self.t, action=action, frm=str(self.split), to=str(target),
                                devices_draining=d["draining"], devices_kept=d["kept"],
                                tokens_lost=lost, breakeven_t=None, breakeven_after_s=None))
        self._drain_frac = d["draining"] / FABRIC
        self._drain_until = self.t + self.t_drain
        self.split = target

    def _record(self, lam, alpha, action, r, target=None, drained=0, kept=None, tokens_lost=0.0):
        s = self.split
        row = dict(t=self.t, lam=lam, alpha=alpha, alpha_s=self.alpha_s, action=action,
                   split=str(s) if s else "", N_L=s.N_L if s else np.nan,
                   N_T=s.N_T if s else np.nan, j=s.j if s else np.nan, m=s.m if s else np.nan,
                   Tput_L=s.Tput_L if s else np.nan, Tput_T=s.Tput_T if s else np.nan,
                   TTFT_L_s=s.TTFT_L_s if s else np.nan,
                   req_lat_T_s=s.req_lat_T_s if s else np.nan,
                   devices_draining=drained, devices_kept=kept, tokens_lost=tokens_lost,
                   target=str(target) if target is not None else "")
        if r is None and s is not None:
            r = route(s, lam, alpha)
        # fixed schema: an all-INFEASIBLE trace (no split ever deployed) must still produce these
        # columns, or the log comes back without them and every consumer breaks
        keys = ("load_L", "load_T", "served_by_L", "served_by_T", "spare_T", "excess_L",
                "spilled", "unmet", "eff_ttft_s", "served_total", "cap_L", "cap_T", "derate")
        if r:
            row.update({k: r[k] for k in keys})
        else:
            row.update({k: np.nan for k in keys})
            row["load_L"], row["load_T"] = alpha * lam, (1 - alpha) * lam
            row["unmet"] = alpha * lam          # nothing deployed -> all latency demand unmet
            row["spilled"] = row["served_total"] = 0.0
        row["ledger"] = self.ledger
        row["ledger_lat"] = self.ledger_lat
        row["shadow"] = str(self._shadow) if self._shadow is not None else ""
        row["draining_now"] = bool(self.t < self._drain_until)
        row["kv_slack_L"] = s.kv_slack_L if s else np.nan
        row["kv_slack_T"] = s.kv_slack_T if s else np.nan
        self.log.append(row)


def make_controller(model, table=None, **kw):
    rungs = ladder(model, table)
    if not rungs:
        return None
    drains = [d for d in (M.drain_seconds(model, r.N_T) for r in rungs) if d]
    return Controller(model=model, rungs=rungs,
                      t_drain=float(np.median(drains)) if drains else 0.0,
                      delta_lat=delta_lat_per_device(model, table), **kw)


def run_trace(model, alpha_t, lam_t, dt=30.0, table=None, **kw):
    """Replay an (alpha, lambda) trace. Returns (controller, per-tick log DataFrame)."""
    c = make_controller(model, table, **kw)
    if c is None:
        return None, pd.DataFrame()
    for a, l in zip(alpha_t, lam_t):
        c.step(float(l), float(a), dt)
    return c, pd.DataFrame(c.log)