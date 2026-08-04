#!/usr/bin/env python3
"""dse_full_table.py — the COMPLETE design-space table: every valid split, every Pool L packing.

Full DSE, not a sampled spectrum: for every model, every integer N_L in [cap_floor, 32-N_T_min],
and EVERY homogeneous factorization j x TP=m of N_L, paired with N_T = 32 - N_L. Splits that
perform badly are kept and reported -- that is the point. Saturated Pool L configs (m > TP_sat)
are flagged, not dropped, so the throughput plateau/decline past the knee is visible as data.

All derived metrics come from explorer/dse_metrics.py -- the SINGLE source of truth shared with
the Streamlit explorer. Do not reimplement any formula here.

Output: results/dda32/dse_full_table.csv
Usage:  python3 dse_full_table.py [--prompt-fraction 0.125]
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "explorer")))
import dse_metrics as M   # noqa: E402


def build(prompt_fraction):
    rows = []
    for model in M.MODELS:
        f = M.model_facts(model)
        cap, L = f["cap_floor"], f["L"]

        # legal Pool T range: cpb = 32//ceil(L/N_T) >= min_cpb, and N_L = 32-N_T >= cap_floor
        legal_NT = [n for n in range(1, M.FABRIC)
                    if M.CHANNELS // -(-L // n) >= f["min_cpb"] and (M.FABRIC - n) >= cap]
        if not legal_NT:
            continue
        NT_min, NT_max = min(legal_NT), max(legal_NT)

        # reference row: entire fabric as Pool T (alpha -> 0)
        t0 = M.pool_T_metrics(model, min(M.FABRIC - cap, NT_max), prompt_fraction)
        if t0:
            rows.append(dict(
                Model=model, N_L=0, N_T=t0["N_T"], n_instances=0, TP_per_instance=0,
                Tput_L=0.0, TTFT_L_s=None, TTFT_L_no_prefill_sampling_s=None, req_lat_L_s=None,
                Tput_T=round(t0["Tput_T"], 1), Tput_T_cent=round(t0["Tput_T_cent"], 1),
                req_lat_T_packed_s=round(t0["req_lat_T_packed_s"], 3),
                req_lat_T_cent_s=round(t0["req_lat_T_cent_s"], 3),
                ttft_T_packed_s=round(t0["ttft_T_packed_s"], 3),
                system_tput=round(t0["Tput_T"], 1),
                idle_T=t0["idle"], cpb_ceil=t0["cpb_ceil"], cpb_floor=t0["cpb_floor"],
                n_ceil=t0["n_ceil"], n_floor=t0["n_floor"], hetero_T=t0["hetero"],
                packing_gain_pct=round(t0["lat_improve_pct"], 1),
                KV_free_per_block_MiB=round(t0["kv_free_per_block_MiB"], 1),
                KV_free_per_L_device_GiB=None,
                C_max_4096_T=round(t0["C_max_4096"], 1), C_max_mean_T=round(t0["C_max_mean"], 1),
                C_need_T=t0["C_need"], kv_slack_T=round(t0["kv_slack"], 3),
                C_max_4096_L=None, C_need_L=None, kv_slack_L=None,
                kv_feasible=bool(t0["kv_feasible"]), saturated=False,
                valid=bool(t0["kv_feasible"]), invalid_reason="" if t0["kv_feasible"] else "kv_slack_T<1",
                cap_floor=cap, TP_sat=f["tp_sat"], min_cpb=f["min_cpb"],
                prompt_fraction=prompt_fraction))

        for N_L in range(cap, M.FABRIC - NT_min + 1):
            N_T = M.FABRIC - N_L
            if N_T < NT_min or N_T > NT_max:
                continue
            t = M.pool_T_metrics(model, N_T, prompt_fraction)

            for fac in M.pool_L_factorizations(model, N_L):
                l = M.pool_L_metrics(model, fac["j"], fac["m"], prompt_fraction)

                reason = []
                if l is None:
                    reason.append(f"poolL TP={fac['m']} sim missing/invalid")
                if t is None:
                    reason.append(f"poolT N_T={N_T} sim missing/invalid")
                if l is not None and not l["kv_feasible"]:
                    reason.append("kv_slack_L<1")
                if t is not None and not t["kv_feasible"]:
                    reason.append("kv_slack_T<1")

                # tri-state: None when a sim is missing (unknown), True/False only when
                # both pools have data. Conflating "not simulated yet" with "KV-infeasible"
                # would silently misreport the capacity gate.
                if l is None or t is None:
                    kv_ok = None
                else:
                    kv_ok = bool(l["kv_feasible"] and t["kv_feasible"])
                rows.append(dict(
                    Model=model, N_L=N_L, N_T=N_T,
                    n_instances=fac["j"], TP_per_instance=fac["m"],
                    Tput_L=round(l["Tput_L"], 1) if l else None,
                    TTFT_L_s=round(l["TTFT_L_s"], 3) if l else None,
                    TTFT_L_no_prefill_sampling_s=round(l["TTFT_L_no_prefill_sampling_s"], 3) if l else None,
                    req_lat_L_s=round(l["req_lat_L_s"], 3) if l else None,
                    Tput_T=round(t["Tput_T"], 1) if t else None,
                    Tput_T_cent=round(t["Tput_T_cent"], 1) if t else None,
                    req_lat_T_packed_s=round(t["req_lat_T_packed_s"], 3) if t else None,
                    req_lat_T_cent_s=round(t["req_lat_T_cent_s"], 3) if t else None,
                    ttft_T_packed_s=round(t["ttft_T_packed_s"], 3) if t else None,
                    system_tput=round((l["Tput_L"] if l else 0) + (t["Tput_T"] if t else 0), 1),
                    idle_T=t["idle"] if t else None,
                    cpb_ceil=t["cpb_ceil"] if t else None,
                    cpb_floor=t["cpb_floor"] if t else None,
                    n_ceil=t["n_ceil"] if t else None, n_floor=t["n_floor"] if t else None,
                    hetero_T=t["hetero"] if t else None,
                    packing_gain_pct=round(t["lat_improve_pct"], 1) if t else None,
                    KV_free_per_block_MiB=round(t["kv_free_per_block_MiB"], 1) if t else None,
                    KV_free_per_L_device_GiB=round(l["kv_free_per_device_GiB"], 2) if l else None,
                    C_max_4096_T=round(t["C_max_4096"], 1) if t else None,
                    C_max_mean_T=round(t["C_max_mean"], 1) if t else None,
                    C_need_T=t["C_need"] if t else None,
                    kv_slack_T=round(t["kv_slack"], 3) if t else None,
                    C_max_4096_L=round(l["C_max_4096"], 1) if l else None,
                    C_need_L=l["C_need"] if l else None,
                    kv_slack_L=round(l["kv_slack"], 3) if l else None,
                    kv_feasible=kv_ok, saturated=bool(fac["saturated"]),
                    valid=bool(kv_ok is True),
                    invalid_reason="; ".join(reason),
                    cap_floor=cap, TP_sat=f["tp_sat"], min_cpb=f["min_cpb"],
                    prompt_fraction=prompt_fraction))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-fraction", type=float, default=M.DEFAULT_PROMPT_FRACTION,
                    help="prompt share of total seqlen; affects TTFT only (default CENT's 0.125)")
    ap.add_argument("--out", default=os.path.join(M.DDA, "dse_full_table.csv"))
    a = ap.parse_args()

    df = build(a.prompt_fraction)
    df.to_csv(a.out, index=False)

    print(f"wrote {a.out}  ({len(df)} rows, prompt_fraction={a.prompt_fraction})\n")
    for model in M.MODELS:
        s = df[df.Model == model]
        if s.empty:
            continue
        v = s[s.valid]
        missing = int(s.invalid_reason.str.contains("missing").sum())
        infeas = int((s.kv_feasible == False).sum())   # noqa: E712 -- tri-state, None != False
        print(f"{model:<11} rows {len(s):4d} | valid {len(v):4d} | saturated {int(s.saturated.sum()):3d} "
              f"| kv-infeasible {infeas:3d} | awaiting-sim {missing:3d}")
        if len(v):
            b = v.loc[v.system_tput.idxmax()]
            print(f"{'':<11}   best system_tput {b.system_tput:8.1f} @ N_L={int(b.N_L)}"
                  f"({int(b.n_instances)}xTP{int(b.TP_per_instance)})/N_T={int(b.N_T)}"
                  f"  TTFT_L {b.TTFT_L_s}s  kv_slack_T {b.kv_slack_T}")