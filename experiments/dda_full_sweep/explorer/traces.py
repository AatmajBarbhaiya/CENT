#!/usr/bin/env python3
"""traces.py — alpha(t) shapes, shared by the asserted eval suite and the Streamlit app.

Lives in explorer/ (not app/ or simulation/) for the same reason dse_metrics.py does: BOTH
simulation/controller_eval.py and app/app.py need these shapes, and a second copy in either
place would be free to drift from the one the paper's numbers came from.

Every motion runs the FULL horizon -- 24 h by default, the natural period of an interactive
workload -- so all of them are judged on one clock and every controller timescale has room to
act (7B T_shrink ~5 min fits ~290x in 24 h, 70B ~75 min fits ~19x).
"""
import numpy as np

MOTIONS = ["diurnal", "steady", "step up", "step down", "brief spike",
           "slow ramp", "flapping", "random walk"]


def alpha_motion(kind, n, mean=0.30, amp=0.25, period_h=24.0, hours=24.0,
                 noise=0.0, rng=None, spike_ticks=None):
    """alpha(t) over `n` ticks spanning `hours`.

    `mean`/`amp` are the shape knobs; `period_h` sets the cycle length of the periodic motions.
    Clipped to [0.02, 0.97] -- alpha is a fraction of requests and both endpoints are degenerate
    (no latency class at all, or no batch class at all).

    `spike_ticks` sizes the 'brief spike' pulse. It MUST be shorter than the controller's T_grow
    for that trace to test what it claims (that a transient is correctly ignored). The n/100
    default is a display convenience; the asserted suite passes an explicit T_grow-derived value,
    because at dt=30s a 7B T_grow is only ~1.6 ticks and n/100 would be 4x too long.
    """
    rng = rng or np.random.default_rng(0)
    t = np.arange(n)
    frac = t / max(1, n - 1)
    lo, hi = max(0.02, mean - amp), min(0.97, mean + amp)

    if kind == "diurnal":
        a = mean + amp * np.sin(2 * np.pi * (hours / max(period_h, 1e-6)) * frac)
    elif kind == "steady":
        a = np.full(n, mean)
    elif kind == "step up":
        a = np.where(frac < 1 / 3, lo, hi)
    elif kind == "step down":
        a = np.where(frac < 1 / 3, hi, lo)
    elif kind == "brief spike":
        # deliberately shorter than T_grow: the controller is RIGHT not to react
        a = np.full(n, lo)
        s0 = n // 3
        a[s0:s0 + spike_len_ticks(n, spike_ticks)] = hi
    elif kind == "slow ramp":
        a = np.linspace(lo, hi, n)
    elif kind == "flapping":
        # A square wave must COMPLETE at least two periods inside the horizon, or it degenerates
        # into a step: at period_h == hours the half-period is n/2, so alpha goes lo -> hi once
        # and the trace ends without ever coming back down. Clamp the period so >= 2 whole cycles
        # always fit, which is also the minimum that can test thrash resistance at all.
        cycles = max(2.0, hours / max(period_h, 1e-6))
        half = max(1, int(round(n / (2 * cycles))))
        a = np.where((t // half) % 2 == 0, lo, hi)
    elif kind == "random walk":
        a = mean + np.cumsum(rng.normal(0, 0.01, n))
    else:
        a = np.full(n, mean)

    if noise > 0:
        a = a + rng.normal(0, noise, n)
    return np.clip(a, 0.02, 0.97)


def spike_len_ticks(n, spike_ticks=None):
    """Length of the 'brief spike' pulse, exposed so a caller can assert it is < T_grow."""
    return max(1, int(spike_ticks) if spike_ticks else n // 100)
