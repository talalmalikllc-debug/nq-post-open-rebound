"""Project 2, variant requested by user: 'large drop' defined by QUANTILE instead of z-score.
NOTE: VALIDATE/TEST were already viewed in the z-score run -> this is NOT a clean out-of-sample test.
Main definition: R_ON <= q-quantile of R_ON over the previous 250 sample days (past only, shift(1)).
Robustness:      R_ON <= q-quantile of R_ON in the TRAIN period (fixed thresholds, frozen)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
src = open("project2.py").read().split("# ---------------- STEP 1")[0]
exec(src.replace("print(", "(lambda *a, **k: None)("))      # same data, costs, periods, entry/exit, stats_of()

# rebuild the full day list (before dropna on z) so the rolling window is identical in construction
for q in (0.05, 0.10, 0.20):
    pass
full = d0[d0.ok_main].copy()
for q in (0.05, 0.10, 0.20):
    full[f"qroll_{int(q*100)}"] = full.R_ON.rolling(250, min_periods=200).quantile(q).shift(1)
D = D.join(full[[c for c in full.columns if c.startswith("qroll_")]])
# leak check: threshold for day t uses only the 250 R_ON values strictly before t
fi = full.dropna(subset=["qroll_10"])
for i in np.random.default_rng(0).choice(np.arange(260, len(full)), 25, replace=False):
    assert np.isclose(full.R_ON.iloc[i-250:i].quantile(.10), full.qroll_10.iloc[i])
trn = period(D, "TRAIN")
fixed = {q: trn.R_ON.quantile(q) for q in (0.05, 0.10, 0.20)}

def rules(kind):
    out = {}
    for q in (0.05, 0.10, 0.20):
        for f in ("none", "R5>0", "R5<0"):
            out[f"{kind} bottom {int(q*100):2d}% | {f}"] = (kind, q, f)
    return out
def sel(df, spec):
    kind, q, f = spec
    thr = df[f"qroll_{int(q*100)}"] if kind == "rolling" else fixed[q]
    m = df.R_ON <= thr
    if f == "R5>0": m &= df.R5 > 0
    if f == "R5<0": m &= df.R5 < 0
    return df[m]

lines = []
P = lambda s="": (print(s), lines.append(s))
Dq = D.dropna(subset=["qroll_20"])
P(f"days with a 250-day history: {len(Dq)} ({Dq.index[0].date()} -> {Dq.index[-1].date()})")
P("fixed TRAIN thresholds (overnight %): " + ", ".join(f"{int(q*100)}% -> {v:+.2f}%" for q, v in fixed.items()))
P(f"rolling 10% threshold ranged from {Dq.qroll_10.min():+.2f}% to {Dq.qroll_10.max():+.2f}% over time")

for kind in ("rolling", "fixed"):
    R = rules(kind)
    P(f"\n=== {kind.upper()} QUANTILE - STEP 1: choose on TRAIN only (GROSS, no costs) ===")
    rows = []
    for name, spec in R.items():
        s = stats_of(sel(period(Dq, "TRAIN"), spec).gross, COSTS["base"]); rows.append((name, s))
        P(f"{name:32s} trades={s['n']:4d} avg gross={s['gross']:+6.2f} pt  t={s['t']:+5.2f}  win={s['win']:5.1f}%")
    best, _ = max([(n, s) for n, s in rows if s["n"] >= 30], key=lambda x: x[1]["net"])
    P(f"FROZEN: {best}")
    for p in PERIODS:
        g = sel(period(Dq, p), R[best]).gross
        sb = stats_of(g, 0.0)
        P(f"  {p:9s} trades={sb['n']:3d} avg gross={sb['gross']:+7.2f} pt (t={sb['t']:+5.2f}, one-sided p={sb['p']:.3f}) "
          f"median={np.median(g) if len(g) else np.nan:+7.2f} win={sb['win']:5.1f}%  avg %={(g/period(Dq,p).loc[g.index].o0935*100).mean():+.3f}%")
    oos = pd.concat([sel(period(Dq, p), R[best]) for p in ("VALIDATE", "TEST")])
    g = oos.gross.sort_values()
    s = stats_of(oos.gross, 0.0)
    P(f"  VAL+TEST: trades={s['n']} gross={s['gross']:+.2f} pt (t={s['t']:+.2f}, one-sided p={s['p']:.3f}); median={g.median():+.2f}; "
      f"without best 3 trades: {g.iloc[:-3].mean():+.2f} pt; in %: {(oos.gross/oos.o0935*100).mean():+.3f}% per trade")
    P("  all rules in all periods (after freezing, transparency):")
    for name, spec in R.items():
        P("    " + f"{name:32s}" + "  ".join(f"{p[:5]} n={len(x):3d} gross={x.gross.mean():+7.2f}" for p in PERIODS for x in [sel(period(Dq, p), spec)]))
    if kind == "rolling":
        best_roll, spec_roll = best, R[best]
open("project2_quantile_results.txt", "w").write("\n".join(lines))

# chart: equity of frozen rolling-quantile rule
x = sel(Dq, spec_roll)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
fig, ax = plt.subplots(figsize=(9, 5))
eq = (x.gross * POINT).cumsum(); ax.step(eq.index, eq.values, where="post", color="#2563eb", lw=1.8, label="gross P&L (no costs)")
for n, c in [("VALIDATE", "#f59e0b"), ("TEST", "#059669")]:
    a, b_ = PERIODS[n]; ax.axvspan(pd.Timestamp(a), min(pd.Timestamp(b_), Dq.index[-1]), color=c, alpha=.08)
ax.axhline(0, color="k", lw=.6); ax.legend(loc="upper left", fontsize=8)
ax.set_ylabel("Cumulative $ gross (1 NQ contract, no costs)")
ax.set_title(f"9. Quantile version, frozen rule: {best_roll}\n(long 09:35 → 10:00; shaded = validate / test, already viewed once)")
fig.tight_layout(); fig.savefig("charts/9_p2_quantile_equity.png")
