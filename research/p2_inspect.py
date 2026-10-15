import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
exec(open("project2.py").read().split("# ---------------- STEP 1")[0].replace("print(", "(lambda *a, **k: None)("))
best = "z<=-2.0 | R5>0"
sel = select(D, best)
oos = sel.loc["2022-01-01":]
t = oos[["prev_close", "open", "R_ON", "z", "R5", "o0935", "p1000", "gross", "fh_rv"]].copy()
t["gross_%"] = t.gross / t.o0935 * 100
t["net_pt(base)"] = t.gross - COSTS["base"]
print(t.round(2).to_string())
g = oos.gross - COSTS["base"]
print(f"\nOOS trades={len(g)}  mean={g.mean():+.2f}  MEDIAN={g.median():+.2f} pt")
s = g.sort_values()
print(f"without the single best trade: mean={s.iloc[:-1].mean():+.2f};  without best 2: {s.iloc[:-2].mean():+.2f};  without best 3: {s.iloc[:-3].mean():+.2f}")
print(f"share of total OOS profit from the 3 best trades: {s.iloc[-3:].sum()/s.sum()*100:.0f}%")
print("dates of top 3:", list(s.index[-3:].date))
# in % terms across all periods
for n in PERIODS:
    x = select(period(D, n), best); pc = x.gross / x.o0935 * 100
    print(f"{n:9s} avg gross {pc.mean():+.3f}%  median {pc.median():+.3f}%  (cost base ≈ {COSTS['base']/x.o0935.mean()*100:.3f}%)")
# how volatile are these mornings? typical 25-min range vs normal days
print(f"\nmedian |09:35->10:00| on rule days: {(sel.gross).abs().median():.1f} pt vs all days {(D.gross).abs().median():.1f} pt")
