"""Is the reversal real, or created by noise in the single 09:30 opening print (shared endpoint)?
Test: measure the overnight return and the morning return WITHOUT sharing the same price."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from data import load_bars
d = pd.read_parquet("daily.parquet"); D = d[d.ok_main & d.open_0931.notna()].copy()
b1 = load_bars("1M")
def at1(hhmm, kind):
    s = b1[b1.index.strftime("%H:%M") == hhmm][kind]; s.index = s.index.tz_localize(None).normalize(); return s
D["c0929"] = at1("09:29", "close"); D["c0930"] = at1("09:30", "close"); D["o0935"] = at1("09:35", "open")
D = D.dropna(subset=["c0929", "c0930", "o0935"])
pct = lambda a, b: (a / b - 1) * 100
def reg(y, x, lab):
    z = pd.concat([y, x], axis=1).dropna(); z.columns = ["y", "x"]
    m = sm.OLS(z.y, sm.add_constant(z.x)).fit(cov_type="HAC", cov_kwds={"maxlags": 8})
    print(f"{lab:72s} beta(x100)={m.params.x*100:7.2f}  t={m.tvalues.x:6.2f}  n={len(z)}")
print(f"1-minute era sample (2020+): {len(D)} days\n")
ron = pct(D.open, D.prev_close); ron_pre = pct(D.c0929, D.prev_close)
reg(pct(D.p1000, D.open), ron, "A  paper: ON(->09:30 open)    vs FH(09:30 open -> 10:00)   [shared price]")
reg(pct(D.c0930, D.open), ron, "B  ON(->09:30 open)    vs first minute only (09:30 open -> 09:31)")
reg(pct(D.p1000, D.open_0931), ron, "C  ON(->09:30 open)    vs 09:31 open -> 10:00  [no shared price]")
reg(pct(D.p1000, D.open), ron_pre, "D  ON(->09:29 close)   vs 09:30 open -> 10:00   [no shared price]")
reg(pct(D.p1000, D.open_0931), ron_pre, "E  ON(->09:29 close)   vs 09:31 open -> 10:00   [1-min gap]")
reg(pct(D.p1000, D.o0935), ron, "F  ON(->09:30 open)    vs 09:35 open -> 10:00   [5-min gap]")
# size of the first-minute move vs overnight
print(f"\nhow different is the 09:30 opening print from the price 1 min earlier? median |open-09:29 close| = {(D.open-D.c0929).abs().median():.2f} pts")
print(f"median |09:30 open -> 09:31 open| = {(D.open_0931-D.open).abs().median():.2f} pts")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = []
for lab, y in [("09:30→09:31\n(first minute)", pct(D.c0930, D.open)), ("09:30→10:00\n(paper's window)", pct(D.p1000, D.open)),
               ("09:31→10:00\n(skip 1 min)", pct(D.p1000, D.open_0931)), ("09:35→10:00\n(skip 5 min)", pct(D.p1000, D.o0935))]:
    z = pd.concat([y, ron], axis=1).dropna(); z.columns = ["y", "x"]
    m = sm.OLS(z.y, sm.add_constant(z.x)).fit(cov_type="HAC", cov_kwds={"maxlags": 8})
    rows.append((lab, m.params.x * 100, m.bse.x * 100))
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.bar(range(4), [r[1] for r in rows], yerr=[1.96 * r[2] for r in rows], capsize=5,
       color=["#dc2626", "#2563eb", "#9ca3af", "#9ca3af"])
ax.axhline(0, color="k", lw=.8); ax.set_xticks(range(4)); ax.set_xticklabels([r[0] for r in rows])
ax.set_ylabel("Reversal strength: slope x100 (below 0 = reversal)")
ax.set_title("6. Where does the reversal happen? (2020-2026, 1-minute data, 95% ranges)\nThe first minute is the only reliable piece; skip 5 minutes and nothing is left")
fig.tight_layout(); fig.savefig("charts/6_first_minute.png")
