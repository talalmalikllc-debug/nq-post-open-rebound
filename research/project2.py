"""Research Project 2: NQ post-open reversal after a LARGE overnight drop (09:35 -> 10:00).

PRE-REGISTERED DESIGN (fixed before any result was looked at):
  overnight return  R_ON = 09:30 open / previous 16:00 close - 1          (same as project 1)
  drop size         z = R_ON / sd(R_ON over previous 60 sample days)     (past data only, shift(1))
  first 5 minutes   R5 = price at 09:35 / 09:30 open - 1                  (known at 09:35:00)
  candidate rules   z <= {-1.0, -1.5, -2.0}  x  R5 filter {none, R5>0, R5<0}   -> 9 rules, LONG only
  entry             OPEN of the 09:35 bar (first trade at/after 09:35:00) + slippage
  exit              CLOSE of the 09:59 bar (last trade before 10:00)        - slippage
  costs             base = 1 tick/side slippage + $4.50 commission RT = 0.725 pt
                    stress = 2 ticks/side + $4.50 = 1.225 pt ; optimistic = 0.5 tick/side = 0.475 pt
  periods           TRAIN 2016-06..2021-12 (choose rule) | VALIDATE 2022-2023 | TEST 2024..2026-10
  selection         highest mean NET points (base cost) in TRAIN among rules with >= 30 train trades
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from data import load_bars

TICK, POINT, COMM = 0.25, 20.0, 4.50
COSTS = {"optimistic": 2 * 0.5 * TICK + COMM / POINT, "base": 2 * 1 * TICK + COMM / POINT, "stress": 2 * 2 * TICK + COMM / POINT}
PERIODS = {"TRAIN": ("2016-06-01", "2021-12-31"), "VALIDATE": ("2022-01-01", "2023-12-31"), "TEST": ("2024-01-01", "2026-12-31")}

d0 = pd.read_parquet("daily.parquet")
D = d0[d0.ok_main].copy()

# extra prices: open of the 09:35 bar (entry), from 5M history + 1M-resampled tail (same construction as build_daily)
b5 = load_bars("5M"); b1 = load_bars("1M")
r5 = b1.resample("5min", label="left", closed="left").agg({"open": "first", "close": "last"}).dropna()
bars = pd.concat([b5[b5.index < "2026-08-26"], r5[r5.index >= "2026-08-26"]]).sort_index()
def at(df, lab, kind):
    s = df[df.index.strftime("%H:%M") == lab][kind]; s.index = s.index.tz_localize(None).normalize(); return s
D["o0935"] = at(bars, "09:35", "open")
D["o0936_1m"] = at(b1, "09:36", "open")          # robustness: 1 more minute of delay (2020+)

# --- features, only past/known information
D["sd60"] = D.R_ON.rolling(60, min_periods=40).std().shift(1)    # yesterday-and-before only
D["z"] = D.R_ON / D.sd60
D["R5"] = (D.p0935 / D.open - 1) * 100
D["gross"] = D.p1000 - D.o0935                                    # long points, 09:35 open -> 10:00
D["gross_delay"] = D.p1000 - D.o0936_1m

# leak check: sd60 on day t must equal the std of the 60 R_ON values STRICTLY BEFORE day t
pos_ = np.random.default_rng(0).choice(np.arange(100, len(D)), 25, replace=False)
for i in pos_:
    manual = D.R_ON.iloc[i - 60:i].std()
    assert np.isclose(manual, D.sd60.iloc[i]), D.index[i]
D = D.dropna(subset=["z", "R5", "gross"])

def period(df, name):
    a, b = PERIODS[name]; return df.loc[a:b]

RULES = {}
for thr in (-1.0, -1.5, -2.0):
    for f in ("none", "R5>0", "R5<0"):
        RULES[f"z<={thr:+.1f} | {f}"] = (thr, f)

def select(df, rule):
    thr, f = RULES[rule]
    m = df.z <= thr
    if f == "R5>0": m &= df.R5 > 0
    if f == "R5<0": m &= df.R5 < 0
    return df[m]

def stats_of(g, cost):
    net = g - cost
    n = len(net)
    if n < 2: return dict(n=n, gross=np.nan, net=np.nan, t=np.nan, p=np.nan, win=np.nan, usd=np.nan)
    t, p2 = stats.ttest_1samp(net, 0)
    return dict(n=n, gross=g.mean(), net=net.mean(), t=t, p=p2 / 2 if t > 0 else 1 - p2 / 2,
                win=(net > 0).mean() * 100, usd=net.sum() * POINT)

lines = []
P = lambda s="": (print(s), lines.append(s))
P(f"sample days with features: {len(D)}  ({D.index[0].date()} -> {D.index[-1].date()})")
P(f"costs per round trip (points): " + ", ".join(f"{k}={v:.3f}" for k, v in COSTS.items()))
for name in PERIODS:
    x = period(D, name); P(f"  {name:9s} days={len(x):4d}  unconditional long 09:35->10:00 avg gross={x.gross.mean():+.2f} pt")

# ---------------- STEP 1: choose on TRAIN only
P("\n=== STEP 1 - TRAIN 2016-06..2021-12: all 9 pre-registered rules (net = base cost) ===")
tr = period(D, "TRAIN")
rows = []
for r in RULES:
    s = stats_of(select(tr, r).gross, COSTS["base"]); rows.append((r, s))
    P(f"{r:20s} trades={s['n']:4d} gross={s['gross']:+6.2f} net={s['net']:+6.2f} pt  t={s['t']:+5.2f}  win={s['win']:5.1f}%")
eligible = [(r, s) for r, s in rows if s["n"] >= 30]
best, bs = max(eligible, key=lambda x: x[1]["net"])
P(f"\nFROZEN RULE (chosen on TRAIN only): {best}")

# ---------------- H0 test across the whole sample, independent of selection
P("\n=== H0 test: after large drops (z<=-1.5), is 09:35->10:00 higher than on other days? (gross, all periods) ===")
big = D.z <= -1.5
a, b_ = D.gross[big], D.gross[~big]
t, p = stats.ttest_ind(a, b_, equal_var=False)
P(f"large-drop days n={len(a)} avg={a.mean():+.2f} pt | other days n={len(b_)} avg={b_.mean():+.2f} pt | Welch t={t:+.2f}, one-sided p={p/2 if t>0 else 1-p/2:.3f}")
reg = stats.linregress(D.z.clip(-4, 4), D.gross)
P(f"slope of 09:35->10:00 points on z (all days): {reg.slope:+.2f} pt per 1 sd, p(two-sided)={reg.pvalue:.3f}  (negative slope = bigger drops -> bigger rebound)")

# ---------------- STEP 2/3: frozen rule on VALIDATE and TEST
P(f"\n=== STEP 2/3 - frozen rule '{best}' on unseen periods ===")
res = {}
for name in PERIODS:
    g = select(period(D, name), best).gross
    for c in COSTS:
        s = stats_of(g, COSTS[c]); res[(name, c)] = s
    s = res[(name, "base")]
    P(f"{name:9s} trades={s['n']:3d} gross={s['gross']:+6.2f} | net optimistic={res[(name,'optimistic')]['net']:+6.2f} "
      f"base={s['net']:+6.2f} (t={s['t']:+5.2f}, p={s['p']:.3f}) stress={res[(name,'stress')]['net']:+6.2f} pt | "
      f"win(base)={s['win']:5.1f}%  total(base)=${s['usd']:+,.0f}")
oos = pd.concat([select(period(D, n), best) for n in ("VALIDATE", "TEST")])
s = stats_of(oos.gross, COSTS["base"])
P(f"VAL+TEST combined: trades={s['n']}  net(base)={s['net']:+.2f} pt  t={s['t']:+.2f}  one-sided p={s['p']:.3f}")
# bootstrap CI of mean net (base) out of sample
rng = np.random.default_rng(1)
bs_means = [rng.choice(oos.gross.values - COSTS["base"], len(oos)).mean() for _ in range(10000)]
P(f"bootstrap 95% range for avg net (base) out of sample: {np.percentile(bs_means, 2.5):+.2f} .. {np.percentile(bs_means, 97.5):+.2f} pt")
# 1-minute extra delay robustness (2020+)
dl = oos.gross_delay.dropna()
P(f"extra 1-minute delay (entry 09:36 open, 2022+ trades n={len(dl)}): gross={dl.mean():+.2f}  net(base)={dl.mean()-COSTS['base']:+.2f} pt")

# ---------------- transparency: every rule in every period (selection already frozen above)
P("\n=== APPENDIX: all 9 rules in every period, net at base cost (shown AFTER freezing, for honesty) ===")
P(f"{'rule':20s} " + " ".join(f"{n:>22s}" for n in PERIODS))
for r in RULES:
    cells = []
    for n in PERIODS:
        s = stats_of(select(period(D, n), r).gross, COSTS["base"]); cells.append(f"n={s['n']:3d} net={s['net']:+6.2f}")
    P(f"{r:20s} " + " ".join(f"{c:>22s}" for c in cells))
open("project2_results.txt", "w").write("\n".join(lines))

# ---------------- charts
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
# 7: equity of frozen rule (net base & stress), with period shading
sel = select(D, best)
fig, ax = plt.subplots(figsize=(9, 5))
for c, col in [("optimistic", "#059669"), ("base", "#2563eb"), ("stress", "#dc2626")]:
    eq = ((sel.gross - COSTS[c]) * POINT).cumsum(); ax.step(eq.index, eq.values, where="post", color=col, lw=1.6, label=f"net, {c} costs ({COSTS[c]:.2f} pt/trade)")
eq = (sel.gross * POINT).cumsum(); ax.step(eq.index, eq.values, where="post", color="#6b7280", lw=1, ls="--", label="gross (no costs)")
for n, c in [("VALIDATE", "#f59e0b"), ("TEST", "#059669")]:
    a, b_ = PERIODS[n]; ax.axvspan(pd.Timestamp(a), min(pd.Timestamp(b_), D.index[-1]), color=c, alpha=.08)
    ax.text(pd.Timestamp(a) + pd.Timedelta(days=20), ax.get_ylim()[1] * .92, n.lower(), fontsize=9)
ax.axhline(0, color="k", lw=.6)
ax.set_ylabel("Cumulative $ (1 NQ contract)"); ax.set_title(f"7. Project 2 frozen rule: {best}  (long 09:35 -> 10:00)\nwhite = training, shaded = unseen periods")
ax.legend(loc="upper left", fontsize=8); fig.tight_layout(); fig.savefig("charts/7_p2_equity.png"); plt.close(fig)
# 8: avg 09:35->10:00 by drop-size bucket (all data, gross) with cost lines
bins = [-np.inf, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, np.inf]
labs = ["≤-2", "-2..-1.5", "-1.5..-1", "-1..-0.5", "-0.5..0", "0..0.5", "0.5..1", "1..1.5", "1.5..2", ">2"]
D["zb"] = pd.cut(D.z, bins, labels=labs)
g = D.groupby("zb", observed=True).gross.agg(["mean", "std", "size"])
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(range(len(g)), g["mean"], yerr=1.96 * g["std"] / np.sqrt(g["size"]), capsize=3, color=["#2563eb" if v > 0 else "#dc2626" for v in g["mean"]], alpha=.85)
ax.axhline(COSTS["base"], color="k", ls="--", lw=1, label=f"base cost {COSTS['base']:.2f} pt (must beat this)")
ax.axhline(0, color="k", lw=.6)
ax.set_xticks(range(len(g))); ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(g.index, g["size"])], fontsize=8)
ax.set_xlabel("Overnight move in units of its recent volatility (z)   ← big drops ... big rises →")
ax.set_ylabel("Avg long points 09:35 → 10:00 (gross, ±95%)")
ax.set_title("8. Does the 09:35→10:00 move depend on the size of the overnight drop? (2016-2026)")
ax.legend(); fig.tight_layout(); fig.savefig("charts/8_p2_buckets.png"); plt.close(fig)
print("charts saved")
