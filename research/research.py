"""Overnight-return -> first-half-hour reversal on NQ futures (replication of Iwanaga & Sakemoto 2026).
Run build_daily.py first. All returns in %. No transaction costs (by design, added later)."""
import warnings; warnings.filterwarnings("ignore")
import json
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d0 = pd.read_parquet("daily.parquet")
D = d0[d0.ok_main].copy()          # main sample: full sessions, complete bars, roll windows removed
ALL = d0[d0.ok_all].copy()         # robustness: roll windows kept
IS_END = "2022-12-31"
POINT = 20.0                       # $ per NQ point
out = {}


def nw_lags(n): return int(np.floor(4 * (n / 100) ** (2 / 9)))


def reg(df, y, xs):
    z = df[[y] + xs].dropna()
    X = sm.add_constant(z[xs])
    m = sm.OLS(z[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(len(z))})
    b, t = m.params[xs[0]], m.tvalues[xs[0]]
    return {"beta": b, "beta_x100": b * 100, "t": t, "p_one_sided_neg": stats.norm.cdf(t),
            "p_two_sided": m.pvalues[xs[0]], "alpha": m.params["const"], "alpha_t": m.tvalues["const"],
            "adjR2_pct": m.rsquared_adj * 100, "n": len(z), "model": m}


def fmt(r, label):
    star = "***" if r["p_two_sided"] < .01 else "**" if r["p_two_sided"] < .05 else "*" if r["p_two_sided"] < .1 else ""
    return (f"{label:34s} beta(x100)={r['beta_x100']:7.2f}{star:3s} t={r['t']:6.2f}  "
            f"p(beta<0)={r['p_one_sided_neg']:.4f}  adjR2={r['adjR2_pct']:5.2f}%  n={r['n']}")


def keep(r): return {k: (float(v) if k != "model" else None) for k, v in r.items() if k != "model"}


lines = []
P = lambda s="": (print(s), lines.append(s))

# ---------------------------------------------------------------- descriptive
P("=== SAMPLE ===")
P(f"main sample days: {len(D)}  ({D.index[0].date()} -> {D.index[-1].date()}); with roll days: {len(ALL)}")
P(f"sd(R_ON)={D.R_ON.std():.3f}%  sd(R_FH)={D.R_FH.std():.3f}%  mean R_ON={D.R_ON.mean():.4f}%  mean R_FH={D.R_FH.mean():.4f}%")
P(f"share of days R_ON<0: {(D.R_ON < 0).mean():.3f}")
P(f"corr(R_ON pre-open signal, R_ON at 09:30 open) = {D[['R_ON', 'R_ON_pre']].corr().iloc[0, 1]:.5f}; "
  f"same sign on {(np.sign(D.R_ON) == np.sign(D.R_ON_pre)).mean():.4f} of days")

# ---------------------------------------------------------------- paper Table 1
P("\n=== PAPER TABLE 1: R_interval = a + b*R_ON  (paper SPY: FH -9.19***, M 3.32, SLH 1.53, LH 6.39***) ===")
t1 = {}
for y in ["R_FH", "R_M", "R_SLH", "R_LH"]:
    r = reg(D, y, ["R_ON"]); t1[y] = keep(r); P(fmt(r, y))
out["table1"] = t1

# ---------------------------------------------------------------- paper Table 2 (lagged)
P("\n=== PAPER TABLE 2: lagged intraday returns (paper: lags insignificant, R_ON stays -8.81***) ===")
for x in ["R_FH_lag", "R_M_lag", "R_SLH_lag", "R_LH_lag"]:
    P(fmt(reg(D, "R_FH", [x]), f"R_FH on {x}"))
r = reg(D, "R_FH", ["R_ON", "R_FH_lag", "R_M_lag", "R_SLH_lag", "R_LH_lag"])
P(fmt(r, "R_FH on R_ON + all lags (R_ON coef)"))
out["table2_full"] = keep(r)

# ---------------------------------------------------------------- horizons (Table A3 + extra)
P("\n=== HORIZONS: R(09:30 -> X) on R_ON  (paper SPY: 15m -5.94, 30m -9.19, 45m -9.47, 60m -9.67, all ***) ===")
hz = {}
for k, lab in [("0935", "5m"), ("0945", "15m"), ("1000", "30m"), ("1015", "45m"), ("1030", "60m"),
               ("1100", "90m"), ("1200", "150m"), ("1600", "to 16:00")]:
    r = reg(D, f"R_open_{k}", ["R_ON"]); hz[lab] = keep(r); P(fmt(r, f"09:30 -> {lab}"))
out["horizons"] = hz

# ---------------------------------------------------------------- sign split (Table A4)
P("\n=== PAPER TABLE A4: split by sign of R_ON (paper SPY: positive -5.41**, negative -14.65***) ===")
for lab, m in [("R_ON > 0", D.R_ON > 0), ("R_ON < 0", D.R_ON < 0)]:
    r = reg(D[m], "R_FH", ["R_ON"]); out[f"sign {lab}"] = keep(r); P(fmt(r, lab))

# ---------------------------------------------------------------- vol / volume split (Table 3)
P("\n=== PAPER TABLE 3: split by first-30-min realized vol / volume (paper: high vol -9.82, low -5.34) ===")
P("   NOTE: like the paper, this split uses 09:30-10:00 information -> descriptive only, NOT tradable.")
hi_rv = D.fh_rv > D.fh_rv.median()
yr_med = D.groupby(D.index.year).fh_volume.transform("median")
hi_vol = D.fh_volume > yr_med
for lab, m in [("low realized vol", ~hi_rv), ("high realized vol", hi_rv), ("low volume (by year)", ~hi_vol), ("high volume (by year)", hi_vol)]:
    r = reg(D[m], "R_FH", ["R_ON"]); out[lab] = keep(r); P(fmt(r, lab))
# tradable alternative: yesterday's full-day realized range known before the open
D["prev_range"] = (D.prev_day.map(d0.p1600) / D.prev_day.map(d0.open) - 1).abs()
hi_prev = D.prev_range > D.prev_range.rolling(250, min_periods=60).median().shift(1)
P("   tradable version: split by |yesterday's 09:30->16:00 return| vs its own trailing 1y median (known pre-open)")
for lab, m in [("calm yesterday", ~hi_prev), ("volatile yesterday", hi_prev)]:
    r = reg(D[m], "R_FH", ["R_ON"]); out[lab] = keep(r); P(fmt(r, lab))

# ---------------------------------------------------------------- subperiods
P("\n=== SUBPERIODS ===")
per = {"2016-06 to 2019-12": ("2016-06-01", "2019-12-31"),
       "COVID phase I (2020-01-21..05-07)": ("2020-01-21", "2020-05-07"),
       "COVID phase II (2020-05-08..12-31)": ("2020-05-08", "2020-12-31"),
       "2021-01 to 2026-10": ("2021-01-01", "2026-12-31"),
       "IN-SAMPLE 2016-06..2022-12": ("2016-06-01", IS_END),
       "OUT-OF-SAMPLE 2023-01..2026-10": ("2023-01-01", "2026-12-31")}
for lab, (a, b) in per.items():
    r = reg(D.loc[a:b], "R_FH", ["R_ON"]); out[lab] = keep(r); P(fmt(r, lab))
r = reg(D.drop(D.loc["2020-01-21":"2020-05-07"].index), "R_FH", ["R_ON"]); out["ex COVID I"] = keep(r); P(fmt(r, "full sample EXCLUDING COVID phase I"))
yearly = {}
for y in sorted(set(D.index.year)):
    r = reg(D[D.index.year == y], "R_FH", ["R_ON"]); yearly[y] = keep(r); P(fmt(r, f"year {y}"))
out["yearly"] = yearly

# ---------------------------------------------------------------- robustness
P("\n=== ROBUSTNESS ===")
r = reg(ALL, "R_FH", ["R_ON"]); out["with roll days"] = keep(r); P(fmt(r, "incl. roll-window days"))
W = D.copy()
for c in ["R_ON", "R_FH"]:
    lo, hi = W[c].quantile([.01, .99]); W[c] = W[c].clip(lo, hi)
r = reg(W, "R_FH", ["R_ON"]); out["winsorized"] = keep(r); P(fmt(r, "winsorized 1%/99%"))
hub = sm.RLM(D.R_FH, sm.add_constant(D.R_ON), M=sm.robust.norms.HuberT()).fit()
P(f"{'Huber robust regression':34s} beta(x100)={hub.params['R_ON']*100:7.2f}    t={hub.tvalues['R_ON']:6.2f}")
out["huber"] = {"beta_x100": float(hub.params['R_ON'] * 100), "t": float(hub.tvalues['R_ON'])}
r = reg(D, "R_FH", ["R_ON_pre"]); out["pre-open signal"] = keep(r); P(fmt(r, "signal = last price before 09:30"))
q = D.R_ON.abs().quantile([1/3, 2/3]).values
for lab, m in [("small |R_ON| (bottom third)", D.R_ON.abs() <= q[0]), ("medium |R_ON|", (D.R_ON.abs() > q[0]) & (D.R_ON.abs() <= q[1])), ("large |R_ON| (top third)", D.R_ON.abs() > q[1])]:
    r = reg(D[m], "R_FH", ["R_ON"]); out[lab] = keep(r); P(fmt(r, lab))
# permutation test: shuffle which overnight return goes with which morning
rng = np.random.default_rng(42)
x, y = D.R_ON.values, D.R_FH.values
b_obs = np.polyfit(x, y, 1)[0]
perm = np.array([np.polyfit(rng.permutation(x), y, 1)[0] for _ in range(5000)])
p_perm = (np.sum(perm <= b_obs) + 1) / (len(perm) + 1)
P(f"permutation test (5000 shuffles): observed beta(x100)={b_obs*100:.2f}, one-sided p={p_perm:.4f}")
out["perm_p"] = float(p_perm)

# ---------------------------------------------------------------- deciles
D["dec"] = pd.qcut(D.R_ON, 10, labels=False) + 1
dec = D.groupby("dec").agg(R_ON=("R_ON", "mean"), R_FH=("R_FH", "mean"), se=("R_FH", lambda s: s.std() / np.sqrt(len(s))), n=("R_FH", "size"))
P("\n=== DECILES of R_ON: average R_FH ===")
P(dec.round(4).to_string())

# ---------------------------------------------------------------- strategy (paper eq. 6), NO costs
P("\n=== STRATEGY: long if overnight<0 else short, exit 10:00, 1 NQ contract, NO COSTS ===")
def strat(df, sig_col, entry_col, label):
    s = df[[sig_col, entry_col, "p1000", "R_FH"]].dropna()
    pos = np.where(s[sig_col] < 0, 1, -1)
    pts = pos * (s.p1000 - s[entry_col])
    pct = pos * (s.p1000 / s[entry_col] - 1) * 100
    eq = (pts * POINT).cumsum()
    dd = (eq - eq.cummax()).min()
    m = sm.OLS(pts, np.ones(len(pts))).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(len(pts))})
    res = {"label": label, "trades": len(s), "win_rate": float((pts > 0).mean()), "avg_pts": float(pts.mean()),
           "t_avg_pts": float(m.tvalues.iloc[0]), "avg_usd": float(pts.mean() * POINT), "total_usd": float(eq.iloc[-1]),
           "max_dd_usd": float(dd), "ann_ret_pct": float(pct.mean() * 252), "sharpe": float(pct.mean() / pct.std() * np.sqrt(252)),
           "long_avg_pts": float(pts[pos == 1].mean()), "short_avg_pts": float(pts[pos == -1].mean())}
    return res, pts
def show(res):
    P(f"{res['label']:44s} trades={res['trades']:5d} win={res['win_rate']*100:5.1f}% avg={res['avg_pts']:6.2f}pt "
      f"(t={res['t_avg_pts']:5.2f}) ${res['avg_usd']:7.1f}/trade total=${res['total_usd']:>10,.0f} maxDD=${res['max_dd_usd']:>9,.0f} "
      f"Sharpe={res['sharpe']:5.2f} | longs {res['long_avg_pts']:.2f}pt shorts {res['short_avg_pts']:.2f}pt")
D["always_long_sig"] = -1.0
strats = {}
curves = {}
for per_lab, a, b in [("FULL", "2016-06-01", "2026-12-31"), ("IN-SAMPLE (<=2022)", "2016-06-01", IS_END), ("OUT-OF-SAMPLE (2023+)", "2023-01-01", "2026-12-31")]:
    P(f"-- {per_lab}")
    S = D.loc[a:b]
    for key, sig, ent, lab in [("V0", "R_ON", "open", "V0 paper-style: signal & entry at 09:30 open"),
                               ("V1", "R_ON_pre", "open", "V1 clean: signal 09:29:59, entry 09:30 open"),
                               ("V2", "R_ON", "open_0931", "V2 delayed: signal 09:30, entry 09:31 (2020+)"),
                               ("BH", "always_long_sig", "open", "Benchmark: always long 09:30->10:00")]:
        res, pts = strat(S, sig, ent, lab); show(res); strats[f"{per_lab}|{key}"] = res
        if per_lab == "FULL": curves[key] = pts
out["strategy"] = strats
# yearly strategy V1
P("-- V1 by year (avg points per trade, win rate)")
pts_v1 = curves["V1"]
yr = pd.DataFrame({"pts": pts_v1}).groupby(pts_v1.index.year).pts.agg(["count", "mean", lambda s: (s > 0).mean(), "sum"])
yr.columns = ["trades", "avg_pts", "win_rate", "total_pts"]
P(yr.round(3).to_string())
out["v1_yearly"] = yr.to_dict()
# same-days comparison V1 vs V2 (2020+ days where both exist)
common = curves["V1"].index.intersection(curves["V2"].index)
P(f"-- same {len(common)} days (2020+): V1 avg {curves['V1'][common].mean():.2f}pt vs V2 (1-min delay) avg {curves['V2'][common].mean():.2f}pt")
out["v1_v2_same_days"] = {"n": len(common), "v1": float(curves['V1'][common].mean()), "v2": float(curves['V2'][common].mean())}
# typical size of a 30-min move
P(f"median |09:30->10:00 move| = {(D.p1000 - D.open).abs().median():.1f} points; median price {D.open.median():.0f}")

json.dump(out, open("results.json", "w"), indent=1, default=float)
open("results.txt", "w").write("\n".join(lines))

# ================================================================= CHARTS
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25})
BLUE, RED, GREY, GREEN = "#2563eb", "#dc2626", "#6b7280", "#059669"

# 1 scatter
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(D.R_ON, D.R_FH, s=6, alpha=.35, color=GREY)
xs = np.linspace(-4, 4, 50); r = t1["R_FH"]
ax.plot(xs, r["alpha"] + r["beta"] * xs, color=RED, lw=2.2, label=f"fitted line: slope = {r['beta']:.3f} (t = {r['t']:.2f})")
ax.axhline(0, color="k", lw=.6); ax.axvline(0, color="k", lw=.6)
ax.set_xlim(-4, 4); ax.set_ylim(-3, 3)
ax.set_xlabel("Overnight return %  (16:00 yesterday -> 09:30 today)"); ax.set_ylabel("First 30-min return %  (09:30 -> 10:00)")
ax.set_title("1. Each dot = one trading day (NQ, 2016-2026)\nA downward-sloping line would mean the overnight move reverses")
ax.legend(loc="upper right"); fig.tight_layout(); fig.savefig("charts/1_scatter.png"); plt.close(fig)

# 2 deciles
fig, ax = plt.subplots(figsize=(8, 5))
cols = [BLUE if v > 0 else RED for v in dec.R_FH]
ax.bar(dec.index, dec.R_FH, yerr=1.96 * dec.se, color=cols, alpha=.85, capsize=3)
ax.set_xticks(dec.index); ax.set_xticklabels([f"{i}\n{v:+.2f}%" for i, v in zip(dec.index, dec.R_ON)], fontsize=8)
ax.axhline(0, color="k", lw=.8)
ax.set_xlabel("Overnight-return group (1 = biggest drops ... 10 = biggest gains), avg overnight return below")
ax.set_ylabel("Average first 30-min return %  (bars = ±95% range)")
ax.set_title("2. Split days into 10 equal groups by overnight move\nReversal = left bars above zero, right bars below zero")
fig.tight_layout(); fig.savefig("charts/2_deciles.png"); plt.close(fig)

# 3 horizons
fig, ax = plt.subplots(figsize=(8, 4.8))
labs = list(hz.keys()); b = [hz[k]["beta_x100"] for k in labs]
se = [abs(hz[k]["beta_x100"] / hz[k]["t"]) for k in labs]
ax.errorbar(range(len(labs)), b, yerr=1.96 * np.array(se), fmt="o-", color=BLUE, capsize=4)
ax.axhline(0, color="k", lw=.8)
ax.axhline(-9.19, color=RED, ls="--", lw=1, label="paper's SPY result at 30 min (-9.19)")
ax.set_xticks(range(len(labs))); ax.set_xticklabels([f"09:30\n+{l}" for l in labs], fontsize=8)
ax.set_ylabel("Reversal strength: slope x100\n(below 0 = reversal)")
ax.set_title("3. How strong is the reversal if you hold from 09:30 for longer?\n(dots = estimate, lines = 95% range; touching 0 = not significant)")
ax.legend(); fig.tight_layout(); fig.savefig("charts/3_horizons.png"); plt.close(fig)

# 4 yearly
fig, ax = plt.subplots(figsize=(8, 4.8))
ys = list(yearly.keys()); b = [yearly[y]["beta_x100"] for y in ys]; se = [abs(yearly[y]["beta_x100"] / yearly[y]["t"]) for y in ys]
ax.errorbar(ys, b, yerr=1.96 * np.array(se), fmt="o", color=BLUE, capsize=4)
ax.axhline(0, color="k", lw=.8); ax.axvspan(2022.5, max(ys) + .5, color=GREEN, alpha=.08, label="out-of-sample (2023+)")
ax.set_xticks(ys); ax.set_ylabel("Reversal strength: slope x100 (below 0 = reversal)")
ax.set_title("4. Is the effect stable year by year?  (95% ranges)")
ax.legend(); fig.tight_layout(); fig.savefig("charts/4_by_year.png"); plt.close(fig)

# 5 equity
fig, ax = plt.subplots(figsize=(9, 5))
for key, lab, c in [("V1", "Reversal strategy (clean signal, entry 09:30)", BLUE), ("V2", "Reversal, 1-minute delayed entry (2020+)", GREEN), ("BH", "Always long 09:30->10:00", GREY)]:
    eq = (curves[key] * POINT).cumsum(); ax.plot(eq.index, eq.values, label=lab, color=c, lw=1.6)
ax.axvline(pd.Timestamp("2023-01-01"), color="k", ls=":", lw=1); ax.text(pd.Timestamp("2023-02-01"), ax.get_ylim()[1] * .9, "out-of-sample →", fontsize=9)
ax.axhline(0, color="k", lw=.6)
ax.set_ylabel("Cumulative profit, $ (1 NQ contract, BEFORE costs)")
ax.set_title("5. Equity curve of the paper's strategy on NQ (no spread / slippage / fees yet)")
ax.legend(loc="upper left"); fig.tight_layout(); fig.savefig("charts/5_equity.png"); plt.close(fig)
print("\ncharts saved")
