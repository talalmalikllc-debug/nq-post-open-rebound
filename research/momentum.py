"""Trend-following (momentum) version: LONG if overnight return > 0, SHORT if < 0.
Signal = last price before 09:30 (known in advance). Same sample/rules as research.py. NO costs."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
d = pd.read_parquet("daily.parquet"); D = d[d.ok_main].copy()
pos = np.where(D.R_ON_pre > 0, 1, -1)
windows = [("09:30 -> 10:00 (paper window)", "open", "p1000"), ("09:30 -> 09:35", "open", "p0935"),
           ("09:35 -> 10:00", "p0935", "p1000"), ("10:00 -> 11:00", "p1000", "p1100"),
           ("10:00 -> 16:00", "p1000", "p1600"), ("09:30 -> 16:00 (whole day)", "open", "p1600"),
           ("15:30 -> 16:00 (last half hour)", "p1530", "p1600")]
rows = []
for lab, a, b in windows:
    for per, lo, hi in [("FULL", "2016", "2026"), ("IN (<=2022)", "2016", "2022"), ("OUT (2023+)", "2023", "2026")]:
        m = (D.index.year >= int(lo)) & (D.index.year <= int(hi))
        pts = (pos * (D[b] - D[a]))[m].dropna()
        bh = (D[b] - D[a])[m].dropna()
        t = sm.OLS(pts, np.ones(len(pts))).fit(cov_type="HAC", cov_kwds={"maxlags": 8}).tvalues.iloc[0]
        rows.append((lab, per, len(pts), pts.mean(), t, (pts > 0).mean() * 100, pts.sum() * 20, bh.mean()))
r = pd.DataFrame(rows, columns=["window", "period", "trades", "avg_pts", "t", "win%", "total_$", "always_long_avg_pts"])
pd.set_option("display.width", 200)
print(r.round(2).to_string(index=False))
