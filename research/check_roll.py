import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from data import load_bars
b5 = load_bars("5M"); b1 = load_bars("1M")
b = pd.concat([b5[b5.index < "2026-08-26"], b1[b1.index >= "2026-08-26"].resample("5min", label="left", closed="left").agg(
    {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()])
b["ret"] = b.close.pct_change() * 100           # bar-to-bar change incl. gaps (prev close -> this close)
b["gap"] = (b.open / b.close.shift() - 1) * 100 # prev close -> this open
def third_friday(y, m):
    d = pd.Timestamp(y, m, 15); return d + pd.Timedelta(days=(4 - d.dayofweek) % 7)
rows = []
for y in range(2016, 2027):
    for m in (3, 6, 9, 12):
        exp = third_friday(y, m)
        w = b[(b.index.tz_localize(None) >= exp - pd.Timedelta(days=16)) & (b.index.tz_localize(None) <= exp + pd.Timedelta(days=1))]
        if len(w) < 100: continue
        k = w.gap.abs().idxmax()
        rows.append((exp.date(), k, round(w.gap[k], 3), int(w.volume[k])))
r = pd.DataFrame(rows, columns=["expiry", "largest_gap_bar", "gap_%", "vol"])
print(r.to_string(index=False))

print("\n--- gap on the bar at 00:00 UTC Monday of expiry week (= Sunday evening ET) vs a normal Sunday")
rows = []
bu = b.copy(); bu.index = bu.index.tz_convert("UTC")
for y in range(2016, 2027):
    for m in (3, 6, 9, 12):
        exp = third_friday(y, m)
        mon = exp - pd.Timedelta(days=4)
        for label, d in [("expiry-week Mon", mon), ("week before Mon", mon - pd.Timedelta(days=7))]:
            k = pd.Timestamp(d.date(), tz="UTC")
            if k in bu.index:
                rows.append((exp.date(), label, round(bu.gap[k], 3)))
r2 = pd.DataFrame(rows, columns=["expiry", "which", "gap_%"]).pivot(index="expiry", columns="which", values="gap_%")
print(r2.to_string())
print("\nmean gap expiry-week Mon:", r2["expiry-week Mon"].mean().round(3), " vs week-before Mon:", r2["week before Mon"].mean().round(3))
