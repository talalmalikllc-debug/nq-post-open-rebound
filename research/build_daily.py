"""Build one row per NYSE trading day with all prices/returns needed.

Timing rules (bar timestamps = bar START, verified in check_ts.py):
  price "at HH:MM"  = CLOSE of the bar that ends at HH:MM  (last trade before HH:MM)
  open at 09:30     = OPEN of the 09:30 bar                (first trade at/after 09:30)

Sample filters:
  - only full NYSE sessions (16:00 close) for both day t and the previous session t-1
  - roll windows excluded (flag `roll_win`): series is NOT back-adjusted (see check_roll.py)
  - every required bar must exist exactly (no forward-filling of stale prices)
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import pandas_market_calendars as mcal
from data import load_bars

b5 = load_bars("5M")
b1 = load_bars("1M")
# 5M history up to its last full day, then 1M resampled to 5M (identical where they overlap, check_data.py)
r5 = b1.resample("5min", label="left", closed="left").agg(
    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
bars = pd.concat([b5[b5.index < "2026-08-26"], r5[r5.index >= "2026-08-26"]]).sort_index()

def at(df, step, hhmm, kind):
    t = pd.Timestamp(f"2000-01-01 {hhmm}")
    if kind == "close":
        t -= pd.Timedelta(minutes=step)
    s = df[df.index.strftime("%H:%M") == t.strftime("%H:%M")][kind]
    s.index = s.index.tz_localize(None).normalize()
    return s

P = pd.DataFrame({
    "pre_open":  at(bars, 5, "09:30", "close"),   # last price BEFORE 09:30 (known at 09:29:59)
    "open":      at(bars, 5, "09:30", "open"),
    "p0935":     at(bars, 5, "09:35", "close"),
    "p0945":     at(bars, 5, "09:45", "close"),
    "p1000":     at(bars, 5, "10:00", "close"),
    "p1015":     at(bars, 5, "10:15", "close"),
    "p1030":     at(bars, 5, "10:30", "close"),
    "p1100":     at(bars, 5, "11:00", "close"),
    "p1200":     at(bars, 5, "12:00", "close"),
    "p1500":     at(bars, 5, "15:00", "close"),
    "p1530":     at(bars, 5, "15:30", "close"),
    "p1600":     at(bars, 5, "16:00", "close"),
})
# 1-minute-only prices (2020+): delayed entry at 09:31
P["open_0931"] = at(b1, 1, "09:31", "open")

# first-30-min volume and realized volatility (5-min returns 09:30-10:00) -- for paper's Table 3 split
hm = bars.index.strftime("%H:%M")
fh = bars[(hm >= "09:30") & (hm < "10:00")].copy()
fh["d"] = fh.index.tz_localize(None).normalize()
fh["r"] = np.log(fh.close / fh.open)
P["fh_volume"] = fh.groupby("d").volume.sum()
P["fh_rv"] = fh.groupby("d").r.apply(lambda x: np.sqrt((x ** 2).sum()))

# NYSE calendar: full sessions only
nyse = mcal.get_calendar("NYSE")
sch = nyse.schedule(start_date="2016-05-01", end_date="2026-10-13")
sch.index = pd.DatetimeIndex(sch.index).tz_localize(None).normalize()
close_ny = sch.market_close.dt.tz_convert("America/New_York")
full = (close_ny.dt.hour == 16) & (close_ny.dt.minute == 0)
cal = pd.DataFrame({"full": full})
cal["prev_day"] = cal.index.to_series().shift(1)
cal["prev_full"] = cal.full.shift(1)

d = cal.join(P, how="left")
d["prev_close"] = d.prev_day.map(P.p1600)
d["prev_p1500"] = d.prev_day.map(P.p1500)
d["prev_p1530"] = d.prev_day.map(P.p1530)
d["prev_p1000"] = d.prev_day.map(P.p1000)
d["prev_open"] = d.prev_day.map(P.open)

pct = lambda a, b: (a / b - 1) * 100
d["R_ON"] = pct(d.open, d.prev_close)
d["R_ON_pre"] = pct(d.pre_open, d.prev_close)          # signal version knowable before 09:30
d["R_FH"] = pct(d.p1000, d.open)
d["R_M"] = pct(d.p1500, d.p1000)
d["R_SLH"] = pct(d.p1530, d.p1500)
d["R_LH"] = pct(d.p1600, d.p1530)
for k in ["0935", "0945", "1000", "1015", "1030", "1100", "1200", "1600"]:
    d[f"R_open_{k}"] = pct(d[f"p{k}"], d.open)
# lagged day t-1 intraday returns (paper Table 2)
d["R_FH_lag"] = pct(d.prev_p1000, d.prev_open)
d["R_M_lag"] = pct(d.prev_p1500, d.prev_p1000)
d["R_SLH_lag"] = pct(d.prev_p1530, d.prev_p1500)
d["R_LH_lag"] = pct(d.prev_close, d.prev_p1530)

# roll windows: [third Friday - 11 days, third Friday] of Mar/Jun/Sep/Dec
def third_friday(y, m):
    x = pd.Timestamp(y, m, 15); return x + pd.Timedelta(days=(4 - x.dayofweek) % 7)
exps = [third_friday(y, m) for y in range(2016, 2027) for m in (3, 6, 9, 12)]
d["roll_win"] = False
for e in exps:
    d.loc[(d.index >= e - pd.Timedelta(days=11)) & (d.index <= e), "roll_win"] = True

need = ["open", "pre_open", "p1000", "p1500", "p1530", "p1600", "prev_close"]
d["complete"] = d[need].notna().all(axis=1)
d = d[(d.index >= "2016-06-01")]
d["ok_all"] = d.full & d.prev_full.astype(bool) & d.complete
d["ok_main"] = d.ok_all & ~d.roll_win

summary = {
    "NYSE sessions in range": len(d),
    "dropped: early close (t or t-1)": int((~(d.full & d.prev_full.astype(bool))).sum()),
    "dropped: missing required bars": int(((d.full & d.prev_full.astype(bool)) & ~d.complete).sum()),
    "usable (all days)": int(d.ok_all.sum()),
    "dropped: roll window": int((d.ok_all & d.roll_win).sum()),
    "usable (main sample)": int(d.ok_main.sum()),
    "first/last day": f"{d[d.ok_main].index[0].date()} -> {d[d.ok_main].index[-1].date()}",
    "days with 1-min 09:31 entry available": int((d.ok_main & d.open_0931.notna()).sum()),
}
for k, v in summary.items():
    print(f"{k:42s} {v}")
d.to_parquet("daily.parquet")
