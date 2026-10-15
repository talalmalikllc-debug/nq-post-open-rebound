# Independent recomputation from the raw CSV (no shared helper code) for random days
import pandas as pd, numpy as np, datetime as dt
from zoneinfo import ZoneInfo
raw5 = pd.read_csv(__import__("pathlib").Path.home()/".config/lse-terminal/data/NQ_F_5M.csv").set_index("ts")
raw1 = pd.read_csv(__import__("pathlib").Path.home()/".config/lse-terminal/data/NQ_F_1M.csv").set_index("ts")
NY = ZoneInfo("America/New_York")
def ts(day, hh, mm): return int(dt.datetime(day.year, day.month, day.day, hh, mm, tzinfo=NY).timestamp())
snap = lambda x: round(x * 4) / 4
d = pd.read_parquet("daily.parquet")
main = d[d.ok_main & (d.index < "2026-08-25")]
rng = np.random.default_rng(7)
for day in sorted(rng.choice(main.index, 6, replace=False)):
    day = pd.Timestamp(day); prev = main.loc[day, "prev_day"]
    o = snap(raw5.loc[ts(day, 9, 30), "open"])            # 09:30 bar open
    p10 = snap(raw5.loc[ts(day, 9, 55), "close"])         # bar 09:55-10:00 close
    pc = snap(raw5.loc[ts(prev, 15, 55), "close"])        # bar 15:55-16:00 close of prev day
    ron, rfh = (o/pc-1)*100, (p10/o-1)*100
    ok = np.isclose(ron, main.loc[day, "R_ON"]) and np.isclose(rfh, main.loc[day, "R_FH"])
    print(day.date(), "prev", prev.date(), f"prevclose={pc} open={o} p10={p10} R_ON={ron:.4f} R_FH={rfh:.4f}", "MATCH" if ok else "MISMATCH")
print(d.loc["2026-10-08":"2026-10-12", ["full","prev_full","open","p1000","p1600","prev_close","roll_win","ok_main"]])
print("1M last bars:", pd.to_datetime(raw1.index[-3:], unit="s", utc=True).tz_convert(NY).tolist())
