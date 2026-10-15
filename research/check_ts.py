import pandas as pd
from pathlib import Path
DATA = Path.home() / ".config/lse-terminal/data"
for tf in ["1M", "5M"]:
    df = pd.read_csv(DATA / f"NQ_F_{tf}.csv")
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("America/New_York")
    df = df.set_index("ts").sort_index()
    df = df[df.index >= "2021-01-01"]
    df = df[df.index.dayofweek < 5]
    hm = df.index.strftime("%H:%M")
    vol = df.groupby(hm)["volume"].median()
    print(f"== {tf}: median volume by clock time around key moments")
    for t in (["09:28","09:29","09:30","09:31","09:32"] if tf=="1M" else ["09:20","09:25","09:30","09:35","09:40"]):
        print("  ", t, vol.get(t))
    for t in (["15:58","15:59","16:00","16:01"] if tf=="1M" else ["15:50","15:55","16:00","16:05"]):
        print("  ", t, vol.get(t))
    # daily maintenance break: which times exist between 16:55 and 18:10?
    present = sorted(set(hm[(hm >= "16:50") & (hm <= "18:10")]))
    cnt = pd.Series(hm).value_counts()
    print("   times present near the 17:00-18:00 break (with counts):", [(p, int(cnt[p])) for p in present])
