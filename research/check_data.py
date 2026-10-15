import pandas as pd, numpy as np
from data import load_bars, price_at
b1, b5 = load_bars("1M"), load_bars("5M")
print("1M range", b1.index[0], "->", b1.index[-1], len(b1))
print("5M range", b5.index[0], "->", b5.index[-1], len(b5))

# --- consistency: same prices from 1M and 5M on the overlap
pts = [("09:30","open",), ("10:00","close"), ("16:00","close")]
for hhmm, kind in pts:
    a = price_at(b1, 1, hhmm, kind); b = price_at(b5, 5, hhmm, kind)
    j = pd.concat([a, b], axis=1, keys=["m1","m5"]).dropna()
    diff = (j.m1 - j.m5).abs()
    print(f"{hhmm} {kind}: overlap days={len(j)}, identical={np.mean(diff==0):.4f}, max abs diff={diff.max()}, days diff>1pt={int((diff>1).sum())}")

# --- continuity of the newest 1M segment (imported later)
cut = pd.Timestamp("2026-08-26", tz="America/New_York")
seg = b1[b1.index >= cut - pd.Timedelta(days=3)]
gaps = seg.close.diff().abs()
print("largest 1-min moves around the 5M end date / newest segment:")
print(seg.assign(move=gaps).sort_values("move", ascending=False).head(5)[["open","close","volume","move"]])

# --- roll check: gap at the daily 18:00 reopen (16:59 close -> 18:00 open), in % 
def reopen_gap(b, step):
    c = price_at(b, step, "17:00", "close")     # last price before 17:00 break
    o = b[b.index.strftime("%H:%M") == "18:00"]["open"]; o.index = o.index.tz_localize(None).normalize()
    j = pd.concat([c, o], axis=1, keys=["c","o"]).dropna()
    return (j.o / j.c - 1) * 100
g = reopen_gap(b5, 5)
print("\n18:00 reopen gap %: abs > 0.4% days:")
print(g[g.abs() > 0.4].round(3).to_string())
