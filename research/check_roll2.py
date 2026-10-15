import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from data import load_bars
b5 = load_bars("5M"); b1 = load_bars("1M")
b = pd.concat([b5[b5.index < "2026-08-26"], b1[b1.index >= "2026-08-26"]])
# "session day" = futures session that starts 18:00 previous evening -> label by date of the 17:00 end
sess = (b.index + pd.Timedelta(hours=6)).tz_localize(None).normalize()
dv = b.volume.groupby(sess).sum()
dv = dv[dv.index.dayofweek < 5]
def third_friday(y, m):
    d = pd.Timestamp(y, m, 15); return d + pd.Timedelta(days=(4 - d.dayofweek) % 7)
rows = []
for y in range(2016, 2027):
    for m in (3, 6, 9, 12):
        exp = third_friday(y, m)
        w = dv[(dv.index >= exp - pd.Timedelta(days=21)) & (dv.index <= exp)]
        if len(w) < 8: continue
        ratio = (w / w.shift(1)).dropna()
        d = ratio.idxmax()
        rows.append((exp.date(), d.date(), d.day_name()[:3], (exp - d).days, round(ratio.max(), 2),
                     " ".join(f"{v/1e3:.0f}" for v in w.values)))
r = pd.DataFrame(rows, columns=["expiry", "roll_day(vol jump)", "dow", "days_before_exp", "vol_ratio", "daily vol (k) in window"])
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
print(r.to_string(index=False))
