import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from data import load_bars
b5 = load_bars("5M")
hm = b5.index.strftime("%H:%M")
rth = b5[(hm >= "09:30") & (hm < "10:00") & (b5.index.dayofweek < 5)]
v = rth.volume.groupby(rth.index.tz_localize(None).normalize()).sum()
print("median first-30-min volume (contracts) by year/quarter:")
print(v.groupby(v.index.to_period("Q")).median().head(10).to_string())
print(v.groupby(v.index.year).median().to_string())
# missing 5M bars inside RTH
cnt = rth.groupby(rth.index.tz_localize(None).normalize()).size()
print("days with <6 bars in 09:30-10:00:", int((cnt < 6).sum()), "of", len(cnt))
