import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
exec(open("project2.py").read().split("# ---------------- STEP 1")[0].replace("print(", "(lambda *a, **k: None)("))
COSTS["extreme"] = 2 * 4 * TICK + COMM / POINT
best = "z<=-2.0 | R5>0"
for name in list(PERIODS) + ["VAL+TEST"]:
    x = select(D.loc["2022-01-01":], best) if name == "VAL+TEST" else select(period(D, name), best)
    cells = []
    for c in ["optimistic", "base", "stress", "extreme"]:
        s = stats_of(x.gross, COSTS[c]); cells.append(f"{c}={s['net']:+6.2f}pt (t={s['t']:+.2f}, win={s['win']:.0f}%)")
    print(f"{name:9s} n={len(x):3d} gross={x.gross.mean():+6.2f} | " + " | ".join(cells))
x = select(D.loc["2022-01-01":], best).dropna(subset=["gross_delay"])
print(f"1-min later entry (09:36) + extreme cost, 2022+: n={len(x)} net={x.gross_delay.mean()-COSTS['extreme']:+.2f} pt")
