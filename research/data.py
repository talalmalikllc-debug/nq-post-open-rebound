"""Load NQ futures bars from LSE Terminal (read-only) and build one row per trading day.

Bar timestamps are bar START times (verified in check_ts.py: volume spike on the 09:30 bar,
last bar before the 17:00 maintenance break is 16:59, first after is 18:00).
So "price at HH:MM" = CLOSE of the bar that starts one bar earlier.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path.home() / ".config/lse-terminal/data"
NY = "America/New_York"


def load_bars(tf: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"NQ_F_{tf}.csv")
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(NY)
    df = df.set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    px = ["open", "high", "low", "close"]
    df[px] = (df[px] * 4).round() / 4  # snap to 0.25 tick
    return df


def price_at(df: pd.DataFrame, minutes_bar: int, hhmm: str, kind: str) -> pd.Series:
    """kind='open'  -> OPEN of the bar starting at hhmm (price exactly at hhmm, first trade)
       kind='close' -> CLOSE of the bar that ENDS at hhmm (last price before hhmm)."""
    t = pd.Timestamp(f"2000-01-01 {hhmm}")
    if kind == "close":
        t = t - pd.Timedelta(minutes=minutes_bar)
    lab = t.strftime("%H:%M")
    sel = df[df.index.strftime("%H:%M") == lab]
    s = sel[kind]
    s.index = s.index.tz_localize(None).normalize()
    return s
