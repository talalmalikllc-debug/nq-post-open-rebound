# Supporting research scripts

These scripts produced the findings summarised in the main notebook. Run them from **inside this folder**. They read the raw bars through `data.py` and write `daily.parquet`, the `*_results.txt` files and `charts/` here.

| Order | Script | Purpose |
|---|---|---|
| 1 | `check_ts.py` | Proves timestamps are bar **start** times (opening-bell volume spike, 17:00 halt) |
| 1 | `check_data.py`, `check_quality.py` | Basic coverage and quality checks on the 1-min and 5-min files |
| 1 | `check_roll.py`, `check_roll2.py` | Detects contract-roll price jumps in the non-adjusted continuous series |
| 1 | `check_firstminute.py` | Tests whether the reversal is real or created by noise in the single 09:30 opening print (`charts/6_first_minute.png`) |
| 2 | `build_daily.py` | Builds one row per NYSE session with all prices needed (`daily.parquet`) |
| 2 | `validate_rows.py` | Independently recomputes random days straight from the raw CSV (no shared code) |
| 3 | `research.py` | **Stage 1:** replication of Iwanaga & Sakemoto (2026) on NQ, plus horizons, robustness and a naive strategy → `results.txt` |
| 3 | `momentum.py` | **Stage 1b:** the trend-following (opposite) version |
| 4 | `project2.py` | **Stage 2:** pre-registered post-open rebound study (9 rules, train/validate/test) → `project2_results.txt` |
| 4 | `project2_quantile.py` | Stage 2 with quantile-based drop definitions → `project2_quantile_results.txt` |
| 4 | `p2_inspect.py`, `p2_costs.py` | Trade-level inspection and cost sensitivity |
| — | `make_notebook.py` | Regenerates `../NQ_PostOpen_Rebound.ipynb`. Run it from the **repository root**: `python research/make_notebook.py`, then execute with `jupyter nbconvert --execute --inplace`. |

## Charts

| File | Shows |
|---|---|
| `1_scatter.png` | Overnight return vs first-half-hour return |
| `2_deciles.png` | First-half-hour return by overnight-return decile |
| `3_horizons.png` | Reversal strength across holding windows |
| `4_by_year.png` | Effect year by year |
| `5_equity.png` | Naive replication strategy equity |
| `6_first_minute.png` | Reversal measured without sharing the 09:30 print |
| `7_p2_equity.png`, `8_p2_buckets.png`, `9_p2_quantile_equity.png` | Stage 2 equity and drop-size buckets |
