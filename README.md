# non_linear_analysis

Chaos classification of a financial return series in one notebook: `chaos_classification.ipynb`.

The notebook reads a price file (read only, the file is never modified), computes log returns, draws the
diagnostic plots (returns vs time, return map, 2-d and 3-d phase space, recurrence plot), chooses the delay with
Average Mutual Information and the dimension with False Nearest Neighbours, runs the largest Lyapunov exponent,
correlation dimension, 0-1 test and a surrogate data test (IAAFT and GARCH(1,1)-t, 99 each), and classifies the
series as Random/Stochastic, Periodic, Quasi-periodic, Potentially Chaotic or Inconclusive with a rule stated
before the results.

## Rerun

```bash
cd non_linear_analysis
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name non_linear_analysis --display-name "Python (non_linear_analysis)"
.venv/bin/python build_notebook.py chaos_classification.py     # builds and runs the notebook (about 1.5 minutes)
```

Or open `chaos_classification.ipynb` in Jupyter / VS Code with the "Python (non_linear_analysis)" kernel and run all.

To analyse another asset, edit the settings cell (`DATA_FILE`, `DATE_COL`, `PRICE_COL` or `RETURN_COL`).
The random seed is fixed in the same cell, so reruns give identical numbers. Figures are saved to `figures/`.

`chaos_classification.py` is the plain-text source of the notebook (cells separated by `# %%`).

## Volatility notebook

`volatility_chaos.ipynb` (source `volatility_chaos.py`, shared methods in `nla_tools.py`) repeats the
classification for four BTC series: log returns, absolute returns, squared returns and 20-day realised
volatility. It adds the ACF, time-reversal asymmetry and S-map nonlinearity gain, GARCH surrogates built by
applying the same transformation to simulated GARCH returns, Benjamini-Hochberg FDR over all tests, and a
split-sample check for non-stationarity. With 999 surrogates per family it runs in about 40 minutes on 8 cores:

```bash
.venv/bin/python build_notebook.py volatility_chaos.py
```

Figures and the comparison table are saved to `figures/volatility/`.

## All assets (61 series)

`nla_pipeline.py` runs the same analysis as `volatility_chaos.ipynb` (including the power spectrum and market period
sections) for one asset, with checkpoints per stage; `run_all.sh` runs it for all assets in parallel and
`aggregate.py` collects the results and applies Benjamini-Hochberg FDR across assets. All settings are in
`config_server.yaml` (seed 20261103; 999 surrogates per family for the full sample, 499 for split halves and
market periods). `ROADMAP_SERVER.md` describes the setup.

```bash
bash run_all.sh                              # all assets (or: bash run_all.sh BTC ETH)
.venv/bin/python aggregate.py --results results_server
```

Assets: BTC, ETH, BNB, SOL, XRP, DOGE, ADA, AVAX, TRX, DOT, the Nifty 50 index and its 50 constituents.

## Results

- `results_server/<asset>/`: per-asset tables (`comparison.csv`, `tests_full.csv`, `tests_half1.csv`,
  `tests_half2.csv`, `spectrum.csv`, `period_tests.csv`, `period_time_stats.csv`, `descriptive.csv`), run
  metadata (`meta.json`) and the checkpoint files of every stage (`*.pkl`).
- `results_server/all_assets/`: cross-asset tables (`classification_table.csv`, `surrogate_tests_long.csv`,
  `spectrum_table.csv`, `period_table.csv`, `summary_by_market.csv`, `assets.csv`) and summary figures.
- `results_server/logs/`: run log, stage timings (`status.csv`) and one log per asset.
- `figures/`: figures of the two BTC notebooks.
- `paper/`: research paper draft (`nonlinear_chaos_paper.tex` and `.pdf`).
- `non_linear_presentation.tex` / `.pdf`: slides.

## Data

The price data are not included in this repository. The code expects them in `../data/` (read only):
`<COIN>/<COIN>_1d.csv` for the cryptocurrencies (daily candles, close at 00:00 UTC),
`nifty50/IDX_NSEI.csv` for the index and `nifty50/nifty50_clean_panel.parquet` for the stocks
(dividend-adjusted close).
