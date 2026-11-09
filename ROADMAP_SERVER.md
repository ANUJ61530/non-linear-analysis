# Roadmap: running the nonlinear analysis on all assets on a server

Goal: run the two notebooks of this folder (`chaos_classification` and `volatility_chaos`, including the power
spectrum and period sections) on **all 10 cryptocurrencies and all 50 Nifty 50 stocks** (plus the Nifty 50 index
as a reference), collect the results in one place, and compare assets with proper multiple-testing control.

The analysis itself does not change. What changes is how it is run: from one notebook per asset run by hand to a
script that runs every asset unattended, in parallel, with checkpoints.

---

## 0. The server and the usage limit

Server: the shared lab workstation (Tyrone workstation, reachable by ssh on the institute network). It is used by
others, so **this project may use at most 50% of the machine**:

- **CPU:** at most half the cores. Every script reads `max_workers = floor(cores / 2)` from the config and never
  starts more worker processes than that (one thread per process: `OMP_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`).
- **Priority:** jobs run with `nice -n 19`, so other users' interactive work always comes first.
- **Memory:** at most half the RAM; with under 1 GB per worker this is far from the limit, and the launcher
  checks free memory before starting each asset.
- **Disk:** results go to the project folder only; surrogate distributions are stored compactly (statistics,
  not series).
- **Monitoring:** a small status file (`results/logs/status.csv`: asset, stage, start, end) so progress can be
  checked without watching the processes.
- Login credentials are never written into any file of the repository; access uses an ssh key.

## 0b. Information needed about the server (before any setup)

| Question | Why it matters |
|---|---|
| Number of CPU cores and RAM | sets how many assets run at once (estimates below) |
| Operating system (Linux distribution) and Python version available | environment setup |
| Is there a job scheduler (SLURM, PBS) or is it a plain machine reached by ssh? | decides between a job array and a simple parallel launcher |
| Wall-time limit per job, if any | long assets need checkpointing |
| How data and code get there (git, scp, shared drive) | step 2 |
| Is internet access allowed from the server (to install packages)? | otherwise packages are copied as wheels |

---

## 1. What has to change in the code

### 1.1 One analysis function per asset (instead of notebook cells)
- Move the computation in the notebooks into `nla_pipeline.py` with one entry point:
  `run_asset(asset, config) -> results folder`. The notebooks stay as they are and become the place to **view**
  results for any single asset.
- Every stage writes its output to disk (`results/<asset>/stage_name.pkl` and `.csv`) and is skipped if the file
  already exists. A job that is killed can be restarted without losing finished stages.

### 1.2 One configuration file
`config_server.yaml` holding everything that is currently a constant in the notebooks:
seed, surrogate counts (999 full sample, 499 halves and periods), Welch segment lengths, slope ranges, realised
volatility window, period boundaries, FDR level, number of worker processes.

### 1.3 Asset-specific rules (decided now, before seeing any result)

| Issue | Crypto | Nifty stocks |
|---|---|---|
| Trading calendar | 7 days a week | NSE trading days (about 250 per year) |
| Price column | close | dividend-adjusted close (`adjclose`), corporate-action corrections from the cleaned panel (TRENT bonus, TMPV demerger) |
| Sample end | 2026-09-02 | 2026-10-16 |
| Sample start | first available day | 2016-01-01 (the common start used in `chaos_markets`) |
| Realised volatility window | 20 days | 20 trading days |
| "Weekly" frequency in the spectrum test | 1/7 cycles per day | 1/5 cycles per trading day |
| Overnight gaps | none | daily close-to-close returns, as for BTC; overnight and intraday parts are a later extension |

### 1.4 Short histories
Six stocks list after 2016 (BSE 2017-02, SBILIFE 2017-10, HDFCLIFE 2017-11, MAXHEALTH 2020-08, ETERNAL 2021-07,
JIOFIN 2023-08), and SOL, AVAX and DOT start in August and September 2020. Fixed rule:
- at least 1500 observations: full analysis;
- 750 to 1499 observations: full analysis, flagged "low power" in every table;
- fewer than 750: only descriptive statistics and spectra, no chaos classification (no current asset falls
  here; JIOFIN has about 780 trading days and is the shortest, so it is "low power").
- The four market periods (2017-2019, 2020-2021, 2022-2023, 2024-2026) are analysed only where the asset has data
  for the whole period; Nifty stocks also get the same four windows for comparability.

### 1.5 Cross-asset multiple testing
Each asset produces about 48 surrogate tests (4 series x 6 statistics x 2 families) plus the spectral tests.
Two levels of FDR control, both reported:
- within an asset (as now);
- across all assets for each test family (for example: "returns, S-map gain vs GARCH" over 60 assets).
A statement such as "asset X is beyond GARCH" is only made if it survives the cross-asset FDR.

---

## 2. Server setup

1. Copy the repository folders `non_linear_analysis/` and `data/` (read only; about the size of the local `data/`
   folder) to the server. Code through git, data through scp or rsync.
2. Create the environment from `requirements.txt` (pinned versions: numpy 2.5.4, pandas 3.1.0, scipy 1.18.1,
   matplotlib 3.11.2, arch 8.0.0, joblib 1.6.0, statsmodels 0.15.0):
   `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
3. Run the unit check: run BTC through the new script and compare every number with the local notebook results
   (they must be identical, since all seeds are fixed). Only then launch the full set.
4. Limit thread oversubscription: set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, because
   parallelism comes from running surrogates and assets in separate processes.

---

## 3. Compute estimate

Measured locally on BTC (n = 3284):

| Part | CPU time per asset |
|---|---|
| `chaos_classification` (returns, 99 surrogates per family, spectrum with 999) | about 0.05 CPU-hours |
| `volatility_chaos` sections 1 to 9 (4 series, 999 surrogates, split halves with 499) | about 4.6 CPU-hours |
| sections 10 and 11 (spectra, 999 and 4 x 499 surrogates) | about 0.5 CPU-hours (estimate) |
| **total** | **about 5 CPU-hours per asset** |

Stocks have about 2700 observations from 2016 (fewer than BTC), so roughly 4 CPU-hours each.

| Assets | CPU-hours | 32 cores | 64 cores |
|---|---|---|---|
| 10 crypto + 50 stocks + index (61) | about 260 to 300 | about 9 to 10 hours | about 5 hours |

With the 50% limit the usable cores are half the machine: on a 32-core workstation that means 16 cores and
about 17 to 19 hours for all assets; on a 64-core workstation, 32 cores and about 9 to 10 hours. The run is
checkpointed per asset and stage, so it can be paused (for example during heavy use by others) and resumed.

Memory is small (under 1 GB per worker). The local run showed that sharing cores with other programs slows the
run several-fold, so the server should not be oversubscribed: one worker process per core.

How to parallelise:
- **Plain server:** one launcher script runs assets one after another, each using all cores for its surrogates
  (simplest, already how the notebooks work), or runs k assets at a time with cores / k workers each.
- **SLURM:** a job array with one task per asset (`--array=0-60`), each task with for example 8 cores.

---

## 4. Outputs

```
results/
  <asset>/            per-asset tables, surrogate distributions, figures (same figures as the notebooks)
  all_assets/
    classification_table.csv    asset x series: classification, nonlinear (yes/no), beyond GARCH (yes/no)
    surrogate_tests_long.csv    every test, with within-asset and cross-asset q-values
    spectrum_table.csv          beta, low-band share, peaks, global tests per asset and series
    period_table.csv            the same per market period
    figures/                    cross-asset summary figures
  logs/                         one log file per asset, parameters, seed, run time, package versions
```

Cross-asset summary figures:
1. heat map: assets x statistics, z-score against GARCH, marked where significant after cross-asset FDR;
2. spectral exponent beta of realised volatility by asset (crypto vs stocks), with GARCH surrogate ranges;
3. share of assets whose nonlinearity (vs IAAFT) is beyond GARCH, by series;
4. period comparison: low-band share and beta per period, crypto vs stocks.

---

## 5. Order of work

| Step | Task | Where | Check before moving on |
|---|---|---|---|
| 1 | Finish the current BTC run and freeze it as the reference | local | notebook runs without errors |
| 2 | Move notebook computations into `nla_pipeline.py` with checkpoints and `config_server.yaml` | local | BTC through the script gives the same numbers as the notebook |
| 3 | Add the asset-specific rules (section 1.3, 1.4) and a loader for Nifty stocks | local | ETH and RELIANCE run end to end with reduced surrogates (for example 49) |
| 4 | Add the cross-asset aggregation and FDR script | local | works on the three test assets |
| 5 | Set up the environment on the server and repeat the BTC check | server | identical BTC numbers |
| 6 | Run all 61 assets | server | all logs end without errors; failed assets rerun from their checkpoints |
| 7 | Aggregate, cross-asset FDR, summary figures | server or local | tables complete for every asset |
| 8 | Report and slides: crypto vs stocks, which (if any) findings survive cross-asset FDR | local | numbers in text match the tables |

Steps 2 to 4 are about one working session; step 6 is unattended compute (see section 3).

---

## 6. Optional second track: the `chaos_markets` pipeline

The earlier `chaos_markets` study (routes to chaos, regimes, coupling, walk-forward test) was run on the BTC and
Nifty 50 index baseline and is waiting for the decision on scaling to all assets. Its full Phase 3 costs about
one CPU-hour per series; with returns and log volatility for 60 assets that is about 120 CPU-hours more, which a
server makes practical without cutting surrogate counts. It can reuse the same server environment and the same
per-asset job pattern.
