"""Per-asset pipeline: the computations of volatility_chaos.ipynb (sections 1 to 11) for any crypto coin, Nifty 50
stock or the Nifty 50 index, run as a script with per-stage checkpoints.

    python nla_pipeline.py --asset BTC --jobs 8
    python nla_pipeline.py --asset RELIANCE --jobs 8 --config config_server.yaml

Every stage writes results/<asset>/<stage>.pkl and is skipped when that file exists, so an interrupted run resumes
where it stopped. Figures are not drawn here; the notebooks and aggregate.py read these results.
The raw data files are only read.
"""
import argparse
import json
import os
import pickle
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

import nla_tools as T

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
NAMES = ["returns", "absolute returns", "squared returns", "realised vol (20d)"]
CRYPTO = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "AVAX", "TRX", "DOT"]
STATS = {"lyapunov": "two-sided", "D2_m5": "less", "K_01": "two-sided", "pred_skill": "greater", "nl_gain": "greater",
         "time_reversal": "two-sided"}
KIND = {"lyapunov": "chaos", "D2_m5": "chaos", "K_01": "chaos", "pred_skill": "nonlinearity", "nl_gain": "nonlinearity",
        "time_reversal": "nonlinearity"}


# --------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------
def market_of(asset):
    return "crypto" if asset in CRYPTO else "nifty"


def load_prices(asset, cfg):
    """Daily close (crypto, index) or dividend-adjusted close from the cleaned panel (stocks). Read only."""
    d = (HERE / cfg["data_dir"]).resolve()
    if asset in CRYPTO:
        raw = pd.read_csv(d / asset / f"{asset}_1d.csv")
        raw["open_time"] = pd.to_datetime(raw["open_time"], utc=True).dt.tz_localize(None)
        s = raw.set_index("open_time").sort_index()["close"].loc[cfg["crypto"]["start"]: cfg["crypto"]["end"]]
    elif asset == "NIFTY50":
        raw = pd.read_csv(d / "nifty50" / "IDX_NSEI.csv", index_col=0, parse_dates=True)
        s = raw["close"].loc[cfg["nifty"]["start"]: cfg["nifty"]["end"]]
    else:
        panel = pd.read_parquet(d / "nifty50" / "nifty50_clean_panel.parquet")
        s = panel[("adjclose", asset)].loc[cfg["nifty"]["start"]: cfg["nifty"]["end"]]
    s = s.astype(float).dropna()
    return s[s > 0]


def make_series_fn(window):
    def make_series(r):
        r = np.asarray(r, float)
        rv = np.sqrt(pd.Series(r ** 2).rolling(window).mean().values)
        k = window - 1
        return {"returns": r[k:], "absolute returns": np.abs(r[k:]), "squared returns": r[k:] ** 2,
                "realised vol (20d)": rv[k:]}
    return make_series


# --------------------------------------------------------------------------------------------
# checkpointed stages
# --------------------------------------------------------------------------------------------
class Stages:
    def __init__(self, out, asset):
        self.out, self.asset = out, asset
        self.status = out.parent / "logs" / "status.csv"
        self.status.parent.mkdir(parents=True, exist_ok=True)

    def run(self, name, fn):
        p = self.out / f"{name}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
        t0 = time.time()
        self._log(name, "start", "")
        res = fn()
        with open(p, "wb") as f:
            pickle.dump(res, f)
        self._log(name, "done", f"{time.time() - t0:.0f}")
        return res

    def _log(self, stage, event, seconds):
        new = not self.status.exists()
        with open(self.status, "a") as f:
            if new:
                f.write("time,asset,stage,event,seconds\n")
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{self.asset},{stage},{event},{seconds}\n")


# --------------------------------------------------------------------------------------------
# computations (identical to the notebook)
# --------------------------------------------------------------------------------------------
def embeddings(S, cfg):
    emb = {}
    for k, v in S.items():
        z = (v - v.mean()) / v.std()
        tau, t_ami, t_acf, A = T.choose_tau(z, cfg["max_lag"])
        theiler = max(10, 2 * tau, cfg["rv_window"] if k.startswith("realised") else 0)
        F = T.fnn(z, tau, cfg["m_max"], theiler=theiler)
        m, reached = T.choose_m(F)
        emb[k] = {"tau": tau, "tau_ami": t_ami, "tau_acf": t_acf, "ami": A, "fnn": F, "m": m,
                  "fnn_reached": reached, "theiler": theiler}
    return emb


def diagnostics(S, emb):
    out = {}
    for k, v in S.items():
        e = emb[k]
        z = (v - v.mean()) / v.std()
        lle, r2, rng_, cur = T.lyapunov(z, e["m"], e["tau"], e["theiler"])
        d2, sat, d2v, _ = T.correlation_dimension(z, e["tau"], e["theiler"])
        K, _ = T.zero_one(v[:: e["tau"]], 50)
        R = T.recurrence_matrix(z, e["m"], e["tau"], 1000, 0.10)
        out[k] = {"lle": lle, "r2": r2, "range": rng_, "curve": cur, "d2": d2, "sat": sat, "d2v": d2v, "K": K,
                  "DET": T.det(R, theiler=1)}
    return out


def statistics(v, e, tau01):
    z = (v - v.mean()) / v.std()
    rr, C = T.corr_sum(z, 5, e["tau"], e["theiler"])
    return {"lyapunov": T.lyapunov(z, e["m"], e["tau"], e["theiler"])[0], "D2_m5": T.d2_slope(rr, C),
            "K_01": T.zero_one(v[::tau01], 20)[0], "pred_skill": T.simplex_skill(z, e["m"], e["tau"]),
            "nl_gain": T.smap_gain(z, e["m"], e["tau"]), "time_reversal": T.time_reversal(z)}


def _surrogate(kind, name, make_series, r, i, seed, garch):
    rs = np.random.default_rng([seed, ["iaaft", "garch"].index(kind), NAMES.index(name), i])
    if kind == "iaaft":
        return T.iaaft(make_series(r)[name], rs)
    return make_series(garch.simulate(len(r), rs))[name]


def run_tests(r, n_surr, seed, label, emb, make_series, jobs, alpha):
    Sd = make_series(r)
    garch = T.Garch(r)
    rows, nulls = [], {}
    for k in NAMES:
        e = {**emb[k]}
        obs = statistics(Sd[k], e, e["tau"])
        for kind in ["iaaft", "garch"]:
            sur = Parallel(n_jobs=jobs)(delayed(lambda i: statistics(_surrogate(kind, k, make_series, r, i, seed, garch), e, e["tau"]))(i)
                                        for i in range(n_surr))
            nd = pd.DataFrame(sur)
            nulls[(k, kind)] = nd
            for st, dirn in STATS.items():
                nv = nd[st].values
                rows.append({"sample": label, "series": k, "surrogate": kind.upper(), "statistic": st, "type": KIND[st],
                             "observed": obs[st], "surr mean": np.nanmean(nv), "surr 2.5%": np.nanquantile(nv, 0.025),
                             "surr 97.5%": np.nanquantile(nv, 0.975), "z": (obs[st] - np.nanmean(nv)) / np.nanstd(nv, ddof=1),
                             "p": T.p_value(obs[st], nv, dirn), "test": dirn})
    tab = pd.DataFrame(rows)
    tab["q (BH)"] = T.bh(tab["p"].values)
    tab["significant"] = tab["q (BH)"] < alpha
    return tab, nulls, garch


def classify(k, tab, diag):
    d = diag[k]
    sig = lambda fam: tab[(tab.series == k) & (tab.surrogate == fam) & tab.significant]["statistic"].tolist()
    si, sg = sig("IAAFT"), sig("GARCH")
    detdir = [s for s in ["D2_m5", "pred_skill", "nl_gain", "lyapunov"] if s in si and s in sg]
    if d["K"] < 0.2 and d["sat"] and d["d2v"] < 0.5:
        lab = "Periodic"
    elif d["K"] < 0.2 and d["sat"]:
        lab = "Quasi-periodic"
    elif d["K"] > 0.8 and d["sat"] and len(detdir) >= 2:
        lab = "Potentially Chaotic"
    elif d["K"] > 0.8 and not sg:
        lab = "Random/Stochastic"
    else:
        lab = "Inconclusive"
    return si, sg, lab


def spectrum_table(S, r, garch, make_series, n_surr, seed, week, jobs, alpha):
    surr = {k: _surrogate_spectra_jobs(make_series, r, k, garch, n_surr, seed, jobs) for k in NAMES}
    rows, spec = [], {}
    for k in NAMES:
        f, P = T.psd_welch(S[k])
        beta, se, r2, npts = T.spectral_slope(f, P)
        d, d_se, m_gph, beta_gph = T.gph(S[k])
        lo_sh, hi_sh = T.band_shares(f, P)
        g, ia = surr[k]["garch"], surr[k]["iaaft"]
        p_glob_g, _, out_g = T.envelope_test(P, g["P"])
        p_glob_i, _, out_i = T.envelope_test(P, ia["P"])
        iw = int(np.argmin(np.abs(f - 1 / week)))
        hi97 = np.quantile(g["P"], 0.975, axis=0)
        peaks = T.spectral_peaks(f, P)
        spec[k] = {"f": f, "P": P, **{f"{kind}_{q}": np.quantile(surr[k][kind]["P"], qq, axis=0)
                                      for kind in ["iaaft", "garch"] for q, qq in [("lo", 0.025), ("med", 0.5), ("hi", 0.975)]}}
        rows.append({"series": k, "beta (Welch, 25-250 d)": beta, "s.e.": se, "R^2": r2, "points": npts,
                     "beta GARCH median": float(np.median(g["beta"])), "beta GARCH 2.5%": float(np.quantile(g["beta"], 0.025)),
                     "beta GARCH 97.5%": float(np.quantile(g["beta"], 0.975)),
                     "p beta vs GARCH": T.p_value(beta, g["beta"], "two-sided"), "beta GPH": beta_gph, "beta GPH s.e.": 2 * d_se,
                     "low-band share (>20 d)": lo_sh, "low share GARCH median": float(np.median(g["low"])),
                     "p low share vs GARCH": T.p_value(lo_sh, g["low"], "two-sided"), "high-band share (<5 d)": hi_sh,
                     "top peaks (period)": "; ".join(f"{per:.1f} d" + (" (above GARCH band)" if P_ > hi97[np.argmin(np.abs(f - fr))] else "")
                                                     for fr, per, P_ in peaks),
                     f"p weekly ({week} d) vs GARCH": T.p_value(P[iw], g["P"][:, iw], "greater"),
                     "global p vs GARCH": p_glob_g, "outside GARCH band": out_g,
                     "global p vs IAAFT": p_glob_i, "outside IAAFT band": out_i})
    tab = pd.DataFrame(rows)
    gp = ["p beta vs GARCH", "p low share vs GARCH", f"p weekly ({week} d) vs GARCH", "global p vs GARCH"]
    qs = T.bh(tab[gp].values.ravel()).reshape(len(NAMES), len(gp))
    for j, c in enumerate(gp):
        tab[c.replace("p ", "q ", 1) if c.startswith("p ") else "q " + c] = qs[:, j]
    return tab, spec


def _surrogate_spectra_jobs(make_series, r, name, garch, n, seed, jobs):
    """Same as nla_tools.surrogate_spectra (same seeds), with a configurable number of worker processes."""
    x = make_series(r)[name]

    def one(kind, i):
        rs = np.random.default_rng([seed, 7, ["iaaft", "garch"].index(kind), i])
        s = T.iaaft(x, rs) if kind == "iaaft" else make_series(garch.simulate(len(r), rs))[name]
        f, P = T.psd_welch(s)
        return P, T.spectral_slope(f, P)[0], *T.band_shares(f, P)

    out = {}
    for kind in ("iaaft", "garch"):
        res = Parallel(n_jobs=jobs)(delayed(one)(kind, i) for i in range(n))
        out[kind] = {"P": np.array([a[0] for a in res]), "beta": np.array([a[1] for a in res]),
                     "low": np.array([a[2] for a in res]), "high": np.array([a[3] for a in res])}
    return out


def _period_surrogates_jobs(x_seg, r_ext, name, make_series, n, seed, nperseg, slope_range, warmup, jobs):
    """Same as nla_tools.period_spectra_surrogates (same seeds), with a configurable number of worker processes."""
    garch = T.Garch(r_ext[warmup:])
    seed = list(np.atleast_1d(seed))

    def one(kind, i):
        rs = np.random.default_rng([*seed, ["iaaft", "garch"].index(kind), i])
        s = T.iaaft(x_seg, rs) if kind == "iaaft" else make_series(garch.simulate(len(r_ext), rs))[name]
        f, P = T.psd_welch(s, nperseg)
        return P, *T.band_shares(f, P), T.spectral_slope(f, P, slope_range)[0]

    out = {}
    for kind in ("iaaft", "garch"):
        res = Parallel(n_jobs=jobs)(delayed(one)(kind, i) for i in range(n))
        out[kind] = {"P": np.array([a[0] for a in res]), "low": np.array([a[1] for a in res]),
                     "high": np.array([a[2] for a in res]), "beta": np.array([a[3] for a in res])}
    out["garch_params"] = garch.params
    return out


def periods_for(asset, cfg, dates):
    """Market periods available for this asset: included if it has at least min_share of the nominal period and at
    least min_obs observations. Returns {label: (index array, nominal position j, coverage)}."""
    pc = cfg["periods"]
    nominal = pc["crypto"] if market_of(asset) == "crypto" else pc["nifty"]
    dser = pd.Series(np.arange(len(dates)), index=dates)
    per_year = 365 if market_of(asset) == "crypto" else 248
    out = {}
    for j, (lab, (a, b)) in enumerate(nominal.items()):
        ix = dser.loc[a:b].values
        span = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25 * per_year
        cov = len(ix) / span if span else 0
        if len(ix) >= pc["min_obs"] and cov >= pc["min_share"]:
            out[lab] = (ix, j, cov)
    return out


def period_analysis(asset, cfg, S, dates, r_full, make_series, jobs, alpha):
    pc = cfg["periods"]
    W = cfg["rv_window"]
    PIDX = periods_for(asset, cfg, dates)
    ts, rows, spec = [], [], {}
    for k in NAMES:
        for lab, (ix, j, cov) in PIDX.items():
            v = S[k][ix]
            ts.append({"series": k, "period": lab, "n": len(ix), "coverage": cov, "start": str(dates[ix[0]].date()),
                       "end": str(dates[ix[-1]].date()), "mean": v.mean(), "std": v.std(), "kurtosis": pd.Series(v).kurt(),
                       "lag-1 ACF": T.acf(v, 1)[1]})
            f, P = T.psd_welch(v, pc["nperseg"], standardised=False)
            fz, Pz = T.psd_welch(v, pc["nperseg"])
            r_ext = r_full[ix[0]: ix[-1] + W]
            su = _period_surrogates_jobs(v, r_ext, k, make_series, cfg["n_surr_period"],
                                         [cfg["seed"], 11, NAMES.index(k), j], pc["nperseg"], tuple(pc["slope_range"]), W - 1, jobs)
            lo_sh, hi_sh = T.band_shares(f, Pz)
            beta, se, r2, npts = T.spectral_slope(f, Pz, tuple(pc["slope_range"]))
            pk = T.spectral_peaks(f, Pz, 3, fmin=2 / pc["nperseg"])
            row = {"series": k, "period": lab, "coverage": cov, "low share (>20 d)": lo_sh, "high share (<5 d)": hi_sh,
                   "beta (25-128 d)": beta, "beta R^2": r2, "dominant period (d)": pk[0][1] if pk else np.nan,
                   "top peaks (d)": ", ".join(f"{q[1]:.1f}" for q in pk)}
            for kind in ["iaaft", "garch"]:
                pg, _, out = T.envelope_test(Pz, su[kind]["P"])
                row[f"global p vs {kind.upper()}"], row[f"outside {kind.upper()} band"] = pg, out
            g = su["garch"]
            row["p low share vs GARCH"] = T.p_value(lo_sh, g["low"], "two-sided")
            row["p beta vs GARCH"] = T.p_value(beta, g["beta"], "two-sided")
            hi97 = np.quantile(g["P"], 0.975, axis=0)
            ip = int(np.argmin(np.abs(f - 1 / row["dominant period (d)"]))) if pk else 0
            row["dominant peak above GARCH 97.5%"] = bool(Pz[ip] > hi97[ip]) if pk else False
            row["GARCH alpha+beta"] = float(su["garch_params"]["alpha[1]"] + su["garch_params"]["beta[1]"])
            rows.append(row)
            spec[(k, lab)] = {"f": f, "P": P, "Pz": Pz, **{f"{kind}_{q}": np.quantile(su[kind]["P"], qq, axis=0)
                                                           for kind in ["iaaft", "garch"] for q, qq in [("lo", 0.025), ("med", 0.5), ("hi", 0.975)]}}
    PT = pd.DataFrame(rows)
    if len(PT):
        gcols = ["global p vs GARCH", "p low share vs GARCH", "p beta vs GARCH"]
        q = T.bh(PT[gcols].values.ravel()).reshape(len(PT), len(gcols))
        for j, c in enumerate(gcols):
            PT["q " + c.replace("p ", "", 1) if c.startswith("p ") else "q " + c] = q[:, j]
    return pd.DataFrame(ts), PT, spec


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------
def run_asset(asset, cfg, jobs):
    out = HERE / cfg["results_dir"] / asset
    out.mkdir(parents=True, exist_ok=True)
    st = Stages(out, asset)
    W = cfg["rv_window"]
    make_series = make_series_fn(W)
    price = load_prices(asset, cfg)
    r_full = np.log(price).diff().dropna()
    dates = r_full.index[W - 1:]
    r = r_full.values
    S = make_series(r)
    n = len(dates)
    power = "full" if n >= cfg["min_obs_full"] else ("low power" if n >= cfg["min_obs_any"] else "too short")
    meta = {"asset": asset, "market": market_of(asset), "n": n, "start": str(dates[0].date()), "end": str(dates[-1].date()),
            "power": power, "seed": cfg["seed"], "n_surr": cfg["n_surr"], "n_surr_half": cfg["n_surr_half"],
            "n_surr_period": cfg["n_surr_period"], "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "host": platform.node()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    alpha = cfg["alpha"]

    stats_tab = st.run("01_descriptive", lambda: pd.DataFrame({k: pd.Series(v).describe() for k, v in S.items()}).T)
    if power == "too short":
        st.run("10_spectrum_only", lambda: spectrum_table(S, r, T.Garch(r), make_series, cfg["n_surr"], cfg["seed"],
                                                          cfg["week"][market_of(asset)], jobs, alpha))
        return meta
    emb = st.run("02_embedding", lambda: embeddings(S, cfg))
    diag = st.run("03_diagnostics", lambda: diagnostics(S, emb))
    tests = st.run("04_tests_full", lambda: run_tests(r, cfg["n_surr"], cfg["seed"], "full", emb, make_series, jobs, alpha)[0])
    half = len(r) // 2
    h1 = st.run("05_tests_half1", lambda: run_tests(r[:half], cfg["n_surr_half"], cfg["seed"] + 1, "first half", emb, make_series, jobs, alpha)[0])
    h2 = st.run("06_tests_half2", lambda: run_tests(r[half:], cfg["n_surr_half"], cfg["seed"] + 2, "second half", emb, make_series, jobs, alpha)[0])

    def comparison():
        rows = []
        for k in NAMES:
            e, d = emb[k], diag[k]
            si, sg, lab = classify(k, tests, diag)
            sig_h = lambda tab, fam: tab[(tab.series == k) & (tab.surrogate == fam) & tab.significant]["statistic"].tolist()
            rows.append({"asset": asset, "market": market_of(asset), "power": power, "series": k, "tau": e["tau"],
                         "m": e["m"], "fnn_reached": e["fnn_reached"], "DET": d["DET"], "lyapunov": d["lle"], "lyapunov R^2": d["r2"],
                         "D2 saturates": d["sat"], "D2": d["d2v"], "K": d["K"],
                         "differs from IAAFT": ", ".join(si) or "none", "differs from GARCH": ", ".join(sg) or "none",
                         "nonlinear": bool(si), "beyond GARCH": bool(sg), "classification": lab,
                         "half1 vs IAAFT": ", ".join(sig_h(h1, "IAAFT")) or "none", "half1 vs GARCH": ", ".join(sig_h(h1, "GARCH")) or "none",
                         "half2 vs IAAFT": ", ".join(sig_h(h2, "IAAFT")) or "none", "half2 vs GARCH": ", ".join(sig_h(h2, "GARCH")) or "none"})
        return pd.DataFrame(rows)
    comp = st.run("07_comparison", comparison)
    garch_full = T.Garch(r)
    spec = st.run("10_spectrum", lambda: spectrum_table(S, r, garch_full, make_series, cfg["n_surr"], cfg["seed"],
                                                        cfg["week"][market_of(asset)], jobs, alpha))
    per = st.run("11_periods", lambda: period_analysis(asset, cfg, S, dates, r, make_series, jobs, alpha))

    for name, df in [("descriptive", stats_tab), ("comparison", comp), ("tests_full", tests), ("tests_half1", h1),
                     ("tests_half2", h2), ("spectrum", spec[0]), ("period_time_stats", per[0]), ("period_tests", per[1])]:
        df.to_csv(out / f"{name}.csv", index=name == "descriptive")
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--config", default="config_server.yaml")
    ap.add_argument("--jobs", type=int, default=None, help="worker processes for surrogates")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(HERE / a.config))
    jobs = a.jobs or cfg["jobs_per_asset"]
    for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ.setdefault(v, "1")
    t0 = time.time()
    meta = run_asset(a.asset, cfg, jobs)
    print(json.dumps({**meta, "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
