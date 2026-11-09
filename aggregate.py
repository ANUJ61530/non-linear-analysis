"""Collect the per-asset results of nla_pipeline.py, apply cross-asset FDR control and draw the summary figures.

    python aggregate.py [--results results]

Cross-asset FDR: for every test family (series x surrogate x statistic for the surrogate tests; series x test for the
spectral tests) the p-values of all assets are adjusted together with Benjamini-Hochberg. A finding is reported as
"beyond GARCH across assets" only if it survives this adjustment.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nla_tools as T

HERE = Path(__file__).resolve().parent
NAMES = ["returns", "absolute returns", "squared returns", "realised vol (20d)"]
STATS = ["lyapunov", "D2_m5", "K_01", "pred_skill", "nl_gain", "time_reversal"]


def load(res):
    metas, comp, tests, spec, per = [], [], [], [], []
    for d in sorted(p for p in res.iterdir() if p.is_dir() and (p / "meta.json").exists()):
        m = json.loads((d / "meta.json").read_text())
        if "finished" not in m:
            continue
        metas.append(m)
        for name, lst in [("comparison", comp), ("tests_full", tests), ("spectrum", spec), ("period_tests", per)]:
            f = d / f"{name}.csv"
            if f.exists() and f.stat().st_size > 1:
                t = pd.read_csv(f)
                t.insert(0, "asset", m["asset"]) if "asset" not in t.columns else None
                t["market"] = m["market"]
                lst.append(t)
    cat = lambda l: pd.concat(l, ignore_index=True) if l else pd.DataFrame()
    return pd.DataFrame(metas), cat(comp), cat(tests), cat(spec), cat(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    res = HERE / a.results
    out = res / "all_assets"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    meta, comp, tests, spec, per = load(res)
    print(f"{len(meta)} finished assets")
    meta.to_csv(out / "assets.csv", index=False)

    # surrogate tests: cross-asset BH within each (series, surrogate, statistic)
    tests["q cross-asset"] = np.nan
    for _, g in tests.groupby(["series", "surrogate", "statistic"]):
        tests.loc[g.index, "q cross-asset"] = T.bh(g["p"].values)
    tests["significant cross-asset"] = tests["q cross-asset"] < a.alpha
    tests.to_csv(out / "surrogate_tests_long.csv", index=False)

    sig = lambda asset, k, fam, col: tests[(tests.asset == asset) & (tests.series == k) & (tests.surrogate == fam) & tests[col]]["statistic"].tolist()
    comp["beyond GARCH cross-asset"] = [", ".join(sig(r.asset, r.series, "GARCH", "significant cross-asset")) or "none" for r in comp.itertuples()]
    comp["nonlinear cross-asset"] = [", ".join(sig(r.asset, r.series, "IAAFT", "significant cross-asset")) or "none" for r in comp.itertuples()]
    comp.to_csv(out / "classification_table.csv", index=False)

    # spectral tests: cross-asset BH within each (series, test)
    if len(spec):
        pcols = [c for c in spec.columns if c.startswith("p ") or c.startswith("global p")]
        for c in pcols:
            spec[f"{c} | q cross-asset"] = np.nan
            for _, g in spec.groupby("series"):
                spec.loc[g.index, f"{c} | q cross-asset"] = T.bh(g[c].values)
        spec.to_csv(out / "spectrum_table.csv", index=False)
    if len(per):
        for c in ["global p vs GARCH", "p low share vs GARCH", "p beta vs GARCH"]:
            per[f"{c} | q cross-asset"] = np.nan
            for _, g in per.groupby(["series", "period"]):
                per.loc[g.index, f"{c} | q cross-asset"] = T.bh(g[c].values)
        per.to_csv(out / "period_table.csv", index=False)

    # summary counts
    summ = comp.groupby(["market", "series"]).agg(
        assets=("asset", "nunique"),
        nonlinear_within=("nonlinear", "mean"),
        beyond_garch_within=("beyond GARCH", "mean"),
        nonlinear_cross=("nonlinear cross-asset", lambda s: float((s != "none").mean())),
        beyond_garch_cross=("beyond GARCH cross-asset", lambda s: float((s != "none").mean())),
        potentially_chaotic=("classification", lambda s: int((s == "Potentially Chaotic").sum())),
        random_stochastic=("classification", lambda s: int((s == "Random/Stochastic").sum())),
        inconclusive=("classification", lambda s: int((s == "Inconclusive").sum()))).reset_index()
    summ.to_csv(out / "summary_by_market.csv", index=False)
    print(summ.to_string(index=False))

    # figure 1: z-scores against GARCH (assets x series/statistic), dots where significant across assets
    g = tests[tests.surrogate == "GARCH"].copy()
    g["col"] = g["series"].str.replace("realised vol (20d)", "RV20").str.replace(" returns", "") + ": " + g["statistic"]
    order_cols = [f"{s.replace('realised vol (20d)', 'RV20').replace(' returns', '')}: {st}" for s in NAMES for st in STATS]
    Z = g.pivot_table(index="asset", columns="col", values="z").reindex(columns=order_cols)
    Sg = g.pivot_table(index="asset", columns="col", values="significant cross-asset", aggfunc="max").reindex(columns=order_cols)
    mk = meta.set_index("asset")["market"]
    Z = Z.loc[sorted(Z.index, key=lambda x: (mk.get(x) != "crypto", x))]
    Sg = Sg.loc[Z.index]
    fig, ax = plt.subplots(figsize=(14, max(6, 0.22 * len(Z))))
    im = ax.imshow(np.clip(Z.values, -4, 4), cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
    yy, xx = np.where(Sg.values.astype(bool))
    ax.scatter(xx, yy, s=8, color="black", label="significant across assets (BH)")
    ax.set_xticks(range(len(order_cols))); ax.set_xticklabels(order_cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(Z))); ax.set_yticklabels(Z.index, fontsize=7)
    ax.set_title("z-score of each statistic against GARCH(1,1)-t surrogates (clipped at +/-4)")
    fig.colorbar(im, ax=ax, label="z (data vs GARCH surrogates)")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout(); fig.savefig(out / "figures" / "01_z_vs_garch_heatmap.png", dpi=150); plt.close(fig)

    # figure 2: spectral exponent of realised volatility with GARCH surrogate ranges
    if len(spec):
        s = spec[spec.series == "realised vol (20d)"].copy()
        s = s.sort_values(["market", "beta (Welch, 25-250 d)"])
        fig, ax = plt.subplots(figsize=(max(8, 0.25 * len(s)), 4.5))
        x = np.arange(len(s))
        ax.vlines(x, s["beta GARCH 2.5%"], s["beta GARCH 97.5%"], color="#1baf7a", lw=3, alpha=0.6, label="GARCH surrogates 95%")
        ax.scatter(x, s["beta (Welch, 25-250 d)"], color=np.where(s.market == "crypto", "#eb6834", "#2a78d6"), s=20, zorder=3, label="data")
        ax.set_xticks(x); ax.set_xticklabels(s.asset, rotation=90, fontsize=7)
        ax.set(title="Spectral exponent beta of 20-day realised volatility (orange: crypto, blue: Nifty)", ylabel="beta (periods 25-250 days)")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "figures" / "02_beta_rv_by_asset.png", dpi=150); plt.close(fig)

    # figure 3: share of assets nonlinear / beyond GARCH (cross-asset FDR), by series and market
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.2
    for i, (mkt, col) in enumerate([("crypto", "#eb6834"), ("nifty", "#2a78d6")]):
        t = summ[summ.market == mkt].set_index("series").reindex(NAMES)
        ax.bar(np.arange(4) + (2 * i - 1.5) * w, t["nonlinear_cross"], w, color=col, alpha=0.5, label=f"{mkt}: nonlinear (vs IAAFT)")
        ax.bar(np.arange(4) + (2 * i - 0.5) * w, t["beyond_garch_cross"], w, color=col, label=f"{mkt}: beyond GARCH")
    ax.set_xticks(range(4)); ax.set_xticklabels(NAMES)
    ax.set(ylabel="share of assets", ylim=(0, 1), title="Share of assets with any statistic significant after cross-asset FDR")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "figures" / "03_share_nonlinear_beyond_garch.png", dpi=150); plt.close(fig)

    # figure 4: period comparison of low-band share and slope, crypto vs stocks
    if len(per):
        fig, ax = plt.subplots(2, 4, figsize=(16, 6.5))
        for j, k in enumerate(NAMES):
            for i, (val, lab) in enumerate([("low share (>20 d)", "share of power at periods > 20 d"), ("beta (25-128 d)", "spectral exponent beta")]):
                a = ax[i, j]
                for mkt, col, off in [("crypto", "#eb6834", -0.15), ("nifty", "#2a78d6", 0.15)]:
                    t = per[(per.series == k) & (per.market == mkt)]
                    labs = sorted(t.period.unique())
                    for pi, p in enumerate(labs):
                        v = t[t.period == p][val].values
                        a.scatter(np.full(len(v), pi + off), v, s=8, color=col, alpha=0.5)
                        a.plot([pi + off - 0.1, pi + off + 0.1], [np.median(v)] * 2, color=col, lw=2)
                    a.set_xticks(range(len(labs))); a.set_xticklabels([l.split()[0] for l in labs])
                a.set(title=k if i == 0 else None, ylabel=lab if j == 0 else None, xlabel="period" if i == 1 else None)
        fig.suptitle("Period spectra across assets (orange: crypto, blue: Nifty; bars: medians)")
        fig.tight_layout(); fig.savefig(out / "figures" / "04_period_spectra_across_assets.png", dpi=150); plt.close(fig)
    print(f"tables and figures written to {out}")


if __name__ == "__main__":
    main()
