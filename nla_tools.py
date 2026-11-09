"""Methods used by volatility_chaos.ipynb: embedding, chaos diagnostics, nonlinearity statistics, surrogates.

Same estimators and rules as chaos_classification.ipynb, collected in one place.
"""
import warnings

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------------------------
# embedding parameters
# --------------------------------------------------------------------------------------------
def embed(x, m, tau):
    n = len(x) - (m - 1) * tau
    return np.column_stack([x[i * tau: i * tau + n] for i in range(m)])


def ami(x, max_lag=30, bins=16):
    """Average mutual information (nats) for lags 0..max_lag, equiprobable bins."""
    edges = np.quantile(x, np.linspace(0, 1, bins + 1)); edges[-1] += 1e-12
    s = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
    out = []
    for lag in range(max_lag + 1):
        a, b = s[: len(s) - lag], s[lag:]
        joint = np.bincount(a * bins + b, minlength=bins * bins).reshape(bins, bins) / len(a)
        pa, pb = joint.sum(1), joint.sum(0)
        nz = joint > 0
        out.append(np.sum(joint[nz] * np.log(joint[nz] / np.outer(pa, pb)[nz])))
    return np.array(out)


def first_minimum(y):
    for i in range(1, len(y) - 1):
        if y[i] < y[i - 1] and y[i] <= y[i + 1]:
            return i
    return len(y) - 1


def acf(x, max_lag):
    s = x - x.mean(); v = s @ s
    return np.array([1.0] + [s[:-k] @ s[k:] / v for k in range(1, max_lag + 1)])


def choose_tau(x, max_lag=30):
    """tau = min(first AMI minimum, first lag with |ACF| < 1/e). Returns (tau, tau_ami, tau_acf, ami curve)."""
    A = ami(x, max_lag)
    a = acf(x, max_lag)
    below = np.where(np.abs(a[1:]) < 1 / np.e)[0]
    t_acf = int(below[0] + 1) if len(below) else max_lag
    t_ami = first_minimum(A)
    return max(1, min(t_ami, t_acf)), t_ami, t_acf, A


def fnn(x, tau, m_max=10, rtol=15.0, atol=2.0, theiler=10):
    """False nearest neighbour fraction for m = 1..m_max (Kennel et al. criteria)."""
    ra = np.std(x); out = []
    for m in range(1, m_max + 1):
        Y = embed(x, m + 1, tau); Ym = Y[:, :m]
        d, idx = cKDTree(Ym).query(Ym, k=min(2 * theiler + 3, len(Ym)))
        false = valid = 0
        for i in range(len(Ym)):
            for dd, j in zip(d[i, 1:], idx[i, 1:]):
                if abs(j - i) > theiler and dd > 0:
                    extra = abs(Y[i, m] - Y[j, m])
                    false += (extra / dd > rtol) or (np.hypot(dd, extra) / ra > atol)
                    valid += 1
                    break
        out.append(false / max(valid, 1))
    return np.array(out)


def choose_m(f, threshold=0.05):
    below = np.where(f < threshold)[0]
    return (int(below[0] + 1), True) if len(below) else (len(f), False)


# --------------------------------------------------------------------------------------------
# recurrence
# --------------------------------------------------------------------------------------------
def recurrence_matrix(x, m, tau, n_last=1000, rate=0.10):
    Y = embed(x, m, tau)[-n_last:]
    D = np.max(np.abs(Y[:, None, :] - Y[None, :, :]), axis=-1)
    eps = np.quantile(D[np.triu_indices(len(Y), 1)], rate)
    return D <= eps


def det(R, lmin=2, theiler=1):
    """Share of recurrence points on diagonal lines of length >= lmin."""
    on, total = 0, 0
    for k in range(theiler + 1, len(R)):
        d = np.diagonal(R, k).astype(int)
        total += d.sum()
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, d, 0])).reshape(-1, 2), axis=1).ravel()
        on += runs[runs >= lmin].sum()
    return on / max(total, 1)


# --------------------------------------------------------------------------------------------
# chaos diagnostics
# --------------------------------------------------------------------------------------------
def divergence(x, m, tau, theiler=10, steps=30):
    Y = embed(x, m, tau); n = len(Y) - steps
    d, idx = cKDTree(Y[:n]).query(Y[:n], k=min(2 * theiler + 12, n))
    nn = np.full(n, -1)
    for i in range(n):
        for dd, j in zip(d[i, 1:], idx[i, 1:]):
            if abs(j - i) > theiler and dd > 0:
                nn[i] = j; break
    i = np.where(nn >= 0)[0]; j = nn[i]
    return np.array([np.mean(np.log(np.linalg.norm(Y[i + k] - Y[j + k], axis=1) + 1e-300)) for k in range(steps + 1)])


def lyapunov(x, m, tau, theiler=10, steps=30):
    """Rosenstein slope over k = 1..k_sat (first k reaching 60% of the total rise, at least 4 points)."""
    cur = divergence(x, m, tau, theiler, steps)
    rise = cur.max() - cur[0]
    hit = np.where(cur - cur[0] >= 0.6 * rise)[0]
    k1 = max(int(hit[0]) if len(hit) else steps, 4)
    k = np.arange(1, k1 + 1)
    p = np.polyfit(k, cur[k], 1)
    r2 = 1 - np.var(cur[k] - np.polyval(p, k)) / np.var(cur[k])
    return float(p[0]), float(r2), (1, k1), cur


def corr_sum(x, m, tau, theiler=10, n_max=2000, n_r=25):
    Y = embed(x, m, tau)
    step = int(np.ceil(len(Y) / n_max)); Y = Y[::step]
    th = max(1, int(np.ceil(theiler / step)))
    D = np.max(np.abs(Y[:, None, :] - Y[None, :, :]), axis=-1)
    d = np.sort(D[np.triu_indices(len(Y), th + 1)])
    radii = np.logspace(np.log10(max(d[-1] * 1e-3, d[d > 0][0])), np.log10(d[-1]), n_r)
    return radii, np.searchsorted(d, radii) / len(d)


def d2_slope(radii, C):
    ok = C > 0
    sl = np.gradient(np.log(C[ok]), np.log(radii[ok]))
    sel = (C[ok] >= 1e-3) & (C[ok] <= 1e-1)
    return float(np.median(sl[sel])) if sel.sum() >= 3 else np.nan


def correlation_dimension(x, tau, theiler=10, m_values=range(1, 11)):
    """D2(m) and the saturation rule: some run of three consecutive m with spread <= 15% of the mean and
    mean < (largest m of the run) - 1."""
    d2, curves = [], {}
    for m in m_values:
        r, C = corr_sum(x, m, tau, theiler)
        curves[m] = (r, C); d2.append(d2_slope(r, C))
    d2 = np.array(d2); ms = list(m_values)
    for i in range(len(d2) - 2):
        run = d2[i:i + 3]
        if np.all(np.isfinite(run)) and run.max() - run.min() <= 0.15 * run.mean() and run.mean() < ms[i + 2] - 1:
            return d2, True, float(run.mean()), curves
    return d2, False, np.nan, curves


def zero_one(x, n_c=50, seed=0):
    """Gottwald-Melbourne 0-1 test (correlation method). Returns median K and one (p, q, c) walk."""
    s = np.asarray(x, float) - np.mean(x)
    N = len(s); ncut = N // 10
    cs = np.random.default_rng(seed).uniform(np.pi / 5, 4 * np.pi / 5, n_c)
    j = np.arange(1, N + 1); n = np.arange(1, ncut + 1)
    Ks, walk = [], None
    for c in cs:
        p, q = np.cumsum(s * np.cos(j * c)), np.cumsum(s * np.sin(j * c))
        Mn = np.array([np.mean((p[k:] - p[:-k]) ** 2 + (q[k:] - q[:-k]) ** 2) for k in n])
        Ks.append(np.corrcoef(n, Mn)[0, 1])
        if walk is None:
            walk = (p, q, c)
    return float(np.median(Ks)), walk


# --------------------------------------------------------------------------------------------
# nonlinearity statistics (they detect nonlinear dependence, not chaos)
# --------------------------------------------------------------------------------------------
def simplex_skill(x, m, tau, frac=0.7):
    """Out-of-sample correlation of a nearest-neighbour forecast of x_{t+1}, library = first 70%."""
    Y = embed(x, m, tau)
    tgt = x[(m - 1) * tau + 1:]; Y = Y[:len(tgt)]
    ntr = int(len(Y) * frac)
    d, idx = cKDTree(Y[:ntr]).query(Y[ntr:], k=m + 1)
    w = np.exp(-d / np.maximum(d[:, :1], 1e-12))
    pred = (w * tgt[:ntr][idx]).sum(1) / w.sum(1)
    return float(np.corrcoef(pred, tgt[ntr:])[0, 1])


def smap_gain(x, m, tau, frac=0.7, thetas=(1.0, 2.0, 4.0), max_test=600):
    """Nonlinearity gain of the S-map: 1 - MSE(best local map) / MSE(global linear map, theta = 0), one-step
    forecasts on the last part of the sample (up to max_test points). The same procedure is applied to the
    surrogates, so the choice of theta on the test part affects data and surrogates alike."""
    Y = embed(x, m, tau)
    tgt = x[(m - 1) * tau + 1:]; Y = Y[:len(tgt)]
    ntr = int(len(Y) * frac)
    Xtr, ytr = Y[:ntr], tgt[:ntr]
    te = np.arange(ntr, len(Y))[-max_test:]
    A0 = np.column_stack([np.ones(ntr), Xtr])
    b0, *_ = np.linalg.lstsq(A0, ytr, rcond=None)
    mse0 = np.mean((np.column_stack([np.ones(len(te)), Y[te]]) @ b0 - tgt[te]) ** 2)
    best = np.inf
    dist = np.linalg.norm(Y[te][:, None, :] - Xtr[None, :, :], axis=-1)
    dbar = dist.mean(1, keepdims=True)
    for th in thetas:
        W = np.exp(-th * dist / dbar)
        err = []
        for i, q in enumerate(te):
            w = W[i]
            b, *_ = np.linalg.lstsq(A0 * w[:, None], ytr * w, rcond=None)
            err.append((np.r_[1.0, Y[q]] @ b - tgt[q]) ** 2)
        best = min(best, np.mean(err))
    return float(1 - best / mse0)


def time_reversal(x, lag=1):
    """Time-reversal asymmetry <(x_{t+lag} - x_t)^3> / <(x_{t+lag} - x_t)^2>^{3/2}; zero for linear Gaussian
    processes and their static transforms."""
    d = x[lag:] - x[:-lag]
    return float(np.mean(d ** 3) / np.mean(d ** 2) ** 1.5)


# --------------------------------------------------------------------------------------------
# surrogates and tests
# --------------------------------------------------------------------------------------------
def iaaft(x, rs, n_iter=200):
    amp = np.abs(np.fft.rfft(x)); srt = np.sort(x)
    s = rs.permutation(x)
    for _ in range(n_iter):
        s2 = np.fft.irfft(amp * np.exp(1j * np.angle(np.fft.rfft(s))), len(x))
        s_new = srt[np.argsort(np.argsort(s2))]
        if np.array_equal(s_new, s):
            break
        s = s_new
    return s


class Garch:
    """GARCH(1,1) with Student-t innovations fitted to the returns; simulates new return paths."""

    def __init__(self, r):
        from arch import arch_model
        self.sd = np.std(r)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res = arch_model(r / self.sd, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        self.params = self.res.params

    def simulate(self, n, rs, burn=500):
        P = self.params
        mu, w, a, b, nu = P["mu"], P["omega"], P["alpha[1]"], P["beta[1]"], max(P["nu"], 2.05)
        z = stats.t.rvs(nu, size=n + burn, random_state=rs) / np.sqrt(nu / (nu - 2))
        s2, e = w / max(1 - a - b, 1e-3), np.empty(n + burn)
        for t in range(n + burn):
            e[t] = np.sqrt(s2) * z[t]
            s2 = w + a * e[t] ** 2 + b * s2
        return (mu + e[burn:]) * self.sd


def p_value(obs, null, direction):
    null = np.asarray(null, float); null = null[np.isfinite(null)]
    if not np.isfinite(obs) or len(null) < 2:
        return np.nan
    hi = (np.sum(null >= obs) + 1) / (len(null) + 1)
    lo = (np.sum(null <= obs) + 1) / (len(null) + 1)
    return hi if direction == "greater" else lo if direction == "less" else min(1.0, 2 * min(hi, lo))


def bh(p):
    """Benjamini-Hochberg adjusted p-values (NaN kept)."""
    p = np.asarray(p, float); q = np.full_like(p, np.nan)
    ok = np.isfinite(p); pv = p[ok]; n = len(pv)
    if n:
        o = np.argsort(pv)
        r = np.minimum.accumulate((pv[o] * n / np.arange(1, n + 1))[::-1])[::-1]
        out = np.empty(n); out[o] = np.minimum(r, 1); q[ok] = out
    return q


# --------------------------------------------------------------------------------------------
# power spectrum analysis
# --------------------------------------------------------------------------------------------
PSD_NPERSEG = 512                 # Welch segment length (days): resolution 1/512 cycles per day
SLOPE_RANGE = (1 / 250, 1 / 25)   # fit range in cycles per day: periods 25 to 250 days
LOW_F, HIGH_F = 1 / 20, 1 / 5     # low band: periods > 20 days; high band: periods < 5 days


def standardise(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std()


def psd_welch(x, nperseg=PSD_NPERSEG, standardised=True):
    """Welch PSD (Hann window, 50% overlap, linear detrend per segment). Frequency in cycles per day.
    standardised=True: PSD of the standardised series, in (standard deviations)^2 per (cycle per day);
    standardised=False: PSD in the series' own units squared per (cycle per day). f = 0 is dropped."""
    from scipy import signal
    z = standardise(x) if standardised else np.asarray(x, float)
    f, P = signal.welch(z, fs=1.0, window="hann", nperseg=nperseg, noverlap=nperseg // 2,
                        detrend="linear", scaling="density")
    return f[1:], P[1:]


def psd_periodogram(x, standardised=True):
    """Raw periodogram (rectangular window, linear detrend), f = 0 dropped; units as in psd_welch."""
    from scipy import signal
    z = standardise(x) if standardised else np.asarray(x, float)
    f, P = signal.periodogram(z, fs=1.0, window="boxcar", detrend="linear", scaling="density")
    return f[1:], P[1:]


def spectral_slope(f, P, frange=SLOPE_RANGE):
    """OLS fit of log10 S = c - beta log10 f over frange. Returns beta, its standard error, R^2, n points."""
    sel = (f >= frange[0]) & (f <= frange[1]) & (P > 0)
    lf, lp = np.log10(f[sel]), np.log10(P[sel])
    A = np.column_stack([np.ones(sel.sum()), lf])
    coef, res, *_ = np.linalg.lstsq(A, lp, rcond=None)
    fit = A @ coef
    r2 = 1 - np.sum((lp - fit) ** 2) / np.sum((lp - lp.mean()) ** 2)
    s2 = np.sum((lp - fit) ** 2) / max(sel.sum() - 2, 1)
    se = np.sqrt(s2 / np.sum((lf - lf.mean()) ** 2))
    return float(-coef[1]), float(se), float(r2), int(sel.sum())


def gph(x, power=0.5):
    """Geweke Porter-Hudak log-periodogram estimate of the memory parameter d over the lowest n^power Fourier
    frequencies; beta = 2 d for S(f) ~ f^(-beta) near zero. Returns d, its asymptotic s.e., m, beta."""
    z = standardise(x); n = len(z)
    m = int(n ** power)
    I = np.abs(np.fft.rfft(z - z.mean())) ** 2 / (2 * np.pi * n)
    lam = 2 * np.pi * np.arange(1, m + 1) / n
    reg = -np.log(4 * np.sin(lam / 2) ** 2)
    A = np.column_stack([np.ones(m), reg])
    coef, *_ = np.linalg.lstsq(A, np.log(I[1:m + 1]), rcond=None)
    d = float(coef[1])
    return d, float(np.pi / np.sqrt(24 * m)), m, 2 * d


def band_shares(f, P, low=LOW_F, high=HIGH_F):
    """Share of spectral power (Welch, integrated) at periods > 1/low days and at periods < 1/high days."""
    tot = np.trapezoid(P, f)
    lo = np.trapezoid(P[f <= low], f[f <= low]) / tot
    hi = np.trapezoid(P[f >= high], f[f >= high]) / tot
    return float(lo), float(hi)


def spectral_peaks(f, P, k=3, fmin=SLOPE_RANGE[0]):
    """The k largest local maxima of the PSD at f >= fmin (periods of at most 250 days, i.e. at least two cycles
    per Welch segment; the lowest bins only show the low-frequency rise): (frequency, period in days, PSD)."""
    i = np.where((P[1:-1] > P[:-2]) & (P[1:-1] >= P[2:]))[0] + 1
    i = i[f[i] >= fmin]
    i = i[np.argsort(P[i])[::-1][:k]]
    return [(float(f[j]), float(1 / f[j]), float(P[j])) for j in i]


def envelope_test(P_obs, P_null):
    """Global (max-deviation) test of a whole spectrum against surrogate spectra.
    D = max_f |log P(f) - median_null(f)| / sd_null(f); the same D is computed for every surrogate.
    Returns p-value, D, and the share of frequencies outside the pointwise 95% band."""
    L = np.log(P_null); lo = np.log(P_obs)
    med, sd = np.median(L, 0), L.std(0, ddof=1) + 1e-12
    D_obs = np.max(np.abs(lo - med) / sd)
    D_null = np.max(np.abs(L - med) / sd, axis=1)
    p = (np.sum(D_null >= D_obs) + 1) / (len(D_null) + 1)
    q025, q975 = np.quantile(L, [0.025, 0.975], axis=0)
    outside = float(np.mean((lo < q025) | (lo > q975)))
    return float(p), float(D_obs), outside


def surrogate_spectra(series_fn, r, name, garch, n, seed, kinds=("iaaft", "garch")):
    """Welch spectra of n surrogates per family. IAAFT is built from the series itself; GARCH surrogates are
    simulated returns put through series_fn (the same transformation as the data). Seeded per family and index."""
    from joblib import Parallel, delayed
    x = series_fn(r)[name]

    def one(kind, i):
        rs = np.random.default_rng([seed, 7, ["iaaft", "garch"].index(kind), i])
        s = iaaft(x, rs) if kind == "iaaft" else series_fn(garch.simulate(len(r), rs))[name]
        f, P = psd_welch(s)
        return P, spectral_slope(f, P)[0], *band_shares(f, P)

    out = {}
    for kind in kinds:
        res = Parallel(n_jobs=-1)(delayed(one)(kind, i) for i in range(n))
        out[kind] = {"P": np.array([a[0] for a in res]), "beta": np.array([a[1] for a in res]),
                     "low": np.array([a[2] for a in res]), "high": np.array([a[3] for a in res])}
    return out


def period_spectra_surrogates(x_seg, r_ext, name, series_fn, n, seed, nperseg, slope_range, warmup):
    """Welch spectra (standardised series) of n IAAFT and n GARCH surrogates for one market period.

    x_seg : the observed series in the period.
    r_ext : returns of the period plus `warmup` earlier days; GARCH(1,1)-t is fitted on the period's own returns
            and each simulated path (length len(r_ext)) is transformed by series_fn, which drops the warm-up days,
            exactly as the data were built."""
    from joblib import Parallel, delayed
    garch = Garch(r_ext[warmup:])

    seed = list(np.atleast_1d(seed))

    def one(kind, i):
        rs = np.random.default_rng([*seed, ["iaaft", "garch"].index(kind), i])
        s = iaaft(x_seg, rs) if kind == "iaaft" else series_fn(garch.simulate(len(r_ext), rs))[name]
        f, P = psd_welch(s, nperseg)
        return P, *band_shares(f, P), spectral_slope(f, P, slope_range)[0]

    out = {}
    for kind in ("iaaft", "garch"):
        res = Parallel(n_jobs=-1)(delayed(one)(kind, i) for i in range(n))
        out[kind] = {"P": np.array([a[0] for a in res]), "low": np.array([a[1] for a in res]),
                     "high": np.array([a[2] for a in res]), "beta": np.array([a[3] for a in res])}
    out["garch_params"] = garch.params
    return out
