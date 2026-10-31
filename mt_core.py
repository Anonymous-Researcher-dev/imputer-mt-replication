"""Subjects, properties (test oracles), randomized experimental conditions and the evaluation harness."""
from __future__ import annotations

import contextlib
import io
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

import mt_mutate as mm

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Subjects under test (real library classes; SoftImpute is the study's own solver)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Subject:
    sid: str
    family: str          # mean | median | most_frequent | knn | iter | softimpute
    target: str          # which mutated source it belongs to
    params: tuple = ()


SUBJECTS = {s.sid: s for s in [
    Subject("SimpleImputer[mean]", "mean", "SimpleImputer"),
    Subject("SimpleImputer[median]", "median", "SimpleImputer"),
    Subject("SimpleImputer[most_frequent]", "most_frequent", "SimpleImputer"),
    Subject("KNNImputer[uniform]", "knn", "KNNImputer", (("weights", "uniform"),)),
    Subject("KNNImputer[distance]", "knn", "KNNImputer", (("weights", "distance"),)),
    Subject("IterativeImputer[BayesianRidge]", "iter", "IterativeImputer"),
    Subject("SoftImpute[custom]", "softimpute", "SoftImpute"),
]}
# KNN subjects are also exposed to faults in the shared distance kernel
KNN_SUBJECTS = ["KNNImputer[uniform]", "KNNImputer[distance]"]

TARGETS = {
    "SimpleImputer": dict(
        module="sklearn.impute._base",
        funcs=["SimpleImputer._validate_input", "SimpleImputer._dense_fit",
               "SimpleImputer.fit", "SimpleImputer.transform"],
        subjects=["SimpleImputer[mean]", "SimpleImputer[median]", "SimpleImputer[most_frequent]"],
        kind="library"),
    "KNNImputer": dict(
        module="sklearn.impute._knn",
        funcs=["KNNImputer._calc_impute", "KNNImputer.fit", "KNNImputer.transform"],
        subjects=KNN_SUBJECTS, kind="library"),
    "nan_euclidean": dict(
        module="sklearn.metrics.pairwise",
        funcs=["nan_euclidean_distances"],
        subjects=KNN_SUBJECTS, kind="library", patch="nan_euclidean"),
    "IterativeImputer": dict(
        module="sklearn.impute._iterative",
        funcs=["IterativeImputer._impute_one_feature", "IterativeImputer._get_neighbor_feat_idx",
               "IterativeImputer._get_ordered_idx", "IterativeImputer._get_abs_corr_mat",
               "IterativeImputer._validate_limit", "IterativeImputer._initial_imputation",
               "IterativeImputer.fit_transform", "IterativeImputer.transform"],
        subjects=["IterativeImputer[BayesianRidge]"], kind="library"),
    "SoftImpute": dict(
        module="mt_softimpute",
        funcs=["SoftImpute.fit_transform", "SoftImpute._shrink"],
        subjects=["SoftImpute[custom]"], kind="custom"),
}


def reference_module_for(target):
    import importlib
    return importlib.import_module(TARGETS[target]["module"])


@contextmanager
def _patched_pairwise(mod):
    import sklearn.metrics.pairwise as pw
    old = pw.PAIRWISE_DISTANCE_FUNCTIONS["nan_euclidean"]
    pw.PAIRWISE_DISTANCE_FUNCTIONS["nan_euclidean"] = mod.nan_euclidean_distances
    try:
        yield
    finally:
        pw.PAIRWISE_DISTANCE_FUNCTIONS["nan_euclidean"] = old


def make_runner(subject: Subject, mod=None):
    """Return run(X, missing_values=np.nan) -> completed matrix. `mod` is a mutated module copy or None."""
    fam = subject.family
    target = subject.target
    if fam in ("mean", "median", "most_frequent"):
        from sklearn.impute import SimpleImputer
        cls = mod.SimpleImputer if (mod is not None and target == "SimpleImputer") else SimpleImputer

        def run(X, missing_values=np.nan):
            return cls(strategy=fam, missing_values=missing_values, keep_empty_features=True).fit_transform(X)
        return run
    if fam == "knn":
        from sklearn.impute import KNNImputer
        params = dict(subject.params)
        patched = mod is not None and mod.__name__.endswith("pairwise")
        cls = mod.KNNImputer if (mod is not None and not patched) else KNNImputer

        def run(X, missing_values=np.nan):
            imp = cls(n_neighbors=5, weights=params["weights"], missing_values=missing_values,
                      keep_empty_features=True)
            if patched:
                with _patched_pairwise(mod):
                    return imp.fit_transform(X)
            return imp.fit_transform(X)
        return run
    if fam == "iter":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge
        cls = mod.IterativeImputer if mod is not None else IterativeImputer

        def run(X, missing_values=np.nan, **extra):
            return cls(estimator=BayesianRidge(), max_iter=5, tol=1e-3, initial_strategy="mean",
                       sample_posterior=False, random_state=0, imputation_order="ascending",
                       keep_empty_features=True, missing_values=missing_values, **extra).fit_transform(X)
        return run
    if fam == "softimpute":
        import mt_softimpute
        cls = mod.SoftImpute if mod is not None else mt_softimpute.SoftImpute

        def run(X, missing_values=np.nan):
            if not (isinstance(missing_values, float) and np.isnan(missing_values)):
                raise NotImplementedError("custom solver only supports NaN as missing marker")
            return cls().fit_transform(X)
        return run
    raise ValueError(fam)


# ----------------------------------------------------------------------------
# Numerical comparison with a scale-aware deviation
# ----------------------------------------------------------------------------
def col_scale(A):
    A = np.asarray(A, dtype=float)
    s = np.where(np.isfinite(A), np.abs(A), 0.0).max(axis=0) if A.size else np.ones(A.shape[1:])
    return np.where(s > 0, s, 1.0)


def scaled_dev(a, b, scale=None, cells=None):
    """max |a-b| / column scale over `cells`; inf on shape/NaN-pattern mismatch."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        return float("inf")
    if scale is None:
        scale = col_scale(b)
    if cells is None:
        cells = np.ones(a.shape, dtype=bool)
    if not cells.any():
        return 0.0
    na, nb = np.isnan(a) & cells, np.isnan(b) & cells
    if (na != nb).any():
        return float("inf")
    use = cells & ~nb
    with np.errstate(invalid="ignore"):
        d = np.abs(a - b) / scale[None, :]
    d = np.where(a == b, 0.0, d)
    d = np.where(use, d, 0.0)
    m = float(d.max())
    return float("inf") if np.isnan(m) else m


# tolerance on the scaled deviation, by implementation family
TOL_FAMILY = {"mean": 1e-9, "median": 1e-9, "most_frequent": 1e-9, "knn": 1e-7, "iter": 1e-4, "softimpute": 1e-6}
TOL_DT = {"mean": 1e-4, "median": 1e-4, "most_frequent": 1e-4, "knn": 1e-4, "iter": 1e-3, "softimpute": 1e-3}


def tolerance(pid, fam):
    if pid in ("NFO", "IV"):
        return 0.0
    if pid == "DT":
        return TOL_DT[fam]
    return TOL_FAMILY[fam]


# ----------------------------------------------------------------------------
# Execution wrapper (memoised calls, exception capture, output log)
# ----------------------------------------------------------------------------
@dataclass
class ExcRecord:
    name: str
    msg: str = ""


class ImplError(Exception):
    def __init__(self, rec):
        super().__init__(rec.name)
        self.rec = rec


class Exec:
    def __init__(self, run, timeout=None):
        self.run, self.timeout, self.outs = run, timeout, {}

    def call(self, key, X, **kw):
        if key not in self.outs:
            try:
                with mm.time_limit(self.timeout), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.outs[key] = np.asarray(self.run(np.array(X, copy=True), **kw))
            except Exception as exc:  # the implementation under test failed
                self.outs[key] = ExcRecord(type(exc).__name__, str(exc)[:160])
        res = self.outs[key]
        if isinstance(res, ExcRecord):
            raise ImplError(res)
        return res


def derangement(n, rng):
    """Random permutation with no fixed point (so *every* row/feature really moves)."""
    if n < 2:
        return np.arange(n)
    while True:
        perm = rng.permutation(n)
        if not (perm == np.arange(n)).any():
            return perm


# ----------------------------------------------------------------------------
# Properties.  Each returns a deviation (0 = holds exactly); the property is violated iff dev > tolerance.
# ----------------------------------------------------------------------------
def _base(ex, c):
    return ex.call("base", c.Xm)


def _fill_complete(Xm):
    with np.errstate(invalid="ignore"):
        means = np.nanmean(np.where(np.isfinite(Xm), Xm, np.nan), axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    return np.where(np.isnan(Xm), means[None, :], Xm)


def p_CI(ex, c, rng, fam):
    complete = _fill_complete(c.Xm)
    return scaled_dev(ex.call("ci", complete), complete)


def p_OVP(ex, c, rng, fam):
    out, obs = _base(ex, c), np.isfinite(c.Xm)
    return scaled_dev(out, c.Xm, scale=col_scale(c.Xm), cells=obs)


def p_NFO(ex, c, rng, fam):
    out = _base(ex, c)
    return 0.0 if (out.shape == c.Xm.shape and np.isfinite(out).all()) else float("inf")


def p_RNG(ex, c, rng, fam):
    out, miss = _base(ex, c), np.isnan(c.Xm)
    if out.shape != c.Xm.shape:
        return float("inf")
    worst, sc = 0.0, col_scale(c.Xm)
    for j in range(c.Xm.shape[1]):
        obs = c.Xm[~miss[:, j], j]
        if obs.size == 0 or not miss[:, j].any():
            continue
        v = out[miss[:, j], j]
        excess = np.maximum(obs.min() - v, v - obs.max()).max()
        if np.isnan(excess):
            return float("inf")
        worst = max(worst, max(float(excess), 0.0) / sc[j])
    return worst


def _closed_form(col_obs, fam):
    if col_obs.size == 0:
        return 0.0
    if fam == "mean":
        return float(np.mean(col_obs))
    if fam == "median":
        return float(np.median(col_obs))
    vals, counts = np.unique(col_obs, return_counts=True)
    return float(vals[np.argmax(counts)])


def p_StatRep(ex, c, rng, fam):
    r = int(rng.integers(c.Xm.shape[0]))
    aug = np.vstack([c.Xm, c.Xm[[r]]])
    out = ex.call("statrep", aug)
    miss = np.isnan(aug)
    expected = np.array([_closed_form(aug[~miss[:, j], j], fam) for j in range(aug.shape[1])])
    return scaled_dev(out, np.broadcast_to(expected, aug.shape), scale=col_scale(aug), cells=miss)


def p_RP(ex, c, rng, fam):
    perm = derangement(c.Xm.shape[0], rng)
    out = ex.call("rp", c.Xm[perm])
    return scaled_dev(out[np.argsort(perm)], _base(ex, c))


def p_FP(ex, c, rng, fam):
    perm = derangement(c.Xm.shape[1], rng)
    out = ex.call("fp", c.Xm[:, perm])
    return scaled_dev(out[:, np.argsort(perm)], _base(ex, c))


def p_DR(ex, c, rng, fam):
    rows = np.flatnonzero(np.isnan(c.Xm).any(axis=1))
    r = int(rng.choice(rows)) if rows.size else 0
    out = ex.call("dr", np.vstack([c.Xm, c.Xm[[r]]]))
    return scaled_dev(out[[r]], out[[-1]], scale=col_scale(out))


def p_SE(ex, c, rng, fam):
    k = 10.0 ** rng.uniform(-3, 3)
    base = _base(ex, c)
    return scaled_dev(ex.call("se", c.Xm * k), base * k, scale=col_scale(base) * k)


def p_TE(ex, c, rng, fam):
    shift = rng.uniform(-3, 3, size=c.Xm.shape[1]) * col_scale(c.Xm)
    base = _base(ex, c)
    has_obs = np.broadcast_to(np.isfinite(c.Xm).any(axis=0)[None, :], c.Xm.shape)   # empty columns are filled with a constant
    return scaled_dev(ex.call("te", c.Xm + shift[None, :]), base + shift[None, :], scale=col_scale(base), cells=has_obs)


def p_PFS(ex, c, rng, fam):
    s = 10.0 ** rng.uniform(-2, 2, size=c.Xm.shape[1])
    base = _base(ex, c)
    return scaled_dev(ex.call("pfs", c.Xm * s[None, :]), base * s[None, :], scale=col_scale(base) * s)


def p_CS(ex, c, rng, fam):
    p = c.Xm.shape[1]
    sub = np.sort(rng.choice(p, size=max(1, p // 2), replace=False))
    base = _base(ex, c)
    return scaled_dev(ex.call("cs", c.Xm[:, sub]), base[:, sub])


def p_MV(ex, c, rng, fam):
    finite = np.abs(c.Xm[np.isfinite(c.Xm)])
    sentinel = -(float(finite.max()) * 10.0 + 12345.678) if finite.size else -12345.678
    out = ex.call("mv", np.where(np.isnan(c.Xm), sentinel, c.Xm), missing_values=sentinel)
    return scaled_dev(out, _base(ex, c))


def p_DT(ex, c, rng, fam):
    out = ex.call("dt", c.Xm.astype(np.float32))
    if out.dtype != np.float32:
        return float("inf")
    return scaled_dev(out.astype(np.float64), _base(ex, c))


def p_IV(ex, c, rng, fam):
    obs = np.argwhere(np.isfinite(c.Xm))
    i, j = obs[int(rng.integers(len(obs)))]
    bad = c.Xm.copy()
    bad[i, j] = np.inf
    try:
        ex.call("iv", bad)
    except ImplError as e:
        return 0.0 if e.rec.name == "ValueError" else float("inf")
    return float("inf")


def p_CB(ex, c, rng, fam):
    """Clip bounds: with min_value/max_value set to the observed range, imputed cells must stay inside it."""
    lo = np.where(np.isfinite(c.Xm).any(axis=0), np.nanmin(np.where(np.isfinite(c.Xm), c.Xm, np.inf), axis=0), -np.inf)
    hi = np.where(np.isfinite(c.Xm).any(axis=0), np.nanmax(np.where(np.isfinite(c.Xm), c.Xm, -np.inf), axis=0), np.inf)
    flat = ~(hi > lo)                                   # constant or empty columns: sklearn requires min < max
    lo, hi = np.where(flat, -np.inf, lo), np.where(flat, np.inf, hi)
    out, miss = ex.call("cb", c.Xm, min_value=lo, max_value=hi), np.isnan(c.Xm)
    if out.shape != c.Xm.shape:
        return float("inf")
    sc = col_scale(c.Xm)
    below = np.where(miss & np.isfinite(lo)[None, :], lo[None, :] - out, 0.0)
    above = np.where(miss & np.isfinite(hi)[None, :], out - hi[None, :], 0.0)
    worst = np.nanmax(np.maximum(np.maximum(below, above), 0.0) / sc[None, :]) if miss.any() else 0.0
    return float("inf") if np.isnan(worst) else float(worst)


def p_NN(ex, c, rng, fam):
    """Asking for p-1 nearest features is asking for all of them: the result must equal the default run."""
    return scaled_dev(ex.call("nn", c.Xm, n_nearest_features=c.Xm.shape[1] - 1), _base(ex, c))


ALL_FAMS = {"mean", "median", "most_frequent", "knn", "iter", "softimpute"}
CLOSED = {"mean", "median", "most_frequent"}
PROPERTIES = {  # id: (function, class, applicable families, description)
    "CI":      (p_CI, "invariant", ALL_FAMS, "Complete-input identity"),
    "OVP":     (p_OVP, "invariant", ALL_FAMS, "Observed-value preservation"),
    "NFO":     (p_NFO, "invariant", ALL_FAMS, "Completeness: same shape, no NaN/inf in output"),
    "RNG":     (p_RNG, "invariant", CLOSED | {"knn"}, "Range containment of imputed values"),
    "StatRep": (p_StatRep, "analytic oracle", CLOSED, "Statistic replication under row duplication"),
    "RP":      (p_RP, "metamorphic", ALL_FAMS, "Row permutation equivariance (derangement)"),
    "FP":      (p_FP, "metamorphic", CLOSED | {"knn", "softimpute"}, "Feature permutation equivariance (derangement)"),
    "DR":      (p_DR, "symmetry", ALL_FAMS, "Duplicate-row symmetry"),
    "SE":      (p_SE, "metamorphic", CLOSED | {"knn", "softimpute"}, "Global scale equivariance (c in [1e-3, 1e3])"),
    "TE":      (p_TE, "metamorphic", CLOSED | {"knn"}, "Per-feature translation equivariance"),
    "PFS":     (p_PFS, "metamorphic", CLOSED, "Per-feature scale equivariance"),
    "CS":      (p_CS, "metamorphic", CLOSED, "Column-subset independence"),
    "MV":      (p_MV, "metamorphic", CLOSED | {"knn", "iter"}, "Missing-marker equivalence (NaN vs sentinel)"),
    "DT":      (p_DT, "metamorphic", CLOSED | {"knn", "iter"}, "float32/float64 consistency + dtype preservation"),
    "IV":      (p_IV, "robustness", ALL_FAMS, "Rejects infinite input with ValueError"),
    "CB":      (p_CB, "invariant", {"iter"}, "Clip bounds (min_value/max_value) are respected"),
    "NN":      (p_NN, "metamorphic", {"iter"}, "n_nearest_features=p-1 equals using all other features"),
}
PROP_IDS = list(PROPERTIES)
PROP_INDEX = {p: i for i, p in enumerate(PROP_IDS)}
PAPER7 = ["CI", "OVP", "RP", "FP", "DR", "SE", "StatRep"]   # the property suite of the original study


def props_for(family):
    return [p for p in PROP_IDS if family in PROPERTIES[p][2]]


# ----------------------------------------------------------------------------
# Evaluation of one (subject, implementation, condition)
# ----------------------------------------------------------------------------
def eval_subject_condition(subject, run, cond, timeout=None, phase="all"):
    """phase: 'main' = every property except IV, 'iv' = only the infinite-input check, 'all' = both.

    IV is isolated because feeding inf to a faulty implementation can hang native LAPACK code."""
    ex = Exec(run, timeout=timeout)
    devs, errs = {}, {}
    for pid in props_for(subject.family):
        if phase == "main" and pid == "IV":
            continue
        if phase == "iv" and pid != "IV":
            continue
        rng = np.random.default_rng([cond.seed, PROP_INDEX[pid]])
        try:
            devs[pid], errs[pid] = float(PROPERTIES[pid][0](ex, cond, rng, subject.family)), ""
        except ImplError as e:
            devs[pid], errs[pid] = float("nan"), e.rec.name
        except Exception as e:  # a bug in the property itself must not be silent
            devs[pid], errs[pid] = float("nan"), "PROPERTY_BUG:" + type(e).__name__ + ":" + str(e)[:80]
    return devs, errs, ex.outs


def compare_outputs(mut_outs, ref_outs, tight=1e-12):
    """Activation analysis: did the mutant change *any* observable output?"""
    differs, base_mag, any_mag, base_exc, n_exc = False, 0.0, 0.0, "", 0
    for key, r in ref_outs.items():
        m = mut_outs.get(key)
        if m is None:
            continue
        r_exc, m_exc = isinstance(r, ExcRecord), isinstance(m, ExcRecord)
        if r_exc or m_exc:
            same = r_exc and m_exc and r.name == m.name
            if not same:
                differs, n_exc = True, n_exc + int(m_exc and not r_exc)
                if key == "base":
                    base_mag, base_exc = float("inf"), (m.name if m_exc else "")
            continue
        mag = scaled_dev(m, r)
        d = mag > tight or m.dtype != r.dtype
        if d:
            differs = True
            if np.isfinite(mag):
                any_mag = max(any_mag, mag)
        if key == "base":
            base_mag = mag if d else 0.0
    return differs, base_mag, any_mag, base_exc, n_exc


# ----------------------------------------------------------------------------
# Randomised experimental conditions
# ----------------------------------------------------------------------------
@dataclass
class Condition:
    cid: int
    seed: int
    Xm: np.ndarray
    X_true: np.ndarray
    meta: dict = field(default_factory=dict)


def _gen_gauss(rng, n, p):
    A = rng.normal(size=(p, p))
    return rng.multivariate_normal(np.zeros(p), A @ A.T + 0.5 * np.eye(p), size=n)


SYNTHETIC = {
    "syn_gaussian": _gen_gauss,
    "syn_student_t": lambda rng, n, p: _gen_gauss(rng, n, p) / np.sqrt(rng.chisquare(3, size=(n, 1)) / 3),
    "syn_lognormal": lambda rng, n, p: np.exp(0.7 * _gen_gauss(rng, n, p)),
    "syn_discrete": lambda rng, n, p: np.clip(np.round(1.2 * _gen_gauss(rng, n, p) + 3), 0, None),
    "syn_lowrank": lambda rng, n, p: rng.normal(size=(n, 2)) @ rng.normal(size=(2, p)) + 0.1 * rng.normal(size=(n, p)),
    "syn_clusters": lambda rng, n, p: rng.normal(size=(n, p)) + 4.0 * rng.normal(size=(3, p))[rng.integers(3, size=n)],
}


def load_pool(csv_paths=None):
    """Real data pool: bundled scikit-learn datasets plus any user-supplied CSV files."""
    import pandas as pd
    from sklearn import datasets as skd
    pool = {
        "real_diabetes": skd.load_diabetes().data,
        "real_wine": skd.load_wine().data,
        "real_breast_cancer": skd.load_breast_cancer().data,
        "real_digits": skd.load_digits().data,
        "real_iris": skd.load_iris().data,
    }
    try:
        pool["real_california"] = skd.fetch_california_housing(download_if_missing=False).data
    except Exception:
        pass
    drop = {"appliances", "lights", "rv1", "rv2", "median_house_value"}     # targets / random diagnostic columns
    for name, path in (csv_paths or {}).items():
        try:
            df = pd.read_csv(path).select_dtypes(include=[np.number])
            df = df[[col for col in df.columns if col.strip().lower() not in drop]]
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            pool[f"csv_{name}"] = df.to_numpy(dtype=float)
        except Exception as exc:
            print(f"[pool] skipping {name}: {exc}")
    return {k: np.asarray(v, dtype=float) for k, v in pool.items()}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def _calibrate(z, rate):
    lo, hi = -40.0, 40.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _sigmoid(z + mid).mean() < rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mask_cells(X, mech, rate, rng):
    """MCAR / MAR (driven by always-observed columns) / MNAR (self-masking) missingness."""
    n, p = X.shape
    sd = X.std(axis=0)
    Z = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    M = np.zeros((n, p), dtype=bool)
    if mech == "MCAR":
        return rng.random((n, p)) < rate
    drivers = rng.choice(p, size=max(1, p // 4), replace=False) if mech == "MAR" else np.array([], dtype=int)
    for j in range(p):
        if mech == "MAR":
            if j in drivers:
                continue
            z = Z[:, drivers].mean(axis=1)
            z = (z - z.mean()) / (z.std() + 1e-12)
        else:
            z = Z[:, j]
        slope = rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 3.0)
        prob = _sigmoid(slope * z + _calibrate(slope * z, rate))
        M[:, j] = rng.random(n) < prob
    return M


def _lhs(n, d, rng):
    u = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        u[:, j] = rng.permutation(u[:, j])
    return u


def make_conditions(n_cond, seed, pool, n_min=30, n_max=200, p_min=3, p_max=8, rate_range=(0.05, 0.55)):
    master = np.random.default_rng(seed)
    sources = list(pool) + list(SYNTHETIC)
    source_cycle = [sources[i % len(sources)] for i in master.permutation(max(n_cond, len(sources)) )]
    mech_cycle = []
    while len(mech_cycle) < n_cond:
        mech_cycle += list(master.permutation(["MCAR", "MAR", "MNAR"]))
    lhs = _lhs(n_cond, 3, master)
    seeds = np.random.SeedSequence(seed).spawn(n_cond)
    conds = []
    for i in range(n_cond):
        rng = np.random.default_rng(seeds[i])
        csd = int(seeds[i].generate_state(1, dtype=np.uint64)[0] % (2 ** 62))
        src, mech = source_cycle[i], mech_cycle[i]
        rate = rate_range[0] + lhs[i, 0] * (rate_range[1] - rate_range[0])
        n = int(round(np.exp(np.log(n_min) + lhs[i, 1] * (np.log(n_max) - np.log(n_min)))))
        p = int(p_min + np.floor(lhs[i, 2] * (p_max - p_min + 1)))
        if src in pool:
            D = pool[src]
            rows = rng.choice(D.shape[0], size=min(n, D.shape[0]), replace=False)
            cols = rng.choice(D.shape[1], size=min(p, D.shape[1]), replace=False)
            X = D[np.ix_(rows, cols)].copy()
        else:
            X = SYNTHETIC[src](rng, n, p)
        n, p = X.shape
        flags = {}
        if rng.random() < 0.30:      # coarse granularity -> exact ties between values and distances
            sd = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
            step = sd * rng.choice([0.5, 0.25, 0.1])
            X = np.round(X / step) * step
            flags["coarse"] = True
        if rng.random() < 0.40:      # heterogeneous column scales and large offsets
            s = 10.0 ** rng.uniform(-2, 3, size=p)
            X = X * s + rng.normal(size=p) * s * rng.uniform(0, 10)
            flags["rescaled"] = True
        if rng.random() < 0.20 and n >= 10:    # duplicated records
            k = max(2, int(0.1 * n))
            X[rng.choice(n, size=k, replace=False)] = X[rng.choice(n, size=k, replace=True)]
            flags["dup_rows"] = True
        if rng.random() < 0.12:      # constant column
            j = int(rng.integers(p))
            X[:, j] = X[0, j]
            flags["const_col"] = True
        M = mask_cells(X, mech, rate, rng)
        empty_col = rng.random() < 0.10
        empty_row = rng.random() < 0.08
        for j in range(p):           # keep >=2 observed per column (except a deliberately empty one)
            if (~M[:, j]).sum() < 2:
                M[rng.choice(n, size=2, replace=False), j] = False
        if not empty_row:
            for r in np.flatnonzero(M.all(axis=1)):
                M[r, rng.integers(p)] = False
        if empty_col:
            M[:, int(rng.integers(p))] = True
            flags["empty_col"] = True
        if empty_row:
            M[int(rng.integers(n)), :] = True
            flags["empty_row"] = True
        if not M.any():
            M[0, 0] = True
        Xm = np.where(M, np.nan, X)
        nun = np.mean([len(np.unique(X[:, j])) / n for j in range(p)])
        cs = col_scale(X)
        meta = dict(cid=i, source=src, mechanism=mech, rate_target=float(rate), rate_eff=float(M.mean()),
                    n=n, p=p, tie_frac=float(1 - nun), scale_spread=float(np.log10(cs.max() / cs.min())),
                    **{f"flag_{k}": v for k, v in flags.items()})
        for k in ("coarse", "rescaled", "dup_rows", "const_col", "empty_col", "empty_row"):
            meta.setdefault(f"flag_{k}", False)
        conds.append(Condition(cid=i, seed=csd, Xm=Xm, X_true=X, meta=meta))
    return conds


# ----------------------------------------------------------------------------
# Parallel tasks
# ----------------------------------------------------------------------------
def _trace_collector(files):
    store = {f: set() for f in files}

    def tracer(frame, event, arg):
        fn = frame.f_code.co_filename
        if fn in store:
            lines = store[fn]
            lines.add(frame.f_lineno)

            def local(fr, ev, a):
                if ev == "line":
                    lines.add(fr.f_lineno)
                return local
            return local
        return None
    return tracer, store


def reference_task(subject_ids, conds, trace_files=()):
    """Run the unmutated implementations; optionally record executed source lines."""
    import time as _time
    rows, t_start = [], _time.time()
    tracer, store = _trace_collector(set(trace_files))
    for sid in subject_ids:
        subj = SUBJECTS[sid]
        run = make_runner(subj, None)
        for c in conds:
            if trace_files:
                sys.settrace(tracer)
            try:
                devs, errs, _ = eval_subject_condition(subj, run, c)
            finally:
                sys.settrace(None)
            for pid in devs:
                rows.append((sid, c.cid, pid, devs[pid], errs[pid]))
    return rows, {k: sorted(v) for k, v in store.items()}, _time.time() - t_start


_REF_CACHE = {}
_SRC_CACHE = {}


def _ref_outputs(subj, c, timeout, phase):
    key = (subj.sid, c.cid, phase)
    if key not in _REF_CACHE:
        _REF_CACHE[key] = eval_subject_condition(subj, make_runner(subj, None), c, timeout, phase)
    return _REF_CACHE[key]


def mutant_task(mu, sid, phase, conds, timeout=20.0, dev_cap=100.0):
    """Evaluate ONE mutant on ONE subject (phase 'main' or 'iv') over all conditions."""
    prop_rows, exec_rows = [], []
    t_cfg = TARGETS[mu["target"]]
    mod_name = t_cfg["module"]
    if mod_name not in _SRC_CACHE:
        _SRC_CACHE[mod_name] = mm.module_source(mod_name)
    src, path = _SRC_CACHE[mod_name]
    try:
        code = mm.build_mutant_code(src, mm.Mutant(**mu), path)
        mod = mm.load_module_copy(mod_name, code, path)
    except Exception as exc:
        return [], [], ("STILLBORN", type(exc).__name__ + ": " + str(exc)[:100])
    subj = SUBJECTS[sid]
    run = make_runner(subj, mod)
    for c in conds:
        rdevs, rerrs, routs = _ref_outputs(subj, c, timeout, phase)
        devs, errs, outs = eval_subject_condition(subj, run, c, timeout, phase)
        differs, bmag, amag, bexc, nexc = compare_outputs(outs, routs)
        exec_rows.append((mu["mutant_id"], sid, c.cid, phase, differs, bmag, amag, bexc, nexc))
        for pid, rd in rdevs.items():
            if rerrs[pid] == "" and rd <= dev_cap * max(tolerance(pid, subj.family), 1e-12):
                prop_rows.append((mu["mutant_id"], sid, c.cid, pid, rd, devs[pid], errs[pid]))
    return prop_rows, exec_rows, ("OK", "")


# ----------------------------------------------------------------------------
# Supervised process pool: a mutated library can hang inside native code (e.g. LAPACK given inf),
# which no in-process timeout can interrupt, so every task has a hard wall-clock limit.
# ----------------------------------------------------------------------------
def _worker_loop(task_q, result_q, conds, soft_timeout):
    while True:
        task = task_q.get()
        if task is None:
            return
        try:
            res = mutant_task(task["mu"], task["sid"], task["phase"], conds, soft_timeout)
            result_q.put((task["tid"], "ok", res))
        except BaseException as exc:  # noqa: BLE001
            result_q.put((task["tid"], "error", f"{type(exc).__name__}: {exc}"[:200]))


def run_supervised(tasks, conds, n_workers=2, soft_timeout=20.0, verbose=True, label=""):
    import multiprocessing as mp
    import queue
    import time as _time
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()

    def start():
        q = ctx.Queue()
        p = ctx.Process(target=_worker_loop, args=(q, result_q, conds, soft_timeout), daemon=True)
        p.start()
        return {"q": q, "p": p, "task": None, "t0": 0.0}

    pending, out = list(tasks)[::-1], {}
    total, t_start, last = len(tasks), _time.time(), _time.time()
    workers = [start() for _ in range(max(1, n_workers))]
    try:
        while len(out) < total:
            for w in workers:
                if w["task"] is None and pending:
                    w["task"], w["t0"] = pending.pop(), _time.time()
                    w["q"].put(w["task"])
            try:
                tid, kind, payload = result_q.get(timeout=0.5)
                out[tid] = (kind, payload)
                for w in workers:
                    if w["task"] is not None and w["task"]["tid"] == tid:
                        w["task"] = None
            except queue.Empty:
                pass
            now = _time.time()
            for i, w in enumerate(workers):
                t = w["task"]
                if t is None:
                    continue
                if now - w["t0"] > t["hard"]:
                    w["p"].terminate(); w["p"].join(5)
                    out[t["tid"]] = ("timeout", None); workers[i] = start()
                elif not w["p"].is_alive():
                    out[t["tid"]] = ("died", None); workers[i] = start()
            if verbose and now - last > 30:
                print(f"  [{label}] {len(out)}/{total} tasks | {(now - t_start) / 60:.1f} min", flush=True)
                last = now
    finally:
        for w in workers:
            try:
                w["q"].put(None)
            except Exception:
                pass
        for w in workers:
            w["p"].join(2)
            if w["p"].is_alive():
                w["p"].terminate()
    return out
