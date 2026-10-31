"""Post-processing: mutant classification, kill matrices, minimal property sets, bootstrap statistics."""
from __future__ import annotations

import itertools
import re

import numpy as np
import pandas as pd

import mt_core as mc

FAMILY_OF = {sid: s.family for sid, s in mc.SUBJECTS.items()}
ANALYSED = ["KILLED_PROP", "KILLED_CRASH", "SURVIVED"]          # non-trivial, behaviour-changing, covered
CLASS_ORDER = ["NO_COVERAGE", "NOT_SAMPLED", "STILLBORN", "TIMEOUT", "EQUIV_CANDIDATE", "CRASH_ALL",
               "KILLED_PROP", "KILLED_CRASH", "SURVIVED"]

_TOL_CACHE = {}


def tol_series(pid, sid):
    keys = pid.astype(str) + "|" + sid.astype(str)
    out = {}
    for k in keys.unique():
        p, s = k.split("|", 1)
        out[k] = mc.tolerance(p, FAMILY_OF[s])
    return keys.map(out).to_numpy(float)


# ----------------------------------------------------------------------------
def subtle_family(cat, stmt, operator, target):
    """Heuristic fault-family label for the subtle fault classes discussed in the paper."""
    t = (stmt or "").lower()
    if cat == "dtype" or re.search(r"dtype|float32|float64|astype|float_dtypes", t):
        return "dtype"
    if cat == "nan" or re.search(r"nan|mask|missing|finite|isinf", t):
        return "NaN / mask"
    if cat == "tolerance" or re.search(r"\btol\b|\.tol|eps|norm|thresh|clip|min_value|max_value|inf_norm", t):
        return "tolerance / numeric"
    if target in ("KNNImputer", "nan_euclidean") and re.search(
            r"argpartition|n_neighbors|donor|neighbor|dist|weight|squared|present_count|sort", t):
        return "tie / neighbour selection"
    return "other logic"


def classify(mut_df, P, E, status, tol_mult=1.0):
    """One row per mutant with its outcome class under tolerance multiplier `tol_mult`."""
    thr = P["tol"].to_numpy() * tol_mult
    elig = P["ref_dev"].to_numpy() <= thr
    err = P["mut_err"].to_numpy() != ""
    viol = elig & ~err & (P["mut_dev"].to_numpy() > thr)
    crash = elig & err
    pm = pd.DataFrame({"mutant_id": P["mutant_id"].to_numpy(), "elig": elig, "viol": viol, "crash": crash})
    pg = pm.groupby("mutant_id").agg(n_elig=("elig", "sum"), killed_prop=("viol", "any"), killed_crash=("crash", "any"))
    em = E[E["phase"] == "main"]
    eg = E.groupby("mutant_id").agg(differs_any=("differs", "any"), n_exec=("differs", "size"))
    cg = em.assign(base_crash=em["base_exc"] != "").groupby("mutant_id").agg(crash_all=("base_crash", "all"))
    st = status.sort_values("status").drop_duplicates("mutant_id", keep="last")   # STILLBORN/TIMEOUT beat OK
    out = (mut_df.set_index("mutant_id").join(pg).join(eg).join(cg).join(st.set_index("mutant_id")[["status"]]))
    for col in ("killed_prop", "killed_crash", "differs_any", "crash_all"):
        out[col] = out[col].fillna(False).astype(bool)
    out["n_elig"] = out["n_elig"].fillna(0).astype(int)

    def cls(r):
        if not r["covered"]:
            return "NO_COVERAGE"
        if not r["selected"]:
            return "NOT_SAMPLED"
        if r["status"] == "STILLBORN":
            return "STILLBORN"
        if r["status"] == "TIMEOUT" and not r["killed_prop"]:
            return "TIMEOUT"
        if not r["differs_any"]:
            return "EQUIV_CANDIDATE"
        if r["crash_all"]:
            return "CRASH_ALL"
        if r["killed_prop"]:
            return "KILLED_PROP"
        if r["killed_crash"]:
            return "KILLED_CRASH"
        return "SURVIVED"
    out["class"] = out.apply(cls, axis=1)
    out["subtle_family"] = [subtle_family(c, s, o, t) for c, s, o, t in
                            zip(out["category"], out["stmt_text"], out["operator"], out["target"])]
    return out.reset_index()


def mutation_score(cls_df, include_crash=False):
    a = cls_df[cls_df["class"].isin(ANALYSED)]
    if a.empty:
        return np.nan
    hit = a["class"].eq("KILLED_PROP") | (a["class"].eq("KILLED_CRASH") if include_crash else False)
    return float(hit.mean())


def boot_ci(values, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of the mean; resampling unit = mutant."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def score_table(cls_df, by, n_boot=2000, include_crash=False):
    rows = []
    for key, g in cls_df.groupby(by):
        a = g[g["class"].isin(ANALYSED)]
        hit = a["class"].eq("KILLED_PROP") | (a["class"].eq("KILLED_CRASH") if include_crash else False)
        m, lo, hi = boot_ci(hit.astype(float), n_boot)
        rows.append({**(dict(zip(by, key if isinstance(key, tuple) else (key,)))),
                     "mutants": len(g), "analysed": len(a), "killed_by_property": int(a["class"].eq("KILLED_PROP").sum()),
                     "killed_by_crash_only": int(a["class"].eq("KILLED_CRASH").sum()),
                     "survived": int(a["class"].eq("SURVIVED").sum()),
                     "score": m, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
def kill_matrix(P, mutant_ids, tol_mult=1.0, by_subject=False):
    thr = P["tol"].to_numpy() * tol_mult
    elig = P["ref_dev"].to_numpy() <= thr
    viol = elig & (P["mut_err"].to_numpy() == "") & (P["mut_dev"].to_numpy() > thr)
    V = P.loc[viol, ["mutant_id", "pid"]].drop_duplicates()
    V["hit"] = True
    K = V.pivot(index="mutant_id", columns="pid", values="hit").reindex(index=list(mutant_ids), columns=mc.PROP_IDS)
    return K.fillna(False).astype(bool)


def property_contributions(K):
    M = K.to_numpy()
    full = int(M.any(axis=1).sum())
    rows = []
    for j, p in enumerate(K.columns):
        others = np.delete(M, j, axis=1)
        rows.append({"property": p, "class": mc.PROPERTIES[p][1], "kills": int(M[:, j].sum()),
                     "unique_kills": int((M[:, j] & ~others.any(axis=1)).sum()),
                     "lost_if_removed": full - int(others.any(axis=1).sum())})
    return pd.DataFrame(rows).sort_values(["unique_kills", "kills"], ascending=False).reset_index(drop=True)


def minimal_covers(K, limit=20):
    """Exact search: smallest property subsets that kill every mutant the full suite kills."""
    M = K.to_numpy()
    props = list(K.columns)
    live = [j for j in range(len(props)) if M[:, j].any()]
    full = int(M.any(axis=1).sum())
    for k in range(1, len(live) + 1):
        sols = [c for c in itertools.combinations(live, k) if int(M[:, list(c)].any(axis=1).sum()) == full]
        if sols:
            return k, [[props[i] for i in c] for c in sols[:limit]], len(sols)
    return 0, [], 0


def greedy_order(K):
    M, props = K.to_numpy(), list(K.columns)
    covered, chosen, out = np.zeros(M.shape[0], bool), [], []
    while True:
        gains = [(~covered & M[:, j]).sum() if props[j] not in chosen else -1 for j in range(len(props))]
        j = int(np.argmax(gains))
        if gains[j] <= 0:
            break
        covered |= M[:, j]
        chosen.append(props[j])
        out.append((props[j], int(covered.sum())))
    return out


def suite_score(K, suite):
    cols = [c for c in suite if c in K.columns]
    return float(K[cols].any(axis=1).mean()) if len(K) and cols else np.nan


# ----------------------------------------------------------------------------
def detection_probability(P, E, analysed_ids, tol_mult=1.0, min_activated=3):
    """P(property violation | fault activated) per (mutant, subject), over randomised conditions."""
    thr = P["tol"].to_numpy() * tol_mult
    elig = P["ref_dev"].to_numpy() <= thr
    viol = elig & (P["mut_err"].to_numpy() == "") & (P["mut_dev"].to_numpy() > thr)
    kc = (P.loc[viol, ["mutant_id", "sid", "cid"]].drop_duplicates().assign(killed=True)
          .set_index(["mutant_id", "sid", "cid"]))
    X = E[E["mutant_id"].isin(analysed_ids) & E["differs"] & (E["base_exc"] == "")].set_index(["mutant_id", "sid", "cid"])
    X = X.join(kc, how="left")
    X["killed"] = X["killed"].fillna(False).astype(bool)
    return X.reset_index()


def budget_curve(X, analysed_ids, cids, n_perm=200, seed=0):
    """Mutation score as a function of the number of randomised conditions used (random orders)."""
    ids = list(analysed_ids)
    hit = X[X["killed"]].groupby(["mutant_id", "cid"]).size().unstack(fill_value=0)
    D = hit.reindex(index=ids, columns=list(cids)).fillna(0).to_numpy() > 0
    rng = np.random.default_rng(seed)
    curves = np.empty((n_perm, len(cids)))
    for i in range(n_perm):
        reached = np.logical_or.accumulate(D[:, rng.permutation(len(cids))], axis=1)
        curves[i] = reached.mean(axis=0)
    return curves


def _hit_matrix(X, analysed_ids, cids):
    hit = X[X["killed"]].groupby(["mutant_id", "cid"]).size().unstack(fill_value=0)
    return hit.reindex(index=list(analysed_ids), columns=list(cids)).fillna(0).to_numpy() > 0


def condition_bootstrap(X, analysed_ids, cids, n_boot=500, seed=0):
    """Variability of the mutation score caused by *which* randomised conditions were drawn."""
    D = _hit_matrix(X, analysed_ids, cids)
    rng = np.random.default_rng(seed)
    scores = np.array([D[:, rng.integers(0, D.shape[1], D.shape[1])].any(axis=1).mean() for _ in range(n_boot)])
    return D.any(axis=1).mean(), scores


def split_half(X, analysed_ids, cids, n_splits=200, seed=0):
    """Reliability: do two disjoint random halves of the conditions agree on how detectable each mutant is?"""
    from scipy.stats import spearmanr
    ids, cids = list(analysed_ids), list(cids)
    act = X.assign(one=1).groupby(["mutant_id", "cid"]).agg(act=("one", "size"), kill=("killed", "any")).reset_index()
    A = act.pivot(index="mutant_id", columns="cid", values="act").reindex(index=ids, columns=cids).fillna(0).to_numpy()
    K = act.pivot(index="mutant_id", columns="cid", values="kill").reindex(index=ids, columns=cids).fillna(0).to_numpy().astype(float)
    rng, rho, dscore = np.random.default_rng(seed), [], []
    for _ in range(n_splits):
        perm = rng.permutation(len(cids)); h1, h2 = perm[: len(cids) // 2], perm[len(cids) // 2:]
        p1 = K[:, h1].sum(1) / np.maximum(A[:, h1].sum(1), 1); p2 = K[:, h2].sum(1) / np.maximum(A[:, h2].sum(1), 1)
        ok = (A[:, h1].sum(1) > 0) & (A[:, h2].sum(1) > 0)
        if ok.sum() > 3 and p1[ok].std() > 0 and p2[ok].std() > 0:
            rho.append(spearmanr(p1[ok], p2[ok])[0])
        dscore.append(abs((K[:, h1].sum(1) > 0).mean() - (K[:, h2].sum(1) > 0).mean()))
    return np.array(rho), np.array(dscore)
