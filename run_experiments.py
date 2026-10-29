"""
run_experiments.py -- all simulations of the paper.  Results are appended to a JSON-lines
file so that an interrupted run resumes where it stopped.

  python3 run_experiments.py <data_dir> <out.jsonl> <experiment> [dataset]

experiments
  main       openness sweep, U = 1, 3, 5, 7 unseen families, 5 random draws per level, all methods
  ablation   component ablation of the proposed head and loss (U = 3)
  imbalance  attack classes of the training pool thinned by 1, 1/4, 1/16, 1/64 (2 draws)
  timing     parameter count, model size and per-flow CPU latency
"""
import sys, os, json, time, math
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score
from iwcevt import (train_encoder, forward_all, hidden_all, MultiEvidenceEVT, TailModel, Geometry,
                    margin_stat, score_msp, score_energy, TiedMahalanobis, OpenMax, EVM, V1Head, softmax)

EPS = 0.05                      # target false-alarm rate on known traffic
ITERS = 2500
GEO_CAP = 20000                 # flows per class used for the covariance / stage-1 tails
FAR_GRID = np.logspace(-3, 0, 31)


def load(ddir, name):
    d = np.load(os.path.join(ddir, name + ".npz"))
    C = [str(c) for c in d["classes"]]
    if name == "unsw_nb15":
        # the official training/testing files are not exchangeable (documented covariate shift),
        # so the 257,673 flows are pooled and re-split 70/30, stratified by class
        X = np.vstack([d["Xtr"], d["Xte"]]); Y = np.r_[d["ytr"], d["yte"]]
        rng = np.random.default_rng(1); tr, te = [], []
        for c in range(len(C)):
            i = rng.permutation(np.where(Y == c)[0]); n = int(0.7 * len(i))
            tr.append(i[:n]); te.append(i[n:])
        tr, te = np.concatenate(tr), np.concatenate(te)
        return X[tr], Y[tr], X[te], Y[te], C, len(C)
    return d["Xtr"], d["ytr"], d["Xte"], d["yte"], C, int(d["n_trainable"])


def draw_unknown(NT, U, trial):
    rng = np.random.default_rng(1000 + 17 * U + trial)
    return sorted(rng.choice(np.arange(1, NT), size=U, replace=False).tolist())   # class 0 = benign, always known


def openness(K, U):
    return 1.0 - math.sqrt(2.0 * K / (2.0 * K + U))


def cap_per_class(y, cap, rng):
    idx = [rng.permutation(np.where(y == c)[0])[:cap] for c in np.unique(y)]
    return np.sort(np.concatenate(idx))


def metrics(score, reject, isu, yk_true, yk_pred, K, cls_ids):
    """score: larger = more novel (all test flows); reject: bool decision at the operating point;
       yk_true/yk_pred: known-class index (true is -1 for unknown flows)."""
    kn = isu == 0
    out = {"auroc": float(roc_auc_score(isu, score))}
    out["macro_auroc"] = float(np.mean([roc_auc_score(np.r_[np.zeros((yk_true == c).sum()), np.ones((~kn).sum())],
                                                      np.r_[score[yk_true == c], score[~kn]])
                                        for c in range(K) if (yk_true == c).sum() > 0]))
    out["far"] = float(reject[kn].mean()); out["udr"] = float(reject[~kn].mean())
    fc = {int(cls_ids[c]): float(reject[yk_true == c].mean()) for c in range(K) if (yk_true == c).sum() > 0}
    out["far_class"] = fc; out["far_worst"] = float(max(fc.values()))
    yt = np.where(kn, yk_true, K); yp = np.where(reject, K, yk_pred)
    out["f1_open"] = float(f1_score(yt, yp, average="macro", labels=list(range(K + 1)), zero_division=0))
    # ROC on a fixed FAR grid (for the averaged ROC figure)
    sk = np.sort(score[kn]); su = score[~kn]
    thr = sk[np.minimum((np.ceil((1 - FAR_GRID) * len(sk))).astype(int), len(sk) - 1)]
    out["roc"] = [float((su > t).mean()) for t in thr]
    return out


def udr_at_worst_far(score_fn, isu, yk_true, K, target=EPS):
    """detection rate when the threshold is tuned (on the test flows, same oracle for every method)
       so that the worst known class has a false-alarm rate of `target`."""
    return score_fn(target, isu, yk_true, K)


def global_wc(score):
    def f(target, isu, yk, K):
        t = max(np.quantile(score[yk == c], 1 - target) for c in range(K) if (yk == c).sum() > 0)
        return float((score[isu == 1] > t).mean())
    return f


def proposed_wc(head, T):
    def f(target, isu, yk, K):
        best = 0.0
        for e in np.logspace(-3.5, -0.3, 60):
            rej = head.decide(T, e)
            w = max(rej[yk == c].mean() for c in range(K) if (yk == c).sum() > 0)
            if w <= target:
                best = max(best, float(rej[isu == 1].mean()))
        return best
    return f


def run_config(D, U, trial, exp, thin=1.0, log=print):
    Xtr, ytr, Xte, yte, C, NT = D
    unk = draw_unknown(NT, U, trial) if U > 0 else []
    unk_all = unk + list(range(NT, len(C)))
    known = [c for c in range(NT) if c not in unk]
    K = len(known); remap = {c: i for i, c in enumerate(known)}
    rng = np.random.default_rng(trial)
    m = np.isin(ytr, known)
    Xk, yk = Xtr[m], np.array([remap[c] for c in ytr[m]])
    if thin < 1.0:                       # imbalance stress: thin every attack class of the pool
        keep = np.ones(len(yk), bool)
        for c in range(1, K):
            i = np.where(yk == c)[0]; n = max(int(round(thin * len(i))), 30)
            keep[rng.permutation(i)[n:]] = False
        Xk, yk = Xk[keep], yk[keep]
    fit, cal = [], []
    for c in range(K):                   # stratified 80/20 fit / calibration
        i = rng.permutation(np.where(yk == c)[0]); n = max(int(0.2 * len(i)), 3)
        cal.append(i[:n]); fit.append(i[n:])
    fit, cal = np.concatenate(fit), np.concatenate(cal)
    Xf, yf, Xc, yc = Xk[fit], yk[fit], Xk[cal], yk[cal]
    isu = np.isin(yte, unk_all).astype(int); kn = isu == 0
    ykt = np.array([remap.get(int(c), -1) for c in yte])
    U_eff = len([u for u in unk_all if (yte == u).sum() > 0])
    base = dict(exp=exp, U=U, trial=trial, K=K, thin=thin, unknown=[C[u] for u in unk_all],
                openness=openness(K, U_eff), n_fit=int(len(yf)), n_cal=int(len(yc)), n_test=int(len(yte)),
                imb_ratio=float(np.bincount(yf).max() / np.bincount(yf).min()),
                cal_min=int(np.bincount(yc).min()))
    rows = []
    gi = cap_per_class(yf, GEO_CAP, rng)

    def add(name, score, reject, pred, wc_fn, extra=None):
        r = dict(base); r["method"] = name
        if isu.sum() > 0:
            r.update(metrics(score, reject, isu, ykt, pred, K, known))
            r["udr_wc"] = wc_fn(EPS, isu, ykt, K)
        else:
            r["far"] = float(reject.mean())
        r["acc_closed"] = float((pred[kn] == ykt[kn]).mean())
        r["f1_closed"] = float(f1_score(ykt[kn], pred[kn], average="macro", labels=list(range(K)), zero_division=0))
        if extra: r.update(extra)
        rows.append(r)

    def glob(score_cal, score_te):
        return score_te > np.quantile(score_cal, 1 - EPS)

    modes = {"main": ["ce", "iw"], "ablation": ["iw", "supcon"], "imbalance": ["ce", "supcon", "iw"]}[exp]
    if os.environ.get("IW_ONLY"):        # re-run only the proposed head (baseline rows are reused)
        modes = [m for m in modes if m == "iw"]
    for mode in modes:
        t0 = time.time()
        model = train_encoder(Xf, yf, K, mode=mode, iters=ITERS, seed=trial)
        Zf, Lf = forward_all(model, Xf[gi], mode); Zc, Lc = forward_all(model, Xc, mode); Ze, Le = forward_all(model, Xte, mode)
        pred = Le.argmax(1); ygi = yf[gi]
        if mode == "ce":
            add("MSP", score_msp(Le), glob(score_msp(Lc), score_msp(Le)), pred, global_wc(score_msp(Le)))
            add("Energy", score_energy(Le), glob(score_energy(Lc), score_energy(Le)), pred, global_wc(score_energy(Le)))
            if exp == "main":
                om = OpenMax().fit(Lf, ygi, K); so = om.score(Le)[0]
                add("OpenMax", so, glob(om.score(Lc)[0], so), pred, global_wc(so))
                mh = TiedMahalanobis().fit(Zf, ygi, K); sm = mh.score(Ze)
                add("Mahalanobis", sm, glob(mh.score(Zc), sm), pred, global_wc(sm))
                ev = EVM(seed=trial).fit(Zf, ygi, K); se = ev.score(Ze)
                add("EVM", se, glob(ev.score(Zc), se), pred, global_wc(se))
            log(f"   [{mode}] {time.time()-t0:.0f}s")
            continue
        Hf, Hc, He = hidden_all(model, Xf[gi]), hidden_all(model, Xc), hidden_all(model, Xte)
        tag = "IW-C-EVT" if mode == "iw" else "C-EVT (no IW)"
        variants = {"": dict(feats=(0, 1), kw={})}
        if exp == "ablation" and mode == "iw":
            variants.update({
                " w/o margin evidence": dict(feats=(0, 1), kw=dict(use_margin=False)),
                " w/o shallow depth": dict(feats=(0,), kw={}),
                " radius only (z)": dict(feats=(0,), kw=dict(use_margin=False)),
                " w/o shrinkage": dict(feats=(0, 1), kw=dict(shrink=False)),
                " Weibull tail": dict(feats=(0, 1), kw=dict(tail="weibull")),
                " fixed tail size": dict(feats=(0, 1), kw=dict(adaptive=False)),
            })
        if exp == "imbalance" and mode == "iw":
            variants[" w/o shrinkage"] = dict(feats=(0, 1), kw=dict(shrink=False))
        for vn, v in variants.items():
            ft = [[Zf, Hf][i] for i in v["feats"]]; fc = [[Zc, Hc][i] for i in v["feats"]]; fe = [[Ze, He][i] for i in v["feats"]]
            head = MultiEvidenceEVT(**v["kw"]).fit(ft, Lf, ygi, fc, Lc, yc, K)
            s, T, Pv = head.evaluate(fe, Le)
            extra = dict(cond_max=float(max(max(g.cond) for g in head.geo)),
                         gamma_mean=float(np.mean([np.mean(g.gamma) for g in head.geo])),
                         xi=[float(x) for x in head.final.xi])
            add(tag + vn, s, head.decide(T, EPS), pred, proposed_wc(head, T), extra)
            if vn == "" and mode == "iw":
                # same score with one global threshold (what class-conditional calibration buys)
                sc = head.evaluate(fc, Lc)[0]
                add(tag + " global threshold", s, glob(sc, s), pred, global_wc(s))
                if exp == "main":
                    add(tag + " marginal radius", s, head.decide(T, EPS, None), pred, proposed_wc(head, T))
                # pure split-conformal radii (no tail extrapolation): infinite when eps(n_c+1) < 1
                if exp != "main":
                    Tc = head._fisher(fc, Lc); r = np.full(K, np.inf)
                    for c in range(K):
                        d = np.sort(Tc[yc == c, c]); k = math.ceil((1 - EPS) * (len(d) + 1))
                        if k <= len(d): r[c] = d[k - 1]
                    add(tag + " conformal only", s, (T > r[None]).all(1), pred, proposed_wc(head, T),
                        dict(n_inf=int(np.isinf(r).sum())))
                # marginal (delta-free) split-conformal/GPD radius: the Version-3.0 operating point
                add(tag + " marginal radius", s, head.decide(T, EPS, None), pred, proposed_wc(head, T))
        if exp == "ablation" and mode == "iw":
            v1 = V1Head().fit(Zf, ygi, K); s1 = v1.score(Ze, Le)
            add("Version-1 head", s1, glob(v1.score(Zc, Lc), s1), pred, global_wc(s1))
        log(f"   [{mode}] {time.time()-t0:.0f}s")
    return rows


def timing(D, out):
    Xtr, ytr, Xte, yte, C, NT = D
    known = list(range(NT)); K = NT
    rng = np.random.default_rng(0)
    i = rng.permutation(len(ytr)); nc = int(0.2 * len(i)); fit, cal = i[nc:], i[:nc]
    model = train_encoder(Xtr[fit], ytr[fit], K, mode="iw", iters=ITERS, seed=0)
    gi = cap_per_class(ytr[fit], GEO_CAP, rng); Xg, yg = Xtr[fit][gi], ytr[fit][gi]
    Zf, Lf = forward_all(model, Xg, "iw"); Zc, Lc = forward_all(model, Xtr[cal], "iw")
    res = {}
    for name, feats in {"IW-C-EVT": (0, 1), "IW-C-EVT (z only)": (0,)}.items():
        H = [Zf, hidden_all(model, Xg)]; Hc = [Zc, hidden_all(model, Xtr[cal])]
        head = MultiEvidenceEVT().fit([H[j] for j in feats], Lf, yg, [Hc[j] for j in feats], Lc, ytr[cal], K)
        xb = Xte[:20000]
        ts = []
        torch.set_num_threads(1)
        for _ in range(5):
            t0 = time.perf_counter()
            Z, L = forward_all(model, xb, "iw"); F_ = [Z, hidden_all(model, xb)]
            s, T, _ = head.evaluate([F_[j] for j in feats], L); head.decide(T, EPS)
            ts.append((time.perf_counter() - t0) / len(xb))
        n_enc = sum(p.numel() for n_, p in model.named_parameters() if not n_.startswith("lin"))
        n_head = sum(g.mu.size + sum(P.size for P in g.prec) for g in head.geo) + 6 * K * (len(feats) + 2)
        res[name] = dict(us_per_flow=float(np.median(ts) * 1e6), params_encoder=int(n_enc), params_head=int(n_head),
                         kB_float32=float((n_enc + n_head) * 4 / 1024))
    json.dump(res, open(out, "w"), indent=1)
    print(res)


if __name__ == "__main__":
    ddir, out, exp = sys.argv[1], sys.argv[2], sys.argv[3]
    dsets = [sys.argv[4]] if len(sys.argv) > 4 else ["cic_ids2017", "unsw_nb15"]
    torch.set_num_threads(1)
    done = set()
    if os.path.exists(out) and exp != "timing":
        for l in open(out):
            r = json.loads(l); done.add((r["dataset"], r["exp"], r["U"], r["trial"], r["thin"]))
    for ds in dsets:
        D = load(ddir, ds)
        if exp == "timing":
            timing(D, out); continue
        grid = {"main": [(U, t, 1.0) for U in (0, 1, 3, 5, 7) for t in range(5 if U else 1)],
                "ablation": [(3, t, 1.0) for t in range(5)],
                "imbalance": [(3, t, th) for th in (1.0, 0.25, 0.0625, 0.015625) for t in range(2)]}[exp]
        for U, t, th in grid:
            if (ds, exp, U, t, th) in done:
                continue
            t0 = time.time()
            rows = run_config(D, U, t, exp, th)
            with open(out, "a") as f:
                for r in rows:
                    r["dataset"] = ds; f.write(json.dumps(r) + "\n")
            msg = " | ".join(f"{r['method']}: {r.get('auroc', float('nan')):.3f}/{r.get('udr', float('nan')):.2f}/{r.get('far_worst', float('nan')):.2f}" for r in rows)
            print(f"{ds} {exp} U={U} t={t} thin={th} ({time.time()-t0:.0f}s) {rows[0]['unknown']}\n   {msg}", flush=True)
