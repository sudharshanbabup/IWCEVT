"""
iwcevt.py -- Imbalance-Weighted Contrastive Extreme Value architecture (IW-C-EVT)
and the open-set baselines it is compared against.  PyTorch (CPU) + NumPy/SciPy.

Proposed pipeline
  1. encoder trained with the imbalance-weighted prototype-anchored supervised
     contrastive loss  (+ open-space reservation with cross-class mixtures)
  2. per-class Ledoit-Wolf shrinkage covariance, blended with the pooled
     within-class covariance in proportion to the class size
  3. peaks-over-threshold GPD tail on the class-conditional Mahalanobis radii,
     tail size adapted to the class size
  4. class-conditional rejection radius: conformal order statistic when the
     class is large enough, GPD tail quantile when it is not
"""
import math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import genpareto, weibull_min, norm
from scipy.stats import beta as beta_dist
from sklearn.covariance import ledoit_wolf

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))


# ----------------------------------------------------------------------------
# network
# ----------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, d, K, p=32, h=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(),
                                  nn.Linear(h, h // 2), nn.BatchNorm1d(h // 2), nn.ReLU(),
                                  nn.Linear(h // 2, p))
        self.proto = nn.Parameter(torch.randn(K, p) * 0.1)   # class prototypes (cosine head)
        self.lin = nn.Linear(p, K)                            # linear head (CE baselines)

    def embed(self, x, normalize=True):
        z = self.body(x)
        return F.normalize(z, dim=1) if normalize else z


def effective_number_weights(counts, beta=0.999):
    """class-balanced weights  w_c = (1-beta)/(1-beta^{N_c}),  normalised to mean 1"""
    counts = np.asarray(counts, dtype=np.float64)
    w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    return w / w.mean()


def iw_supcon_loss(z, y, proto, w, tau=0.1, alpha=0.5, weighted=True):
    """Imbalance-weighted prototype-anchored supervised contrastive loss (eq. 3-5).
       z: (B,p) unit embeddings, y: (B,), proto: (K,p), w: (K,) class weights."""
    B = z.shape[0]
    P = F.normalize(proto, dim=1)
    wm = w[y] if weighted else torch.ones(B, device=z.device)
    # --- sample-to-sample term
    sim = z @ z.t() / tau
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~eye
    npos = pos.sum(1)
    l_ss = -(logp * pos).sum(1) / npos.clamp(min=1)
    has = (npos > 0).float()
    l_ss = (wm * has * l_ss).sum() / (wm * has).sum().clamp(min=1e-8)
    # --- sample-to-prototype term (always has a positive, even for a singleton class)
    logit = z @ P.t() / tau
    l_sp = (wm * F.cross_entropy(logit, y, reduction="none")).sum() / wm.sum()
    return (1 - alpha) * l_ss + alpha * l_sp


def open_space_loss(model, x, y, tau=0.1):
    """Open-space reservation: cross-class mixtures are pushed away from every
       prototype (soft-max of the prototype similarities is minimised)."""
    perm = torch.randperm(x.shape[0], device=x.device)
    diff = y != y[perm]
    if diff.sum() < 2:
        return x.new_zeros(())
    lam = torch.empty(x.shape[0], 1, device=x.device).uniform_(0.35, 0.65)
    xm = (lam * x + (1 - lam) * x[perm])[diff]
    zm = model.embed(xm)
    P = F.normalize(model.proto, dim=1)
    return tau * torch.logsumexp(zm @ P.t() / tau, dim=1).mean()


def train_encoder(X, y, K, mode="iw", iters=3000, bs=512, lr=2e-3, tau=0.1, alpha=0.5,
                  beta=0.999, lam_os=0.5, seed=0, p=32, log=None):
    """mode: 'ce'      linear head, plain cross-entropy (closed-world DNN)
             'supcon'  prototype SupCon, no imbalance weights, no open-space term
             'iw'      + imbalance weights
             'iw_os'   + imbalance weights + open-space reservation (proposed)
             'os'      open-space reservation without imbalance weights"""
    torch.manual_seed(seed); np.random.seed(seed)
    model = Encoder(X.shape[1], K, p=p)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    counts = np.bincount(y, minlength=K)
    w = torch.tensor(effective_number_weights(counts, beta), dtype=torch.float32)
    n = X.shape[0]
    model.train()
    for it in range(iters):
        idx = torch.randint(0, n, (bs,))
        xb, yb = Xt[idx], yt[idx]
        if mode == "ce":
            loss = F.cross_entropy(model.lin(model.embed(xb, normalize=False)), yb)
        else:
            z = model.embed(xb)
            loss = iw_supcon_loss(z, yb, model.proto, w, tau, alpha, weighted=mode in ("iw", "iw_os"))
            if mode in ("iw_os", "os"):
                loss = loss + lam_os * open_space_loss(model, xb, yb, tau)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log and (it + 1) % 1000 == 0:
            log(f"      it {it+1} loss {loss.item():.4f}")
    model.eval()
    return model


@torch.no_grad()
def forward_all(model, X, mode, tau=0.1, bs=8192):
    """returns embedding (unit-norm for contrastive models, raw for CE) and logits"""
    Z, L = [], []
    for i in range(0, X.shape[0], bs):
        xb = torch.from_numpy(X[i:i + bs])
        if mode == "ce":
            z = model.embed(xb, normalize=False); l = model.lin(z)
        else:
            z = model.embed(xb); l = z @ F.normalize(model.proto, dim=1).t() / tau
        Z.append(z.numpy()); L.append(l.numpy())
    return np.concatenate(Z).astype(np.float64), np.concatenate(L).astype(np.float64)


@torch.no_grad()
def hidden_all(model, X, bs=8192):
    """first hidden representation h1 (shallow depth, retains flow detail the head discards)"""
    out = []
    b = model.body
    for i in range(0, X.shape[0], bs):
        out.append(b[2](b[1](b[0](torch.from_numpy(X[i:i + bs])))).numpy())
    return np.concatenate(out)                      # float32, promoted chunk-wise in _maha


# ----------------------------------------------------------------------------
# proposed EVT head
# ----------------------------------------------------------------------------
def shrink_cov(Zc, pooled=None, kappa=0.0, shrink=True):
    """Ledoit-Wolf shrinkage towards (tr/p) I, then empirical-Bayes blend with the pooled
       within-class covariance:  S = (N_c S_LW + kappa S_W)/(N_c + kappa)."""
    n, p = Zc.shape
    if shrink:
        S, gam = ledoit_wolf(Zc)
    else:
        S, gam = np.cov(Zc.T) + 1e-9 * np.eye(p), 0.0
    if pooled is not None and kappa > 0:
        S = (n * S + kappa * pooled) / (n + kappa)
    return S, gam


def tail_size(n, rho=0.10, nmin=25, nmax=1000):
    """imbalance-adaptive number of exceedances used for the tail fit"""
    return int(min(max(nmin, math.ceil(rho * n)), nmax, max(n // 2, 5)))


def fit_gpd(exc):
    """GPD(xi, sigma) fit to threshold excesses (MLE; PWM fallback for tiny tails)"""
    exc = np.asarray(exc, dtype=np.float64)
    exc = exc[exc > 0]
    if len(exc) < 5:
        return 0.0, max(exc.mean() if len(exc) else 1e-3, 1e-6)
    try:
        xi, _, sg = genpareto.fit(exc, floc=0.0)
        if not np.isfinite(xi) or not np.isfinite(sg) or sg <= 0 or abs(xi) > 1.0:
            raise ValueError
    except Exception:
        s = np.sort(exc); m = len(s)
        a0 = s.mean(); a1 = np.sum((1 - (np.arange(1, m + 1) - 0.35) / m) * s) / m
        xi = 2 - a0 / max(a0 - 2 * a1, 1e-12); sg = 2 * a0 * a1 / max(a0 - 2 * a1, 1e-12)
        xi = float(np.clip(xi, -0.5, 0.9)); sg = float(max(sg, 1e-6))
    return float(xi), float(sg)


def _maha(Z, mu, prec):
    """Mahalanobis radius through a matrix square root of the precision (BLAS, fast)"""
    w, V = np.linalg.eigh((prec + prec.T) / 2)
    W = V * np.sqrt(np.maximum(w, 0.0))
    out = np.empty(Z.shape[0])
    for i in range(0, Z.shape[0], 50000):          # chunked: keeps 10^6-flow test sets in RAM
        R = (Z[i:i + 50000] - mu) @ W
        out[i:i + 50000] = np.sqrt((R * R).sum(1))
    return out


class TailModel:
    """Class-conditional semi-parametric tail model of a statistic S[:, c] (larger = more novel).
       Body: conformal rank.  Beyond the threshold u_c: peaks-over-threshold GPD
       (or the version-1 Weibull-on-excess model, tail='weibull')."""

    def __init__(self, tail="gpd", rho=0.10, adaptive=True, m0=20):
        self.tail, self.rho, self.adaptive, self.m0 = tail, rho, adaptive, m0

    def fit(self, S, y, K):
        self.K = K
        self.cal, self.u, self.xi, self.sg, self.nt, self.n = [], [], [], [], [], []
        for c in range(K):
            d = np.sort(S[y == c, c]); n = len(d)
            nt = tail_size(n, self.rho) if self.adaptive else max(int(math.ceil(self.rho * n)), 3)
            nt = max(min(nt, n - 1), 1)
            u = d[n - nt - 1]
            exc = d[n - nt:] - u
            if self.tail == "gpd":
                xi, sg = fit_gpd(exc)
            else:
                e = exc[exc > 0]
                try:
                    k, _, lam = weibull_min.fit(e, floc=0.0)
                except Exception:
                    k, lam = 1.0, max(e.mean() if len(e) else 1e-3, 1e-6)
                xi, sg = float(k), float(lam)
            self.cal.append(d); self.u.append(u); self.xi.append(xi); self.sg.append(sg)
            self.nt.append(nt); self.n.append(n)
        return self

    def _tail_sf(self, c, exc):
        if self.tail == "gpd":
            xi, sg = self.xi[c], self.sg[c]
            if abs(xi) < 1e-8:
                return np.exp(-exc / sg)
            return np.power(np.maximum(1 + xi * exc / sg, 1e-300), -1.0 / xi)
        return np.exp(-np.power(exc / self.sg[c], self.xi[c]))

    def pvalues(self, S):
        """pi_c(x): conformal rank in the body, (n_t+1)/(n+1) * tail survival beyond u_c"""
        Pv = np.empty_like(S, dtype=np.float64)
        for c in range(self.K):
            d, cal, n, nt, u = S[:, c], self.cal[c], self.n[c], self.nt[c], self.u[c]
            ge = n - np.searchsorted(cal, d, side="left")
            body = (ge + 1.0) / (n + 1.0)
            tail = (nt + 1.0) / (n + 1.0) * self._tail_sf(c, np.maximum(d - u, 0.0))
            Pv[:, c] = np.where(d > u, tail, body)
        return np.maximum(Pv, 1e-300)

    def radius(self, eps, delta=None):
        """class-conditional rejection threshold r_c(eps).
           delta=None : split-conformal rank k = ceil((1-eps)(n+1))  (false alarms <= eps on average
                        over calibration sets)
           delta in (0,1): training-conditional (PAC) rank: the smallest k whose coverage, which is
                        Beta(k, n+1-k) distributed, is >= 1-eps with probability >= 1-delta, so the
                        false-alarm rate of *this* calibrated detector is <= eps with confidence 1-delta.
           When fewer than m0 calibration points lie beyond the rank the class is too small for an
           order statistic and the GPD tail quantile is used (with the same finite-sample margin)."""
        r = np.empty(self.K)
        for c in range(self.K):
            n, nt, u = self.n[c], self.nt[c], self.u[c]
            k = math.ceil((1 - eps) * (n + 1)); eps_c = eps
            if delta is not None:
                while k <= n and beta_dist.cdf(1 - eps, k, n + 1 - k) > delta:
                    k += 1
                eps_c = max(eps - norm.ppf(1 - delta) * math.sqrt(eps * (1 - eps) / n), eps / 2)
            if (n + 1 - k) >= self.m0 and k <= n:      # enough exceedances: order statistic
                r[c] = self.cal[c][k - 1]
            else:                                      # minority class: extrapolate along the GPD tail
                q = min(max(eps_c * (n + 1.0) / (nt + 1.0), 1e-12), 1.0)
                if self.tail == "gpd":
                    xi, sg = self.xi[c], self.sg[c]
                    r[c] = u + (sg * (-math.log(q)) if abs(xi) < 1e-8 else sg / xi * (q ** (-xi) - 1))
                else:
                    r[c] = u + self.sg[c] * (-math.log(q)) ** (1.0 / self.xi[c])
        return r


class Geometry:
    """class means + shrinkage precision matrices of one representation layer"""

    def __init__(self, shrink=True, kappa=50.0):
        self.shrink, self.kappa = shrink, kappa

    def fit(self, Z, y, K):
        self.K = K; p = Z.shape[1]
        self.mu = np.stack([Z[y == c].mean(0) for c in range(K)])
        Z = Z.astype(np.float64)
        resid = Z - self.mu[y]
        pooled = resid.T @ resid / max(len(Z) - K, 1) + 1e-9 * np.eye(p)
        self.prec, self.gamma, self.cond = [], [], []
        for c in range(K):
            S, g = shrink_cov(Z[y == c], pooled, self.kappa if self.shrink else 0.0, self.shrink)
            self.gamma.append(g); self.cond.append(np.linalg.cond(S)); self.prec.append(np.linalg.pinv(S))
        return self

    def dist(self, Z):
        return np.stack([_maha(Z, self.mu[c], self.prec[c]) for c in range(self.K)], 1)


def margin_stat(L):
    """discriminative nonconformity of class c:  log-sum-exp of the other logits minus logit c"""
    M = np.empty_like(L)
    for c in range(L.shape[1]):
        o = np.delete(L, c, axis=1); m = o.max(1)
        M[:, c] = m + np.log(np.exp(o - m[:, None]).sum(1)) - L[:, c]
    return M


class MultiEvidenceEVT:
    """Proposed open-set head.
       stage 1 (training split):   per evidence channel e (Mahalanobis radius at each depth,
                                   discriminative margin) a class-conditional GPD tail turns the
                                   statistic into a tail probability pi_c^e(x)
       stage 2 (calibration split): Fisher statistic T_c(x) = -sum_e ln pi_c^e(x) is re-calibrated
                                   class-conditionally (conformal body + GPD tail)
       decision: UNKNOWN  iff  T_c(x) > r_c(eps) for every known class c."""

    def __init__(self, shrink=True, kappa=50.0, tail="gpd", rho=0.10, adaptive=True, use_margin=True):
        self.kw_geo = dict(shrink=shrink, kappa=kappa)
        self.kw_tail = dict(tail=tail, rho=rho, adaptive=adaptive)
        self.use_margin = use_margin

    def _fisher(self, feats, L):
        T = 0.0
        for g, tm, F_ in zip(self.geo, self.tm, feats):
            T = T - np.log(tm.pvalues(g.dist(F_)))
        if self.use_margin:
            T = T - np.log(self.tm_m.pvalues(margin_stat(L)))
        return T

    def fit(self, feats_tr, L_tr, y_tr, feats_cal, L_cal, y_cal, K):
        self.K = K
        self.geo = [Geometry(**self.kw_geo).fit(F_, y_tr, K) for F_ in feats_tr]
        self.tm = [TailModel(**self.kw_tail).fit(g.dist(F_), y_tr, K) for g, F_ in zip(self.geo, feats_tr)]
        if self.use_margin:
            self.tm_m = TailModel(**self.kw_tail).fit(margin_stat(L_tr), y_tr, K)
        self.final = TailModel(**self.kw_tail).fit(self._fisher(feats_cal, L_cal), y_cal, K)
        return self

    def evaluate(self, feats, L):
        T = self._fisher(feats, L)
        Pv = self.final.pvalues(T)
        return -np.log(Pv.max(1)), T, Pv        # unknown-threat score, statistic, p-values

    def decide(self, T, eps, delta=0.10):
        return (T > self.final.radius(eps, delta)[None, :]).all(1)   # True = UNKNOWN


# ----------------------------------------------------------------------------
# baselines (all on the same backbone)
# ----------------------------------------------------------------------------
def softmax(L):
    L = L - L.max(1, keepdims=True)
    e = np.exp(L)
    return e / e.sum(1, keepdims=True)


def score_msp(L):
    return 1.0 - softmax(L).max(1)


def score_energy(L):
    m = L.max(1)
    return -(m + np.log(np.exp(L - m[:, None]).sum(1)))


class TiedMahalanobis:
    """class-conditional Gaussian with one shared (tied) covariance, no shrinkage"""

    def fit(self, Z, y, K):
        self.mu = np.stack([Z[y == c].mean(0) for c in range(K)])
        r = Z - self.mu[y]
        self.prec = np.linalg.pinv(r.T @ r / len(Z))
        return self

    def score(self, Z):
        D = np.stack([_maha(Z, m, self.prec) ** 2 for m in self.mu], 1)
        return D.min(1)


class OpenMax:
    """OpenMax (Bendale & Boult): Weibull on the largest distances of correctly classified
       training activation vectors to the class mean activation vector, top-alpha revision."""

    def __init__(self, tail=20, alpha=3):
        self.tail, self.alpha = tail, alpha

    def fit(self, L, y, K):
        self.K = K
        pred = L.argmax(1)
        self.mav, self.wb = [], []
        for c in range(K):
            A = L[(y == c) & (pred == c)]
            if len(A) < 5:
                A = L[y == c]
            m = A.mean(0); d = np.sort(np.linalg.norm(A - m, axis=1))[-self.tail:]
            try:
                k, loc, lam = weibull_min.fit(d, floc=d.min() - 1e-6)
            except Exception:
                k, loc, lam = 1.0, d.min() - 1e-6, d.std() + 1e-6
            self.mav.append(m); self.wb.append((k, loc, lam))
        return self

    def score(self, L):
        K, a = self.K, min(self.alpha, self.K)
        rank = np.argsort(-L, axis=1)
        wgt = np.zeros_like(L)
        for i in range(a):
            wgt[np.arange(len(L)), rank[:, i]] = (a - i) / a
        Wc = np.empty_like(L)
        for c in range(K):
            k, loc, lam = self.wb[c]
            Wc[:, c] = weibull_min.cdf(np.linalg.norm(L - self.mav[c], axis=1), k, loc=loc, scale=lam)
        om = 1.0 - wgt * Wc
        Lhat = L * om
        unk = (L * (1 - om)).sum(1)
        P = softmax(np.hstack([Lhat, unk[:, None]]))
        return P[:, -1] + (1 - P[:, :-1].max(1)) * 1e-6, P


class EVM:
    """Extreme Value Machine (Rudd et al.), margin Weibulls on a class-balanced subsample
       of extreme-vector candidates; psi(x) = exp(-(d/lam)^k), score = 1 - max psi."""

    def __init__(self, per_class=300, tail=75, seed=0):
        self.m, self.tail, self.seed = per_class, tail, seed

    def fit(self, Z, y, K):
        rng = np.random.default_rng(self.seed)
        idx = np.concatenate([rng.permutation(np.where(y == c)[0])[: self.m] for c in range(K)])
        A, ya = Z[idx], y[idx]
        D = np.sqrt(np.maximum((A * A).sum(1)[:, None] + (A * A).sum(1)[None] - 2 * A @ A.T, 0))
        self.A, self.k, self.lam = A, np.empty(len(A)), np.empty(len(A))
        for i in range(len(A)):
            d = np.sort(D[i, ya != ya[i]])[: self.tail] * 0.5
            d = np.maximum(d, 1e-9)
            try:
                k, _, lam = weibull_min.fit(d, floc=0.0)
                if not (np.isfinite(k) and np.isfinite(lam) and lam > 0):
                    raise ValueError
            except Exception:
                k, lam = 1.0, d.mean()
            self.k[i], self.lam[i] = min(k, 50.0), lam
        return self

    def score(self, Z, bs=4096):
        out = np.empty(len(Z)); a2 = (self.A * self.A).sum(1)
        for i in range(0, len(Z), bs):
            z = Z[i:i + bs]
            d = np.sqrt(np.maximum((z * z).sum(1)[:, None] + a2[None] - 2 * z @ self.A.T, 0))
            psi = np.exp(-np.power(d / self.lam[None], self.k[None]))
            out[i:i + bs] = 1.0 - psi.max(1)
        return out


class V1Head:
    """Version-1 scoring rule, kept for the ablation: fixed-gamma shrinkage to (tr/p) I,
       85th-percentile threshold on the *training* radii, 2-parameter Weibull on the excesses,
       score 1 - max_c w_c(z) P(c|x)."""

    def fit(self, Z, y, K, gamma=0.1):
        self.K = K; p = Z.shape[1]
        self.mu, self.prec, self.u, self.wb = [], [], [], []
        for c in range(K):
            Zc = Z[y == c]; m = Zc.mean(0); S = np.cov(Zc.T)
            S = (1 - gamma) * S + gamma * np.trace(S) / p * np.eye(p)
            P = np.linalg.pinv(S)
            d = _maha(Zc, m, P)
            u = np.percentile(d, 85); e = d[d > u] - u
            try:
                k, _, lam = weibull_min.fit(e, floc=0.0)
            except Exception:
                k, lam = 1.0, e.mean() + 1e-6
            self.mu.append(m); self.prec.append(P); self.u.append(u); self.wb.append((k, lam))
        return self

    def score(self, Z, L):
        Pk = softmax(L); W = np.empty_like(Pk)
        for c in range(self.K):
            d = _maha(Z, self.mu[c], self.prec[c])
            k, lam = self.wb[c]
            W[:, c] = np.exp(-np.power(np.maximum(d - self.u[c], 0) / lam, k))
        return 1.0 - (W * Pk).max(1)
