"""check_rank.py -- reproduces the rank facts quoted in Section II-C of the version_11 paper.

  python3 check_rank.py            (needs scipy)

For eps = 0.05, delta = 0.1, m0 = 20 it prints
  * that I_{1-eps}(k, n+1-k) decreases in k (so eq. (13) is searched upwards from ceil((1-eps)(n+1))),
  * the smallest calibration size for which an admissible rank exists (45 = ln(delta)/ln(1-eps)),
  * the smallest calibration size for which the order statistic branch of eq. (14) is used (515),
  * the FAR budget a class actually spends for the 140,000 benign and 272 Bot calibration flows.
"""
import math
from scipy.stats import beta, norm

EPS, DELTA, M0 = 0.05, 0.1, 20


def rank(n, eps=EPS, delta=DELTA):
    k = math.ceil((1 - eps) * (n + 1))
    while k <= n and beta.cdf(1 - eps, k, n + 1 - k) > delta:
        k += 1
    return k                                        # k = n + 1 means "no admissible rank"


if __name__ == "__main__":
    n = 200
    vals = [beta.cdf(1 - EPS, k, n + 1 - k) for k in range(185, 201)]
    print("I_{1-eps}(k, n+1-k) decreasing in k:", all(a >= b for a, b in zip(vals, vals[1:])))
    n_exist = next(n for n in range(1, 10000) if rank(n) <= n)
    print("smallest n_c with an admissible rank:", n_exist, " (ln d / ln(1-e) = %.2f)" % (math.log(DELTA) / math.log(1 - EPS)))
    n_os = next(n for n in range(1, 10000) if rank(n) <= n and n + 1 - rank(n) >= M0)
    print("smallest n_c using the order statistic branch (m0 = %d):" % M0, n_os)
    for name, nc in (("benign", 140000), ("Bot", 272)):
        k = rank(nc)
        approx = EPS - norm.ppf(1 - DELTA) * math.sqrt(EPS * (1 - EPS) / nc)
        print("%-6s n_c=%6d  exact budget %.4f  normal approx eps_c %.4f" % (name, nc, (nc + 1 - k) / (nc + 1), approx))
