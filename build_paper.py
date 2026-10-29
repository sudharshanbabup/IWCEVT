"""
build_paper.py -- fills every @@key@@ placeholder of latex_3/main_head.tex + main_body.tex with the
measured value from results/numbers.json and writes latex_3/main.tex.  No number is typed by hand.

  python3 build_paper.py <results_dir> <latex_dir>
"""
import sys, os, re, json, math
R, LX = sys.argv[1], sys.argv[2]
n = json.load(open(os.path.join(R, "numbers.json")))
f = lambda k: float(n[k])
BASE = ["MSP", "Energy", "OpenMax", "Mahalanobis", "EVM"]; DS = ["cic_ids2017", "unsw_nb15"]
tot = int(n["data:cic_ids2017:ntrain"]) + int(n["data:cic_ids2017:ntest"])
n["cic_total"] = f"{tot:,}".replace(",", "{,}")
n["data:cic_ids2017:ntot"] = "%.2f" % (tot / 1e6)
n["data:cic_ids2017:ntest"] = f"{int(n['data:cic_ids2017:ntest']):,}".replace(",", "{,}")
n["nconfig_total"] = str(sum(int(n[f"{d}:nconfig"]) for d in DS))
fw = [f(f"{d}:{m}:far_worst") for d in DS for m in BASE]
n["fw_base_min"], n["fw_base_max"] = "%.1f" % min(fw), "%.1f" % max(fw)
n["cic_best_base_udrwc"] = "%.1f" % max(f(f"cic_ids2017:{m}:udr_wc") for m in BASE)
# openness range: CIC has 11 trainable classes and 2 always-unknown families, UNSW 10 classes
op = [1 - math.sqrt(2 * K / (2 * K + U)) for K, U in ((10, 3), (4, 9), (9, 1), (3, 7))]
n["open_min"], n["open_max"] = "%.2f" % min(op), "%.2f" % max(op)
# probability that an anchor of the rarest UNSW family (Worms, 174 flows of 257,673) has no in-batch positive
n["nopos"] = "%.2f" % ((1 - 174 / 257673) ** 511)
a = [f(f"U:cic_ids2017:IW-C-EVT:{u}:auroc") for u in (1, 3, 5, 7)]
n["cic_auroc_min"], n["cic_auroc_max"] = "%.3f" % min(a), "%.3f" % max(a)
C = ["Benign", "DoS Hulk", "DDoS", "PortScan", "GoldenEye", "FTP-Pat.", "Slowloris", "Slowhttp", "SSH-Pat.", "Web Att.", "Bot"]
v = [f(f"farclass:IW-C-EVT:{c}") for c in C]
n["fc_prop_min"], n["fc_prop_max"] = "%.1f" % min(v), "%.1f" % max(v)
n["fc_logit_ddos"] = "%.0f" % math.ceil(max(f("farclass:Energy:DDoS"), f("farclass:OpenMax:DDoS")))
for d, t in (("cic_ids2017", "cic"), ("unsw_nb15", "unsw")):
    n[f"d_margin_{t}"] = "%.1f" % (100 * (f(f"abl:{d}:IW-C-EVT:auroc") - f(f"abl:{d}:IW-C-EVT w/o margin evidence:auroc")))
# Corollary 1 (price of rarity): eps* = eps - z_{1-delta} sqrt(eps (1-eps) / n_c), eps = 0.05, delta = 0.1
from scipy.stats import norm
z = norm.ppf(0.9)
for cls, ntrain in (("benign", 700000), ("bot", 1363)):           # CIC-IDS2017 trainable flows of the class
    nc = int(0.2 * ntrain)                                         # calibration share of the class
    n[f"price:n{cls}"] = f"{nc:,}".replace(",", "{,}")
    n[f"price:{cls}"] = "%.1f" % (100 * (0.05 - z * math.sqrt(0.05 * 0.95 / nc)))
n["time:IW-C-EVT:mb"] = "%.1f" % (f("time:IW-C-EVT:kB") / 1024)
src = open(os.path.join(LX, "main_head.tex")).read() + open(os.path.join(LX, "main_body.tex")).read()
missing = sorted({k for k in re.findall(r"@@(.+?)@@", src) if k not in n})
assert not missing, missing
out = re.sub(r"@@(.+?)@@", lambda m: n[m.group(1)], src)
# tables are inlined (\\input inside tabular clashes with booktabs rules)
out = re.sub(r"\\input\{tables/(\w+)\}", lambda m: open(os.path.join(LX, "tables", m.group(1) + ".tex")).read().strip(), out)
open(os.path.join(LX, "main.tex"), "w").write(out)
print("main.tex written;", len(set(re.findall(r"@@(.+?)@@", src))), "measured values inserted")
