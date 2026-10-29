"""
make_figures.py -- aggregates the JSON-lines results and writes
   * one JSON per panel for matlab_fig_1.m           -> codes_3/matlab/data/*.json
   * LaTeX tables and the \\newcommand numbers file   -> <latex_dir>/tables/*.tex
Nothing is typed by hand: every number in the paper comes from these files.

  python3 make_figures.py <results_dir> <latex_dir>
  then:  cd matlab;  matlab_fig_1   (MATLAB)   or
         xvfb-run -a -s "-screen 0 1920x1200x24 -dpi 72" octave --no-gui --eval "graphics_toolkit('qt'); matlab_fig_1"
"""
import sys, os, json, glob, collections
import numpy as np

R, LX = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
JD = os.path.join(HERE, "matlab", "data"); os.makedirs(JD, exist_ok=True)
TD = os.path.join(LX, "tables"); os.makedirs(TD, exist_ok=True)
for f in glob.glob(os.path.join(JD, "*.json")):
    os.remove(f)
DS = {"cic_ids2017": "CIC-IDS2017", "unsw_nb15": "UNSW-NB15"}
PROP = "IW-C-EVT"
# one colour / marker / line per scheme, identical in every figure
STYLE = collections.OrderedDict([
    (PROP,          dict(color=[0.000, 0.447, 0.698], marker="o", line="-", proposed=True)),
    ("MSP",         dict(color=[0.835, 0.369, 0.000], marker="s", line="--")),
    ("Energy",      dict(color=[0.000, 0.620, 0.451], marker="d", line="-.")),
    ("OpenMax",     dict(color=[0.482, 0.196, 0.580], marker="^", line=":")),
    ("Mahalanobis", dict(color=[0.800, 0.475, 0.655], marker="v", line="--")),
    ("EVM",         dict(color=[0.902, 0.624, 0.000], marker="p", line="-.")),
    (PROP + " global threshold", dict(color=[0.337, 0.706, 0.914], marker=">", line="--")),
    ("C-EVT (no IW)", dict(color=[0.835, 0.369, 0.000], marker="s", line="--")),
    (PROP + " w/o shrinkage", dict(color=[0.350, 0.350, 0.350], marker="h", line=":")),
    (PROP + " conformal only", dict(color=[0.482, 0.196, 0.580], marker="^", line=":")),
])
LABEL = {PROP: "IW-C-EVT (proposed)", PROP + " global threshold": "Proposed score, global threshold",
         "C-EVT (no IW)": "Without imbalance weights", PROP + " w/o shrinkage": "Without shrinkage",
         PROP + " conformal only": "Conformal only (no GPD)", "MSP": "MSP", "Energy": "Energy"}
json.dump({k: v for k, v in STYLE.items()}, open(os.path.join(HERE, "matlab", "palette.json"), "w"), indent=1)


def rd(name):
    p = os.path.join(R, name)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def agg(rows, key, **filt):
    v = [r[key] for r in rows if key in r and all(r[k] == x for k, x in filt.items())]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (float("nan"), float("nan"), 0)


def series(method, x, y, label=None):
    s = dict(STYLE[method]); s.update(label=label or LABEL.get(method, method), x=list(map(float, x)), y=list(map(float, y)))
    return s


def panel(name, **kw):
    kw = {k: v for k, v in kw.items() if v is not None}
    kw["name"] = name
    json.dump(kw, open(os.path.join(JD, name + ".json"), "w"), indent=1)


num = {}                                   # numbers quoted in the text


def setnum(k, v, fmt="%.1f"):
    num[k] = fmt % v


USET = (0, 1, 3, 5, 7)
main = {d: [r for r in rd(f"main_{d}.jsonl") if r["U"] in USET] for d in DS}
abl = {d: rd(f"ablation_{d}.jsonl") for d in DS}
imb = {d: rd(f"imbalance_{d}.jsonl") for d in DS}
BASE = ["MSP", "Energy", "OpenMax", "Mahalanobis", "EVM"]
ALL = BASE + [PROP]

# ---------------------------------------------------------------- Fig. 2: AUROC and matched-FAR detection vs openness
for d, tag in (("cic_ids2017", "cic"), ("unsw_nb15", "unsw")):
    rows = main[d]
    Us = [u for u in USET if u >= 1 and agg(rows, "auroc", U=u, method=PROP)[2] >= 1]
    if not Us:
        continue
    for key, fn, yl in (("auroc", "a", "AUROC (known vs. unknown)"), ("udr_wc", "b", "UDR at worst class FAR of 5%")):
        ser = [series(m, Us, [agg(rows, key, U=u, method=m)[0] for u in Us]) for m in ALL]
        name = f"fig2{fn}_{key}_cic" if d == "cic_ids2017" else f"extra_unsw_{key}"
        panel(name, layout="pair", xlabel="Unseen attack families {\\itU}", ylabel=yl,
              xticks=Us, xlim=[0.6, 7.4], ylim=[0, 1.0] if key == "udr_wc" else [0.4, 1.0], ytickformat="%.1f",
              legend="auto", legend_cols=2, series=ser)
        for u in Us:
            for m in ALL:
                setnum(f"U:{d}:{m}:{u}:{key}", agg(rows, key, U=u, method=m)[0] * (100 if key == "udr_wc" else 1), "%.1f" if key == "udr_wc" else "%.3f")

# ---------------------------------------------------------------- Table I: operating point, averaged over all open-set draws
def table_main():
    L = []
    for d in DS:
        rows = [r for r in main[d] if "auroc" in r and r["U"] >= 1]
        if not rows:
            continue
        best = {k: (max if hi else min)(agg(rows, k, method=m)[0] for m in ALL)
                for k, hi in (("auroc", 1), ("macro_auroc", 1), ("udr_wc", 1), ("f1_open", 1), ("far_worst", 0))}
        for i, m in enumerate(ALL):
            c = []
            for k, sc in (("auroc", 1), ("macro_auroc", 1), ("far", 100), ("far_worst", 100), ("udr", 100), ("udr_wc", 100), ("f1_open", 100)):
                v = agg(rows, k, method=m)[0]
                s = ("%.3f" % v) if sc == 1 else ("%.1f" % (v * sc))
                if k in best and abs(v - best[k]) < 1e-12:
                    s = "\\textbf{%s}" % s
                c.append(s)
                setnum(f"{d}:{m}:{k}", v * sc, "%.3f" if sc == 1 else "%.1f")
            lead = ("\\multirow{6}{*}{\\rotatebox{90}{%s}} " % DS[d]) if i == 0 else ""
            nm = "\\textbf{IW-C-EVT (proposed)}" if m == PROP else m
            L.append(f"{lead}& {nm} & " + " & ".join(c) + " \\\\")
        L.append("\\midrule")
        setnum(f"{d}:nconfig", len({(r['U'], r['trial']) for r in rows}), "%d")
    open(os.path.join(TD, "table_main.tex"), "w").write("\n".join(L[:-1]) + "\n")


table_main()

# ---------------------------------------------------------------- Fig. 4a: per-class false-alarm rate (closed taxonomy order = frequency)
d = "cic_ids2017"
rows = [r for r in main[d] if "far_class" in r and r["U"] >= 1]
if rows:
    npz_classes = ["Benign", "DoS Hulk", "DDoS", "PortScan", "GoldenEye", "FTP-Pat.", "Slowloris", "Slowhttp", "SSH-Pat.", "Web Att.", "Bot"]
    ser = []
    for m in ("Energy", "OpenMax", "Mahalanobis", PROP):
        y = []
        for c in range(len(npz_classes)):
            v = [r["far_class"][str(c)] for r in rows if r["method"] == m and str(c) in r["far_class"]]
            y.append(100 * float(np.mean(v)) if v else float("nan"))
        ser.append(series(m, range(1, len(npz_classes) + 1), [max(v, 0.05) for v in y]))
        for c, v in enumerate(y):
            setnum(f"farclass:{m}:{npz_classes[c]}", v)
    # single-draw behaviour of the class-conditional radius: confidence rank vs. plain conformal rank
    for m, t in ((PROP, "prop"), (PROP + " marginal radius", "marg")):
        v = np.array([x for r in rows if r["method"] == m for x in r["far_class"].values()]) * 100
        if len(v):
            setnum(f"exceed:{t}", 100 * float(np.mean(v > 5.0)), "%.0f"); setnum(f"maxdraw:{t}", float(v.max()))
    setnum("marg:cic:udr", 100 * agg(rows, "udr", method=PROP + " marginal radius")[0])
    ser.append(dict(kind="hline", y=5.0, line="--", label="Target {\\it\\epsilon} = 5%"))
    panel("fig3a_far_per_class", layout="pair", xlabel="Known class index (decreasing frequency)",
          ylabel="False alarm rate (%)", yscale="log", ylim=[0.05, 5000], lock_ylim=True, yticks=[0.1, 1, 10, 100],
          yticklabels=["0.1", "1", "10", "100"], xticks=list(range(1, 12)),
          xlim=[0.5, 11.5], legend="north", legend_cols=2, series=ser)

# ---------------------------------------------------------------- Fig. 4b/5: imbalance stress test
for d, tag in (("cic_ids2017", "cic"), ("unsw_nb15", "unsw")):
    rows = imb[d]
    if not rows:
        continue
    th = sorted({r["thin"] for r in rows}, reverse=True)
    x = [1.0 / t for t in th]
    for key, yl, fn in (("udr", "UDR at target {\\it\\epsilon} = 5% (%)", "udr"), ("far_worst", "Worst-class false alarm rate (%)", "far")):
        ser = []
        for m in (PROP, "C-EVT (no IW)", PROP + " w/o shrinkage", PROP + " conformal only"):
            ser.append(series(m, x, [max(100 * agg(rows, key, thin=t, method=m)[0], 0.0) for t in th]))
        if key == "far_worst":
            ser.append(dict(kind="hline", y=5.0, line="--", label="Target {\\it\\epsilon} = 5%"))
        name = "fig3b_imbalance_udr" if (d == "cic_ids2017" and fn == "udr") else f"extra_{tag}_imbalance_{fn}"
        panel(name, layout="pair", ylim=[0, 125] if fn == "udr" else None, lock_ylim=(fn == "udr"),
              yticks=[0, 20, 40, 60, 80, 100] if fn == "udr" else None, legend="northeast" if fn == "udr" else "auto", xlabel="Attack class thinning factor", ylabel=yl,
              xscale="log", xticks=x, xticklabels=["1", "4", "16", "64"], xlim=[0.8, 80], series=ser)
    for m in (PROP, PROP + " conformal only", PROP + " global threshold", "C-EVT (no IW)", PROP + " w/o shrinkage", "Energy", "MSP"):
        for t in th:
            for k in ("far_worst", "udr", "auroc", "f1_closed", "udr_wc", "f1_open", "far"):
                setnum(f"imb:{d}:{m}:{int(round(1/t))}:{k}", agg(rows, k, thin=t, method=m)[0] * (1 if k == "auroc" else 100), "%.3f" if k == "auroc" else "%.1f")
    for t in th:
        setnum(f"imb:{d}:ratio:{int(round(1/t))}", agg(rows, "imb_ratio", thin=t, method=PROP)[0], "%.0f")
        setnum(f"imb:{d}:calmin:{int(round(1/t))}", agg(rows, "cal_min", thin=t, method=PROP)[0], "%.0f")

# ---------------------------------------------------------------- Table II: ablation (U = 3, five draws, both data sets)
def table_abl():
    order = [PROP, PROP + " w/o margin evidence", PROP + " w/o shallow depth", PROP + " radius only (z)",
             PROP + " w/o shrinkage", PROP + " Weibull tail", PROP + " fixed tail size", PROP + " marginal radius", PROP + " global threshold",
             "C-EVT (no IW)", "Version-1 head"]
    names = ["\\textbf{Full IW-C-EVT}", "\\;w/o margin evidence", "\\;w/o shallow depth", "\\;latent radius only",
             "\\;w/o shrinkage", "\\;Weibull tail", "\\;fixed tail fraction", "\\;plain conformal rank", "\\;global threshold",
             "\\;w/o imbalance weights", "Version 1 rule"]
    L = []
    for m, nm in zip(order, names):
        c = []
        for d in DS:
            for k, sc in (("auroc", 1), ("udr_wc", 100), ("far_worst", 100)):
                v = agg(abl[d], k, method=m)[0]
                c.append(("%.3f" % v) if sc == 1 else ("%.1f" % (v * sc)))
                setnum(f"abl:{d}:{m}:{k}", v * sc, "%.3f" if sc == 1 else "%.1f")
        L.append(nm + " & " + " & ".join(c) + " \\\\")
    open(os.path.join(TD, "table_ablation.tex"), "w").write("\n".join(L) + "\n")
    for d in DS:
        for m in (PROP, PROP + " w/o shrinkage"):
            setnum(f"abl:{d}:{m}:cond", np.log10(agg(abl[d], "cond_max", method=m)[0]) if abl[d] else float("nan"))
        setnum(f"abl:{d}:f1c_iw", 100 * agg(abl[d], "f1_closed", method=PROP)[0])
        setnum(f"abl:{d}:f1c_noiw", 100 * agg(abl[d], "f1_closed", method="C-EVT (no IW)")[0])


if all(abl[d] for d in DS):
    table_abl()

# closed-world reference and data statistics
for d in DS:
    r0 = [r for r in main[d] if r["U"] == 0]
    for r in r0:
        if r["method"] in (PROP, "MSP"):
            setnum(f"closed:{d}:{r['method']}:f1", 100 * r["f1_closed"]); setnum(f"closed:{d}:{r['method']}:acc", 100 * r["acc_closed"])
            setnum(f"closed:{d}:{r['method']}:far", 100 * r["far"])
        if "auroc" in r:
            setnum(f"rare:{d}:{r['method']}:auroc", r["auroc"], "%.3f"); setnum(f"rare:{d}:{r['method']}:udr", 100 * r["udr"])
    if r0:
        setnum(f"data:{d}:ntest", r0[0]["n_test"], "%d"); setnum(f"data:{d}:ntrain", r0[0]["n_fit"] + r0[0]["n_cal"], "%d")
        setnum(f"data:{d}:imb", r0[0]["imb_ratio"], "%.0f")
tp = os.path.join(R, "timing.json")
if os.path.exists(tp):
    for k, v in json.load(open(tp)).items():
        setnum(f"time:{k}:us", v["us_per_flow"]); setnum(f"time:{k}:kB", v["kB_float32"], "%.0f")
        setnum(f"time:{k}:params", (v["params_encoder"] + v["params_head"]) / 1e3, "%.0f")

json.dump(num, open(os.path.join(R, "numbers.json"), "w"), indent=1, sort_keys=True)
print("panels:", sorted(os.path.basename(f) for f in glob.glob(os.path.join(JD, "*.json"))))
print("numbers:", len(num))
