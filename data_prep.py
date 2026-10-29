"""
data_prep.py  --  loads the two public benchmarks used in the paper and writes
compact .npz files (float32 features, int labels) that the experiments read.

  UNSW-NB15   official training-set (175,341 flows) / testing-set (82,332 flows)
  CIC-IDS2017 MachineLearningCVE flow files (all 8 capture files, 2.83 M flows),
              de-duplicated; benign capped at 10^6 flows, each attack family at 2x10^5

Usage:  python3 data_prep.py <data_dir>
        <data_dir>/unsw/UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv
        <data_dir>/cic/*.parquet
"""
import sys, glob, os
import numpy as np
import pandas as pd

SEED = 2026


def _signed_log(a):
    return np.sign(a) * np.log1p(np.abs(a))


def prep_unsw(ddir, out):
    tr = pd.read_csv(os.path.join(ddir, "unsw", "UNSW_NB15_training-set.csv"), encoding="utf-8-sig")
    te = pd.read_csv(os.path.join(ddir, "unsw", "UNSW_NB15_testing-set.csv"), encoding="utf-8-sig")
    for d in (tr, te):
        d["attack_cat"] = d["attack_cat"].astype(str).str.strip().replace({"Backdoors": "Backdoor"})
    classes = ["Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
               "Analysis", "Backdoor", "Shellcode", "Worms"]
    cat_cols = ["proto", "service", "state"]
    num_cols = [c for c in tr.columns if c not in cat_cols + ["id", "attack_cat", "label"]]
    # categorical: one-hot of the levels that cover the training set (rare -> 'other')
    blocks_tr, blocks_te, names = [], [], []
    for c in cat_cols:
        top = tr[c].value_counts()
        keep = list(top[top >= 200].index)
        for lv in keep:
            blocks_tr.append((tr[c] == lv).values.astype(np.float32))
            blocks_te.append((te[c] == lv).values.astype(np.float32))
            names.append(f"{c}={lv}")
        blocks_tr.append((~tr[c].isin(keep)).values.astype(np.float32))
        blocks_te.append((~te[c].isin(keep)).values.astype(np.float32))
        names.append(f"{c}=other")
    Xn_tr = _signed_log(tr[num_cols].values.astype(np.float64))
    Xn_te = _signed_log(te[num_cols].values.astype(np.float64))
    mu, sd = Xn_tr.mean(0), Xn_tr.std(0) + 1e-6
    Xtr = np.hstack([(Xn_tr - mu) / sd, np.stack(blocks_tr, 1)]).astype(np.float32)
    Xte = np.hstack([(Xn_te - mu) / sd, np.stack(blocks_te, 1)]).astype(np.float32)
    cmap = {c: i for i, c in enumerate(classes)}
    ytr = tr["attack_cat"].map(cmap).values.astype(np.int64)
    yte = te["attack_cat"].map(cmap).values.astype(np.int64)
    np.savez_compressed(out, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                        classes=np.array(classes), features=np.array(num_cols + names))
    print("UNSW-NB15", Xtr.shape, Xte.shape)
    for i, c in enumerate(classes):
        print(f"  {c:15s} train {np.sum(ytr == i):7d}  test {np.sum(yte == i):7d}")


def prep_cic(ddir, out, cap_benign=1000000, cap_attack=200000):
    fs = sorted(glob.glob(os.path.join(ddir, "cic", "*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    lab = df["Label"].astype(str).str.strip()
    lab = lab.str.replace("�", "-", regex=False).str.replace("\x96", "-", regex=False)
    lab = lab.replace({l: "Web Attack" for l in lab.unique() if l.startswith("Web Attack")})
    df["Label"] = lab
    print("CIC-IDS2017 raw", df.shape)
    print(df["Label"].value_counts())
    feat = [c for c in df.columns if c != "Label"]
    for c in feat:                                   # float32 keeps 2.8 M x 78 inside RAM
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    # drop exact duplicates (well-known artefact of the CICFlowMeter export)
    df = df.drop_duplicates(ignore_index=True)
    print("CIC-IDS2017 after cleaning / de-duplication", df.shape)
    lab = df["Label"].values
    X = df[feat].values
    del df
    keep = X.std(0) > 0
    X, feat = X[:, keep], [f for f, k in zip(feat, keep) if k]
    classes = ["BENIGN", "DoS Hulk", "DDoS", "PortScan", "DoS GoldenEye", "FTP-Patator",
               "DoS slowloris", "DoS Slowhttptest", "SSH-Patator", "Web Attack", "Bot"]
    rare = ["Infiltration", "Heartbleed"]          # too few flows to train on: always-unknown
    rng = np.random.default_rng(SEED)
    itr, ite, ytr, yte = [], [], [], []
    for i, c in enumerate(classes + rare):
        idx = np.where(lab == c)[0]
        rng.shuffle(idx)
        cap = cap_benign if c == "BENIGN" else cap_attack
        idx = idx[:cap]
        if c in rare:
            ntr = 0
        else:
            ntr = int(0.7 * len(idx))
        itr.append(idx[:ntr]); ite.append(idx[ntr:])
        ytr += [i] * ntr; yte += [i] * (len(idx) - ntr)
    itr, ite = np.concatenate(itr), np.concatenate(ite)
    Xa, Xb = _signed_log(X[itr].astype(np.float64)), _signed_log(X[ite].astype(np.float64))
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    Xtr = ((Xa - mu) / sd).astype(np.float32)
    Xte = ((Xb - mu) / sd).astype(np.float32)
    ytr, yte = np.array(ytr), np.array(yte)
    np.savez_compressed(out, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                        classes=np.array(classes + rare), features=np.array(feat),
                        n_trainable=len(classes))
    print("CIC-IDS2017 used", Xtr.shape, Xte.shape)
    for i, c in enumerate(classes + rare):
        print(f"  {c:18s} train {np.sum(ytr == i):7d}  test {np.sum(yte == i):7d}")


if __name__ == "__main__":
    ddir = sys.argv[1]
    prep_unsw(ddir, os.path.join(ddir, "unsw_nb15.npz"))
    prep_cic(ddir, os.path.join(ddir, "cic_ids2017.npz"))
