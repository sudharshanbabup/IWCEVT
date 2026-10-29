# Imbalance Weighted Contrastive Extreme Value Architecture for Open Set Network Intrusion Detection

This repository contains the official Python implementation of the paper **"Imbalance Weighted Contrastive Extreme Value Architecture for Open Set Network Intrusion Detection"** (IW-C-EVT).

IW-C-EVT addresses open set network intrusion detection under extreme class imbalance. Rather than setting a single global rejection threshold on pooled traffic—which disproportionately causes false alarms on rare known attack classes—IW-C-EVT controls the false alarm rate (FAR) individually for each known class using:
1. **Imbalance-Weighted Prototype-Anchored Contrastive Loss (IW-SupCon)** with open-space synthetic mixtures.
2. **Shrinkage Mahalanobis Geometry** blending per-class Ledoit-Wolf covariance with pooled within-class covariance.
3. **Class-Adapted Generalized Pareto Distribution (GPD)** tail models for extreme score estimation.
4. **PAC-Conformal Order Statistic Rejection Radii** bounding per-class false alarm rates under statistical confidence.

---

## Repository Structure

| File | Description |
|---|---|
| `iwcevt.py` | Core implementation: network encoder, IW-SupCon loss, shrinkage geometry, GPD tail estimation, conformal rejection rule, and baselines (MSP, Energy, OpenMax, Mahalanobis, EVM). |
| `data_prep.py` | Ingestion, cleaning, deduplication, log-scaling, and train/test partitioning of UNSW-NB15 and CIC-IDS2017 into compact `.npz` archives. |
| `download_data.py` | Downloads the public benchmark datasets (UNSW-NB15 and CIC-IDS2017) from Hugging Face mirrors. |
| `run_experiments.py` | Experiment runner for main openness sweeps ($U \in \{1, 3, 5, 7\}$), component ablation, imbalance stress testing (down to 1/64 thinning), and timing benchmarks. |
| `check_rank.py` | Exact verification of conformal calibration ranks and theoretical bounds. |
| `make_figures.py` | Aggregates experiment results, computes summary metrics, and generates LaTeX tables. |
| `build_paper.py` | Injects measured numbers from experiment runs into LaTeX publication sources. |

---

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- scikit-learn
- pandas
- pyarrow

Install dependencies via:
```bash
pip install -r requirements.txt
```

---

## Workflow

### 1. Download Datasets
Download the UNSW-NB15 and CIC-IDS2017 benchmark datasets into a target directory (e.g., `data`):
```bash
python download_data.py data
```

### 2. Preprocess Data
Clean, scale, and build the `.npz` archives:
```bash
python data_prep.py data
```

### 3. Verify Theoretical Conformal Bounds
Verify the admissible calibration sizes, beta incomplete bounds, and class budget properties:
```bash
python check_rank.py
```

### 4. Run Experiments
Execute experimental evaluations on either `cic_ids2017` or `unsw_nb15` (runs are resumable via JSON-lines outputs):

- **Main Openness Sweep ($U = 1, 3, 5, 7$):**
  ```bash
  python run_experiments.py data results/main_cic_ids2017.jsonl main cic_ids2017
  python run_experiments.py data results/main_unsw_nb15.jsonl main unsw_nb15
  ```

- **Component Ablation Study:**
  ```bash
  python run_experiments.py data results/ablation_cic_ids2017.jsonl ablation cic_ids2017
  python run_experiments.py data results/ablation_unsw_nb15.jsonl ablation unsw_nb15
  ```

- **Imbalance Stress Test:**
  ```bash
  python run_experiments.py data results/imbalance_cic_ids2017.jsonl imbalance cic_ids2017
  python run_experiments.py data results/imbalance_unsw_nb15.jsonl imbalance unsw_nb15
  ```

- **Timing Benchmark:**
  ```bash
  python run_experiments.py data results/timing.json timing cic_ids2017
  ```

### 5. Aggregate Results
Aggregate output files and generate LaTeX tables and summary statistics:
```bash
python make_figures.py results ./latex
```
