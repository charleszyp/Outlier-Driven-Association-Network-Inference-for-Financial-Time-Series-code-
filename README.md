# Outlier-Driven Network Inference (ODNI) – Core Code

This repo contains a small, cleaned version of the core Python code used in the ODNI workflow.

## What is included

- `synth_core.py`  
  Core functions for generating synthetic multivariate time series with VAR(1) mean dynamics and cross-GARCH(1,1) volatility.

- `simulate_synthetic_data.py`  
  CLI script for simulating synthetic data, saving the series, parameter files, outlier mask, and Phi matrix.

- `pairwise_em_core.py`  
  Core EM-based pairwise inference for binary outlier-activation matrices.

- `run_pairwise_em.py`  
  CLI script for running pairwise EM inference on an input binary matrix and saving `Pcond`, `Pedge`, and `Peff`.

- `requirements.txt`  
  Minimal Python dependencies.

## Simple workflow

### 1) Generate synthetic data
```bash
python simulate_synthetic_data.py --base_name demo --out_dir ./outputs
```

This will save files such as:
- `demo_y.csv`
- `demo_params.npz`
- `demo_outliers.csv`
- `demo_Phi.xlsx`
- `demo_meta.json`

### 2) Run pairwise EM inference
```bash
python run_pairwise_em.py --input ./outputs/demo_outliers.csv --out_dir ./binstat_outputs --lag 1
```

This will save:
- `demo_outliers_Pcond_lag1.xlsx`
- `demo_outliers_Pedge_lag1.xlsx`
- `demo_outliers_Peff_lag1.xlsx`

## Notes

- This is only the core code bundle, not the full project.
- File names were cleaned a little for GitHub upload.
- The code is kept close to the working version and is not heavily repackaged.

## Environment

Python 3.10+ is recommended.

Install dependencies:
```bash
pip install -r requirements.txt
```
