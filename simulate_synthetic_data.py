"""
Command-line script for generating synthetic multivariate time series.

This script:
1. Sets the simulation parameters
2. Calls `simulate_var_cross_garch(...)`
3. Saves the simulated series, parameter files, outlier mask, and Phi matrix

It is a light wrapper around `synth_core.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

from synth_core import simulate_var_cross_garch, boxplot_outliers, save_outputs


def load_mu_excel(path: str) -> np.ndarray:
    df = pd.read_excel(path, index_col=0)
    mu = df.values.astype(float)
    if mu.shape[0] != mu.shape[1]:
        raise ValueError(f"mu matrix must be square, got {mu.shape}")
    return mu


def _auto_seed() -> int:
    """
    Generate a different seed each run (best-effort, no external deps).
    Uses time_ns + PID + 32 bits of OS entropy.
    """
    pid = os.getpid()
    t = time.time_ns()
    rnd = int.from_bytes(os.urandom(4), "little", signed=False)
    # fold into 31-bit positive int (NumPy accepts up to 2**32-1, but keep safe)
    seed = (t ^ (pid << 16) ^ rnd) & 0x7FFFFFFF
    if seed == 0:
        seed = 1
    return int(seed)


def _save_matrix(path: str, mat: np.ndarray, prefix: str = "Node") -> None:
    df = pd.DataFrame(
        mat,
        index=[f"{prefix}{i+1}" for i in range(mat.shape[0])],
        columns=[f"{prefix}{i+1}" for i in range(mat.shape[1])],
    )
    if path.lower().endswith(".csv"):
        df.to_csv(path, index=True)
    elif path.lower().endswith((".xlsx", ".xls")):
        df.to_excel(path, index=True)
    else:
        raise ValueError(f"Unsupported matrix format for: {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # core sizes
    p.add_argument("--n", type=int, default=10, help="dimension n (ignored if --mu_excel is provided)")
    p.add_argument("--T", type=int, default=10000, help="length T (after burn-in)")
    p.add_argument("--burnin", type=int, default=500, help="burn-in length")

    # seed: <0 means auto
    p.add_argument("--seed", type=int, default=-1, help="random seed; set <0 for auto-seed each run")

    # dynamics strength
    p.add_argument("--phi_scale", type=float, default=0.2, help="mean network strength")
    p.add_argument("--vol_scale", type=float, default=0.1, help="volatility network strength")

    # innovation
    p.add_argument("--innovation", type=str, choices=["normal", "t"], default="t")
    p.add_argument("--t_df", type=float, default=8.0)

    # mu input / outputs
    p.add_argument("--mu_excel", type=str, default="", help="path to Excel file containing mu (square matrix)")
    p.add_argument("--out_dir", type=str, default="./outputs", help="output directory")
    p.add_argument("--base_name", type=str, default="demo", help="base filename for saved outputs")

    # naming
    p.add_argument("--append_seed", action="store_true",
                   help="append _seed<SEED> to base_name for this run (recommended for many runs)")
    p.add_argument("--no_append_seed", action="store_true",
                   help="force keeping the original base name")

    # outliers
    p.add_argument("--outlier_k", type=float, default=1.5, help="IQR multiplier for boxplot outliers")
    p.add_argument("--no_save_outliers", action="store_true", help="skip saving the outlier mask CSV")

    # Phi saving
    p.add_argument("--no_save_phi", action="store_true", help="skip saving the Phi matrix file")
    p.add_argument("--phi_fmt", type=str, choices=["xlsx", "csv"], default="xlsx", help="file format for Phi")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # choose seed
    seed = int(args.seed)
    if seed < 0:
        seed = _auto_seed()

    # load mu if given
    mu: Optional[np.ndarray] = None
    if args.mu_excel.strip():
        mu = load_mu_excel(args.mu_excel.strip())
        n = mu.shape[0]
    else:
        n = args.n

    # decide output base_name
    base_name = args.base_name
    if args.append_seed and (not args.no_append_seed):
        base_name = f"{base_name}_seed{seed}"

    # simulate
    y, params = simulate_var_cross_garch(
        n=n,
        T=args.T,
        mu=mu,
        phi_scale=args.phi_scale,
        vol_scale=args.vol_scale,
        innovation=args.innovation,
        t_df=args.t_df,
        burnin=args.burnin,
        seed=seed,
    )

    # outliers
    mask, _, _ = boxplot_outliers(y, k=args.outlier_k)
    out_rate = float(mask.mean())

    os.makedirs(args.out_dir, exist_ok=True)

    # save y + params + outliers
    save_outputs(
        args.out_dir,
        y,
        params,
        base_name=base_name,
        outlier_mask=(None if args.no_save_outliers else mask),
    )

    # save Phi separately (easy to inspect)
    phi_path = None
    if not args.no_save_phi:
        phi = np.asarray(params.get("phi"))
        phi_path = os.path.join(args.out_dir, f"{base_name}_Phi.{args.phi_fmt}")
        _save_matrix(phi_path, phi)

    # save a small metadata json (seed, args, outlier rate, timestamp)
    meta = {
        "base_name": base_name,
        "seed": seed,
        "n": int(n),
        "T": int(args.T),
        "burnin": int(args.burnin),
        "phi_scale": float(args.phi_scale),
        "vol_scale": float(args.vol_scale),
        "innovation": str(args.innovation),
        "t_df": float(args.t_df),
        "outlier_k": float(args.outlier_k),
        "outlier_rate": float(out_rate),
        "timestamp_unix": time.time(),
    }
    meta_path = os.path.join(args.out_dir, f"{base_name}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # prints
    print(f"[OK] Saved to: {os.path.abspath(args.out_dir)}")
    print(f"Seed used: {seed}")
    print(f"y shape: {y.shape}, outlier rate: {out_rate:.6f}")

    files = [f"{base_name}_y.csv", f"{base_name}_params.npz", f"{base_name}_meta.json"]
    if not args.no_save_outliers:
        files.append(f"{base_name}_outliers.csv")
    if phi_path is not None:
        files.append(os.path.basename(phi_path))
    print("Files:", ", ".join(files))


if __name__ == "__main__":
    main()
