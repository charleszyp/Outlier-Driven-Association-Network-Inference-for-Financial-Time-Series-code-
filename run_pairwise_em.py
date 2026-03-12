"""
Command-line runner for pairwise EM-based inference on a binary outlier matrix.

Outputs three matrices for lag τ:
- Pcond: observable lagged conditional probability
- Pedge: EM attribution probability
- Peff:  effective score = Pcond * Pedge
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pairwise_em_core import reconstruct_pairwise_all


DEFAULT_INPUT = "./outputs/demo_outliers.csv"          # e.g. "./outputs/demo_outliers.csv"
DEFAULT_LAG = 1
DEFAULT_OUT_DIR = "./binstat_outputs"
DEFAULT_FMT = "xlsx"        # "xlsx" or "csv"


def _auto_drop_nonbinary_first_col(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] <= 1:
        return df
    v0 = df.iloc[:, 0].dropna().unique()
    if not set(v0).issubset({0, 1, True, False}):
        return df.iloc[:, 1:]
    return df


def load_binary_matrix(path: str) -> np.ndarray:
    path = str(path)
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    elif path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    df = _auto_drop_nonbinary_first_col(df)
    return df.to_numpy()


def save_matrix(path: str, M: np.ndarray, node_prefix: str = "Node") -> None:
    df = pd.DataFrame(
        M,
        index=[f"{node_prefix}{i+1}" for i in range(M.shape[0])],
        columns=[f"{node_prefix}{i+1}" for i in range(M.shape[1])],
    )
    if path.lower().endswith(".csv"):
        df.to_csv(path, index=True)
    else:
        df.to_excel(path, index=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=DEFAULT_INPUT)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--lag", type=int, default=DEFAULT_LAG)
    p.add_argument("--fmt", type=str, choices=["csv", "xlsx"], default=DEFAULT_FMT)

    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--max_iter", type=int, default=2000)
    p.add_argument("--init_p", type=float, default=0.5)
    p.add_argument("--init_eps", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input:
        raise ValueError('Please provide --input, e.g. --input "./outputs/demo_outliers.csv"')

    X = load_binary_matrix(args.input)
    os.makedirs(args.out_dir, exist_ok=True)

    Pcond, Pedge, Peff, info = reconstruct_pairwise_all(
        X,
        lag=args.lag,
        tol=args.tol,
        max_iter=args.max_iter,
        init_p=args.init_p,
        init_eps=args.init_eps,
        show_progress=True,
    )



    # Sanity checks: Pcond must be in [0,1]
    mn, mx = float(np.nanmin(Pcond)), float(np.nanmax(Pcond))
    if mn < -1e-9 or mx > 1.0 + 1e-9:
        raise ValueError(f"Pcond out of range [0,1]. min={mn}, max={mx}. "
                         f"This indicates input is not binary or there is a bug.")
    base = Path(args.input).stem
    out_pcond = os.path.join(args.out_dir, f"{base}_Pcond_lag{args.lag}.{args.fmt}")
    out_pedge = os.path.join(args.out_dir, f"{base}_Pedge_lag{args.lag}.{args.fmt}")
    out_peff  = os.path.join(args.out_dir, f"{base}_Peff_lag{args.lag}.{args.fmt}")

    save_matrix(out_pcond, Pcond)
    save_matrix(out_pedge, Pedge)
    save_matrix(out_peff, Peff)

    info_df = pd.DataFrame({
        "Node": [f"Node{i+1}" for i in range(Pedge.shape[0])],
        "iters": info["iters"],
        "epsilon": info["epsilon"],
        "M_used": info["M"],
    })
    out_info = os.path.join(args.out_dir, f"{base}_info_lag{args.lag}.csv")
    info_df.to_csv(out_info, index=False)

    print(f"[OK] Saved to: {os.path.abspath(args.out_dir)}")
    print("Outputs:")
    print(" -", os.path.abspath(out_pcond))
    print(" -", os.path.abspath(out_pedge))
    print(" -", os.path.abspath(out_peff))
    print(" -", os.path.abspath(out_info))


if __name__ == "__main__":
    main()
