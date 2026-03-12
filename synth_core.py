# -*- coding: utf-8 -*-
"""
Core utilities for generating synthetic multivariate financial time series.

Main model:
- VAR(1) mean dynamics
- cross-GARCH(1,1) volatility dynamics

This file only contains reusable core functions. There is no CLI logic here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal

import numpy as np
import pandas as pd

Array = np.ndarray
Innovation = Literal["normal", "t"]


def spectral_radius(A: Array) -> float:
    """Largest absolute eigenvalue (spectral radius)."""
    vals = np.linalg.eigvals(A)
    return float(np.max(np.abs(vals)))


def scale_to_spectral_radius(A: Array, target_radius: float = 0.98) -> Tuple[Array, float]:
    """
    Scale matrix A so that spectral_radius(A) <= target_radius.
    Returns (A_scaled, scale_factor).
    """
    r = spectral_radius(A)
    if r <= 0 or r <= target_radius:
        return A.copy(), 1.0
    s = target_radius / r
    return A * s, float(s)


def make_sparse_mu(
    n: int,
    p_nonzero: float = 0.2,
    weight_scale: float = 0.05,
    diag: float = 0.1,
    seed: int = 1,
) -> Array:
    """
    Create a sparse VAR(1) coefficient template mu (n x n):
      - Off-diagonal entries: random uniform in [-weight_scale, weight_scale] with sparsity p_nonzero
      - Diagonal entries: fixed `diag`
    """
    rng = np.random.default_rng(seed)
    mu = rng.uniform(-weight_scale, weight_scale, size=(n, n))
    mask = rng.random((n, n)) < p_nonzero
    np.fill_diagonal(mask, False)
    mu = mu * mask
    np.fill_diagonal(mu, diag)
    return mu


@dataclass
class CrossGarchConfig:
    """Config for generating cross-GARCH parameters."""
    alpha0_min: float = 1e-4
    alpha0_max: float = 1e-2
    persistence: float = 0.95     # volatility clustering (alpha+beta) total per row
    cross_share: float = 0.05     # fraction of mass assigned to cross terms (off-diagonal)
    seed: int = 1


def make_cross_garch_params(n: int, cfg: CrossGarchConfig) -> Tuple[Array, Array, Array]:
    """
    Build (alpha0, alpha, beta) for cross-GARCH:
        sigma2_i(t) = alpha0_i + sum_j alpha_ij * gamma_j(t-1)^2 + sum_j beta_ij * sigma2_j(t-1)
    Row sums of (alpha+beta) are about cfg.persistence (before any further scaling).
    """
    rng = np.random.default_rng(cfg.seed)
    alpha0 = rng.uniform(cfg.alpha0_min, cfg.alpha0_max, size=n)

    alpha = np.zeros((n, n), dtype=float)
    beta = np.zeros((n, n), dtype=float)

    alpha_share = 0.08  # typical: alpha small, beta large

    for i in range(n):
        total = float(cfg.persistence)
        total_alpha = total * alpha_share
        total_beta = total * (1.0 - alpha_share)

        cross_total_alpha = total_alpha * cfg.cross_share
        cross_total_beta = total_beta * cfg.cross_share

        diag_alpha = total_alpha - cross_total_alpha
        diag_beta = total_beta - cross_total_beta

        alpha[i, i] = diag_alpha
        beta[i, i] = diag_beta

        if n > 1 and (cross_total_alpha > 0 or cross_total_beta > 0):
            w = rng.random(n)
            w[i] = 0.0
            s = float(w.sum())
            if s > 0:
                w /= s
                alpha[i, :] += cross_total_alpha * w
                beta[i, :] += cross_total_beta * w

    return alpha0, alpha, beta


def _standardize_t(rng: np.random.Generator, df: float, size: Tuple[int, ...]) -> Array:
    """Student-t with unit variance (when df>2)."""
    x = rng.standard_t(df, size=size)
    if df > 2:
        x = x / math.sqrt(df / (df - 2))
    return x


def simulate_var_cross_garch(
    n: int,
    T: int,
    *,
    mu: Optional[Array] = None,
    # mean dynamics strength
    phi_scale: float = 0.1,
    # volatility coupling strength
    vol_scale: float = 0.1,
    # garch params (if None, auto-generate via garch_cfg)
    alpha0: Optional[Array] = None,
    alpha: Optional[Array] = None,
    beta: Optional[Array] = None,
    garch_cfg: Optional[CrossGarchConfig] = None,
    # innovations
    innovation: Innovation = "t",
    t_df: float = 8.0,
    # simulation controls
    burnin: int = 300,
    y0: Optional[Array] = None,
    seed: int = 1,
    # stability targets / safety
    target_var_radius: float = 0.98,
    target_vol_row_sum: float = 0.99,
    clip_sigma2: Tuple[float, float] = (1e-12, 1e6),
    clip_gamma: float = 1e3,
) -> Tuple[Array, Dict[str, Array]]:
    """
    Simulate y in R^{T x n}:

        y[t]      = phi @ y[t-1] + gamma[t]
        sigma2[t] = alpha0 + vol_scale * (alpha @ gamma[t-1]^2 + beta @ sigma2[t-1])
        gamma[t]  = sqrt(sigma2[t]) * eps[t]

    Notes:
    - `phi_scale` controls mean-network strength (VAR coupling)
    - `vol_scale` controls volatility-network strength (cross-GARCH coupling)
    - We enforce stability: spectral_radius(phi) <= target_var_radius (via scaling)
    - We enforce volatility stability: vol_scale * max_row_sum(alpha+beta) < target_vol_row_sum

    Returns:
      y_out: (T, n) after burn-in
      params: dict of used parameters and latent arrays (sigma2, gamma)
    """
    if n <= 0 or T <= 0:
        raise ValueError("n and T must be positive.")

    rng = np.random.default_rng(seed)

    # --- mu / VAR stability ---
    if mu is None:
        mu = make_sparse_mu(n, p_nonzero=0.25, weight_scale=0.05, diag=0.1, seed=seed)
    mu = np.asarray(mu, dtype=float)
    if mu.shape != (n, n):
        raise ValueError(f"mu must be shape {(n, n)}, got {mu.shape}")

    phi = phi_scale * mu
    phi_scaled, s_phi = scale_to_spectral_radius(phi, target_radius=target_var_radius)
    if s_phi != 1.0:
        # adjust mu consistently (so mu stays interpretable as the "template")
        mu = phi_scaled / max(phi_scale, 1e-12)
        phi = phi_scaled

    # --- GARCH params / stability ---
    if (alpha0 is None) or (alpha is None) or (beta is None):
        if garch_cfg is None:
            garch_cfg = CrossGarchConfig(seed=seed)
        alpha0, alpha, beta = make_cross_garch_params(n, garch_cfg)

    alpha0 = np.asarray(alpha0, dtype=float).reshape(-1)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if alpha0.shape != (n,) or alpha.shape != (n, n) or beta.shape != (n, n):
        raise ValueError("alpha0/alpha/beta have incompatible shapes.")

    # scale rows so vol_scale*row_sum < target
    row_sum = (alpha + beta).sum(axis=1)
    max_row = float(np.max(row_sum)) if float(np.max(row_sum)) > 0 else 1e-12
    if vol_scale * max_row >= target_vol_row_sum:
        s = target_vol_row_sum / (vol_scale * max_row)
        alpha *= s
        beta *= s
        row_sum = (alpha + beta).sum(axis=1)

    # --- allocate with burn-in ---
    TT = T + burnin
    y = np.zeros((TT, n), dtype=float)
    sigma2 = np.zeros((TT, n), dtype=float)
    gamma = np.zeros((TT, n), dtype=float)

    y[0] = rng.uniform(-1.0, 1.0, size=n) if y0 is None else np.asarray(y0, dtype=float).reshape(-1)

    # init sigma2 approx unconditional
    denom = 1.0 - vol_scale * row_sum
    denom = np.clip(denom, 1e-6, None)
    sigma2[0] = np.clip(alpha0 / denom, clip_sigma2[0], clip_sigma2[1])

    for t in range(1, TT):
        if innovation == "normal":
            eps = rng.normal(0.0, 1.0, size=n)
        elif innovation == "t":
            eps = _standardize_t(rng, t_df, size=(n,))
        else:
            raise ValueError("innovation must be 'normal' or 't'")

        gamma2_prev = gamma[t - 1] ** 2
        sigma2_t = alpha0 + vol_scale * (alpha @ gamma2_prev + beta @ sigma2[t - 1])
        sigma2_t = np.clip(sigma2_t, clip_sigma2[0], clip_sigma2[1])
        sigma2[t] = sigma2_t

        gamma_t = np.sqrt(sigma2_t) * eps
        gamma_t = np.clip(gamma_t, -clip_gamma, clip_gamma)
        gamma[t] = gamma_t

        y[t] = phi @ y[t - 1] + gamma_t

    y_out = y[burnin:]
    sigma2_out = sigma2[burnin:]
    gamma_out = gamma[burnin:]

    params: Dict[str, Array] = {
        "mu": mu,
        "phi": phi,
        "alpha0": alpha0,
        "alpha": alpha,
        "beta": beta,
        "sigma2": sigma2_out,
        "gamma": gamma_out,
        "phi_scale": np.array([phi_scale], dtype=float),
        "vol_scale": np.array([vol_scale], dtype=float),
        "innovation_df": np.array([t_df], dtype=float),
    }
    return y_out, params


def boxplot_outliers(y: Array, k: float = 1.5) -> Tuple[Array, Array, Array]:
    """
    Boxplot/IQR outlier mask per dimension.
    Returns (mask, lower, upper) where mask is (T,n) boolean.
    """
    y = np.asarray(y, dtype=float)
    q1 = np.quantile(y, 0.25, axis=0)
    q3 = np.quantile(y, 0.75, axis=0)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = (y < lower) | (y > upper)
    return mask, lower, upper


def to_dataframe(
    y: Array,
    start: str = "2020-01-01",
    freq: str = "D",
    col_prefix: str = "Series_",
) -> pd.DataFrame:
    """Convert y (T,n) to a DataFrame with a Time column."""
    T, n = y.shape
    dates = pd.date_range(start=start, periods=T, freq=freq)
    df = pd.DataFrame(y, columns=[f"{col_prefix}{i+1}" for i in range(n)])
    df.insert(0, "Time", dates)
    return df


def save_outputs(
    out_dir: str,
    y: Array,
    params: Dict[str, Array],
    *,
    base_name: str = "synthetic",
    outlier_mask: Optional[Array] = None,
) -> None:
    """Save y to CSV and params to NPZ; optionally save outlier mask to CSV."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    df = to_dataframe(y)
    df.to_csv(os.path.join(out_dir, f"{base_name}_y.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, f"{base_name}_params.npz"), **params)

    if outlier_mask is not None:
        out_df = pd.DataFrame(outlier_mask.astype(int), columns=[f"Series_{i+1}" for i in range(outlier_mask.shape[1])])
        out_df.insert(0, "Time", df["Time"].values)
        out_df.to_csv(os.path.join(out_dir, f"{base_name}_outliers.csv"), index=False)


__all__ = [
    "CrossGarchConfig",
    "simulate_var_cross_garch",
    "boxplot_outliers",
    "save_outputs",
    "to_dataframe",
    "make_sparse_mu",
    "make_cross_garch_params",
]
