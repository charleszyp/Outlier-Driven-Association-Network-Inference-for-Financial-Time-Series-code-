"""
Pairwise EM-based inference for binary outlier-activation matrices S ∈ {0,1}^{T×N}.

Outputs for lag τ:
1. Pcond[j, i] = observable lagged conditional probability
2. Pedge[j, i] = EM attribution / direct-trigger probability
3. Peff[j, i]  = Pcond * Pedge

Convention:
- rows = target node/channel j
- columns = source node/channel i
- diagonal = 0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

Array = np.ndarray


def _as_binary_matrix(S: Array) -> Array:
    S = np.asarray(S)
    if S.ndim != 2:
        raise ValueError(f"S must be 2D, got shape {S.shape}")
    uniq = np.unique(S)
    if not np.all(np.isin(uniq, [0, 1, False, True])):
        raise ValueError(f"S must be binary (0/1). Found unique values: {uniq[:10]}")
    return S.astype(np.int8, copy=False)


def _safe_div(num: Array, den: Array, eps: float = 1e-12) -> Array:
    return num / (den + eps)


@dataclass
class PairwiseEMConfig:
    lag: int = 1
    tol: float = 1e-6
    max_iter: int = 2000
    init_p: float = 0.5
    init_eps: float = 0.2
    eps_floor: float = 1e-9
    eps_ceiling: float = 0.999999
    clip_p: Tuple[float, float] = (0.0, 1.0)
    tiny: float = 1e-12


def compute_P_i_given_j0(
    S: Array,
    target_j: int,
    *,
    lag: int = 1,
    tiny: float = 1e-12,
) -> Tuple[Array, Array, Array]:
    """
    Vectorized computation of P_i^j(τ):

        P_i^j(τ) = P( s_j(t+τ)=1 | s_j(t)=0, s_i(t)=1 )

    Returns:
      pid: indices of candidate parents i (0-based), shape (N-1,)
      P_i_j: conditional probabilities aligned with pid, shape (N-1,)
      denom: counts of occurrences {s_j(t)=0 and s_i(t)=1}, shape (N-1,)
    """
    S = _as_binary_matrix(S)
    T, N = S.shape
    if not (0 <= target_j < N):
        raise ValueError("target_j out of range")
    if lag < 1 or lag >= T:
        raise ValueError(f"lag must be in [1, T-1], got lag={lag}, T={T}")

    pid = np.delete(np.arange(N), target_j)
    S0 = S[: T - lag]
    y_next = S[lag:, target_j]

    # IMPORTANT: cast to int64 before any dot-products to prevent overflow
    j0 = (S0[:, target_j] == 0).astype(np.int64)     # (T-lag,)
    X = S0[:, pid].astype(np.int64)                  # (T-lag, N-1)
    y_next64 = y_next.astype(np.int64)

    denom = X.T @ j0                                 # int64 counts
    numer = X.T @ (j0 * y_next64)                    # int64 counts

    P_i_j = _safe_div(numer.astype(float), denom.astype(float), eps=tiny)

    # sanity clip to [0,1] to guard any tiny numeric noise
    P_i_j = np.clip(P_i_j, 0.0, 1.0)
    return pid, P_i_j, denom.astype(float)


def emscr_pairwise_target(
    S: Array,
    target_j: int,
    cfg: PairwiseEMConfig = PairwiseEMConfig(),
) -> Tuple[Array, Array, float, int, Array, int]:
    """
    Pairwise EM for a single target j (0-based).

    Returns:
      pid: candidate parent indices (0-based), length N-1
      P_ij: estimated P_{i->j} aligned with pid
      epsilon_j: spontaneous activation probability
      k: iterations used
      P_i_j: observable P_i^j(τ) aligned with pid
      M: number of time points used (those with s_j(t)=0)
    """
    S = _as_binary_matrix(S)
    T, N = S.shape
    lag = cfg.lag
    if lag < 1 or lag >= T:
        raise ValueError(f"lag must be in [1, T-1], got lag={lag}, T={T}")
    if not (0 <= target_j < N):
        raise ValueError("target_j out of range")

    pid, P_i_j, denom_counts = compute_P_i_given_j0(S, target_j, lag=lag, tiny=cfg.tiny)

    valid_t = np.where(S[: T - lag, target_j] == 0)[0]
    M = int(valid_t.size)
    if M == 0:
        return pid, np.zeros_like(pid, dtype=float), 0.0, 0, P_i_j, 0

    X = S[valid_t[:, None], pid].astype(np.int8)      # (M, N-1)
    B = S[valid_t + lag, target_j].astype(np.int8)    # (M,)

    identifiable = denom_counts > cfg.tiny
    if not np.any(identifiable):
        eps_hat = float(B.mean())
        return pid, np.zeros_like(pid, dtype=float), eps_hat, 0, P_i_j, M

    pid_sub = pid[identifiable]
    P_i_j_sub = P_i_j[identifiable]
    X_sub = X[:, identifiable]

    P_ij_sub = np.full(pid_sub.shape[0], float(cfg.init_p), dtype=float)
    epsilon_j = float(cfg.init_eps)

    denom_m = (X_sub.astype(float) * P_i_j_sub[None, :]).sum(axis=0).astype(float)
    denom_m = np.maximum(denom_m, cfg.tiny)

    k = 0
    delta = np.inf
    while (delta > cfg.tol) and (k < cfg.max_iter):
        k += 1
        P_old = P_ij_sub.copy()
        eps_old = epsilon_j

        w = P_ij_sub * P_i_j_sub
        denom_e = (X_sub.astype(float) @ w) + epsilon_j
        denom_e = np.maximum(denom_e, cfg.tiny)

        rho_i = (X_sub.astype(float) * w[None, :]) / denom_e[:, None]
        rho_eps = epsilon_j / denom_e

        numer = (B.astype(float)[:, None] * rho_i).sum(axis=0)
        P_ij_sub = _safe_div(numer, denom_m, eps=cfg.tiny)
        P_ij_sub = np.clip(P_ij_sub, cfg.clip_p[0], cfg.clip_p[1])

        epsilon_j = float((B.astype(float) * rho_eps).sum() / max(M, 1))
        epsilon_j = float(np.clip(epsilon_j, cfg.eps_floor, cfg.eps_ceiling))

        delta = float(np.sum(np.abs(P_ij_sub - P_old)) + abs(epsilon_j - eps_old))

    P_ij = np.zeros_like(pid, dtype=float)
    P_ij[identifiable] = P_ij_sub
    return pid, P_ij, epsilon_j, k, P_i_j, M


def reconstruct_pairwise_all(
    S: Array,
    *,
    lag: int = 1,
    tol: float = 1e-6,
    max_iter: int = 2000,
    init_p: float = 0.5,
    init_eps: float = 0.2,
    show_progress: bool = True,
) -> Tuple[Array, Array, Array, Dict[str, Array]]:
    """
    Reconstruct three matrices for lag τ:
      Pcond[j,i] = P_i^j(τ)
      Pedge[j,i] = P_{i->j}
      Peff [j,i] = Pcond * Pedge
    """
    S = _as_binary_matrix(S)
    _, N = S.shape

    cfg = PairwiseEMConfig(lag=lag, tol=tol, max_iter=max_iter, init_p=init_p, init_eps=init_eps)

    Pcond = np.zeros((N, N), dtype=float)
    Pedge = np.zeros((N, N), dtype=float)

    iters = np.zeros(N, dtype=int)
    eps = np.zeros(N, dtype=float)
    M_used = np.zeros(N, dtype=int)

    nodes: Iterable[int] = range(N)
    if show_progress:
        try:
            from tqdm import tqdm
            nodes = tqdm(nodes, desc=f"Pairwise BSI/EM (lag={lag})", total=N)
        except Exception:
            pass

    for j in nodes:
        pid, P_ij, eps_j, k, P_i_j, M = emscr_pairwise_target(S, j, cfg)
        Pcond[j, pid] = P_i_j
        Pedge[j, pid] = P_ij
        iters[j] = k
        eps[j] = eps_j
        M_used[j] = M

    np.fill_diagonal(Pcond, 0.0)
    np.fill_diagonal(Pedge, 0.0)

    Peff = Pcond * Pedge
    np.fill_diagonal(Peff, 0.0)

    info = {"iters": iters, "epsilon": eps, "M": M_used}
    return Pcond, Pedge, Peff, info


__all__ = [
    "PairwiseEMConfig",
    "compute_P_i_given_j0",
    "emscr_pairwise_target",
    "reconstruct_pairwise_all",
]
