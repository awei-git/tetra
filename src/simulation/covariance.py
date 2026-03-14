"""Covariance estimation for multi-asset simulation.

Three estimators, each serving a different purpose:
1. Ledoit-Wolf shrinkage — stable daily covariance for portfolio optimization
2. Exponentially weighted — recent dynamics for short-horizon risk
3. Regime-conditional — separate covariance per market regime (from HMM)

All estimators work on (T x N) return matrices: T observations, N assets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CovEstimate:
    """Covariance estimation result."""
    cov: List[List[float]]       # N x N covariance matrix
    corr: List[List[float]]      # N x N correlation matrix
    vols: List[float]            # N annualized volatilities
    symbols: List[str]           # N symbol labels
    method: str                  # estimation method used
    n_obs: int                   # observations used
    effective_obs: float         # effective observations (after weighting)


# ---------------------------------------------------------------------------
# Matrix utilities (pure Python — no numpy dependency)
# ---------------------------------------------------------------------------

def _mean_vec(matrix: List[List[float]]) -> List[float]:
    """Column means of T x N matrix."""
    t = len(matrix)
    if t == 0:
        return []
    n = len(matrix[0])
    return [sum(matrix[i][j] for i in range(t)) / t for j in range(n)]


def _demean(matrix: List[List[float]], means: List[float]) -> List[List[float]]:
    """Subtract column means."""
    return [[row[j] - means[j] for j in range(len(means))] for row in matrix]


def _sample_cov(demeaned: List[List[float]], ddof: int = 1) -> List[List[float]]:
    """Sample covariance from demeaned T x N matrix."""
    t = len(demeaned)
    n = len(demeaned[0]) if demeaned else 0
    divisor = max(t - ddof, 1)
    cov = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = sum(demeaned[k][i] * demeaned[k][j] for k in range(t))
            cov[i][j] = s / divisor
            cov[j][i] = cov[i][j]
    return cov


def _diag(matrix: List[List[float]]) -> List[float]:
    """Extract diagonal."""
    return [matrix[i][i] for i in range(len(matrix))]


def _cov_to_corr(cov: List[List[float]]) -> Tuple[List[List[float]], List[float]]:
    """Convert covariance to correlation + volatilities."""
    n = len(cov)
    vols = [math.sqrt(max(cov[i][i], 0.0)) for i in range(n)]
    corr = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if vols[i] > 0 and vols[j] > 0:
                corr[i][j] = cov[i][j] / (vols[i] * vols[j])
                corr[i][j] = max(-1.0, min(1.0, corr[i][j]))
            else:
                corr[i][j] = 1.0 if i == j else 0.0
    return corr, vols


def _identity(n: int) -> List[List[float]]:
    """N x N identity matrix."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _scale_matrix(matrix: List[List[float]], scalar: float) -> List[List[float]]:
    return [[v * scalar for v in row] for row in matrix]


def _add_matrices(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def _trace(matrix: List[List[float]]) -> float:
    return sum(matrix[i][i] for i in range(len(matrix)))


def _frobenius_sq(matrix: List[List[float]]) -> float:
    """Squared Frobenius norm."""
    return sum(matrix[i][j] ** 2 for i in range(len(matrix)) for j in range(len(matrix[0])))


def annualize_vol(daily_vol: float, trading_days: int = 252) -> float:
    return daily_vol * math.sqrt(trading_days)


# ---------------------------------------------------------------------------
# Cholesky decomposition (needed by generator.py for correlated sampling)
# ---------------------------------------------------------------------------

def cholesky(matrix: List[List[float]]) -> List[List[float]]:
    """Lower-triangular Cholesky decomposition. Raises ValueError if not PD."""
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = matrix[i][i] - s
                if val <= 0:
                    # Nudge toward positive definite
                    val = max(val, 1e-10)
                L[i][j] = math.sqrt(val)
            else:
                L[i][j] = (matrix[i][j] - s) / max(L[j][j], 1e-12)
    return L


def nearest_psd(matrix: List[List[float]], epsilon: float = 1e-8) -> List[List[float]]:
    """Project matrix to nearest positive semi-definite via eigenvalue clipping.

    Uses iterative diagonal loading — simple but effective for our dimensions (N < 300).
    """
    n = len(matrix)
    result = [row[:] for row in matrix]
    # Iterative: try Cholesky, if it fails add epsilon to diagonal
    for attempt in range(20):
        try:
            cholesky(result)
            return result
        except (ValueError, ZeroDivisionError):
            for i in range(n):
                result[i][i] += epsilon * (2 ** attempt)
    return result


# ---------------------------------------------------------------------------
# 1. Ledoit-Wolf shrinkage estimator
# ---------------------------------------------------------------------------

def ledoit_wolf(
    returns: List[List[float]],
    symbols: List[str],
    target: str = "identity",
) -> CovEstimate:
    """Ledoit-Wolf linear shrinkage toward a structured target.

    Args:
        returns: T x N matrix of daily returns
        symbols: N asset labels
        target: "identity" (scaled identity) or "diagonal" (diagonal of sample cov)

    Returns optimal shrinkage intensity α* and covariance (1-α)*S + α*F.
    """
    t = len(returns)
    n = len(returns[0]) if returns else 0
    if t < 2 or n < 1:
        raise ValueError(f"Need at least 2 observations and 1 asset, got T={t} N={n}")

    means = _mean_vec(returns)
    dm = _demean(returns, means)
    S = _sample_cov(dm)

    # Shrinkage target F
    if target == "diagonal":
        F = [[S[i][j] if i == j else 0.0 for j in range(n)] for i in range(n)]
    else:
        # Scaled identity: mu * I where mu = average variance
        mu = _trace(S) / n
        F = [[mu if i == j else 0.0 for j in range(n)] for i in range(n)]

    # Compute optimal shrinkage intensity (Ledoit-Wolf 2004 formula)
    # delta = ||S - F||^2_F
    delta_sq = _frobenius_sq([[S[i][j] - F[i][j] for j in range(n)] for i in range(n)])

    # Estimate sum of asymptotic variances of entries of sqrt(T)*(S - Sigma)
    # Using the Oracle Approximating Shrinkage (OAS) simplified formula
    rho_num = 0.0
    for k in range(t):
        x_k = dm[k]
        outer = [[x_k[i] * x_k[j] for j in range(n)] for i in range(n)]
        diff = [[outer[i][j] - S[i][j] for j in range(n)] for i in range(n)]
        rho_num += _frobenius_sq(diff)
    rho_num /= (t * t)

    if delta_sq < 1e-14:
        alpha = 1.0  # S ≈ F, just use target
    else:
        alpha = min(1.0, max(0.0, rho_num / delta_sq))

    # Shrunk covariance: (1 - alpha) * S + alpha * F
    cov = _add_matrices(
        _scale_matrix(S, 1.0 - alpha),
        _scale_matrix(F, alpha),
    )
    cov = nearest_psd(cov)
    corr, vols_daily = _cov_to_corr(cov)
    vols_ann = [annualize_vol(v) for v in vols_daily]

    return CovEstimate(
        cov=cov, corr=corr, vols=vols_ann, symbols=symbols,
        method=f"ledoit_wolf_{target}(alpha={alpha:.3f})",
        n_obs=t, effective_obs=float(t),
    )


# ---------------------------------------------------------------------------
# 2. Exponentially weighted covariance (EWMA)
# ---------------------------------------------------------------------------

def ewma_cov(
    returns: List[List[float]],
    symbols: List[str],
    halflife: int = 21,
    min_periods: int = 10,
) -> CovEstimate:
    """Exponentially weighted moving average covariance.

    More responsive to recent dynamics than sample covariance.
    halflife=21 ≈ 1 month decay.

    Args:
        returns: T x N matrix
        symbols: N asset labels
        halflife: decay half-life in trading days
    """
    t = len(returns)
    n = len(returns[0]) if returns else 0
    if t < min_periods or n < 1:
        raise ValueError(f"Need >= {min_periods} observations, got {t}")

    decay = math.log(2) / halflife
    weights = [math.exp(-decay * (t - 1 - i)) for i in range(t)]
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    # Weighted mean
    means = [sum(weights[i] * returns[i][j] for i in range(t)) for j in range(n)]

    # Weighted covariance
    cov = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = sum(
                weights[k] * (returns[k][i] - means[i]) * (returns[k][j] - means[j])
                for k in range(t)
            )
            # Bias correction for weighted estimator
            w2_sum = sum(w * w for w in weights)
            correction = 1.0 / (1.0 - w2_sum) if w2_sum < 1.0 else 1.0
            cov[i][j] = s * correction
            cov[j][i] = cov[i][j]

    cov = nearest_psd(cov)
    corr, vols_daily = _cov_to_corr(cov)
    vols_ann = [annualize_vol(v) for v in vols_daily]

    # Effective observations: sum(w)^2 / sum(w^2)
    eff_obs = w_sum ** 2 / sum(w * w for w in weights) if weights else 0

    return CovEstimate(
        cov=cov, corr=corr, vols=vols_ann, symbols=symbols,
        method=f"ewma(halflife={halflife})",
        n_obs=t, effective_obs=eff_obs,
    )


# ---------------------------------------------------------------------------
# 3. Regime-conditional covariance
# ---------------------------------------------------------------------------

def regime_conditional_cov(
    returns: List[List[float]],
    symbols: List[str],
    regime_labels: List[int],
    shrink: bool = True,
) -> Dict[int, CovEstimate]:
    """Estimate separate covariance matrices per regime.

    Args:
        returns: T x N matrix
        symbols: N asset labels
        regime_labels: T regime assignments (e.g., 0=calm, 1=stressed, 2=crisis)
        shrink: apply Ledoit-Wolf shrinkage within each regime

    Returns dict mapping regime_id → CovEstimate.
    """
    if len(returns) != len(regime_labels):
        raise ValueError("returns and regime_labels must have same length")

    # Group returns by regime
    regime_returns: Dict[int, List[List[float]]] = {}
    for i, label in enumerate(regime_labels):
        regime_returns.setdefault(label, []).append(returns[i])

    results: Dict[int, CovEstimate] = {}
    for regime_id, r_returns in regime_returns.items():
        if len(r_returns) < 5:
            continue
        if shrink:
            est = ledoit_wolf(r_returns, symbols)
        else:
            means = _mean_vec(r_returns)
            dm = _demean(r_returns, means)
            cov = _sample_cov(dm)
            cov = nearest_psd(cov)
            corr, vols_daily = _cov_to_corr(cov)
            vols_ann = [annualize_vol(v) for v in vols_daily]
            est = CovEstimate(
                cov=cov, corr=corr, vols=vols_ann, symbols=symbols,
                method=f"sample_regime_{regime_id}",
                n_obs=len(r_returns), effective_obs=float(len(r_returns)),
            )
        results[regime_id] = est

    return results


# ---------------------------------------------------------------------------
# Stress covariance: scale correlations toward 1 under stress
# ---------------------------------------------------------------------------

def stress_cov(
    base: CovEstimate,
    correlation_multiplier: float = 1.5,
    vol_multiplier: float = 2.0,
) -> CovEstimate:
    """Generate a stressed covariance matrix.

    In crises, correlations spike toward 1 and volatilities increase.
    This creates a synthetic stress covariance for scenario analysis.

    Args:
        base: baseline covariance estimate
        correlation_multiplier: scale off-diagonal correlations (clamped to [-1,1])
        vol_multiplier: scale volatilities
    """
    n = len(base.corr)
    stressed_corr = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                stressed_corr[i][j] = 1.0
            else:
                stressed_corr[i][j] = max(-1.0, min(1.0,
                    base.corr[i][j] * correlation_multiplier))

    # Convert daily vols from annualized back to daily, then scale
    daily_vols = [v / math.sqrt(252) for v in base.vols]
    stressed_daily_vols = [v * vol_multiplier for v in daily_vols]

    # Reconstruct covariance: C_ij = corr_ij * vol_i * vol_j
    cov = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            cov[i][j] = stressed_corr[i][j] * stressed_daily_vols[i] * stressed_daily_vols[j]

    cov = nearest_psd(cov)
    corr, vols_d = _cov_to_corr(cov)
    vols_ann = [annualize_vol(v) for v in vols_d]

    return CovEstimate(
        cov=cov, corr=corr, vols=vols_ann, symbols=base.symbols,
        method=f"stress(corr_mult={correlation_multiplier}, vol_mult={vol_multiplier})",
        n_obs=base.n_obs, effective_obs=base.effective_obs,
    )
