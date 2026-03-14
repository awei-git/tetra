"""Hidden Markov Model regime detection from market observables.

Detects market regimes (calm / stressed / crisis) from observable features:
- Realized volatility (20d rolling)
- Return momentum (sign and magnitude)
- VIX level / credit spreads (if available)
- Correlation regime (average pairwise correlation)

Uses a Gaussian HMM with 3 states, fitted via Baum-Welch (EM).
Pure Python implementation — no hmmlearn dependency.

The key insight: regimes aren't just about volatility levels, they're about
the *joint dynamics* — in a crisis, vol is high AND correlations spike AND
momentum is negative AND credit spreads widen simultaneously.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class RegimeState:
    """Description of a detected regime."""
    label: str                   # "calm", "stressed", "crisis"
    index: int                   # 0, 1, 2
    mean_vol: float              # average annualized vol in this regime
    mean_return: float           # average daily return
    mean_corr: float             # average pairwise correlation
    duration_days: float         # average duration in days
    frequency: float             # fraction of time spent in this regime


@dataclass
class RegimeModel:
    """Fitted HMM regime model."""
    n_states: int
    transition_matrix: List[List[float]]   # K x K
    means: List[List[float]]               # K x D (state means)
    covs: List[List[List[float]]]          # K x D x D (state covariances)
    initial_probs: List[float]             # K
    states: List[RegimeState]              # K regime descriptions
    log_likelihood: float
    n_iter: int

    # Outputs
    regime_path: List[int] = field(default_factory=list)    # T Viterbi path
    regime_probs: List[List[float]] = field(default_factory=list)  # T x K smoothed probs


# ---------------------------------------------------------------------------
# Feature extraction from returns
# ---------------------------------------------------------------------------

def extract_regime_features(
    returns: List[List[float]],
    window: int = 20,
) -> List[List[float]]:
    """Extract regime-relevant features from T x N return matrix.

    Features per timestep:
    1. Cross-sectional realized volatility (annualized)
    2. Cross-sectional mean return (rolling window)
    3. Average pairwise correlation (rolling window)
    4. Return dispersion (cross-sectional std of returns)

    Returns (T - window + 1) x 4 feature matrix.
    """
    t = len(returns)
    n = len(returns[0]) if returns else 0
    if t < window or n < 2:
        raise ValueError(f"Need T >= {window} and N >= 2, got T={t} N={n}")

    features = []
    sqrt_252 = math.sqrt(252)

    for end in range(window, t + 1):
        chunk = returns[end - window:end]

        # 1. Realized vol: average across assets of rolling vol
        asset_vols = []
        for j in range(n):
            col = [chunk[i][j] for i in range(window)]
            mu = sum(col) / window
            var = sum((x - mu) ** 2 for x in col) / max(window - 1, 1)
            asset_vols.append(math.sqrt(var) * sqrt_252)
        avg_vol = sum(asset_vols) / n

        # 2. Mean return (cross-sectional average of rolling mean)
        asset_means = []
        for j in range(n):
            col = [chunk[i][j] for i in range(window)]
            asset_means.append(sum(col) / window)
        avg_return = sum(asset_means) / n

        # 3. Average pairwise correlation
        corr_sum = 0.0
        corr_count = 0
        for j1 in range(min(n, 20)):  # cap at 20 assets for speed
            for j2 in range(j1 + 1, min(n, 20)):
                col1 = [chunk[i][j1] for i in range(window)]
                col2 = [chunk[i][j2] for i in range(window)]
                mu1, mu2 = sum(col1) / window, sum(col2) / window
                cov = sum((col1[i] - mu1) * (col2[i] - mu2) for i in range(window))
                v1 = sum((x - mu1) ** 2 for x in col1)
                v2 = sum((x - mu2) ** 2 for x in col2)
                if v1 > 0 and v2 > 0:
                    corr_sum += cov / math.sqrt(v1 * v2)
                    corr_count += 1
        avg_corr = corr_sum / corr_count if corr_count > 0 else 0.0

        # 4. Return dispersion (cross-sectional std)
        today_returns = [returns[end - 1][j] for j in range(n)]
        xmean = sum(today_returns) / n
        dispersion = math.sqrt(sum((r - xmean) ** 2 for r in today_returns) / max(n - 1, 1))

        features.append([avg_vol, avg_return, avg_corr, dispersion])

    return features


# ---------------------------------------------------------------------------
# Gaussian HMM (Baum-Welch)
# ---------------------------------------------------------------------------

def _log_sum_exp(values: List[float]) -> float:
    """Numerically stable log-sum-exp."""
    if not values:
        return float("-inf")
    m = max(values)
    if m == float("-inf"):
        return float("-inf")
    return m + math.log(sum(math.exp(v - m) for v in values))


def _gaussian_log_pdf(x: List[float], mean: List[float], cov: List[List[float]]) -> float:
    """Log PDF of multivariate Gaussian. Uses diagonal approximation for speed."""
    d = len(x)
    log_det = sum(math.log(max(cov[i][i], 1e-12)) for i in range(d))
    quad = sum((x[i] - mean[i]) ** 2 / max(cov[i][i], 1e-12) for i in range(d))
    return -0.5 * (d * math.log(2 * math.pi) + log_det + quad)


def _init_params(
    features: List[List[float]],
    n_states: int,
    rng: random.Random,
) -> Tuple[List[float], List[List[float]], List[List[float]], List[List[List[float]]]]:
    """Initialize HMM parameters via k-means-style clustering."""
    t = len(features)
    d = len(features[0])

    # Sort by first feature (volatility) and split into n_states groups
    sorted_idx = sorted(range(t), key=lambda i: features[i][0])
    group_size = t // n_states

    means = []
    covs = []
    for s in range(n_states):
        start = s * group_size
        end = (s + 1) * group_size if s < n_states - 1 else t
        group = [features[sorted_idx[i]] for i in range(start, end)]
        gm = [sum(g[j] for g in group) / len(group) for j in range(d)]
        means.append(gm)
        # Diagonal covariance
        gc = [[0.0] * d for _ in range(d)]
        for j in range(d):
            var = sum((g[j] - gm[j]) ** 2 for g in group) / max(len(group) - 1, 1)
            gc[j][j] = max(var, 1e-6)
        covs.append(gc)

    # Initial state probs: uniform
    pi = [1.0 / n_states] * n_states

    # Transition matrix: high self-transition (regimes are persistent)
    trans = [[0.0] * n_states for _ in range(n_states)]
    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                trans[i][j] = 0.9
            else:
                trans[i][j] = 0.1 / max(n_states - 1, 1)

    return pi, trans, means, covs


def fit_hmm(
    features: List[List[float]],
    n_states: int = 3,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 42,
) -> RegimeModel:
    """Fit Gaussian HMM via Baum-Welch (EM) algorithm.

    Args:
        features: T x D observation matrix
        n_states: number of hidden states (default 3: calm/stressed/crisis)
        max_iter: maximum EM iterations
        tol: convergence tolerance on log-likelihood
        seed: random seed for initialization

    Returns fitted RegimeModel.
    """
    rng = random.Random(seed)
    t = len(features)
    d = len(features[0]) if features else 0
    k = n_states

    if t < k * 5:
        raise ValueError(f"Need at least {k * 5} observations for {k} states, got {t}")

    pi, trans, means, covs = _init_params(features, k, rng)

    prev_ll = float("-inf")
    n_iter = 0

    for iteration in range(max_iter):
        # ---- E-step: forward-backward ----

        # Log emission probabilities: T x K
        log_emit = [[_gaussian_log_pdf(features[i], means[s], covs[s])
                      for s in range(k)] for i in range(t)]

        # Forward pass (log scale)
        log_alpha = [[0.0] * k for _ in range(t)]
        for s in range(k):
            log_alpha[0][s] = math.log(max(pi[s], 1e-300)) + log_emit[0][s]

        for i in range(1, t):
            for s in range(k):
                vals = [log_alpha[i - 1][s2] + math.log(max(trans[s2][s], 1e-300))
                        for s2 in range(k)]
                log_alpha[i][s] = _log_sum_exp(vals) + log_emit[i][s]

        # Log-likelihood
        ll = _log_sum_exp(log_alpha[t - 1])

        # Backward pass (log scale)
        log_beta = [[0.0] * k for _ in range(t)]
        # log_beta[t-1] = 0 (log(1) = 0)

        for i in range(t - 2, -1, -1):
            for s in range(k):
                vals = [math.log(max(trans[s][s2], 1e-300)) + log_emit[i + 1][s2] + log_beta[i + 1][s2]
                        for s2 in range(k)]
                log_beta[i][s] = _log_sum_exp(vals)

        # Posterior (gamma): T x K
        gamma = [[0.0] * k for _ in range(t)]
        for i in range(t):
            log_denom = _log_sum_exp([log_alpha[i][s] + log_beta[i][s] for s in range(k)])
            for s in range(k):
                gamma[i][s] = math.exp(log_alpha[i][s] + log_beta[i][s] - log_denom)

        # Xi: (T-1) x K x K transition posteriors
        xi_sum = [[0.0] * k for _ in range(k)]
        for i in range(t - 1):
            log_vals = []
            for s1 in range(k):
                for s2 in range(k):
                    log_vals.append(
                        log_alpha[i][s1] +
                        math.log(max(trans[s1][s2], 1e-300)) +
                        log_emit[i + 1][s2] +
                        log_beta[i + 1][s2]
                    )
            log_denom = _log_sum_exp(log_vals)
            idx = 0
            for s1 in range(k):
                for s2 in range(k):
                    xi_sum[s1][s2] += math.exp(log_vals[idx] - log_denom)
                    idx += 1

        # ---- M-step ----

        # Initial probs
        pi = [gamma[0][s] for s in range(k)]
        pi_sum = sum(pi)
        pi = [p / pi_sum for p in pi]

        # Transition matrix
        for s1 in range(k):
            row_sum = sum(xi_sum[s1])
            for s2 in range(k):
                trans[s1][s2] = xi_sum[s1][s2] / max(row_sum, 1e-300)

        # Means and covariances
        for s in range(k):
            gamma_sum = sum(gamma[i][s] for i in range(t))
            if gamma_sum < 1e-10:
                continue

            # Mean
            means[s] = [
                sum(gamma[i][s] * features[i][j] for i in range(t)) / gamma_sum
                for j in range(d)
            ]

            # Covariance (diagonal)
            covs[s] = [[0.0] * d for _ in range(d)]
            for j in range(d):
                var = sum(
                    gamma[i][s] * (features[i][j] - means[s][j]) ** 2
                    for i in range(t)
                ) / gamma_sum
                covs[s][j][j] = max(var, 1e-6)

        # Convergence check
        n_iter = iteration + 1
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    # ---- Viterbi decoding (MAP state sequence) ----
    log_delta = [[0.0] * k for _ in range(t)]
    psi = [[0] * k for _ in range(t)]

    for s in range(k):
        log_delta[0][s] = math.log(max(pi[s], 1e-300)) + log_emit[0][s]

    for i in range(1, t):
        for s in range(k):
            candidates = [log_delta[i - 1][s2] + math.log(max(trans[s2][s], 1e-300))
                          for s2 in range(k)]
            best = max(range(k), key=lambda s2: candidates[s2])
            log_delta[i][s] = candidates[best] + log_emit[i][s]
            psi[i][s] = best

    # Backtrack
    path = [0] * t
    path[t - 1] = max(range(k), key=lambda s: log_delta[t - 1][s])
    for i in range(t - 2, -1, -1):
        path[i] = psi[i + 1][path[i + 1]]

    # ---- Label regimes by volatility (state 0 = lowest vol = calm) ----
    vol_order = sorted(range(k), key=lambda s: means[s][0])  # feature 0 = avg_vol
    label_map = {vol_order[i]: i for i in range(k)}
    label_names = {0: "calm", 1: "stressed", 2: "crisis"}
    if k == 2:
        label_names = {0: "calm", 1: "stressed"}
    elif k > 3:
        label_names = {i: f"regime_{i}" for i in range(k)}

    # Remap path
    path = [label_map[s] for s in path]

    # Remap everything to sorted order
    sorted_means = [means[vol_order[i]] for i in range(k)]
    sorted_covs = [covs[vol_order[i]] for i in range(k)]
    sorted_trans = [[trans[vol_order[i]][vol_order[j]] for j in range(k)] for i in range(k)]
    sorted_pi = [pi[vol_order[i]] for i in range(k)]

    # Remap gamma
    sorted_gamma = [[gamma[i][vol_order[s]] for s in range(k)] for i in range(t)]

    # Build regime descriptions
    states = []
    for s in range(k):
        freq = sum(1 for p in path if p == s) / t

        # Average duration: 1 / (1 - self-transition prob)
        self_trans = sorted_trans[s][s]
        avg_duration = 1.0 / max(1.0 - self_trans, 0.01)

        states.append(RegimeState(
            label=label_names.get(s, f"regime_{s}"),
            index=s,
            mean_vol=sorted_means[s][0],
            mean_return=sorted_means[s][1],
            mean_corr=sorted_means[s][2] if len(sorted_means[s]) > 2 else 0.0,
            duration_days=avg_duration,
            frequency=freq,
        ))

    return RegimeModel(
        n_states=k,
        transition_matrix=sorted_trans,
        means=sorted_means,
        covs=sorted_covs,
        initial_probs=sorted_pi,
        states=states,
        log_likelihood=ll,
        n_iter=n_iter,
        regime_path=path,
        regime_probs=sorted_gamma,
    )


def current_regime(model: RegimeModel) -> Tuple[RegimeState, List[float]]:
    """Get the current (latest) regime and its probability distribution."""
    if not model.regime_probs:
        raise ValueError("Model has no regime probabilities (was it fitted?)")
    latest_probs = model.regime_probs[-1]
    current_idx = model.regime_path[-1]
    return model.states[current_idx], latest_probs


def regime_transition_forecast(
    model: RegimeModel,
    horizon: int = 5,
) -> List[List[float]]:
    """Forecast regime probabilities over next `horizon` days.

    Returns list of K probability vectors, one per future day.
    Uses current regime probs and iterates transition matrix.
    """
    if not model.regime_probs:
        raise ValueError("Model has no regime probabilities")

    probs = model.regime_probs[-1][:]
    k = model.n_states
    forecast = []

    for _ in range(horizon):
        new_probs = [0.0] * k
        for s2 in range(k):
            new_probs[s2] = sum(
                probs[s1] * model.transition_matrix[s1][s2]
                for s1 in range(k)
            )
        forecast.append(new_probs)
        probs = new_probs

    return forecast
