"""
Performance metrics and APEX composite scoring.

Score = Win Rate × 0.35 + Profit Factor × 0.30 + (1 − Max Drawdown) × 0.20
        + Confluence Score × 0.15

All metric functions accept a trades DataFrame and/or equity curve Series
and return plain floats — no pandas I/O, no side effects.

Overfitting-aware metrics (T4.2)
---------------------------------
- deflated_sharpe_ratio: Bailey & Lopez de Prado DSR corrected for multiple trials.
- probability_backtest_overfitting: CSCV (Combinatorially Symmetric Cross-Validation).
- minimum_track_record_length: minimum n_obs needed to trust an observed SR.

Risk-adjusted scoring (T4.5)
------------------------------
- compute_score_v2: demotes win rate; weights profit_factor, return/drawdown,
  and (optionally) deflated Sharpe.
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import SCORE_WEIGHTS

# ── scipy shim — works with or without scipy installed ───────────────────────

try:
    from scipy.stats import norm as _scipy_norm  # type: ignore
    def _Phi(x: float) -> float:
        """Standard-normal CDF."""
        return float(_scipy_norm.cdf(x))

    def _Phi_inv(p: float) -> float:
        """Standard-normal quantile (percent-point function)."""
        return float(_scipy_norm.ppf(p))
except ImportError:
    # Pure math/numpy fallback — accurate to ~1e-7 for |x| < 5.
    _SQRT2 = math.sqrt(2.0)

    def _Phi(x: float) -> float:  # type: ignore[misc]
        """Standard-normal CDF via erf (no scipy)."""
        return 0.5 * (1.0 + math.erf(x / _SQRT2))

    def _Phi_inv(p: float) -> float:  # type: ignore[misc]
        """
        Standard-normal quantile via Peter Acklam's rational approximation
        (max absolute error < 4.5e-9 on the central range).
        Reference: https://web.archive.org/web/20151030215612/
                   http://home.online.no/~pjacklam/notes/invnorm/
        """
        p = float(np.clip(p, 1e-15, 1.0 - 1e-15))
        # Coefficients for the rational approximation
        a = [-3.969683028665376e+01,  2.209460984245205e+02,
             -2.759285104469687e+02,  1.383577518672690e+02,
             -3.066479806614716e+01,  2.506628277459239e+00]
        b = [-5.447609879822406e+01,  1.615858368580409e+02,
             -1.556989798598866e+02,  6.680131188771972e+01,
             -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01,
             -2.400758277161838e+00, -2.549732539343734e+00,
              4.374664141464968e+00,  2.938163982698783e+00]
        d = [ 7.784695709041462e-03,  3.224671290700398e-01,
              2.445134137142996e+00,  3.754408661907416e+00]
        p_low, p_high = 0.02425, 1.0 - 0.02425
        if p_low <= p <= p_high:
            q = p - 0.5
            r = q * q
            num = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q
            den = ((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0
            return num / den
        elif p < p_low:
            q = math.sqrt(-2.0 * math.log(p))
            num = ((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]
            den = (((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0
            return num / den
        else:
            q = math.sqrt(-2.0 * math.log(1.0 - p))
            num = ((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]
            den = (((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0
            return -(num / den)


# ── Individual metrics ────────────────────────────────────────────────────────

def win_rate(trades: pd.DataFrame) -> float:
    """Fraction of trades that closed in profit."""
    if len(trades) == 0:
        return 0.0
    return float((trades["pnl"] > 0).sum() / len(trades))


def profit_factor(trades: pd.DataFrame, cap: float = 10.0) -> float:
    """Gross profit / gross loss, capped to avoid inf on perfect runs."""
    if len(trades) == 0:
        return 0.0
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss   = trades.loc[trades["pnl"] < 0, "pnl"].abs().sum()
    if gross_loss == 0:
        return cap if gross_profit > 0 else 0.0
    return float(min(gross_profit / gross_loss, cap))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough decline as a fraction [0, 1]."""
    if len(equity_curve) < 2:
        return 0.0
    peak     = equity_curve.cummax()
    drawdown = (peak - equity_curve) / peak.replace(0, np.nan)
    return float(drawdown.max())


def average_rr(trades: pd.DataFrame) -> float:
    """Average realized reward-to-risk ratio."""
    if len(trades) == 0:
        return 0.0
    return float(trades["rr_realized"].mean())


def confluence_score(trades: pd.DataFrame, max_possible: int = 6) -> float:
    """
    Normalized average confluence count of all executed trades.
    Returns value in [0, 1].
    """
    if len(trades) == 0 or "confluence_count" not in trades.columns:
        return 0.0
    avg = trades["confluence_count"].mean()
    return float(min(avg / max_possible, 1.0))


def _clean_equity_curve(equity_curve: pd.Series) -> pd.Series:
    """
    Remove stale between-trade bars from a sparse equity curve.

    BacktestEngine initializes ``equity_curve = [account_size] * n_bars`` and
    only updates bars that fall inside a trade window.  Bars between trades keep
    the initial account-size value.  This causes phantom returns -- for example,
    a trade that exits at a realized loss of 9 800 is followed by bars that still
    show 10 000, making it look like a +2% recovery that never happened.

    Fix: any bar *after* the first equity deviation that still equals the initial
    value is replaced with NaN, then forward-filled from the last realized equity.
    Bars before the first trade are untouched (they correctly hold the starting
    equity and contribute 0% return).

    A dense equity curve (all bars distinct) passes through unchanged because no
    bar after the first deviation equals the initial value.
    """
    eq = equity_curve.astype(float)
    initial = float(eq.iloc[0])

    deviates = eq != initial
    if not deviates.any():
        # Flat curve with no trades -- return as-is; callers will return 0.
        return eq

    first_dev_pos = int(np.argmax(deviates.values))

    positions = np.arange(len(eq))
    stale = (positions > first_dev_pos) & (eq.values == initial)

    eq_clean = eq.copy()
    eq_clean.iloc[stale] = np.nan
    eq_clean = eq_clean.ffill()
    return eq_clean


def sharpe_ratio(equity_curve: pd.Series, periods_per_year: int = 252 * 26) -> float:
    """
    Annualized Sharpe ratio on per-bar equity returns (0 risk-free rate).

    Annualization convention
    ------------------------
    ``periods_per_year`` defaults to 252 * 26 = 6 552, reflecting intraday
    15-minute bars with 26 bars per regular trading-hours session and 252
    trading days per year.  Override for different bar sizes or instruments.

    Sparse equity-curve handling
    ----------------------------
    The backtesting engine produces a sparse equity curve: bars between trades
    keep the initial account-size value instead of being forward-filled with the
    last realized equity.  Those stale bars create phantom returns (e.g. a
    realized loss followed by a jump back to the initial value) that corrupt the
    mean and sign of the naive ``pct_change()`` Sharpe.

    This function calls ``_clean_equity_curve`` to replace stale between-trade
    bars with the last realized equity before computing returns, so:

    * a net-losing equity curve always yields Sharpe <= 0, and
    * a net-winning equity curve always yields Sharpe >= 0.

    Sign contract
    -------------
    mean(per-bar returns) / std(per-bar returns) * sqrt(periods_per_year).
    A strategy whose equity ends below its starting value has a negative mean
    return on the cleaned curve and therefore a negative (or zero) Sharpe.
    """
    if len(equity_curve) < 2:
        return 0.0

    eq = _clean_equity_curve(equity_curve)
    returns = eq.pct_change().dropna()
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


def per_observation_sharpe_ratio(equity_curve: pd.Series) -> Tuple[float, int]:
    """
    Per-observation (NON-annualized) Sharpe ratio and its observation count.

    Returns ``(sr_per_obs, n_obs)`` where:

    * ``sr_per_obs = mean(per-bar returns) / std(per-bar returns)`` -- the raw
      per-period Sharpe with **no** ``sqrt(periods_per_year)`` annualization
      factor, and
    * ``n_obs`` is the number of return observations the ratio was computed
      from (``len(returns)``).

    Why this exists (DSR units contract)
    ------------------------------------
    The Deflated Sharpe Ratio multiplies ``observed_sr`` by ``sqrt(n_obs - 1)``
    internally.  Feeding it an *annualized* Sharpe (which already embeds a
    ``sqrt(periods_per_year)`` factor) together with a large ``n_obs`` double-
    counts the time scaling and makes ``Phi(...)`` saturate to ~1.0 even for a
    zero-edge strategy -- a false-confidence guardrail failure.  DSR must be fed
    a Sharpe and ``n_obs`` on the *same per-period basis*; this helper returns
    exactly that matched pair.

    Uses ``_clean_equity_curve`` so the same sign contract as ``sharpe_ratio``
    holds (net-losing curve => sr_per_obs <= 0).
    """
    if len(equity_curve) < 2:
        return 0.0, 2

    eq = _clean_equity_curve(equity_curve)
    returns = eq.pct_change().dropna()
    n_obs = int(len(returns))
    if returns.std() == 0 or n_obs < 2:
        return 0.0, max(2, n_obs)
    return float(returns.mean() / returns.std()), n_obs


def calmar_ratio(equity_curve: pd.Series) -> float:
    """
    Total return / max drawdown (same sign contract as Sharpe).

    Uses ``_clean_equity_curve`` so that stale between-trade bars do not
    inflate the final equity to the initial account size and do not create
    phantom drawdown from post-win flat regions.  A net-losing strategy
    yields Calmar <= 0; a net-winning strategy yields Calmar >= 0.
    """
    eq = _clean_equity_curve(equity_curve)
    mdd = max_drawdown(eq)
    if mdd == 0:
        return 0.0
    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1.0
    return float(total_return / mdd)


def expectancy(trades: pd.DataFrame) -> float:
    """Dollar expectancy per trade = avg(win) * win_rate − avg(loss) * loss_rate."""
    if len(trades) == 0:
        return 0.0
    wins   = trades.loc[trades["pnl"] > 0, "pnl"]
    losses = trades.loc[trades["pnl"] < 0, "pnl"].abs()
    wr     = len(wins) / len(trades)
    avg_w  = wins.mean()   if len(wins)   > 0 else 0.0
    avg_l  = losses.mean() if len(losses) > 0 else 0.0
    return float(wr * avg_w - (1 - wr) * avg_l)


# ── Overfitting-aware metrics (T4.2) ─────────────────────────────────────────

_EULER_GAMMA = 0.5772156649  # Euler-Mascheroni constant


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    sr_variance: Optional[float] = None,
) -> float:
    """
    Deflated Sharpe Ratio (DSR) -- Bailey & Lopez de Prado (2014).

    Returns the probability in [0, 1] that the observed Sharpe Ratio is
    genuine after correcting for selection bias across `n_trials` independent
    parameter configurations.

    Parameters
    ----------
    observed_sr  : Annualised (or unnormalised) Sharpe Ratio of the chosen config.
    n_trials     : Number of independent configurations tried (backtest trials).
    n_obs        : Number of observations (bars / periods) in the backtest.
    skew         : Skewness of the returns distribution (default 0).
    kurt         : Excess kurtosis + 3 (i.e. normal = 3, default 3).
    sr_variance  : Variance of the SR estimator across trials.  When None (the
                   default), approximated as 1/(n_obs-1), which is the sampling
                   variance of an SR estimate under the null of i.i.d. returns.
                   This approximation is conservative (overstates SR0) when
                   returns are fat-tailed.

    Returns
    -------
    float in [0, 1]  (higher means the SR is more likely genuine).

    Notes
    -----
    SR0 (expected maximum SR under H0) is derived from extreme-value theory:
        SR0 = sqrt(V) * [(1-gamma)*Phi_inv(1-1/N) + gamma*Phi_inv(1-1/(N*e))]
    where V is sr_variance, gamma = Euler-Mascheroni, N = n_trials.

    DSR = Phi( [(observed_sr - SR0) * sqrt(n_obs - 1)]
               / sqrt(1 - skew*observed_sr + ((kurt-1)/4)*observed_sr**2) )
    """
    n_trials = max(1, int(n_trials))
    n_obs    = max(2, int(n_obs))

    if sr_variance is None:
        # Conservative approximation under the i.i.d. null
        sr_variance = 1.0 / (n_obs - 1)

    sr_std = math.sqrt(max(sr_variance, 1e-15))

    # Expected maximum SR across N trials (extreme-value approximation)
    if n_trials == 1:
        sr0 = 0.0  # no multiple-testing penalty with a single trial
    else:
        sr0 = sr_std * (
            (1.0 - _EULER_GAMMA) * _Phi_inv(1.0 - 1.0 / n_trials)
            + _EULER_GAMMA       * _Phi_inv(1.0 - 1.0 / (n_trials * math.e))
        )

    # Non-centrality denominator (handles non-normality of returns)
    denom_sq = 1.0 - skew * observed_sr + ((kurt - 1.0) / 4.0) * observed_sr ** 2
    denom    = math.sqrt(max(denom_sq, 1e-15))

    z = (observed_sr - sr0) * math.sqrt(n_obs - 1) / denom
    return float(np.clip(_Phi(z), 0.0, 1.0))


def probability_backtest_overfitting(
    perf_matrix: np.ndarray,
    S: int = 16,
) -> float:
    """
    Probability of Backtest Overfitting (PBO) via CSCV
    (Bailey, Borwein, Lopez de Prado & Zhu, 2014).

    Parameters
    ----------
    perf_matrix : ndarray of shape (T, N)
                  Rows = contiguous time periods (observations); columns = configs.
                  Each cell is the per-period performance of config n at time t
                  (e.g. bar returns, block Sharpe, etc.).
    S           : Number of contiguous blocks to split T rows into.
                  Must be even; default 16.  C(S, S/2) partitions are formed.
                  Capped internally to keep C(S, S/2) <= 12_870 (S=16).

    Returns
    -------
    float in [0, 1].  Near 0 = low overfitting; near 0.5 = random selection.

    Algorithm
    ---------
    1. Split T rows into S contiguous blocks of equal size (drop remainder rows).
    2. For each of the C(S, S/2) ways to choose S/2 blocks as IS:
         a. IS performance  = mean across IS blocks per config.
         b. OOS performance = mean across OOS blocks per config.
         c. IS-best config  = argmax of IS means.
         d. OOS rank        = rank of IS-best in sorted OOS means (1=worst).
         e. logit(rank / (N+1))  -- negative logit => IS-best is below OOS median.
    3. PBO = fraction of partitions where logit <= 0.

    Notes
    -----
    C(16, 8) = 12 870 partitions -- tractable.  S is capped at 16 to ensure
    runtime stays below a few seconds.  Use S=8 for very large matrices (T*N big).
    """
    perf_matrix = np.asarray(perf_matrix, dtype=float)
    if perf_matrix.ndim != 2:
        raise ValueError("perf_matrix must be 2-D (T_observations x N_configs)")

    T, N = perf_matrix.shape
    if N < 2:
        raise ValueError("Need at least 2 configs to compute PBO")

    # Force S even and cap at 16
    S = max(2, int(S))
    if S % 2 != 0:
        S -= 1
    S = min(S, 16)

    block_size = T // S
    if block_size < 1:
        raise ValueError(
            f"perf_matrix has only {T} rows but S={S} blocks requested; "
            "reduce S or provide more observations"
        )

    # Trim to S * block_size rows so blocks are equal
    trimmed = perf_matrix[: S * block_size, :]

    # Split into S contiguous blocks: list of (block_size x N) arrays
    blocks = [trimmed[i * block_size : (i + 1) * block_size, :] for i in range(S)]

    half = S // 2
    all_indices = list(range(S))
    partitions  = list(combinations(all_indices, half))  # IS block indices

    logit_vals = []
    for is_idx in partitions:
        oos_idx = [i for i in all_indices if i not in is_idx]

        is_perf  = np.mean(np.vstack([blocks[i] for i in is_idx]),  axis=0)  # (N,)
        oos_perf = np.mean(np.vstack([blocks[i] for i in oos_idx]), axis=0)  # (N,)

        best_is  = int(np.argmax(is_perf))
        oos_val  = oos_perf[best_is]

        # Rank of IS-best in OOS (1=worst, N=best)
        rank = int(np.sum(oos_perf <= oos_val))  # configs with OOS <= IS-best OOS
        rank = max(1, min(rank, N))

        logit = math.log(rank / (N + 1 - rank + 1e-15))
        logit_vals.append(logit)

    if not logit_vals:
        return 0.0

    pbo = float(np.mean(np.array(logit_vals) <= 0.0))
    return float(np.clip(pbo, 0.0, 1.0))


def pbo_from_oos_matrix(
    perf_matrix: Optional[np.ndarray],
    S: int = 16,
    min_configs: int = 4,
    min_rows: int = 4,
) -> Optional[float]:
    """
    PBO wrapper that requires a GENUINE per-period OOS performance matrix.

    Unlike calling :func:`probability_backtest_overfitting` directly, this
    helper refuses to fabricate a result from degenerate input.  It returns
    ``None`` (clearly-marked "undefined") rather than a misleading ``0.0`` when:

    * ``perf_matrix`` is ``None`` or not 2-D, or
    * there are fewer than ``min_configs`` configs (columns), or
    * there are fewer than ``min_rows`` per-period observations (rows), or
    * every row is identical (a tiled / constant matrix carries no genuine
      out-of-sample dispersion, so the IS-best is mechanically the OOS-best and
      PBO would collapse to 0.0 by construction -- the original miscalibration).

    Parameters
    ----------
    perf_matrix : ndarray of shape (T_observations, N_configs) or None.
                  Each cell is the *out-of-sample* per-period performance of
                  config n at observation t (per-fold or per-bar OOS score).
    S           : Number of CSCV blocks (passed through; auto-reduced to an even
                  value <= number of rows).
    min_configs : Minimum number of configs required to report a number.
    min_rows    : Minimum number of per-period observations required.

    Returns
    -------
    float in [0, 1], or ``None`` when the matrix is too small / degenerate to
    yield a genuine estimate.
    """
    if perf_matrix is None:
        return None

    pm = np.asarray(perf_matrix, dtype=float)
    if pm.ndim != 2:
        return None

    T, N = pm.shape
    if N < min_configs or T < min_rows:
        return None

    # Reject a tiled / constant-in-time matrix: no genuine OOS dispersion.
    if np.allclose(pm, pm[0:1, :]):
        return None

    # Force S even and no larger than the number of rows (need >= 1 row/block).
    S = max(2, int(S))
    if S % 2 != 0:
        S -= 1
    S = min(S, T)
    if S % 2 != 0:
        S -= 1
    if S < 2:
        return None

    try:
        return probability_backtest_overfitting(pm, S=S)
    except (ValueError, ZeroDivisionError):
        return None


def minimum_track_record_length(
    observed_sr: float,
    sr_benchmark: float = 0.0,
    n_obs: Optional[int] = None,
    skew: float = 0.0,
    kurt: float = 3.0,
    prob: float = 0.95,
) -> float:
    """
    Minimum Track Record Length (MinTRL) -- Bailey & Lopez de Prado (2014).

    Returns the minimum number of observations needed so that the observed SR
    is statistically significant at `prob` confidence, accounting for non-
    normality via skew and kurtosis.

    Parameters
    ----------
    observed_sr    : Observed annualised Sharpe Ratio.
    sr_benchmark   : Benchmark SR to beat (default 0).
    n_obs          : Current track-record length (informational only; unused
                     in the formula -- present for the caller's reference).
    skew           : Skewness of returns.
    kurt           : Kurtosis (3 = normal).
    prob           : Required confidence level (default 0.95).

    Returns
    -------
    float : Minimum required number of observations.
    """
    z = _Phi_inv(prob)
    sr_excess = observed_sr - sr_benchmark
    if abs(sr_excess) < 1e-12:
        return float("inf")

    non_normality = 1.0 - skew * observed_sr + ((kurt - 1.0) / 4.0) * observed_sr ** 2
    min_trl = 1.0 + non_normality * (z / sr_excess) ** 2
    return float(max(min_trl, 1.0))


# ── Composite score ───────────────────────────────────────────────────────────

def compute_score(
    trades:       pd.DataFrame,
    equity_curve: pd.Series,
    min_trades:   int = 20,
) -> float:
    """
    APEX composite score in [0, 1] (higher = better).

    Penalizes studies with too few trades to be statistically meaningful.
    Returns 0.0 for invalid/empty backtests.

    .. deprecated::
        This function over-weights win_rate (0.35) and ignores risk-adjusted
        return metrics, making it susceptible to high-win-rate / low-edge
        strategies.  Use :func:`compute_score_v2` for overfitting-aware,
        risk-adjusted scoring.
    """
    if len(trades) < min_trades:
        return 0.0

    eq_clean = _clean_equity_curve(equity_curve)
    wr   = win_rate(trades)
    pf   = profit_factor(trades, cap=5.0)
    mdd  = max_drawdown(eq_clean)
    conf = confluence_score(trades)

    # Normalize profit factor to [0, 1] (capped at 5)
    pf_norm = min(pf / 5.0, 1.0)

    w = SCORE_WEIGHTS
    score = (
        wr       * w["win_rate"]      +
        pf_norm  * w["profit_factor"] +
        (1 - mdd) * w["max_drawdown"] +
        conf     * w["confluence"]
    )

    # Hard penalty gates — realistic minimum thresholds
    if wr < 0.30:
        score *= 0.5
    if pf < 1.0:
        score *= 0.5
    if mdd > 0.25:
        score *= max(0.0, 1.0 - (mdd - 0.25) * 4)

    return float(np.clip(score, 0.0, 1.0))


# Default weights for compute_score_v2.
# win_rate is intentionally near-zero: a high win rate with small wins /
# large losses is a common overfitting signature.  profit_factor and
# return-over-drawdown capture actual edge; deflated Sharpe penalises
# multiple-trial selection bias when available.
_SCORE_V2_DEFAULT_WEIGHTS: Dict[str, float] = {
    "profit_factor":    0.40,   # gross-profit / gross-loss (normalised to [0,1])
    "return_over_mdd":  0.35,   # total_return / max_drawdown (normalised)
    "deflated_sharpe":  0.20,   # DSR in [0,1]; falls back to raw Sharpe if not supplied
    "win_rate":         0.05,   # intentionally low -- win rate alone is not edge
}


def compute_score_v2(
    trades:          pd.DataFrame,
    equity_curve:    pd.Series,
    *,
    deflated_sharpe: Optional[float] = None,
    n_trials:        Optional[int]   = None,
    weights:         Optional[Dict[str, float]] = None,
    min_trades:      int = 20,
) -> float:
    """
    Risk-adjusted composite score in [0, 1] (higher = better).

    Supersedes :func:`compute_score`.  Key differences:

    * **win_rate weight is 0.05** (was 0.35) -- avoids rewarding high-WR /
      negative-edge strategies.
    * **profit_factor weight 0.40** -- directly captures gross edge.
    * **return_over_mdd weight 0.35** -- rewards strategies that earn more
      per unit of drawdown (Calmar-style, but using total return).
    * **deflated_sharpe weight 0.20** -- when `deflated_sharpe` or `n_trials`
      is provided, incorporates the Bailey & Lopez de Prado DSR to penalise
      multiple-trial selection bias.  Falls back to a normalised raw Sharpe
      when not supplied.

    Parameters
    ----------
    trades          : DataFrame with at least a ``pnl`` column.
    equity_curve    : Portfolio equity indexed to the same bars as trades.
    deflated_sharpe : Pre-computed DSR in [0, 1].  If None and `n_trials` is
                      provided, DSR is computed automatically.  If neither is
                      provided, the raw annualised Sharpe (clipped to [0, 1])
                      is used as the ``deflated_sharpe`` component.
    n_trials        : Number of independent parameter configurations tried;
                      used only when `deflated_sharpe` is None.
    weights         : Override the component weights dict (keys: profit_factor,
                      return_over_mdd, deflated_sharpe, win_rate).  Falls back
                      first to this arg, then to ``settings.SCORE_WEIGHTS`` if
                      all four keys are present, then to the built-in defaults.
    min_trades      : Return 0.0 if fewer trades than this threshold.

    Returns
    -------
    float in [0, 1].
    """
    if len(trades) < min_trades:
        return 0.0

    # Resolve weights
    if weights is not None:
        w = weights
    else:
        # Accept settings.SCORE_WEIGHTS only if it has the v2 keys
        v2_keys = set(_SCORE_V2_DEFAULT_WEIGHTS)
        try:
            if v2_keys.issubset(set(SCORE_WEIGHTS)):
                w = SCORE_WEIGHTS
            else:
                w = _SCORE_V2_DEFAULT_WEIGHTS
        except Exception:
            w = _SCORE_V2_DEFAULT_WEIGHTS

    # ── Component metrics ────────────────────────────────────────────────────
    eq_clean = _clean_equity_curve(equity_curve)
    wr  = win_rate(trades)
    pf  = profit_factor(trades, cap=5.0)
    mdd = max_drawdown(eq_clean)

    # Normalise profit_factor to [0, 1] (cap=5)
    pf_norm = float(np.clip(pf / 5.0, 0.0, 1.0))

    # Return-over-max-drawdown: total_return / mdd, normalised to [0, 1].
    # Cap at 10x to avoid inf on very low drawdown runs, then scale.
    if len(eq_clean) >= 2 and eq_clean.iloc[0] > 0:
        total_return = (eq_clean.iloc[-1] / eq_clean.iloc[0]) - 1.0
    else:
        total_return = 0.0
    if mdd > 1e-9:
        romd = float(np.clip(total_return / mdd / 10.0, 0.0, 1.0))
    else:
        # No drawdown: reward if positive return, penalise if zero/negative
        romd = float(np.clip(total_return * 10.0, 0.0, 1.0))

    # Deflated Sharpe component
    if deflated_sharpe is not None:
        dsr_component = float(np.clip(deflated_sharpe, 0.0, 1.0))
    elif n_trials is not None:
        sr_raw  = sharpe_ratio(equity_curve)
        n_obs   = max(2, len(equity_curve))
        dsr_component = deflated_sharpe_ratio(sr_raw, n_trials=n_trials, n_obs=n_obs)
    else:
        # Fallback: raw annualised Sharpe clipped to [0, 3] then normalised
        sr_raw = sharpe_ratio(equity_curve)
        dsr_component = float(np.clip(sr_raw / 3.0, 0.0, 1.0))

    # ── Weighted sum ─────────────────────────────────────────────────────────
    score = (
        pf_norm       * w.get("profit_factor",   _SCORE_V2_DEFAULT_WEIGHTS["profit_factor"])   +
        romd          * w.get("return_over_mdd",  _SCORE_V2_DEFAULT_WEIGHTS["return_over_mdd"]) +
        dsr_component * w.get("deflated_sharpe",  _SCORE_V2_DEFAULT_WEIGHTS["deflated_sharpe"]) +
        wr            * w.get("win_rate",         _SCORE_V2_DEFAULT_WEIGHTS["win_rate"])
    )

    # Hard-gate penalties (same thresholds as v1 but applied multiplicatively)
    if pf < 1.0:
        score *= 0.5
    if mdd > 0.25:
        score *= max(0.0, 1.0 - (mdd - 0.25) * 4.0)
    if wr < 0.20:
        score *= 0.7

    return float(np.clip(score, 0.0, 1.0))


# ── Full metrics dict ─────────────────────────────────────────────────────────

def compute_all_metrics(
    trades:       pd.DataFrame,
    equity_curve: pd.Series,
) -> Dict[str, float]:
    """Return every metric as a flat dict (for reporting / JSON output)."""
    if len(trades) == 0:
        return {k: 0.0 for k in (
            "total_trades", "win_rate", "profit_factor", "max_drawdown",
            "avg_rr", "confluence_score", "sharpe", "calmar",
            "expectancy", "score", "gross_profit", "gross_loss",
            "avg_trade_duration_bars",
        )}

    wins   = trades.loc[trades["pnl"] > 0, "pnl"]
    losses = trades.loc[trades["pnl"] < 0, "pnl"].abs()

    eq_clean = _clean_equity_curve(equity_curve)

    return {
        "total_trades":            int(len(trades)),
        "winning_trades":          int((trades["pnl"] > 0).sum()),
        "losing_trades":           int((trades["pnl"] < 0).sum()),
        "win_rate":                round(win_rate(trades), 4),
        "profit_factor":           round(profit_factor(trades), 4),
        "max_drawdown":            round(max_drawdown(eq_clean), 4),
        "avg_rr":                  round(average_rr(trades), 4),
        "confluence_score":        round(confluence_score(trades), 4),
        "sharpe":                  round(sharpe_ratio(equity_curve), 4),
        "calmar":                  round(calmar_ratio(equity_curve), 4),
        "expectancy":              round(expectancy(trades), 2),
        "score":                   round(compute_score(trades, equity_curve), 6),
        "gross_profit":            round(float(wins.sum()), 2),
        "gross_loss":              round(float(losses.sum()), 2),
        "avg_win":                 round(float(wins.mean()) if len(wins) > 0 else 0.0, 2),
        "avg_loss":                round(float(losses.mean()) if len(losses) > 0 else 0.0, 2),
        "avg_trade_duration_bars": round(float(trades["duration_bars"].mean()), 1)
                                   if "duration_bars" in trades.columns else 0.0,
        "final_equity":            round(float(eq_clean.iloc[-1]), 2),
    }
