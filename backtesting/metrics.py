"""
Performance metrics and APEX composite scoring.

Score = Win Rate × 0.35 + Profit Factor × 0.30 + (1 − Max Drawdown) × 0.20
        + Confluence Score × 0.15

All metric functions accept a trades DataFrame and/or equity curve Series
and return plain floats — no pandas I/O, no side effects.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config.settings import SCORE_WEIGHTS


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


def sharpe_ratio(equity_curve: pd.Series, periods_per_year: int = 252 * 26) -> float:
    """Annualized Sharpe on bar-by-bar returns (0 risk-free rate)."""
    returns = equity_curve.pct_change().dropna()
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


def calmar_ratio(equity_curve: pd.Series) -> float:
    """Annualized return / max drawdown."""
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return 0.0
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0
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
    """
    if len(trades) < min_trades:
        return 0.0

    wr   = win_rate(trades)
    pf   = profit_factor(trades, cap=5.0)
    mdd  = max_drawdown(equity_curve)
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

    return {
        "total_trades":            int(len(trades)),
        "winning_trades":          int((trades["pnl"] > 0).sum()),
        "losing_trades":           int((trades["pnl"] < 0).sum()),
        "win_rate":                round(win_rate(trades), 4),
        "profit_factor":           round(profit_factor(trades), 4),
        "max_drawdown":            round(max_drawdown(equity_curve), 4),
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
        "final_equity":            round(float(equity_curve.iloc[-1]), 2),
    }
