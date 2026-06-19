"""
Structured run + outcome logging to Supabase (and the local file log).

Every important pipeline outcome is recorded so nothing is lost:
  - runs            one row per top-level invocation (backtest/optimize/fetch/live/test)
  - events          structured log stream (phases, validation, risk halts, notes)
  - backtest_runs   one row per backtest evaluation, incl. honesty metrics
  - optimization_runs / data_fetches

Design goals:
  - Never raise. Logging must not crash the trading pipeline.
  - Graceful no-op when SUPABASE_URL/KEY are unset (mirrors config.supabase_client).
  - Always mirror to the local file logger, so outcomes survive a Supabase outage.
  - Sanitize numpy/pandas/datetime values so JSONB inserts never fail.

Backend auth: set SUPABASE_KEY to the service_role key for reliable server-side
writes (it bypasses RLS). The anon key also works while the tables have RLS
disabled (the current default), but service_role is recommended for production.

Usage:
    from config import run_logger as rl

    run = rl.start_run("backtest", symbol="MES", timeframe="15m", config=params)
    rl.log_event("Loaded 52k bars from Databento", run_uuid=run, kind="data")
    rl.log_backtest(symbol="MES", timeframe="15m", run_uuid=run,
                    strategy_name="order_block", data_source="databento",
                    is_holdout=False, params=params, costs=costs, metrics=metrics)
    rl.finish_run(run, status="ok", summary={"net_pnl": 1234.5})
"""
from __future__ import annotations

import math
import subprocess
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Optional

from config.logger import get_logger
from config.supabase_client import get_client

log = get_logger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_uuid() -> str:
    return str(uuid.uuid4())


@lru_cache(maxsize=1)
def git_sha() -> Optional[str]:
    """Short git SHA of the working tree, cached. None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _jsonable(obj: Any) -> Any:
    """Recursively coerce a value into something the Supabase client can serialize.

    Handles numpy scalars/arrays, pandas Timestamps, datetimes, and non-finite
    floats (NaN/Inf -> None) so a stray value never breaks a JSONB insert.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return _jsonable(obj.item())
        if isinstance(obj, np.ndarray):
            return [_jsonable(v) for v in obj.tolist()]
    except Exception:
        pass
    try:
        import pandas as pd
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
    except Exception:
        pass
    return str(obj)


def _insert(table: str, row: dict) -> None:
    """Best-effort insert. Never raises."""
    client = get_client()
    if client is None:
        return
    try:
        client.table(table).insert(_jsonable(row)).execute()
    except Exception as exc:  # noqa: BLE001
        log.warning("Supabase insert into %s failed: %s", table, exc)


def _update(table: str, match: dict, changes: dict) -> None:
    """Best-effort update. Never raises."""
    client = get_client()
    if client is None:
        return
    try:
        q = client.table(table).update(_jsonable(changes))
        for k, v in match.items():
            q = q.eq(k, v)
        q.execute()
    except Exception as exc:  # noqa: BLE001
        log.warning("Supabase update on %s failed: %s", table, exc)


# ── Runs ────────────────────────────────────────────────────────────────────

def start_run(
    kind: str,
    *,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    config: Optional[dict] = None,
) -> str:
    """Open a run and return its run_uuid. Always returns a uuid, even offline."""
    run_uuid = new_run_uuid()
    log.info("[run:%s] start kind=%s symbol=%s tf=%s", run_uuid[:8], kind, symbol, timeframe)
    _insert("runs", {
        "run_uuid": run_uuid,
        "kind": kind,
        "status": "running",
        "symbol": symbol,
        "timeframe": timeframe,
        "git_sha": git_sha(),
        "config": config,
        "started_at": _now_iso(),
    })
    return run_uuid


def finish_run(run_uuid: str, *, status: str = "ok", summary: Optional[dict] = None) -> None:
    log.info("[run:%s] finish status=%s", run_uuid[:8] if run_uuid else "????", status)
    _update("runs", {"run_uuid": run_uuid}, {
        "status": status,
        "summary": summary,
        "finished_at": _now_iso(),
    })


# ── Events ──────────────────────────────────────────────────────────────────

def log_event(
    message: str,
    *,
    run_uuid: Optional[str] = None,
    level: str = "INFO",
    kind: str = "note",
    payload: Optional[dict] = None,
) -> None:
    """Record an important event to Supabase and mirror it to the file log."""
    lvl = level.upper()
    _file_log = log.error if lvl == "ERROR" else (log.warning if lvl == "WARN" else log.info)
    _file_log("[event:%s] %s", kind, message)
    _insert("events", {
        "run_uuid": run_uuid,
        "ts": _now_iso(),
        "level": lvl,
        "kind": kind,
        "message": message,
        "payload": payload,
    })


# ── Outcome tables ──────────────────────────────────────────────────────────

def log_backtest(
    *,
    symbol: str,
    timeframe: str,
    run_uuid: Optional[str] = None,
    strategy_name: Optional[str] = None,
    data_source: Optional[str] = None,
    is_holdout: bool = False,
    params: Optional[dict] = None,
    costs: Optional[dict] = None,
    metrics: Optional[dict] = None,
) -> None:
    """Log one backtest evaluation. `metrics` may carry the headline fields below
    plus any extras (the whole dict is also stored in the `metrics` JSONB)."""
    m = metrics or {}
    _insert("backtest_runs", {
        "run_uuid": run_uuid,
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy_name": strategy_name,
        "data_source": data_source,
        "is_holdout": is_holdout,
        "params": params,
        "costs": costs,
        "bar_start": m.get("bar_start"),
        "bar_end": m.get("bar_end"),
        "n_bars": m.get("n_bars"),
        "total_trades": m.get("total_trades"),
        "win_rate": m.get("win_rate"),
        "profit_factor": m.get("profit_factor"),
        "max_drawdown": m.get("max_drawdown"),
        "sharpe": m.get("sharpe"),
        "deflated_sharpe": m.get("deflated_sharpe"),
        "pbo": m.get("pbo"),
        "effective_trials": m.get("effective_trials"),
        "net_pnl": m.get("net_pnl"),
        "metrics": m,
        "git_sha": git_sha(),
    })


def log_optimization(
    *,
    symbol: str,
    timeframe: str,
    strategy_name: str,
    run_uuid: Optional[str] = None,
    **fields: Any,
) -> None:
    """Log one Optuna study result. Pass any columns of optimization_runs as kwargs
    (best_score, best_params, deflated_sharpe, pbo, effective_trials, oos_score,
    holdout_score, data_source, status, win_rate, profit_factor, max_drawdown, ...)."""
    row = {
        "run_uuid": run_uuid,
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy_name": strategy_name,
        "git_sha": git_sha(),
    }
    row.update(fields)
    _insert("optimization_runs", row)


def log_data_fetch(
    *,
    source: str,
    symbol: str,
    run_uuid: Optional[str] = None,
    timeframe: Optional[str] = None,
    **fields: Any,
) -> None:
    """Log one data pull. Pass any columns of data_fetches as kwargs
    (bar_start, bar_end, n_bars, gaps_found, cost_units, ok, message)."""
    row = {
        "run_uuid": run_uuid,
        "source": source,
        "symbol": symbol,
        "timeframe": timeframe,
    }
    row.update(fields)
    _insert("data_fetches", row)
