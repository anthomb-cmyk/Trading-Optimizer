"""Central configuration for APEX Optimizer."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent
DATA_DIR    = ROOT_DIR / "data" / "cache"
OUTPUT_DIR  = ROOT_DIR / "output"
LOG_DIR     = ROOT_DIR / "logs"
REPORT_DIR  = ROOT_DIR / "reports" / "output"

for _d in (DATA_DIR, OUTPUT_DIR, LOG_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Instruments ───────────────────────────────────────────────────────────────
INSTRUMENTS = {
    # ── Live trading symbols (NinjaTrader 8 / CME Globex) ────────────────────
    "MES": {
        "ticker":          "MES=F",
        "fallback_ticker": "ES=F",   # full-size contract, same price series
        "tick_size":       0.25,
        "tick_value":      1.25,
        "point_value":     5.0,
        "description":     "Micro E-mini S&P 500",
        "live_only":       False,
    },
    "MNQ": {
        "ticker":          "MNQ=F",
        "fallback_ticker": "NQ=F",
        "tick_size":       0.25,
        "tick_value":      0.50,
        "point_value":     2.0,
        "description":     "Micro E-mini Nasdaq-100",
        "live_only":       False,
    },
    # ── Test / backtest proxy (stooq: SPY.US) ────────────────────────────────
    # SPY tracks the S&P 500 and is used for strategy development and
    # optimization until a reliable futures data feed is connected.
    # Switch DEFAULT_TEST_SYMBOL back to "MES" when futures data is stable.
    "SPY": {
        "ticker":          "SPY",
        "fallback_ticker": "",
        "tick_size":       0.01,
        "tick_value":      0.01,
        "point_value":     1.0,      # $1 per point per share
        "description":     "SPDR S&P 500 ETF (backtest proxy for MES)",
        "live_only":       False,
    },
    "QQQ": {
        "ticker":          "QQQ",
        "fallback_ticker": "",
        "tick_size":       0.01,
        "tick_value":      0.01,
        "point_value":     1.0,      # $1 per point per share
        "description":     "Invesco QQQ ETF (Nasdaq-100, backtest proxy for MNQ)",
        "live_only":       False,
    },
    "GLD": {
        "ticker":          "GLD",
        "fallback_ticker": "",
        "tick_size":       0.01,
        "tick_value":      0.01,
        "point_value":     1.0,      # $1 per point per share
        "description":     "SPDR Gold Shares ETF (US stock, SMART exchange)",
        "live_only":       False,
    },
    "MGC": {
        "ticker":          "MGC=F",
        "fallback_ticker": "GLD",    # ETF proxy for backtesting
        "tick_size":       0.10,     # $0.10 per troy oz
        "tick_value":      1.00,     # $1.00 per tick (0.10 × $10/oz × 10 oz ÷ 10)
        "point_value":     10.0,     # $10 per full point (10 troy oz × $1/oz)
        "description":     "Micro Gold Futures (COMEX)",
        "live_only":       False,
    },
}

# Symbol used by default for backtesting and optimization.
# Set to "SPY" (stooq: SPY.US) until MES=F futures data is reliably accessible.
# Change to "MES" when switching to live futures data.
DEFAULT_INSTRUMENT  = "SPY"
TEST_SYMBOL         = "SPY"
LIVE_SYMBOLS        = ["MES", "MNQ", "MGC"]  # micro futures — paper + live

# ── Timeframes ────────────────────────────────────────────────────────────────
TIMEFRAMES = {
    "1m":  "1m",
    "3m":  "3m",
    "5m":  "5m",
    "15m": "15m",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1d",
    "1w":  "1wk",
}

HTF_BIAS_TIMEFRAMES  = ["1d", "4h"]
ENTRY_TIMEFRAMES     = ["15m", "5m", "3m"]
CONTEXT_TIMEFRAMES   = ["1h", "15m"]

# ── Data fetch ────────────────────────────────────────────────────────────────
LOOKBACK_DAYS        = 365
WARMUP_BARS          = 200  # excluded from backtesting
CACHE_EXPIRY_HOURS   = 4

# ── Kill zones (UTC) ──────────────────────────────────────────────────────────
KILL_ZONES = {
    "asian":  {"start": "00:00", "end": "04:00"},
    "london": {"start": "07:00", "end": "10:00"},
    "ny_am":  {"start": "13:30", "end": "17:00"},
    "ny_pm":  {"start": "18:00", "end": "20:00"},
}

ACTIVE_KILL_ZONES = ["london", "ny_am"]

# ── Session times (UTC) ───────────────────────────────────────────────────────
SESSIONS = {
    "asian":    {"start": "00:00", "end": "09:00"},
    "london":   {"start": "07:00", "end": "16:00"},
    "new_york": {"start": "13:30", "end": "22:00"},
}

ASIAN_SESSION_LOOKBACK_BARS = 48  # bars to define Asian range

# ── Scoring weights ───────────────────────────────────────────────────────────
SCORE_WEIGHTS = {
    "win_rate":       0.35,
    "profit_factor":  0.30,
    "max_drawdown":   0.20,   # applied as (1 - max_drawdown)
    "confluence":     0.15,
}

MIN_CONFLUENCE_CONFIRMATIONS = 3

# Fixed dollar risk per trade for micro futures position sizing.
# qty = FUTURES_RISK_PER_TRADE / (SL_points × point_value)
FUTURES_RISK_PER_TRADE = 10_000   # USD

# ── Risk management ───────────────────────────────────────────────────────────
RISK = {
    "max_risk_per_trade_pct":   1.0,    # % of account per trade
    "max_daily_drawdown_pct":   3.0,    # halt trading for the day
    "max_weekly_drawdown_pct":  6.0,    # halt trading for the week
    "max_open_positions":       2,
    "default_rr_ratio":         2.0,    # reward:risk
    "min_rr_ratio":             1.5,
    "account_size":             1_000_000.0,
    "commission_per_contract":  0.62,   # NinjaTrader default MES/MNQ
}

# ── Optimizer (Sunday) ────────────────────────────────────────────────────────
OPTIMIZER = {
    "n_trials":             500_000,
    "n_jobs":               -1,         # use all CPU cores
    "timeout_hours":        8,
    "sampler":              "TPE",      # TPE | CmaEs | Random
    "pruner":               "hyperband",
    "walk_forward_splits":  5,
    "walk_forward_gap_bars": 20,        # gap between IS and OOS
    "min_trades_per_period": 20,
    "study_name":           "apex_optimizer",
    "storage":              f"sqlite:///{OUTPUT_DIR}/optuna_study.db",
}

# ── Volatility circuit breaker ────────────────────────────────────────────────
VOLATILITY = {
    "atr_period":            14,
    "atr_multiplier_max":    3.0,   # pause if ATR > multiplier * avg ATR
    "vix_pause_threshold":   35.0,
    "low_volatility_filter": 0.3,   # skip if ATR < this multiplier * avg
}

# ── News filter ───────────────────────────────────────────────────────────────
NEWS = {
    "api_url":          os.getenv("NEWS_API_URL", "https://nfs.faireconomy.media/ff_calendar_thisweek.json"),
    "high_impact_only": True,
    "blackout_before_mins": 30,
    "blackout_after_mins":  15,
    "currencies":       ["USD"],
}

# ── NT8 Bridge (kept for reference — live trading now uses IB) ─────────────────
NT8 = {
    "host":             os.getenv("NT8_HOST", "127.0.0.1"),
    "port":             int(os.getenv("NT8_PORT", "36973")),
    "heartbeat_secs":   10,
    "reconnect_delay":  5,
    "max_reconnects":   10,
}

# ── Interactive Brokers Bridge ────────────────────────────────────────────────
IB = {
    "host":          os.getenv("IB_HOST",          "127.0.0.1"),
    "port_paper":    int(os.getenv("IB_PORT_PAPER", "7497")),   # TWS paper / Gateway paper
    "port_live":     int(os.getenv("IB_PORT_LIVE",  "7496")),   # TWS live  / Gateway live
    "client_id":     int(os.getenv("IB_CLIENT_ID",  "1")),
    "timeout_secs":  20,
    "warmup_days":   15,   # days of 15-min bars to fetch for strategy warmup (~1 500 bars)
    "readonly":      False,
}

# ── SMC structure parameters ──────────────────────────────────────────────────
SMC = {
    "swing_lookback":       10,     # bars each side for swing high/low
    "fvg_min_gap_atr":      0.3,    # FVG must be > x * ATR
    "ob_body_pct":          0.6,    # order block body must be > 60% of bar
    "bos_confirmation_bars": 2,     # bars to confirm break of structure
    "choch_lookback":       20,     # bars to look back for CHoCH
    "premium_zone":         0.618,  # fib level for premium
    "discount_zone":        0.382,  # fib level for discount
    "ote_low":              0.618,
    "ote_high":             0.786,
    "liquidity_equal_pct":  0.001,  # % diff to consider swing highs/lows equal
    "breaker_lookback":     30,
    "stop_hunt_atr_mult":   1.5,    # wick must exceed x * ATR to qualify
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOGGING = {
    "level":        os.getenv("LOG_LEVEL", "INFO"),
    "file":         str(LOG_DIR / "apex.log"),
    "max_bytes":    10 * 1024 * 1024,  # 10 MB
    "backup_count": 5,
}
