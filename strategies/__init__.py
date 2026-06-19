"""
APEX Strategy Registry — 10 SMC strategies in system hierarchy order.

System roles:
  bias_filter    → S10  DailyBiasIntraday         sets daily direction
  timing_filter  → S6   KillZoneReversal           gates entry windows
  primary        → S1-3 LiquiditySweep, OrderBlock, FVGRetest
  confirmation   → S4,5,7,8,9 Breaker, MSS, StopHunt, OTE, Asian
"""
from strategies.liquidity_sweep         import LiquiditySweepReversal       # S1  primary
from strategies.order_block_reversal    import OrderBlockReversal            # S2  primary
from strategies.fvg_retest              import FVGRetest                     # S3  primary
from strategies.breaker_block           import BreakerBlockReversal          # S4  confirmation
from strategies.market_structure_shift  import MarketStructureShiftEntry     # S5  confirmation
from strategies.kill_zone_reversal      import KillZoneReversal              # S6  timing_filter
from strategies.stop_hunt_continuation  import StopHuntContinuation          # S7  confirmation
from strategies.ote_fibonacci           import OTEFibonacciEntry             # S8  confirmation
from strategies.asian_session           import AsianSessionRangeBreakout     # S9  confirmation
from strategies.daily_bias_intraday     import DailyBiasIntraday             # S10 bias_filter

# Additional intraday strategies
from strategies.opening_range_breakout  import OpeningRangeBreakout          # opening_range_breakout
from strategies.vwap_reversion          import VWAPReversion                 # vwap_reversion
from strategies.gap_fade                import GapFade                       # gap_fade
from strategies.short_term_reversal     import ShortTermReversal             # short_term_reversal

ALL_STRATEGIES = [
    LiquiditySweepReversal,    # 1
    OrderBlockReversal,         # 2
    FVGRetest,                  # 3
    BreakerBlockReversal,       # 4
    MarketStructureShiftEntry,  # 5
    KillZoneReversal,           # 6
    StopHuntContinuation,       # 7
    OTEFibonacciEntry,          # 8
    AsianSessionRangeBreakout,  # 9
    DailyBiasIntraday,          # 10
    OpeningRangeBreakout,       # opening_range_breakout
    VWAPReversion,              # vwap_reversion
    GapFade,                    # gap_fade
    ShortTermReversal,          # short_term_reversal
]

STRATEGY_REGISTRY: dict = {cls.name: cls for cls in ALL_STRATEGIES}

# Grouped by system role
BY_ROLE = {
    "bias_filter":   [DailyBiasIntraday],
    "timing_filter": [KillZoneReversal],
    "primary":       [LiquiditySweepReversal, OrderBlockReversal, FVGRetest],
    "confirmation":  [BreakerBlockReversal, MarketStructureShiftEntry,
                      StopHuntContinuation, OTEFibonacciEntry, AsianSessionRangeBreakout],
}


def get_strategy(name: str, symbol: str = "MES"):
    """Instantiate a strategy by its name string."""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {sorted(STRATEGY_REGISTRY)}"
        )
    return cls(symbol=symbol)
