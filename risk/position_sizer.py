"""
Fixed-fractional position sizer with optional confluence scaling.

Base formula:
    contracts = floor(equity × risk_pct / (sl_distance_pts × point_value))

Confluence scaling (optional, default ON):
    Each confirmation signal above min_confluence adds scale_per_confluence × base
    additional contracts, capped at max_scale_steps extra tiers.

    Example at risk_pct=1%, equity=$25k, base=2 contracts, scale=0.25, conf=5, min=3:
        extra_steps = min(5-3, 3) = 2
        scaled = floor(2 × (1 + 2 × 0.25)) = floor(3.0) = 3 contracts

Result is always clamped to [min_contracts, max_contracts].
"""
import math

from config.logger import get_logger
from config.settings import RISK

log = get_logger(__name__)


class PositionSizer:

    def __init__(
        self,
        point_value:            float,
        tick_size:              float,
        risk_pct:               float = RISK["max_risk_per_trade_pct"] / 100,
        min_contracts:          int   = 1,
        max_contracts:          int   = 500,
        use_confluence_scaling: bool  = True,
        scale_per_confluence:   float = 0.25,
        max_scale_steps:        int   = 3,
    ):
        self.point_value            = point_value
        self.tick_size              = tick_size
        self.risk_pct               = risk_pct
        self.min_contracts          = min_contracts
        self.max_contracts          = max_contracts
        self.use_confluence_scaling = use_confluence_scaling
        self.scale_per_confluence   = scale_per_confluence
        self.max_scale_steps        = max_scale_steps

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_contracts(
        self,
        equity:           float,
        sl_distance_pts:  float,
        confluence_count: int = 0,
        min_confluence:   int = 3,
    ) -> int:
        """
        Return number of contracts to trade.

        equity           : current account equity ($)
        sl_distance_pts  : abs(entry_price - stop_loss) in price points
        confluence_count : number of confirmations present on this setup
        min_confluence   : minimum required (from MIN_CONFLUENCE_CONFIRMATIONS)
        """
        if sl_distance_pts < self.tick_size or equity <= 0:
            return self.min_contracts

        risk_dollars = equity * self.risk_pct
        base         = risk_dollars / (sl_distance_pts * self.point_value)

        if self.use_confluence_scaling and confluence_count > min_confluence:
            extra  = min(confluence_count - min_confluence, self.max_scale_steps)
            scale  = 1.0 + extra * self.scale_per_confluence
            base  *= scale

        contracts = int(math.floor(base))
        return max(self.min_contracts, min(contracts, self.max_contracts))

    def dollar_risk(self, sl_distance_pts: float, contracts: int) -> float:
        """Dollar amount risked before commissions."""
        return sl_distance_pts * self.point_value * contracts

    def effective_risk_pct(self, equity: float, sl_distance_pts: float, contracts: int) -> float:
        """Actual fraction of equity being risked."""
        if equity <= 0:
            return 0.0
        return self.dollar_risk(sl_distance_pts, contracts) / equity
