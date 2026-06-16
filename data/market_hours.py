"""
Market hours utilities: trading day detection, holiday calendar,
futures roll detection, and weekend/holiday bar filtering.
"""
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd

# CME Globex closes: Friday 17:00 CT / Saturday 22:00 UTC
# reopens: Sunday 17:00 CT / Sunday 22:00 UTC
_CME_CLOSE_UTC_DOW = 5   # Saturday
_CME_CLOSE_UTC_H   = 22
_CME_OPEN_UTC_DOW  = 6   # Sunday
_CME_OPEN_UTC_H    = 22

# US market holidays (approximate — expand as needed)
_US_HOLIDAYS = {
    date(2024, 1, 1),   date(2024, 1, 15),  date(2024, 2, 19),
    date(2024, 5, 27),  date(2024, 6, 19),  date(2024, 7, 4),
    date(2024, 9, 2),   date(2024, 11, 28), date(2024, 12, 25),
    date(2025, 1, 1),   date(2025, 1, 20),  date(2025, 2, 17),
    date(2025, 5, 26),  date(2025, 6, 19),  date(2025, 7, 4),
    date(2025, 9, 1),   date(2025, 11, 27), date(2025, 12, 25),
    date(2026, 1, 1),   date(2026, 1, 19),  date(2026, 2, 16),
    date(2026, 5, 25),  date(2026, 6, 19),  date(2026, 7, 4),
    date(2026, 9, 7),   date(2026, 11, 26), date(2026, 12, 25),
}


def is_trading_bar(dt: datetime) -> bool:
    """Return True if *dt* (UTC-aware) falls within CME Globex trading hours."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    # Saturday after 22:00 UTC or Sunday before 22:00 UTC → closed
    if dt.weekday() == 5 and dt.hour >= _CME_CLOSE_UTC_H:
        return False
    if dt.weekday() == 6 and dt.hour < _CME_OPEN_UTC_H:
        return False
    # Public holidays (check UTC date)
    if dt.date() in _US_HOLIDAYS:
        return False
    return True


def filter_trading_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Remove weekend / holiday / off-hours rows from *df*."""
    mask = pd.Series(
        [is_trading_bar(ts) for ts in df.index],
        index=df.index,
        dtype=bool,
    )
    return df[mask]


def is_sunday_optimization_day(dt: Optional[datetime] = None) -> bool:
    """Return True if *dt* is Sunday (optimizer trigger day)."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.weekday() == 6


def next_monday_open_utc(dt: Optional[datetime] = None) -> datetime:
    """Return the next Monday CME open (Sunday 22:00 UTC)."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    days_until_sunday = (6 - dt.weekday()) % 7
    if days_until_sunday == 0 and dt.hour >= _CME_OPEN_UTC_H:
        days_until_sunday = 7
    target = dt.replace(hour=_CME_OPEN_UTC_H, minute=0, second=0, microsecond=0)
    target += pd.Timedelta(days=days_until_sunday)
    return target
