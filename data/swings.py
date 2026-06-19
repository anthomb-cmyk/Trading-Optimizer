"""
Causal swing pivot detection — numpy-only, no heavy dependencies.

Design
------
A swing high at bar ``i`` is the center of a symmetric window
``[i - lb, i + lb]``.  That pivot is ONLY confirmable at bar ``i + lb``,
because that is the first bar at which all ``lb`` right-side neighbours are
available.

Causality guarantee
-------------------
The flag at output index ``t`` depends solely on bars with index ``<= t``.
Specifically:

* The pivot is detected at the CENTERED position ``i`` (standard algorithm).
* The ``True`` flag is PLACED at index ``i + lb`` (the confirmation bar).
* Any pivot whose confirmation bar ``i + lb`` would exceed the array bounds is
  discarded — it cannot yet be confirmed.

Consequence (warmup)
--------------------
The first ``lb`` bars of the output array are always ``False`` because no
pivot can be confirmed that early.  Consumers that look for the MOST RECENT
swing should also note that the latest ``lb`` bars of the price series will
never carry a confirmed swing, even if a genuine pivot exists there — those
bars are still within the confirmation window of any pivot centred near the
end of the series.
"""
import numpy as np


def _swing_highs(highs: np.ndarray, lookback: int) -> np.ndarray:
    """
    Return a boolean array marking CONFIRMED swing-high bars.

    A swing high at pivot bar ``i`` (where ``highs[i]`` is the maximum of the
    window ``highs[i-lb : i+lb+1]``) is reported as ``True`` at index
    ``i + lb`` — the first bar at which the confirmation is available from
    already-seen data.

    Parameters
    ----------
    highs : np.ndarray
        Array of bar high prices.
    lookback : int
        Number of bars on each side of the pivot used for confirmation (``lb``
        in the codebase; typically ``SMC["swing_lookback"]``).

    Returns
    -------
    np.ndarray of bool, same length as ``highs``.
        ``result[t]`` is ``True`` only if bar ``t - lb`` is a swing-high pivot
        that is now confirmed at bar ``t``.  No future bars are referenced.
    """
    n = len(highs)
    result = np.zeros(n, dtype=bool)
    lb = lookback
    # i is the pivot center; confirmation lands at i + lb.
    # i must satisfy: i - lb >= 0  →  i >= lb
    #                 i + lb < n   →  i + lb <= n - 1  (confirmation fits)
    for i in range(lb, n - lb):
        window = highs[i - lb : i + lb + 1]
        if highs[i] == window.max():
            result[i + lb] = True   # place flag at confirmation bar
    return result


def _swing_lows(lows: np.ndarray, lookback: int) -> np.ndarray:
    """
    Return a boolean array marking CONFIRMED swing-low bars.

    A swing low at pivot bar ``i`` (where ``lows[i]`` is the minimum of the
    window ``lows[i-lb : i+lb+1]``) is reported as ``True`` at index
    ``i + lb`` — the first bar at which the confirmation is available from
    already-seen data.

    Parameters
    ----------
    lows : np.ndarray
        Array of bar low prices.
    lookback : int
        Number of bars on each side of the pivot used for confirmation (``lb``
        in the codebase; typically ``SMC["swing_lookback"]``).

    Returns
    -------
    np.ndarray of bool, same length as ``lows``.
        ``result[t]`` is ``True`` only if bar ``t - lb`` is a swing-low pivot
        that is now confirmed at bar ``t``.  No future bars are referenced.
    """
    n = len(lows)
    result = np.zeros(n, dtype=bool)
    lb = lookback
    for i in range(lb, n - lb):
        window = lows[i - lb : i + lb + 1]
        if lows[i] == window.min():
            result[i + lb] = True   # place flag at confirmation bar
    return result
