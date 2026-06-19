# Phase 1 Spec: The Honest Measurement System

**Status:** Draft for review
**Owner / Architect:** Anthony (PM) + Claude (architect)
**Date:** 2026-06-19
**Project:** Trading-Optimizer (APEX), ICT/SMC intraday micro-futures bot on Interactive Brokers

---

## Context

A two-part audit (code + external research) found the project has two independent problems:

1. **The engine lies.** Look-ahead bias in the data layer contaminates every signal; fills are frictionless; the optimizer runs 500,000 trials (the False Strategy Theorem says 7 trials on 2yr of data is already enough to make an in-sample Sharpe of 1.0 meaningless). Net effect: no backtest number produced so far is trustworthy.
2. **The strategy thesis is unproven.** ICT/SMC has no peer-reviewed validation in 15+ years; its signals are not reproducible between coders.

**Strategic decision (locked): Direction A.** Build the honest measurement system first (this Phase 1), then honest-test the 3 most defensible existing strategies (Phase 2), keep only survivors, and pivot the strategy layer only if nothing survives (Phase 3). Phase 1 is identical under every direction, so it starts now.

**Data decision (locked): Databento**, pay-as-you-go, dataset `GLBX.MDP3` (CME Globex). $125 free credits to start.

---

## Phase 1 Goal

Produce a backtesting + optimization pipeline whose numbers can be trusted. Concretely: an engine with strict temporal causality, realistic costs, real CME data, overfitting-aware validation, and a test suite that proves the engine is honest.

## Definition of Done (Phase 1 is complete when ALL are true)

- [ ] Engine passes the **causality/repaint test**: signals computed on `data[:k]` are identical to signals on `data[:k+N]` for all indices `< k - warmup`. No signal repaints when future bars arrive.
- [ ] Engine passes the **buy-and-hold reproduction test**: an always-long signal reproduces the instrument's return within costs.
- [ ] Engine passes the **noise test**: on shuffled / phase-randomized data, the optimizer reports no significant edge (Deflated Sharpe < 0.5). If it finds edge in noise, it still leaks.
- [ ] Engine passes **known-answer fixtures**: hand-built bars with one expected trade reconcile exactly, including commissions and slippage.
- [ ] Full pipeline runs end-to-end on **real Databento MES/ES data** (no yfinance in the optimize/backtest path, no silent SPY proxy).
- [ ] Every optimization result reports **Deflated Sharpe Ratio (DSR)** and **Probability of Backtest Overfitting (PBO)**.
- [ ] A **locked out-of-sample holdout** exists that the optimizer cannot read; only the final chosen config is scored on it, once.
- [ ] The optimizer enforces a **data-derived trial budget** (tens to low hundreds, not 500,000).
- [ ] Equity curve is **mark-to-market per bar**; reported max drawdown reflects intratrade adverse excursion.

---

## Tickets

Effort key: **S** < half a day, **M** half to two days, **L** two to five days.

### Epic 1: Eliminate look-ahead bias (highest correctness priority)

**T1.1 - Causal swing detection** | Effort: M
- **Problem:** `_swing_highs` / `_swing_lows` use a centered window `highs[i-lb : i+lb+1]`, confirming a swing at bar `i` with `lb` (default 10) **future** bars. `data/loader.py:233` and `:245`. Swings feed order blocks, BOS, CHoCH, OTE, liquidity, so every downstream signal sees the future.
- **Fix:** A swing pivot may only be marked once it is confirmed by past bars, and the boolean must become `True` only at the confirmation bar (not retroactively at the pivot bar). Two acceptable implementations: (a) keep centered detection for labeling but **shift the result forward by `lb`** so it is only available `lb` bars later; (b) causal detection: pivot confirmed at bar `i` if it is the max/min of the trailing `2*lb` window and the last `lb` bars are lower/higher. Document which is chosen.
- **Acceptance:** Passes T5.3 repaint test for swings.

**T1.2 - Audit all detectors for forward leakage** | Effort: M | depends: T1.1
- **Problem:** `base_strategy` detectors use forward shifts (e.g., `detect_order_blocks` uses `shift(-j)`). Some are legitimate (the order block is only traded on a future retest), but this must be proven, not assumed. `detect_fvg`, `detect_order_blocks`, `detect_bos`, `detect_choch` all need a causality review.
- **Fix:** For each detector, document the exact bar at which its output is first knowable in real time, and ensure the backtest never acts on it before that bar. Where a detector marks a zone at a past bar, the *tradeable* event is the retest, which is fine; encode that explicitly.
- **Acceptance:** Each detector has a one-line causality note; all strategies pass T5.3.

**T1.3 - Per-fold signal generation** | Effort: M | depends: T1.1
- **Problem:** Signals are generated on the **full dataset** before the walk-forward split (`sunday_optimizer.py` calls `generate_signals(self.df, ...)`, then slices). Indicators on the OOS fold are computed with statistics that include OOS (and future) data.
- **Fix:** Generate signals inside each fold from only that fold's bars (plus a causal warmup prefix taken from the IS side). The OOS signal stream must be identical whether or not later folds exist in the frame.
- **Acceptance:** Append-invariance test on OOS signals (subset of T5.3).

### Epic 2: Realistic cost and fill model

**T2.1 - Slippage on both sides, regime-aware** | Effort: S
- **Problem:** Slippage applied at entry only; exits are frictionless. `backtesting/engine.py` (~`:225`).
- **Fix:** Configurable slippage in ticks on **entry and exit**. Default 1 tick in normal conditions; escalate to 3 to 5 ticks when the bar's ATR is in the top regime band or a news-blackout flag is set (research: news-event slippage runs 3 to 5 ticks on MES/MNQ).
- **Acceptance:** Round-trip cost in a fixture matches hand calc.

**T2.2 - Real commission schedule** | Effort: S
- **Problem:** `commission_per_contract = 0.62` understates true cost.
- **Fix:** Model IB micro-futures round-trip realistically: ~$0.25/contract/side IB + ~$0.55/side exchange+NFA = ~$1.60 round-trip. Make it a per-instrument config. Note for the team: on MNQ that is ~3.2 ticks just to break even, which rules out small-target scalping by design.
- **Acceptance:** Fixture P&L includes correct per-contract round-trip.

**T2.3 - Conservative (pessimistic) fill logic** | Effort: M | depends: T2.1
- **Problem:** Stops and limit targets fill at the exact price; when a single bar spans both stop and target, the engine can credit the favorable one.
- **Fix:** (a) Stop fills at stop +/- slippage, and if the bar gaps past the stop, fill at the gap (open). (b) Limit-target fills require the bar to trade **through** the level, not merely touch it. (c) When one bar contains both SL and TP, assume the **stop** hits first (pessimistic intrabar assumption) unless intrabar (1m) data is available to resolve order.
- **Acceptance:** Documented intrabar rule; fixtures for gap-through and both-in-one-bar cases.

**T2.4 - Mark-to-market equity curve** | Effort: S
- **Problem:** Equity is stamped only at exit; intratrade bars carry post-exit equity, so drawdown is understated. `backtesting/engine.py` (~`:262`).
- **Fix:** Update equity each bar inside a trade using unrealized P&L = `(close - fill) * direction * point_value * contracts`.
- **Acceptance:** Max drawdown reflects open-trade adverse excursion in a fixture.

### Epic 3: Real data pipeline (Databento)

**T3.1 - Databento ingestion** | Effort: L
- **Fix:** Add a `databento` client integration pulling `GLBX.MDP3`. Start with `ohlcv-1m`, `ohlcv-5m`, `ohlcv-15m`; keep the option to pull `trades`/`mbp-10` later for tick-level fill modeling. Cache to partitioned parquet keyed by `symbol / timeframe / date`. Respect pay-as-you-go (use the $125 credits; log bytes/credits per pull).
- **History note:** Micros (MES/MNQ/MGC) only exist from ~2019. For longer backtests use full-size **ES / NQ / GC** continuous series (same price path, back to 2010) and apply **micro** point/tick values for P&L and sizing.
- **Acceptance:** One command fetches and caches multi-year MES + ES 15m/5m/1m; second run hits cache.

**T3.2 - Continuous-contract roll handling** | Effort: M | depends: T3.1
- **Fix:** Use Databento continuous symbology (e.g., volume-based lead `ES.v.0` or calendar `ES.c.0`) to build a back-adjusted continuous series; document the roll rule and adjustment so order-block / FVG price levels are consistent across rolls. (Confirm exact symbology against current Databento docs.)
- **Acceptance:** No price discontinuities at roll dates beyond the documented adjustment; roll method written down.

**T3.3 - Remove yfinance + SPY proxy from the research path** | Effort: S | depends: T3.1
- **Fix:** yfinance allowed only for throwaway quick-looks, never for optimize/backtest. Delete the silent `_FUTURES_PROXY_MAP` substitution from the optimization path. **Optimize on the instrument actually traded.** Update `config/settings.py` defaults (`DEFAULT_INSTRUMENT`, `TEST_SYMBOL`) off SPY.
- **Acceptance:** Optimizing "MES" loads MES/ES Databento data; no SPY anywhere in the run logs.

**T3.4 - Data integrity checks** | Effort: S | depends: T3.1
- **Fix:** Validate CME session alignment (CT timezone), detect/flag gaps, reject unexpected weekend bars, assert monotonic timestamps. Fail loudly rather than silently proceeding on bad data.
- **Acceptance:** Bar counts per session match expectations on a sample week; corrupt input raises.

### Epic 4: Validation and overfitting control

**T4.1 - Purged + embargoed CV / CPCV** | Effort: L | depends: T1.3
- **Fix:** Replace bare walk-forward with purged, embargoed cross-validation (Lopez de Prado): purge training rows whose labels overlap the test window, add an embargo gap proportional to the trade-holding horizon. Add Combinatorial Purged CV (CPCV) to produce a **distribution** of OOS Sharpes (input to DSR/PBO), not a single number.
- **Acceptance:** CV produces N OOS paths; embargo/purge configurable; documented.

**T4.2 - Deflated Sharpe + PBO metrics** | Effort: M | depends: T4.1
- **Fix:** Implement DSR (Bailey/Lopez de Prado) and PBO via CSCV in `backtesting/metrics.py`. DSR needs the number of effective trials and the cross-trial Sharpe variance; PBO needs the CPCV path matrix.
- **Acceptance:** Both computed and logged on every optimization run; unit-tested against a worked example.

**T4.3 - Trial-budget governance** | Effort: M | depends: T4.2
- **Fix:** Cap Optuna trials at a number **derived from data length** (False Strategy Theorem), not 500,000. Track the *effective* number of trials (cluster correlated configs). Surface DSR/PBO alongside the best value so an overfit search is visibly flagged.
- **Acceptance:** Default trial count is data-derived (tens to low hundreds); run report shows effective-N, DSR, PBO.

**T4.4 - Locked out-of-sample holdout** | Effort: S | depends: T3.1
- **Fix:** Carve off the most recent ~20% of data as a holdout that is **not loaded** during optimize. Only the single final chosen config is scored on it, exactly once, and that number is the headline result.
- **Acceptance:** Optimizer cannot access holdout rows (enforced in the loader, not by convention); a final `evaluate-holdout` step exists.

**T4.5 - Fix the scoring function** | Effort: S
- **Problem:** `SCORE_WEIGHTS` weights win_rate 0.35, a classic trap that rewards strategies that win small and lose big.
- **Fix:** Reweight toward risk-adjusted, overfitting-aware metrics (DSR, profit factor, return/maxDD); demote raw win rate.
- **Acceptance:** New scoring documented; re-scoring a known win-rate-trap strategy ranks it lower.

### Epic 5: Engine-honesty acceptance tests (the proof)

**T5.1 - Buy-and-hold reproduction** | Effort: S
- Always-long signal reproduces instrument return within costs.

**T5.2 - Noise / shuffle test** | Effort: M | depends: T4.2
- On phase-randomized or block-shuffled returns, the optimizer must report DSR < 0.5 (no edge in noise). This is the single most important honesty check.

**T5.3 - Repaint / causality harness** | Effort: M | depends: T1.1, T1.3
- Generic harness: for any strategy, assert signals on `data[:k]` equal signals on the full series at all indices `< k - warmup`.

**T5.4 - Known-answer fixtures** | Effort: S | depends: T2.x
- Tiny hand-built OHLCV series with a known single trade; assert exact P&L including commission and slippage.

### Epic 6: Config realism (supports honest sizing)

**T6.1 - Realistic account and risk config** | Effort: S
- **Problem:** `account_size = 1_000_000`, `FUTURES_RISK_PER_TRADE = 10_000`. Fantasy numbers distort backtest sizing.
- **Fix:** Set account size and per-trade risk to Anthony's real intended figures; express risk as a percentage with a hard contract cap. (Live-bridge sizing safety belongs to Phase 4, but the backtest must size realistically now.)
- **Acceptance:** Backtest sizing matches a hand calc at the real account size.

---

## Suggested execution order

1. **T3.1, T3.3** (real data in, yfinance out) and **T1.1** (causal swings) in parallel. These unblock everything.
2. **T1.2, T1.3** (finish causality) then **T5.3** (lock it with the repaint harness).
3. **T2.1 to T2.4** (costs/fills) then **T5.1, T5.4** (reproduction + fixtures).
4. **T3.2, T3.4** (rolls + integrity), **T6.1** (config).
5. **T4.1 to T4.5** (validation stack) then **T5.2** (noise test) as the final gate.

When T5.1 to T5.4 are green on Databento data, Phase 1 is done and Phase 2 (honest-test the 3 salvageable strategies on the locked holdout) begins.

---

## Out of scope for Phase 1 (later phases)

- **Phase 2:** honest-test `order_block_reversal`, `kill_zone_reversal`, `daily_bias_intraday` (as a filter) on real data with the locked holdout; fix strategy-specific logic bugs (see appendix); keep survivors.
- **Phase 3 (fork):** harden survivors + regime filter (VIX/ADX), OR pivot strategy layer to evidence-based edges (ORB+volume, VWAP reversion, gap fade, short-term reversal).
- **Phase 4:** live-bridge risk hardening (wire `DrawdownManager`, position-size cap in `_calc_qty`, fractional Kelly, kill switch, startup reconciliation).
- **Phase 5:** weeks of paper trading; live-vs-backtest fill comparison.
- **Phase 6:** tiny real capital + monitoring.
- **Phase 7 (differentiator):** LLM news/sentiment overlay, measured additively.

---

## Appendix A: Known strategy bugs (Phase 2 backlog, do not lose)

- `asian_session.py:143` and `:151` - operator-precedence bug: in default "breakout" mode `long_conf`/`short_conf` max out at 2 but the gate needs 3, so the strategy **never fires**. Wrap the ternary in parentheses.
- `fvg_retest.py` (~`:149`) - confluence slot `[2]` duplicates `in_bull_fvg`, secretly lowering the strategy's own confluence threshold.
- `ote_fibonacci.py` - `retracement_min` declared but never used (missing filter); swing-size math uses a fixed `shift(lb)` instead of the paired swing low.
- `market_structure_shift.py` - RSI-divergence window ignores its own parameter and uses a 50-bar window, making that confluence slot almost always true.
- `stop_hunt_continuation.py:94` - hardcoded `0.6` trend threshold, not tunable.
- `breaker_block.py` - displacement check lacks a body filter (a large-range doji qualifies).

## Appendix B: Evidence base (one-line citations)

- False Strategy Theorem / DSR / PBO: Bailey & Lopez de Prado (Deflated Sharpe 2014; PBO 2016; AFML 2018). 7 trials on 2yr daily, 45 on 5yr, make an IS Sharpe of 1.0 meaningless.
- ICT has no peer-reviewed edge; founder blew the 2024 Robbins Cup (audited); signals non-reproducible across coders.
- Realistic retail intraday Sharpe after costs: 0.5 to 1.2. MNQ round-trip ~3.2 ticks to break even.
- Data: yfinance caps intraday at 60 days and lacks real futures; Databento `GLBX.MDP3` from 2010, ~$2/week of ES history, $125 free credits.
- Regime filters (VIX/ADX) have real, if modest, supporting evidence (~+34% return/DD in one published test).
