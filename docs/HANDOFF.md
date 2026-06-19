# APEX / Trading-Optimizer - Conversation Handoff (2026-06-19)

Read this top to bottom before doing anything. It is the complete state so a fresh session can continue with zero prior context.

## TL;DR
We took an Interactive Brokers ICT/SMC futures bot and rebuilt it into a TRUSTWORTHY honest backtesting + validation engine, then used that engine to test strategies on real MES data. Verdict so far: NO validated edge in ICT, nor in four textbook price-only edges (even under a rigorous multi-year re-test). A sentiment overlay is built but cannot be honestly tested with free data. The durable, valuable deliverable is the honest engine itself. Price-only technical-analysis edges are a dead end here; do not re-test them.

## Repo, environment, infra
- GitHub: https://github.com/anthomb-cmyk/Trading-Optimizer  |  working clone: `/Users/anthonymakeen/Documents/Trading-Optimizer-AH`  |  branch `main`  |  latest commit `23f91d7`.
- Python: the SYSTEM `python3` is 3.13 and runs everything (the engine is pure numpy/pandas, no vectorbt). `python3.11` exists at `~/.local/bin/python3.11` if ever needed for optuna-heavy runs. Run tests with `for f in tests/test_*.py; do python3 "$f"; done` (PYTHONPATH=. for app scripts).
- Data: Databento (dataset `GLBX.MDP3`), continuous `ES.c.0`/`NQ.c.0`/`GC.c.0` (MES/MNQ/MGC map to full-size for history). Key is in `.env` as `DATABENTO_API_KEY` (validated, working). Cache: `data/cache/`. yfinance is OFF the futures path (legacy quick-look only). NOTE: there are real data holes in the Databento history (a few degraded/missing days) and the cache fix refetches when a longer `days` window isn't covered.
- Supabase: project `umqdxhvilenqmrbqawhi`. Logging via `config/run_logger.py` -> tables `runs`, `events`, `backtest_runs`, `optimization_runs`, `data_fetches`, `trades`. `.env` has `SUPABASE_URL`; you must add `SUPABASE_KEY` (service_role) for logs to persist, and confirm the migration deployed (push to `main` triggers `.github/workflows/supabase-deploy.yml`, or run `supabase db push`).
- GDELT (sentiment news): FREE but hard rate-limited (persistent HTTP 429) and recent-biased. `data/gdelt_news.py` degrades gracefully (returns empty). Not reliable for history; live needs a paid feed.
- `.env` (gitignored, never commit): `DATABENTO_API_KEY` (set), `SUPABASE_KEY` (add service_role), `ANTHROPIC_API_KEY` (NOT set; only needed to activate Claude-Haiku headline scoring, else `config/llm_sentiment.py` falls back to free GDELT tone).

## The honest engine (the real asset)
- Causal signals (no look-ahead): `data/swings.py` (confirm-and-shift swings, flagged at the confirmation bar + pivot price carried causally), `strategies/detectors.py` (FVG/OB/BOS/CHoCH; OB zone carried to the displacement bar). Every feature has an append-invariance (no-repaint) test.
- Realistic costs: `backtesting/engine.py` - two-sided slippage, round-trip commission, pessimistic fills (gap-through stops fill at the open, TP needs trade-through, SL assumed before TP intrabar), mark-to-market equity. Pure numpy/pandas (NO vectorbt).
- Validation: `backtesting/walk_forward.py` (per-fold causal signal generation, purge/embargo, CPCV, `recommended_max_trials`), `backtesting/metrics.py` (`deflated_sharpe_ratio`, `probability_backtest_overfitting`, `compute_score_v2` which demotes win-rate, and `_clean_equity_curve` which fixed a Sharpe phantom-return bug). TRUST the locked-holdout score + Deflated Sharpe.
- Per-strategy optimizer: `optimizer/strategy_optimizer.py`. Run: `PYTHONPATH=. python3 -m optimizer.strategy_optimizer --strategy <name> --symbol MES --timeframe 15m --days 912 --trials 400 --jobs 1 --timeout 0.25 --min-trades-per-year 50` (locked 20% holdout, trade-frequency floor, DSR/PBO, Supabase logging). Also `python3 main.py optimize-strategy ...`. System optimizer: `optimizer/sunday_optimizer.py`.
- ~15 test suites in `tests/` (causality + honesty). All pass; the GDELT live test SKIPs on 429.

## Strategies (all registered in strategies/__init__.py)
- ICT (legacy) - NO EDGE: liquidity_sweep, order_block_reversal, fvg_retest, breaker_block, market_structure_shift, kill_zone_reversal, asian_session, daily_bias_intraday, ote_fibonacci, stop_hunt_continuation; combined in `strategies/system.py`.
- Evidence-based - NO VALIDATED EDGE: opening_range_breakout, vwap_reversion, gap_fade, short_term_reversal; plus `strategies/regime.py` (ADX + ATR-percentile, `regime_mode` param). `gap_fade` is the ONLY one positive on the holdout (PF 1.14, Sharpe 0.35) but DSR 0.21 = most likely noise.
- Sentiment (built, untested): `strategies/sentiment_momentum.py`, `strategies/sentiment_feature.py` (causal, decayed news tone), `data/gdelt_news.py`, `config/llm_sentiment.py`.

## The verdict (measured on real MES 15m)
- ICT: holdout 0, walk-forward 100% degradation. No edge.
- 4 textbook edges: rigorous 2.5yr (58k bars) + powered 183-day holdout + trade-frequency floor + regime gating + 400 trials each -> all Deflated Sharpe 0.13-0.25 (< 0.5), all is_robust=False. No validated edge. gap_fade is the only flicker.
- Sentiment: cannot be honestly backtested (free GDELT rate-limited + LLM training-overlap bias). The only clean test is FORWARD paper trading.

## Hard-won learnings (do not relearn these)
- Look-ahead bias was the original killer (swings used future bars). Any new feature MUST be causal and have an append-invariance test.
- DSR must use the PER-OBSERVATION Sharpe (not annualized x large n_obs, which falsely saturates DSR to 1.0). PBO returns None when too few OOS configs.
- Costs are pessimistic on purpose. Backtests should read worse and truer.
- 8GB M1: run optimizations SEQUENTIALLY, single-core (`--jobs 1`). Do not parallelize heavy/data-loading jobs (memory).
- LLM-sentiment backtests are training-overlap-compromised; forward paper is the only valid test.
- Workflow sub-agents that make live GDELT calls HANG on the 429 rate limit. Keep live network calls out of agent builds/tests.
- Conventions: NO em-dashes. Commit/push only when asked; gate on tests; never stage `.env` / `*.parquet` / `*.db`. End commit messages with the Co-Authored-By line.

## Open items
- Add `SUPABASE_KEY` (service_role) to `.env`; confirm the Supabase migration deployed (Actions tab).
- Confirm real account size (placeholder $50k in `config/settings.py` RISK; `MAX_CONTRACTS_PER_TRADE=10` cap is defined but enforcement in the live IB bridge is still Phase-4 work).
- The live IB bridge (`ib_bridge.py`) still needs risk hardening before any real/paper trading: wire `DrawdownManager`, enforce the contract cap in `_calc_qty`, add startup order/position reconciliation.
- A stale "phase3-evidence-based-strategies" workflow may show as running in the UI panel; harmless (its work is committed), stop it from the panel.

## Honest odds (my calibrated estimate)
- A profitable strategy in the CURRENT repo: ~5% (gap_fade maybe 15-20% it is a weak real edge).
- EVER building a working bot, with sustained disciplined effort riding on this engine: ~20-30% for modest durable profit, ~5-10% for meaningful income, ~1-2% for "set and forget rich."
- Price-only TA is a dead end here. The realistic shots are alt-data/sentiment (forward-tested) or genuinely different markets/mechanisms. Treat this as a low-time, high-discipline experiment; real estate (BRRRR) and Socle are the higher-probability wealth engines.

## Recommended next goal
Get ONE genuinely different, honestly-validated edge to forward-paper, or conclude cheaply that none exists. Concretely: (1) set up IB paper forward trading (port 7497) for gap_fade + log to Supabase for real in-time out-of-sample evidence (harden `ib_bridge.py` first); (2) use the honest per-strategy optimizer to test 2-3 genuinely DIFFERENT edge ideas (event/alt-data/cross-asset/microstructure, NOT price-only TA), killing anything with DSR < 0.5 on the untouched holdout; (3) decide go/no-go on the sentiment strategy, which needs a reliable paid news feed. Be brutally honest, that is what the engine is for.

Full build log: `docs/PROGRESS.md`. Original plan: `docs/PHASE1_SPEC.md`.
