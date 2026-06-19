"""
APEX Optimizer — entry point.

Usage:
    python main.py [command] [options]

Commands:
    fetch       Download and cache market data
    backtest    Run backtest for a strategy
    optimize    Run Sunday optimizer (Optuna)
    report      Generate HTML performance report
    connect     Test IB TWS connection (paper port 7497)
    live        Start IB live bridge (MES / MNQ / MGC micro futures via Interactive Brokers)
"""
import argparse
import sys

from config.logger import get_logger
from config.settings import TEST_SYMBOL, LIVE_SYMBOLS, INSTRUMENTS

log = get_logger("apex")

_TEST_CHOICES = [TEST_SYMBOL] + LIVE_SYMBOLS        # SPY, MES, MNQ
_LIVE_CHOICES = [TEST_SYMBOL] + LIVE_SYMBOLS        # SPY (paper only), MES, MNQ
_FUTURES      = set(LIVE_SYMBOLS)                   # MES, MNQ — can trade live
_ALL_SYMBOLS  = set(INSTRUMENTS)                    # SPY, QQQ, MES, MNQ


def cmd_fetch(args):
    from data.loader import load_multi_timeframe
    log.info("Fetching data for %s…", args.symbol)
    dfs = load_multi_timeframe(
        symbol=args.symbol,
        timeframes=args.timeframes.split(",") if args.timeframes else None,
        use_cache=False,
    )
    for tf, df in dfs.items():
        log.info("  %s: %d bars (%.2f MB)", tf, len(df),
                 df.memory_usage(deep=True).sum() / 1e6)


def cmd_backtest(args):
    from data.loader import load_bars
    from backtesting.engine import BacktestEngine
    from strategies import get_strategy, STRATEGY_REGISTRY
    from strategies.system import StrategySystem

    names = list(STRATEGY_REGISTRY) if args.strategy == "all" else [args.strategy]
    engine = BacktestEngine(args.symbol)

    for name in names:
        log.info("Backtesting strategy: %s on %s", name, args.symbol)
        strat = get_strategy(name, args.symbol)
        df    = load_bars(args.symbol, "15m")
        p     = strat.default_params
        sigs  = strat.generate_signals(df, p)
        result = engine.run(df, sigs, p)
        log.info("  %s", result.summary())


def cmd_optimize(args):
    from optimizer.sunday_optimizer import SundayOptimizer
    from config.settings import OPTIMIZER

    opt = SundayOptimizer(
        symbol    = args.symbol,
        timeframe = args.timeframe,
        n_trials  = args.trials or OPTIMIZER["n_trials"],
        n_jobs    = args.jobs   or OPTIMIZER["n_jobs"],
        timeout_h = args.timeout or OPTIMIZER["timeout_hours"],
        use_wf    = not args.no_wf,
    )
    opt.run()


def cmd_optimize_strategy(args):
    from optimizer.strategy_optimizer import StrategyOptimizer

    optimizer = StrategyOptimizer(
        strategy_name = args.strategy,
        symbol        = args.symbol,
        timeframe     = args.timeframe,
        n_trials      = args.trials,
        n_jobs        = args.jobs,
        timeout_h     = args.timeout,
        use_wf        = not args.no_wf,
        holdout_frac  = args.holdout_frac,
    )
    result = optimizer.run()

    # Headline verdict
    print("\n" + "=" * 60)
    print(f"STRATEGY OPTIMIZER RESULT: {args.strategy}")
    print("=" * 60)
    print(f"  Best train OOS score : {result.get('best_score', 0):.4f}")
    print(f"  Holdout score        : {result.get('holdout_score', 0):.4f}")
    print(f"  Deflated Sharpe (DSR): {result.get('deflated_sharpe', 0):.4f}")
    pbo_val = result.get("pbo")
    print(f"  PBO                  : {'UNDEFINED' if pbo_val is None else f'{pbo_val:.4f}'}")
    wf = result.get("walk_forward", {})
    print(f"  WF robust            : {wf.get('is_robust', 'N/A')}  "
          f"(pass_rate={wf.get('pass_rate', 0):.0%})")
    hm = result.get("holdout_metrics", {})
    print(f"  Holdout PF           : {hm.get('profit_factor', 0):.2f}")
    print(f"  Holdout Sharpe       : {hm.get('sharpe', 0):.2f}")
    print(f"  Holdout Trades       : {hm.get('total_trades', 0)}")
    print(f"  Holdout MaxDD        : {hm.get('max_drawdown', 0):.1%}")
    print(f"  Holdout net PnL      : "
          f"{hm.get('gross_profit', 0) - hm.get('gross_loss', 0):.0f}")
    print(f"  Is robust            : {result.get('is_robust', False)}")
    print("=" * 60)


def cmd_report(args):
    log.info("Reporting not yet implemented — coming in Phase 7.")


def cmd_connect(args):
    """Verify TWS connectivity and report account/server info."""
    try:
        from ib_insync import IB, util
    except ImportError:
        log.error("ib_insync not installed. Run: pip install ib_insync")
        sys.exit(1)

    from config.settings import IB as IB_CFG

    port = args.port or IB_CFG["port_paper"]
    log.info("Connecting to IB TWS at %s:%d (clientId=%d)…",
             IB_CFG["host"], port, args.client_id)

    ib = IB()

    async def _test():
        try:
            await ib.connectAsync(
                IB_CFG["host"], port,
                clientId = args.client_id,
                timeout  = 10,
                readonly = True,
            )
        except Exception as exc:
            log.error("Connection failed: %s", exc)
            log.error(
                "Make sure TWS is running with API enabled on port %d.\n"
                "  TWS: File > Global Configuration > API > Settings\n"
                "  Enable: 'Enable ActiveX and Socket Clients'\n"
                "  Socket port: %d", port, port,
            )
            return

        log.info("Connected!")
        log.info("  Server version  : %s", ib.client.serverVersion())
        log.info("  Managed accounts: %s", ib.managedAccounts())

        # Quick account summary — NetLiquidation, AvailableFunds, BuyingPower
        try:
            summary = await ib.accountSummaryAsync()
            for item in summary:
                if item.tag in ("NetLiquidation", "AvailableFunds", "BuyingPower"):
                    log.info("  %-20s %s %s", item.tag, item.value, item.currency)
        except Exception as exc:
            log.warning("Account summary unavailable: %s", exc)

        ib.disconnect()
        log.info("Disconnected. Connection test PASSED.")

    util.run(_test())


def cmd_live(args):
    from ib_bridge import IBBridge
    from config.settings import OUTPUT_DIR

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    for sym in symbols:
        if sym not in _ALL_SYMBOLS:
            log.error(
                "Unknown symbol '%s'. Valid symbols: %s", sym, sorted(_ALL_SYMBOLS)
            )
            sys.exit(1)
        if sym not in _FUTURES and not args.paper:
            log.error(
                "'%s' is an ETF/stock — add --paper for paper trading.", sym
            )
            sys.exit(1)

    etfs = [s for s in symbols if s not in _FUTURES]
    if etfs:
        log.info("Paper trading ETF(s) %s via SMART exchange.", etfs)

    bridge = IBBridge(
        symbols     = symbols,
        params_file = OUTPUT_DIR / "best_params.json",
        paper       = args.paper,
    )
    bridge.run()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="apex",
        description="APEX Optimizer — SMC algo trading system",
    )
    sub = p.add_subparsers(dest="command")

    # fetch
    pf = sub.add_parser("fetch", help="Download and cache market data")
    pf.add_argument("--symbol",     default=TEST_SYMBOL, choices=_TEST_CHOICES,
                    help=f"Symbol to fetch (default: {TEST_SYMBOL})")
    pf.add_argument("--timeframes", default=None,
                    help="Comma-separated timeframes, e.g. 15m,1h,4h")
    pf.set_defaults(func=cmd_fetch)

    # backtest
    pb = sub.add_parser("backtest", help="Run a strategy backtest")
    pb.add_argument("--strategy",  default="all")
    pb.add_argument("--symbol",    default=TEST_SYMBOL, choices=_TEST_CHOICES,
                    help=f"Symbol to backtest (default: {TEST_SYMBOL})")
    pb.add_argument("--timeframe", default="15m")
    pb.set_defaults(func=cmd_backtest)

    # optimize
    po = sub.add_parser("optimize", help="Run Sunday Optuna optimizer")
    po.add_argument("--symbol",    default=TEST_SYMBOL, choices=_TEST_CHOICES,
                    help=f"Symbol to optimize (default: {TEST_SYMBOL})")
    po.add_argument("--timeframe", default="15m",
                    help="Timeframe for optimization (default 15m; SMC strategies designed for intraday)")
    po.add_argument("--trials",    type=int,   default=None)
    po.add_argument("--jobs",      type=int,   default=None)
    po.add_argument("--timeout",   type=float, default=None, help="Hours")
    po.add_argument("--no-wf",     action="store_true", help="Skip walk-forward")
    po.set_defaults(func=cmd_optimize)

    # optimize-strategy
    pos = sub.add_parser(
        "optimize-strategy",
        help="Run single-strategy honest optimizer (honest DSR/PBO, locked holdout)",
    )
    pos.add_argument("--strategy",  required=True,
                     help="Strategy name, e.g. short_term_reversal")
    pos.add_argument("--symbol",    default="MES")
    pos.add_argument("--timeframe", default="15m")
    pos.add_argument("--trials",    type=int,   default=None,
                     help="Optuna trial budget (default: data-derived rec)")
    pos.add_argument("--jobs",      type=int,   default=1)
    pos.add_argument("--timeout",   type=float, default=None,
                     help="Timeout in hours (0 / omit = none)")
    pos.add_argument("--no-wf",     action="store_true",
                     help="Skip walk-forward validation inside the objective")
    pos.add_argument("--holdout-frac", type=float, default=0.20,
                     dest="holdout_frac",
                     help="Fraction of bars reserved as locked holdout (default 0.20)")
    pos.set_defaults(func=cmd_optimize_strategy)

    # report
    pr = sub.add_parser("report", help="Generate HTML performance report")
    pr.set_defaults(func=cmd_report)

    # connect — IB TWS connection test
    pc = sub.add_parser("connect", help="Test IB TWS connection")
    pc.add_argument("--port",      type=int, default=None,
                    help="TWS socket port (default: 7497 paper)")
    pc.add_argument("--client-id", type=int, default=10, dest="client_id",
                    help="IB client ID (default: 10)")
    pc.set_defaults(func=cmd_connect)

    # live — IB live/paper trading
    pl = sub.add_parser("live", help="Start IB live bridge")
    pl.add_argument(
        "--symbols", default="MES,MNQ,MGC",
        help=(
            "Comma-separated micro futures to trade, e.g. MES,MNQ,MGC  "
            "(add --paper for paper trading; ETFs SPY/QQQ/GLD also supported with --paper)"
        ),
    )
    pl.add_argument("--paper",  action="store_true",
                    help="Paper trading mode (port 7497; required for ETFs)")
    pl.set_defaults(func=cmd_live)

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
