# walkforward-signal-discovery

Offline research scripts that mine the Polymarket history archive for tradeable signals on 5-minute crypto up/down markets (BTC/XRP) and validate them out-of-sample before they reach the live bot.

Every candidate rule must survive a train-on-past / test-on-future split with multiple-testing correction (Bonferroni/BH, permutation, bootstrap, Wilson CIs, Bayesian Beta posteriors, SPRT). No network, no keys; pure analysis over local `.jsonl` files.

## Scripts

Walk-forward signal search (TRAIN ts <= deploy, TEST ts > deploy):

- `big_data_signal_search.py`: master search, 1108 pre-deploy TRAIN vs 112 post-deploy TEST. Sweeps rule_skip grids plus per-strategy/per-side/per-cd-bucket gating; reports only t-stat > 3.0 signals.
- `stuck_loser_fix_search.py`: three loss-cutters (cd=240 checkpoint, per-side fire cap, real-time unrealized stop), TRAIN to TEST.
- `imbalanced_cap_exit_search.py`: imbalanced cap-exit signal. Sells capped-side positions when the opposite side is under-filled but still bid. Loads ticks.
- `flat_up_strategy_vps.py`: scans the full `market_history.jsonl` (1GB+) for flat-UP $0.43-$0.65 entries. argparse CLI (`--file`, `--out`), Wilson CI, binomial and Mann-Whitney p-values.
- `vps_s1only_extract.py`: extracts the S1only BN-gate subset (bn200 fires, bn120 does not confirm).

Statistical validators:

- `dnlive_threshold_oos.py`: calibrates the `dn_live` threshold on IS (first 60%), validates on OOS (last 40%); Fisher exact plus 10k permutation.
- `part15_decisive.py`: decisive battery on the early-BN to DN-loss prior: binomial-on-fires, Bayesian Beta, SPRT, bootstrap CI, Fisher, BH-FDR, utility decision theory.
- `regime_validator.py`: monster-vs-disaster separability, structural feature analysis, CL-gate side prediction, cold-streak detection. Takes history and recap paths as argv.
- `flip_observer.py`: replays the flip-gate proba distribution per market using each market's true timestamp (fixes a "now"-stamping leak), split pre/post regime cutoff.

Reference modules (importable, no data load):

- `btc_polymarket_patterns.py`: validated BTC candle-sequence patterns (DNDNDN to UP, etc.) with 3-fold walk-forward grades; `get_pattern_signal()`.
- `up_strategies.py`: UP-side strategy functions (`STRATEGY_REGISTRY`, `TIER_A/B_STRATEGIES`) that passed K-fold gates in the scanner.

## Requirements

Python 3.8+. `pip install numpy scipy` for the statistical scripts; the rest are stdlib-only. `flip_observer.py` needs `flip_gate_eval` and `up_strategies.py` needs `up_strategy_scanner_v3_2` output; both live in the trading engine, not here.

## Usage

Most scripts take no args (`python3 big_data_signal_search.py`). Two take paths:

```bash
python3 regime_validator.py market_history.jsonl market_recap_history.jsonl
python3 flat_up_strategy_vps.py --file $DATA_DIR/market_history.jsonl --out flat_up_report.json
```

## Data

Reads `market_history.jsonl` (full per-tick book, 1GB+) and `market_recap_history.jsonl` (per-market fires, winners, `hypo_pnl`) from the private polymarket-data repo. Several scripts hard-code `./data`; point that at the checkout or override the path arguments where exposed. The `.jsonl`/`.json` files are git-ignored and never committed.

## Feedback

Suggestions, improvements, or something not working? Please
[open an issue](https://github.com/needmoneyforhoes/walkforward-signal-discovery/issues/new). Pull requests are welcome too.
