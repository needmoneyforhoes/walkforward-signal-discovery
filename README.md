# walkforward-signal-discovery

Walk-forward / out-of-sample signal discovery over full Polymarket history, with
statistical validators and per-tick extractors.

Part of a private Polymarket quant-trading suite. This repo is the **offline research
arm**: it mines the historical tick + recap archive for tradeable signals on
**Polymarket 5-minute crypto up/down markets** (BTC/XRP) and subjects every candidate
to honest, leak-free statistical validation before it is ever wired into the live bot.

## Why it exists

Live-bot edges decay and overfit. Before any rule is deployed it must survive a strict
**train-on-the-past / test-on-the-future** split with proper multiple-testing
correction. These scripts run that gauntlet — Bonferroni/BH correction, permutation
and bootstrap tests, Wilson CIs, Bayesian Beta posteriors, SPRT — so only signals with
real out-of-sample (OOS) edge graduate to the trading engine.

## What's inside

The scripts fall into three groups.

### Walk-forward signal search (TRAIN ≤ deploy_ts → TEST > deploy_ts)

| Script | What it does |
| --- | --- |
| `big_data_signal_search.py` | Master honest search: 1108 pre-deploy TRAIN vs 112 post-deploy TEST. Sweeps rule_skip grids, per-strategy/per-side/per-cd-bucket gating; reports only t-stat > 3 signals. |
| `stuck_loser_fix_search.py` | Tests three loss-cutters (early cd=240 checkpoint, per-side fire cap, real-time unrealized stop), TRAIN→TEST. |
| `imbalanced_cap_exit_search.py` | "Imbalanced cap exit" signal — sell capped-side positions when the opposite side is under-filled but still bid. Loads ticks for the search universe. |
| `flat_up_strategy_vps.py` | VPS-scale scan of the 1GB+ history for the flat-UP $0.43–0.65 entry; argparse CLI (`--file`, `--out`), Wilson CI + binomial/Mann-Whitney p-values. |
| `vps_s1only_extract.py` | Extracts the "S1only" BN-gate market subset (bn200 fires, bn120 doesn't confirm) for offline validation. |

### Statistical validators (OOS / decisive significance)

| Script | What it does |
| --- | --- |
| `dnlive_threshold_oos.py` | Calibrates the `dn_live` discriminator threshold on IS (first 60%), validates on OOS (last 40%); Fisher exact + 10k permutation. |
| `part15_decisive.py` | Decisive battery on the early-BN→DN-loss prior: binomial-on-fires, Bayesian Beta, SPRT, bootstrap CI, Fisher, BH-FDR, utility decision theory. |
| `regime_validator.py` | Monster-vs-disaster regime separability: structural feature analysis, CL-gate side prediction, cold-streak detection. Takes history + recap paths as argv. |
| `flip_observer.py` | Replays the flip-gate proba distribution per market using each market's TRUE timestamp (fixes a "now"-stamping leak), split pre/post regime cutoff. |

### Reference signal modules (importable, no data load)

| Module | What it does |
| --- | --- |
| `btc_polymarket_patterns.py` | Validated BTC candle-sequence patterns (DNDNDN→UP, etc.) with 3-fold walk-forward grades; `get_pattern_signal()`. |
| `up_strategies.py` | Deployment-ready UP-side strategy functions (`STRATEGY_REGISTRY`, `TIER_A/B_STRATEGIES`) that passed K-fold gates in the scanner. |

## Requirements

- **Python 3.8+**
- `pip install numpy scipy` (statistical scripts; the rest are stdlib-only)
- **Data**: the private historical archive (see below). No wallet keys, signer, or
  network access — every script here is pure offline analysis.
- `flip_observer.py` additionally needs `flip_gate_eval` and `up_strategies.py` needs
  the `up_strategy_scanner_v3_2` output; these live in the trading engine, not here.

## Usage

```bash
# Master walk-forward signal search (recap-driven, prints TRAIN/TEST verdicts)
python3 big_data_signal_search.py

# Loss-cutter and cap-exit searches
python3 stuck_loser_fix_search.py
python3 imbalanced_cap_exit_search.py

# OOS-validated dn_live threshold + decisive BN-gate battery
python3 dnlive_threshold_oos.py
python3 part15_decisive.py

# Regime separability (explicit history + recap paths)
python3 regime_validator.py market_history.jsonl market_recap_history.jsonl

# Large-history flat-UP scan (VPS, argparse CLI)
python3 flat_up_strategy_vps.py \
    --file /path/to/market_history.jsonl \
    --out  flat_up_report.json
```

## Data

These scripts load the historical tick + recap archive — primarily
`market_history.jsonl` (full per-tick book, 1GB+) and `market_recap_history.jsonl`
(per-market fires, winners, and pre-computed `hypo_pnl`). Several scripts hard-code
`~/polymarket-bot/data/`; point that at a checkout of the private **polymarket-data**
repo (or override the path arguments where exposed). These `.jsonl`/`.json` files are
git-ignored here and are never committed.

> Private research software. No warranty; trades/handles real funds at your own risk.
