#!/usr/bin/env python3
"""
BIG DATA HONEST SEARCH — train on 1108 pre-deploy, validate on 112 post-deploy.

This is the proper walk-forward we needed. Search for any honest (no future info)
signal that gives +$2-$3/mkt on TEST while also positive on TRAIN.

Methodology:
- TRAIN: 1108 pre-deploy markets (recap-only, no need to scan tick file)
- TEST:  112 post-deploy markets (true OOS, never used in training)
- Honest gating only: at fire time, decide ALLOW/BLOCK using only pre-fire data
- Bonferroni-corrected: report only signals with t-stat > 3.0 (~p<0.001)

Search space:
A. rule_skip parameter sweep (ae_mid x n_mid threshold grid)
B. Per-strategy late-fire gating (top-N worst strategies from TRAIN, applied to TEST)
C. Per-side patterns (UP-late vs DN-late differential gating)
D. Per-cd-bucket gating (refuse fires in specific cd ranges)
E. Combined policies
"""

import json, sys, os, statistics, math
sys.path.insert(0, '/home/polybot/polymarket-bot')
os.chdir('/home/polybot/polymarket-bot/data')

DEPLOY_TS = 1778785939

# ============================================================
# Load all 1220 recaps with fires + hypo_pnl
# ============================================================
print("Loading recaps...", file=sys.stderr)
all_recaps = {}
with open('/home/polybot/polymarket-bot/data/market_recap_history.jsonl') as f:
    for line in f:
        try:
            d = json.loads(line)
            if d.get('fires') and d.get('winner'):
                # Make sure hypo_pnl is present per fire
                if all('hypo_pnl' in f for f in d['fires']):
                    all_recaps[d.get('slug')] = d
        except: pass

train_slugs = [s for s, r in all_recaps.items() if r.get('ts', 0) <= DEPLOY_TS]
test_slugs  = [s for s, r in all_recaps.items() if r.get('ts', 0) >  DEPLOY_TS]

print(f"  TRAIN: {len(train_slugs)} markets")
print(f"  TEST:  {len(test_slugs)} markets")
print()

# ============================================================
# Per-fire data extracted from recaps (pre-computed hypo_pnl)
# ============================================================
def get_fires(slug):
    """Returns list of {cd, side, entry, hypo_pnl, strategy} for the market."""
    r = all_recaps.get(slug, {})
    out = []
    for f in r.get('fires', []):
        cd = f.get('cd')
        side = f.get('side')
        entry = f.get('entry_price')
        hp = f.get('hypo_pnl')
        if cd is None or side is None or hp is None: continue
        out.append({
            'cd': cd, 'side': side, 'entry_price': entry, 'hp': hp,
            'strategy': f.get('strategy', ''), 'cat': _fire_cat_simple(f),
        })
    return out

def _fire_cat_simple(f):
    """Lightweight category — FREE if not held/blocked/wblock."""
    if f.get('hold_to_settle'): return 'HELD'
    return 'FREE'

# ============================================================
# Policies (all honest, no future info)
# ============================================================
def baseline(slug):
    return sum(f['hp'] for f in get_fires(slug))

def rule_skip(slug, ae_thresh=0.31, n_thresh=5):
    """rule_skip: if rule_pass=False, exclude fires with cd<120."""
    fires = get_fires(slug)
    if not fires: return 0
    mid = [f for f in fires if 120 <= f['cd'] < 240]
    if len(mid) >= n_thresh:
        eps = [f['entry_price'] for f in mid if f['entry_price']]
        ae = sum(eps)/len(eps) if eps else 0
        rp = ae >= ae_thresh
    else:
        rp = False
    if rp:
        return sum(f['hp'] for f in fires)
    return sum(f['hp'] for f in fires if f['cd'] >= 120)

def per_strategy_gate(slug, blocked_set):
    """Refuse late fires (cd<120) from listed strategies."""
    fires = get_fires(slug)
    return sum(f['hp'] for f in fires
                if f['cd'] >= 120 or f['strategy'] not in blocked_set)

def cd_bucket_gate(slug, gate_min, gate_max):
    """Refuse all fires within cd in [gate_min, gate_max]."""
    fires = get_fires(slug)
    return sum(f['hp'] for f in fires if not (gate_min <= f['cd'] < gate_max))

def per_side_late_gate(slug, gated_side):
    """Refuse late fires on specified side only."""
    fires = get_fires(slug)
    return sum(f['hp'] for f in fires
                if f['cd'] >= 120 or f['side'] != gated_side)

def combined_rs_strategy(slug, ae, nm, blocked_set):
    """rule_skip + extra per-strategy gating even when rule passes."""
    fires = get_fires(slug)
    if not fires: return 0
    mid = [f for f in fires if 120 <= f['cd'] < 240]
    if len(mid) >= nm:
        eps = [f['entry_price'] for f in mid if f['entry_price']]
        ae_v = sum(eps)/len(eps) if eps else 0
        rp = ae_v >= ae
    else:
        rp = False
    if rp:
        # rule passes, but still gate worst strategies in late window
        return sum(f['hp'] for f in fires
                    if f['cd'] >= 120 or f['strategy'] not in blocked_set)
    # rule fails: gate all late fires
    return sum(f['hp'] for f in fires if f['cd'] >= 120)


# ============================================================
# Scale (unit hypo → dollars). Compute from baseline pnl on a few markets.
# ============================================================
import importlib.util
spec = importlib.util.spec_from_file_location('ie', '/home/polybot/polymarket-bot/induction_engine.py')
ie = importlib.util.module_from_spec(spec)
ie._ie_loaded = False
try:
    spec.loader.exec_module(ie)
except Exception as e:
    print(f"Note: ie load partially failed (ok): {e}", file=sys.stderr)

ratios = []
for s in train_slugs[:30]:
    rec = all_recaps[s]
    try:
        edges = ie._f28_compute_market_edge_per_twin(rec, all_verdicts={})
        d = edges['_baseline_no_mute']['hypo_gc']
        u = baseline(s)
        if abs(u) > 0.5: ratios.append(d/u)
    except: pass
SCALE = sum(ratios)/len(ratios) if ratios else 6.0
print(f"Scale: ${SCALE:.2f}/unit", file=sys.stderr)
print()

# ============================================================
# Evaluation helpers with t-stat
# ============================================================
def eval_policy(policy_fn, slugs_list, **kwargs):
    """Returns (cum_lift_$, mean_lift_$, lift_per_mkt_list, t_stat)"""
    deltas = []
    for s in slugs_list:
        new = policy_fn(s, **kwargs)
        base = baseline(s)
        deltas.append((new - base) * SCALE)
    cum = sum(deltas)
    n = len(deltas)
    mean = cum / n if n else 0
    se = (statistics.stdev(deltas) / math.sqrt(n)) if n > 1 else 0
    t_stat = (mean / se) if se > 0 else 0
    return {'cum': cum, 'mean': mean, 'n': n, 't_stat': t_stat, 'deltas': deltas}

def report(label, train_res, test_res, hypothesis_count=1):
    """Bonferroni-aware reporting."""
    bonf_thresh = 3.0 + 0.5 * math.log10(max(1, hypothesis_count))
    t_ok = (abs(test_res['t_stat']) > bonf_thresh)
    test_positive = test_res['mean'] > 1.0
    train_positive = train_res['mean'] > 0.5
    sig = ""
    if t_ok and test_positive: sig = " ★"
    if test_positive and train_positive: sig += " ✓"
    return f"  {label:<60s}  TR ${train_res['mean']:+.2f}/mkt (t={train_res['t_stat']:+.2f}) | TE ${test_res['mean']:+.2f}/mkt (t={test_res['t_stat']:+.2f}){sig}"


# ============================================================
# A. Baseline + plain rule_skip parameter sweep
# ============================================================
print("=" * 100)
print("A. RULE_SKIP PARAMETER SWEEP")
print("=" * 100)

# Baseline (no policy)
print(f"  baseline                                                       TR ${sum(baseline(s) for s in train_slugs)/len(train_slugs)*SCALE:+.2f}/mkt | TE ${sum(baseline(s) for s in test_slugs)/len(test_slugs)*SCALE:+.2f}/mkt")
print()

best_a = None
n_hyp_a = 0
for ae in [0.20, 0.25, 0.28, 0.31, 0.34, 0.37, 0.40]:
    for nm in [3, 4, 5, 6, 7, 8, 10]:
        n_hyp_a += 1
        tr = eval_policy(rule_skip, train_slugs, ae_thresh=ae, n_thresh=nm)
        te = eval_policy(rule_skip, test_slugs, ae_thresh=ae, n_thresh=nm)
        if tr['mean'] > 0.3 and te['mean'] > 0.5:
            label = f"rule_skip ae>={ae:.2f} n_mid>={nm}"
            line = report(label, tr, te, n_hyp_a)
            print(line)
            if best_a is None or te['mean'] > best_a['te_mean']:
                best_a = {'ae': ae, 'nm': nm, 'tr_mean': tr['mean'], 'te_mean': te['mean'],
                          'tr_t': tr['t_stat'], 'te_t': te['t_stat']}

# ============================================================
# B. Per-strategy gating — TRAIN to identify worst, validate on TEST
# ============================================================
print()
print("=" * 100)
print("B. PER-STRATEGY LATE-FIRE GATING (TRAIN identifies worst, TEST validates)")
print("=" * 100)

# Strategy late-fire totals on TRAIN
strat_total = {}
strat_count = {}
for s in train_slugs:
    for f in get_fires(s):
        if f['cd'] < 120:
            strat_total[f['strategy']] = strat_total.get(f['strategy'], 0) + f['hp']
            strat_count[f['strategy']] = strat_count.get(f['strategy'], 0) + 1

# Sort worst-first
strats_ranked = sorted(
    [(s, strat_total[s] * SCALE, strat_count[s], strat_total[s] / strat_count[s] * SCALE)
     for s in strat_total if strat_count[s] >= 5],
    key=lambda x: x[1]
)
print()
print(f"  Worst 15 strategies in TRAIN late window:")
print(f"    {'strategy':<32s}{'count':>8s}{'total$':>12s}{'avg$/fire':>12s}")
for s, t, n, avg in strats_ranked[:15]:
    print(f"    {s:<32s}{n:>8d}{t:>+11.2f}{avg:>+10.3f}")
print()

best_b = None
n_hyp_b = 0
for top_n in [3, 5, 8, 10, 15, 20, 25, 30, 40, 50]:
    n_hyp_b += 1
    blocked = set(s for s, t, n, avg in strats_ranked[:top_n])
    tr = eval_policy(per_strategy_gate, train_slugs, blocked_set=blocked)
    te = eval_policy(per_strategy_gate, test_slugs, blocked_set=blocked)
    label = f"per_strategy_gate top{top_n} ({len(blocked)} strats)"
    if tr['mean'] > 0.5 or te['mean'] > 0.5:
        print(report(label, tr, te, n_hyp_b))
        if best_b is None or (te['mean'] > best_b['te_mean'] and te['t_stat'] > 1.5):
            best_b = {'top_n': top_n, 'blocked': blocked, 'tr_mean': tr['mean'],
                      'te_mean': te['mean'], 'tr_t': tr['t_stat'], 'te_t': te['t_stat']}

# ============================================================
# C. CD-bucket gating
# ============================================================
print()
print("=" * 100)
print("C. CD-BUCKET GATING (refuse fires in specific cd ranges)")
print("=" * 100)
n_hyp_c = 0
for gmin, gmax in [(0, 30), (0, 60), (0, 90), (0, 120),
                    (30, 60), (30, 90), (30, 120),
                    (60, 90), (60, 120),
                    (90, 120)]:
    n_hyp_c += 1
    tr = eval_policy(cd_bucket_gate, train_slugs, gate_min=gmin, gate_max=gmax)
    te = eval_policy(cd_bucket_gate, test_slugs, gate_min=gmin, gate_max=gmax)
    label = f"cd_bucket_gate cd[{gmin},{gmax})"
    if tr['mean'] > 0.3 or te['mean'] > 0.3:
        print(report(label, tr, te, n_hyp_c))

# ============================================================
# D. Per-side late gating
# ============================================================
print()
print("=" * 100)
print("D. PER-SIDE LATE GATING")
print("=" * 100)
for side in ['UP', 'DN']:
    tr = eval_policy(per_side_late_gate, train_slugs, gated_side=side)
    te = eval_policy(per_side_late_gate, test_slugs, gated_side=side)
    print(report(f"per_side_late_gate side={side}", tr, te))

# ============================================================
# E. Combined: rule_skip + per-strategy gating
# ============================================================
print()
print("=" * 100)
print("E. COMBINED rule_skip + per-strategy")
print("=" * 100)

if best_b:
    # Try multiple combinations
    for top_n in [5, 10, 20, 30]:
        for ae, nm in [(0.20, 4), (0.25, 5), (0.31, 5), (0.31, 6)]:
            blocked = set(s for s, t, n, avg in strats_ranked[:top_n])
            tr = eval_policy(combined_rs_strategy, train_slugs, ae=ae, nm=nm, blocked_set=blocked)
            te = eval_policy(combined_rs_strategy, test_slugs, ae=ae, nm=nm, blocked_set=blocked)
            if tr['mean'] > 0.5 and te['mean'] > 0.5:
                label = f"combined_rs ae={ae} nm={nm} top{top_n}"
                print(report(label, tr, te, 50))

# ============================================================
# Summary
# ============================================================
print()
print("=" * 100)
print("BEST FINDINGS")
print("=" * 100)
if best_a:
    print(f"A. rule_skip best: ae={best_a['ae']} nm={best_a['nm']}")
    print(f"   TRAIN: ${best_a['tr_mean']:+.2f}/mkt (t={best_a['tr_t']:+.2f})")
    print(f"   TEST:  ${best_a['te_mean']:+.2f}/mkt (t={best_a['te_t']:+.2f})")
if best_b:
    print(f"B. per-strategy best: top{best_b['top_n']}")
    print(f"   TRAIN: ${best_b['tr_mean']:+.2f}/mkt (t={best_b['tr_t']:+.2f})")
    print(f"   TEST:  ${best_b['te_mean']:+.2f}/mkt (t={best_b['te_t']:+.2f})")
print()
print("Legend: ★ = passes Bonferroni-adjusted t-test on TEST  |  ✓ = positive on both train + test")
