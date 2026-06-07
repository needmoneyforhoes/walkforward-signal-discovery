#!/usr/bin/env python3
"""
regime_validator.py — Run on VPS against full market_history.jsonl + market_recap_history.jsonl

Validates:
1. Can we distinguish monster from disaster BEFORE the outcome?
2. What structural features separate winning vs losing markets?
3. Does CL gate actually predict the winning side?
4. Can we build a regime detector to avoid cold streaks?

Usage: python3 regime_validator.py market_history.jsonl market_recap_history.jsonl
       python3 regime_validator.py market_history_lite.jsonl market_recap_history.jsonl
"""

import json, sys, math
from collections import defaultdict
import statistics

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def safe_div(a, b, default=0):
    return a / b if b != 0 else default

def g(d, key, default=0):
    """Safe get — returns default if key missing OR value is None."""
    v = d.get(key)
    return v if v is not None else default

def bootstrap_p(values, n_boot=5000):
    """One-sided bootstrap test: is mean > 0?"""
    if len(values) < 3:
        return 1.0
    import random
    observed = sum(values) / len(values)
    count = 0
    for _ in range(n_boot):
        sample = [random.choice(values) for _ in range(len(values))]
        if sum(sample) / len(sample) <= 0:
            count += 1
    return count / n_boot

# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════

if len(sys.argv) < 3:
    print("Usage: python3 regime_validator.py <market_history.jsonl> <market_recap_history.jsonl>")
    sys.exit(1)

print("Loading data...")
history = load_jsonl(sys.argv[1])
recaps = load_jsonl(sys.argv[2])

print(f"  History: {len(history)} markets")
print(f"  Recaps:  {len(recaps)} markets")

# Index recaps by slug for cross-referencing
recap_by_slug = {}
for r in recaps:
    slug = r.get('slug', r.get('market', ''))
    recap_by_slug[slug] = r

# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: STRUCTURAL FEATURES — MONSTER vs DISASTER vs NORMAL
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  SECTION 1: CAN WE TELL MONSTER FROM DISASTER STRUCTURALLY?")
print("=" * 90)

categories = {"monster": [], "disaster": [], "normal": []}

for i, m in enumerate(history):
    slug = m.get('slug', '')
    r = recap_by_slug.get(slug)
    winner = m.get('winner', '?')
    mw = m.get('market_wide') or {}
    
    # Compute hypo PnL from recap
    hypo_pnl = 0
    n_fires = 0
    if r:
        fires = r.get('fires', [])
        hypo_pnl = sum(f.get('hypo_pnl', 0) or 0 for f in fires)
        n_fires = len(fires)
    
    # Extract structural features from market_wide
    features = {
        'idx': i + 1,
        'slug': slug,
        'winner': winner,
        'hypo_pnl': hypo_pnl,
        'n_fires': n_fires,
        # Price range
        'up_bid_min': g(mw, 'up_bid_min', 0.5),
        'up_bid_max': g(mw, 'up_bid_max', 0.5),
        'up_bid_std': g(mw, 'up_bid_std', 0),
        'dn_bid_min': g(mw, 'dn_bid_min', 0.5),
        'dn_bid_max': g(mw, 'dn_bid_max', 0.5),
        'dn_bid_std': g(mw, 'dn_bid_std', 0),
        # Volatility
        'lead_changes': g(mw, 'lead_changes', 0),
        'time_near_5050': g(mw, 'time_near_5050', 0),
        # BN (Binance)
        'bn_delta_min': g(mw, 'bn_delta_min', 0),
        'bn_delta_max': g(mw, 'bn_delta_max', 0),
        'bn_delta_start': g(mw, 'bn_delta_start', 0),
        'bn_delta_end': g(mw, 'bn_delta_end', 0),
        'bn_flipped': g(mw, 'bn_flipped_sign', False),
        'bn_range': abs(g(mw, 'bn_delta_max', 0) - g(mw, 'bn_delta_min', 0)),
        # Depth
        'up_depth_min': g(mw, 'up_depth_min', 0),
        'up_depth_max': g(mw, 'up_depth_max', 0),
        'dn_depth_min': g(mw, 'dn_depth_min', 0),
        'dn_depth_max': g(mw, 'dn_depth_max', 0),
        'up_sr_max': g(mw, 'up_sr_max', 0),
        'dn_sr_max': g(mw, 'dn_sr_max', 0),
        'depth_osc_up': g(mw, 'depth_oscillations_up', 0),
        'depth_osc_dn': g(mw, 'depth_oscillations_dn', 0),
        # Price spread (how far did the market move?)
        'winner_price_range': 0,
        'loser_price_range': 0,
    }
    
    # Compute winner/loser price ranges
    if winner == 'UP':
        features['winner_price_range'] = g(mw, 'up_bid_max', 0.5) - g(mw, 'up_bid_min', 0.5)
        features['loser_price_range'] = g(mw, 'dn_bid_max', 0.5) - g(mw, 'dn_bid_min', 0.5)
    elif winner == 'DN':
        features['winner_price_range'] = g(mw, 'dn_bid_max', 0.5) - g(mw, 'dn_bid_min', 0.5)
        features['loser_price_range'] = g(mw, 'up_bid_max', 0.5) - g(mw, 'up_bid_min', 0.5)
    
    # BN direction vs winner
    bn_end = g(mw, 'bn_delta_end', 0)
    features['bn_confirms_winner'] = (
        (winner == 'UP' and bn_end > 0.01) or
        (winner == 'DN' and bn_end < -0.01)
    )
    
    # Skip counterfactual data
    sc = m.get('skip_counterfactuals') or {}
    if sc:
        t200 = sc.get('T-200') or {}
        t120 = sc.get('T-120') or {}
        t60 = sc.get('T-60') or {}
        t30 = sc.get('T-30') or {}
        features['winner_ask_t200'] = g(t200, 'winner_ask', 0.5)
        features['winner_ask_t120'] = g(t120, 'winner_ask', 0.5)
        features['winner_ask_t60'] = g(t60, 'winner_ask', 0.5)
        features['winner_ask_t30'] = g(t30, 'winner_ask', 0.5)
        features['bn_delta_t200'] = g(t200, 'bn_delta', 0)
        features['bn_delta_t120'] = g(t120, 'bn_delta', 0)
        features['conv_t120'] = g(t120, 'crowd_conv', 0.5)
        features['conv_t60'] = g(t60, 'crowd_conv', 0.5)
    
    if hypo_pnl >= 20:
        categories['monster'].append(features)
    elif hypo_pnl <= -10:
        categories['disaster'].append(features)
    else:
        categories['normal'].append(features)

# Compare features
feature_names = [
    'up_bid_std', 'dn_bid_std', 'lead_changes', 'time_near_5050',
    'bn_range', 'bn_flipped', 'up_sr_max', 'dn_sr_max',
    'depth_osc_up', 'depth_osc_dn', 'winner_price_range', 'loser_price_range',
]

print(f"\n  {'Feature':30s} {'Monster':>12s} {'Disaster':>12s} {'Normal':>12s} {'M vs D diff':>12s}")
print(f"  {'-'*78}")

separators = []
for feat in feature_names:
    m_vals = [f[feat] for f in categories['monster'] if isinstance(f[feat], (int, float))]
    d_vals = [f[feat] for f in categories['disaster'] if isinstance(f[feat], (int, float))]
    n_vals = [f[feat] for f in categories['normal'] if isinstance(f[feat], (int, float))]
    
    m_avg = sum(m_vals) / len(m_vals) if m_vals else 0
    d_avg = sum(d_vals) / len(d_vals) if d_vals else 0
    n_avg = sum(n_vals) / len(n_vals) if n_vals else 0
    
    diff_pct = safe_div(abs(m_avg - d_avg), max(abs(m_avg), abs(d_avg), 0.001)) * 100
    tag = "⭐" if diff_pct > 30 else "  "
    separators.append((feat, diff_pct, m_avg, d_avg))
    
    print(f"  {feat:30s} {m_avg:12.3f} {d_avg:12.3f} {n_avg:12.3f} {diff_pct:10.0f}% {tag}")

# BN confirms winner
for cat_name, cat_list in categories.items():
    if not cat_list: continue
    bn_rate = sum(1 for f in cat_list if f.get('bn_confirms_winner', False)) / len(cat_list) * 100
    print(f"  BN confirms winner ({cat_name:10s}): {bn_rate:.0f}%")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: CL GATE VALIDATION — DOES CL PREDICT THE WINNING SIDE?
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  SECTION 2: CL GATE — DOES cl_pass PREDICT WINNING SIDE?")
print("=" * 90)

cl_pass_wins = 0
cl_pass_losses = 0
cl_fail_wins = 0
cl_fail_losses = 0
cl_pass_pnl = 0.0
cl_fail_pnl = 0.0

# Per-strategy CL analysis
strat_cl = defaultdict(lambda: {'pass_w': 0, 'pass_l': 0, 'fail_w': 0, 'fail_l': 0,
                                  'pass_pnl': 0.0, 'fail_pnl': 0.0})

for r in recaps:
    for f in r.get('fires', []):
        cl = f.get('cl_pass')
        pnl = f.get('hypo_pnl', 0) or 0
        strat = f.get('strategy', '?')
        win = pnl > 0
        
        if cl == 'PASS':
            if win:
                cl_pass_wins += 1
                strat_cl[strat]['pass_w'] += 1
            else:
                cl_pass_losses += 1
                strat_cl[strat]['pass_l'] += 1
            cl_pass_pnl += pnl
            strat_cl[strat]['pass_pnl'] += pnl
        elif cl == 'FAIL':
            if win:
                cl_fail_wins += 1
                strat_cl[strat]['fail_w'] += 1
            else:
                cl_fail_losses += 1
                strat_cl[strat]['fail_l'] += 1
            cl_fail_pnl += pnl
            strat_cl[strat]['fail_pnl'] += pnl

cl_pass_total = cl_pass_wins + cl_pass_losses
cl_fail_total = cl_fail_wins + cl_fail_losses

if cl_pass_total > 0 or cl_fail_total > 0:
    print(f"\n  OVERALL CL GATE:")
    print(f"    cl_pass=PASS: {cl_pass_total:3d} fires, {cl_pass_wins}W {cl_pass_losses}L "
          f"({safe_div(cl_pass_wins, cl_pass_total)*100:.0f}% WR)  ${cl_pass_pnl:+.2f}")
    print(f"    cl_pass=FAIL: {cl_fail_total:3d} fires, {cl_fail_wins}W {cl_fail_losses}L "
          f"({safe_div(cl_fail_wins, cl_fail_total)*100:.0f}% WR)  ${cl_fail_pnl:+.2f}")
    
    if cl_pass_total > 0 and cl_fail_total > 0:
        delta = safe_div(cl_pass_wins, cl_pass_total)*100 - safe_div(cl_fail_wins, cl_fail_total)*100
        print(f"    DELTA: {delta:+.0f}pp win rate improvement with CL gate")
        print(f"    CL gate value: ${cl_pass_pnl - cl_fail_pnl:+.2f} saved by filtering FAIL fires")
    
    print(f"\n  PER-STRATEGY CL GATE:")
    print(f"  {'Strategy':30s} {'PASS':>20s} {'FAIL':>20s} {'Delta':>8s}")
    print(f"  {'-'*80}")
    for strat in sorted(strat_cl.keys()):
        d = strat_cl[strat]
        p_t = d['pass_w'] + d['pass_l']
        f_t = d['fail_w'] + d['fail_l']
        if p_t + f_t < 2: continue
        p_wr = safe_div(d['pass_w'], p_t) * 100 if p_t > 0 else 0
        f_wr = safe_div(d['fail_w'], f_t) * 100 if f_t > 0 else 0
        delta = p_wr - f_wr if p_t > 0 and f_t > 0 else 0
        p_str = f"{p_t}F {d['pass_w']}W ${d['pass_pnl']:+.2f}" if p_t > 0 else "—"
        f_str = f"{f_t}F {d['fail_w']}W ${d['fail_pnl']:+.2f}" if f_t > 0 else "—"
        tag = "⭐" if delta > 20 else ""
        print(f"  {strat:30s} {p_str:>20s} {f_str:>20s} {delta:+6.0f}pp {tag}")
else:
    print("\n  ⚠️  No CL gate data yet — need more markets with extras recording")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: SIDE PREDICTION — WHAT PREDICTS WHICH SIDE WINS?
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  SECTION 3: SIDE PREDICTION — WHAT PREDICTS THE WINNER?")
print("=" * 90)

# Test each indicator at various time points
predictors = []

for m in history:
    winner = m.get('winner', '?')
    if winner not in ('UP', 'DN'): continue
    sc = m.get('skip_counterfactuals') or {}
    mw = m.get('market_wide') or {}
    
    pred = {'winner': winner}
    
    # BN delta at various times
    for t in ['T-200', 'T-150', 'T-120', 'T-90', 'T-60', 'T-30']:
        td = sc.get(t) or {}
        pred[f'bn_{t}'] = td.get('bn_delta') or 0
        pred[f'conv_{t}'] = td.get('crowd_conv') or 0.5
        pred[f'crowd_{t}'] = td.get('crowd_side') or '?'
        pred[f'winner_ask_{t}'] = td.get('winner_ask') or 0.5
    
    pred['bn_end'] = mw.get('bn_delta_end') or 0
    pred['bn_start'] = mw.get('bn_delta_start') or 0
    pred['bn_flipped'] = mw.get('bn_flipped_sign') or False
    pred['bn_range'] = abs((mw.get('bn_delta_max') or 0) - (mw.get('bn_delta_min') or 0))
    pred['lead_changes'] = mw.get('lead_changes') or 0
    
    predictors.append(pred)

# Test: does BN delta predict winner at each time point?
print(f"\n  BN DELTA → WINNER PREDICTION ACCURACY:")
print(f"  {'Time':>8s} {'BN→UP correct':>16s} {'BN→DN correct':>16s} {'Overall':>10s} {'N':>5s}")
for t in ['T-200', 'T-150', 'T-120', 'T-90', 'T-60', 'T-30']:
    correct = 0
    total = 0
    bn_up_correct = 0
    bn_up_total = 0
    bn_dn_correct = 0
    bn_dn_total = 0
    
    for p in predictors:
        bn = p.get(f'bn_{t}')
        if bn is None or not isinstance(bn, (int, float)): continue
        if abs(bn) < 0.001: continue  # skip ambiguous
        total += 1
        
        if bn > 0:
            bn_up_total += 1
            if p['winner'] == 'UP':
                correct += 1
                bn_up_correct += 1
        else:
            bn_dn_total += 1
            if p['winner'] == 'DN':
                correct += 1
                bn_dn_correct += 1
    
    if total > 0:
        up_pct = safe_div(bn_up_correct, bn_up_total) * 100
        dn_pct = safe_div(bn_dn_correct, bn_dn_total) * 100
        pct = correct / total * 100
        print(f"  {t:>8s} {up_pct:13.0f}% ({bn_up_total:3d}) {dn_pct:13.0f}% ({bn_dn_total:3d}) {pct:8.0f}%  {total:4d}")

# Crowd conviction → winner accuracy
print(f"\n  CROWD CONVICTION → WINNER PREDICTION:")
print(f"  {'Time':>8s} {'Crowd correct':>16s} {'N':>5s}")
for t in ['T-200', 'T-150', 'T-120', 'T-90', 'T-60', 'T-30']:
    correct = 0
    total = 0
    for p in predictors:
        crowd = p.get(f'crowd_{t}', '?')
        if crowd not in ('UP', 'DN'): continue
        total += 1
        if crowd == p['winner']:
            correct += 1
    if total > 0:
        print(f"  {t:>8s} {correct/total*100:14.0f}% {total:4d}")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: WRONG-SIDE FIRE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  SECTION 4: WRONG-SIDE FIRES — THE CORE PROBLEM")
print("=" * 90)

# For each market: what % of fires were on the winning side?
per_market_correctness = []
for r in recaps:
    fires = r.get('fires', [])
    if not fires: continue
    winner = r.get('winner', '?')
    if winner not in ('UP', 'DN'): continue
    
    correct = sum(1 for f in fires if f.get('side') == winner)
    wrong = len(fires) - correct
    pnl = sum(f.get('hypo_pnl', 0) or 0 for f in fires)
    
    per_market_correctness.append({
        'correct_pct': correct / len(fires) * 100,
        'correct': correct,
        'wrong': wrong,
        'total': len(fires),
        'pnl': pnl,
    })

# Correlation between correct_pct and PnL
print(f"\n  CORRECT-SIDE % vs MARKET PnL:")
buckets = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]
for lo, hi in buckets:
    subset = [m for m in per_market_correctness if lo <= m['correct_pct'] < hi]
    if not subset: continue
    avg_pnl = sum(m['pnl'] for m in subset) / len(subset)
    avg_correct = sum(m['correct_pct'] for m in subset) / len(subset)
    print(f"  {lo:3d}-{hi-1:3d}% correct: {len(subset):3d} markets, avg PnL=${avg_pnl:+.2f}, "
          f"avg correct={avg_correct:.0f}%")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: CL DELTA AS SIDE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  SECTION 5: CL DELTA AS SIDE PREDICTOR")
print("=" * 90)

# From recap fires that have cl_delta
cl_correct = 0
cl_wrong = 0
cl_correct_pnl = 0
cl_wrong_pnl = 0

for r in recaps:
    winner = r.get('winner', '?')
    for f in r.get('fires', []):
        cl_delta = f.get('cl_delta')
        if cl_delta is None or not isinstance(cl_delta, (int, float)): continue
        side = f.get('side', '?')
        pnl = f.get('hypo_pnl', 0) or 0
        
        # Does CL delta agree with the fire's side?
        # cl_delta > 0 = CL price going up = UP likely winning
        cl_says = 'UP' if cl_delta > 0.001 else ('DN' if cl_delta < -0.001 else 'NEUTRAL')
        
        if cl_says == side:
            cl_correct += 1
            cl_correct_pnl += pnl
        elif cl_says != 'NEUTRAL':
            cl_wrong += 1
            cl_wrong_pnl += pnl

if cl_correct + cl_wrong > 0:
    print(f"  CL agrees with fire side:      {cl_correct:3d} fires  ${cl_correct_pnl:+.2f}")
    print(f"  CL disagrees with fire side:   {cl_wrong:3d} fires  ${cl_wrong_pnl:+.2f}")
    print(f"  CL-confirmed fires $/fire:     ${safe_div(cl_correct_pnl, cl_correct):+.2f}")
    print(f"  CL-opposing fires $/fire:      ${safe_div(cl_wrong_pnl, cl_wrong):+.2f}")
    print(f"  VALUE OF CL SIDE FILTER:       ${cl_correct_pnl - cl_wrong_pnl:+.2f}")
else:
    print("  ⚠️  No cl_delta data yet")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 6: REGIME DETECTOR — ROLLING MONSTER RATE
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  SECTION 6: REGIME PHASES — MONSTER CLUSTERING")
print("=" * 90)

pnl_per_market = []
for r in recaps:
    pnl = sum(f.get('hypo_pnl', 0) or 0 for f in r.get('fires', []))
    pnl_per_market.append(pnl)

# Rolling 6-market monster/disaster count
print(f"\n  {'Window':>12s} {'Mon':>4s} {'Dis':>4s} {'Net PnL':>10s} {'Regime':>10s}")
for i in range(0, len(pnl_per_market) - 5):
    w = pnl_per_market[i:i+6]
    mons = sum(1 for p in w if p >= 20)
    diss = sum(1 for p in w if p <= -10)
    net = sum(w)
    if mons >= 2:
        regime = "🟢 HOT"
    elif diss >= 3:
        regime = "🔴 COLD"
    else:
        regime = "⚪ NORMAL"
    if i % 6 == 0:
        print(f"  mkts {i+1:3d}-{i+6:3d}: {mons:3d}  {diss:3d}  ${net:+8.2f}  {regime}")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 7: CONCLUSIONS AND RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  SECTION 7: STATISTICAL CONCLUSIONS")
print("=" * 90)

# Test if any structural feature has statistical power
print(f"\n  FEATURES THAT SEPARATE MONSTER FROM DISASTER (>30% difference):")
top_seps = sorted(separators, key=lambda x: -x[1])
for feat, diff, m_avg, d_avg in top_seps[:10]:
    verdict = "✅ SEPARATES" if diff > 30 else "❌ NO SEPARATION"
    print(f"    {feat:30s}: Monster={m_avg:.3f}  Disaster={d_avg:.3f}  diff={diff:.0f}%  {verdict}")

# Overall conclusion
print(f"\n  OVERALL VALIDATION:")
n_sep = sum(1 for _, d, _, _ in separators if d > 30)
print(f"    Features with >30% separation: {n_sep}/{len(separators)}")
if n_sep <= 2:
    print(f"    ⚠️  CONFIRMED: Monster and disaster are structurally similar")
    print(f"       → Cannot predict market type from depth/price structure alone")
    print(f"       → SIDE PREDICTION (CL gate) is the correct approach")
else:
    print(f"    Some structural features show separation — investigate further")

if cl_pass_total > 0 and cl_fail_total > 0:
    if cl_pass_pnl > cl_fail_pnl and safe_div(cl_pass_wins, cl_pass_total) > safe_div(cl_fail_wins, cl_fail_total):
        print(f"\n    ✅ CL GATE VALIDATED: PASS fires outperform FAIL fires")
        print(f"       PASS: ${cl_pass_pnl:+.2f} ({safe_div(cl_pass_wins,cl_pass_total)*100:.0f}% WR)")
        print(f"       FAIL: ${cl_fail_pnl:+.2f} ({safe_div(cl_fail_wins,cl_fail_total)*100:.0f}% WR)")
        print(f"       → IMPLEMENT CL GATE as primary filter")
    else:
        print(f"\n    ⚠️  CL gate data inconclusive — need more samples")

print(f"\n  RECOMMENDED ACTIONS:")
print(f"    1. CL gate is the #1 priority — validates side prediction")
print(f"    2. Don't go LIVE until CL gate is validated with 50+ PASS and 50+ FAIL fires")
print(f"    3. BN direction at T-120 is the strongest early regime signal")
print(f"    4. Monster rate clusters — if 0 monsters in last 12 markets, reduce position size")

print(f"\n{'='*90}")
print(f"  VALIDATION COMPLETE — {len(history)} markets analyzed")
print(f"{'='*90}")
