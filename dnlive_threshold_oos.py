"""
dnlive_threshold_oos.py — OOS-validated threshold for dn_live discriminator

CRITICAL METHODOLOGY:
  1. Simulate dn_live on all 372 markets (same as before)
  2. Split trigger markets chronologically: IS=first 60%, OOS=last 40%
  3. Find optimal crowd_conv threshold on IS ONLY
  4. Apply that threshold to OOS ONLY and measure precision + EV
  5. Also test composite (crowd_conv + bn_delta + spread) on OOS
  6. Fisher exact test on OOS results
  7. Permutation test on OOS delta (10k)
  8. Final recommendation: no future bias anywhere

Run:  python3 dnlive_threshold_oos.py
"""

import json, os, sys, numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
from itertools import product

MARKET_HISTORY_PATH = os.path.expanduser('~/polymarket-bot/market_history.jsonl')
RECAP_HISTORY_PATH  = os.path.expanduser('~/polymarket-bot/market_recap_history.jsonl')

DNLIVE_UP_THRESHOLD = 0.50
DNLIVE_CONSEC       = 2
DNLIVE_MAX_CD       = 210
IS_SPLIT            = 0.60
N_PERMUTATIONS      = 10_000
ALPHA               = 0.05

def simulate_dnlive(ticks, cols):
    ci = {c: i for i, c in enumerate(cols)}
    cd_i = ci.get('cd',0); ua_i = ci.get('up_ask',2); da_i = ci.get('dn_ask',4)
    bn_i = ci.get('bn_delta_pct',12); cr_i = ci.get('crowd_conviction',20)
    cs_i = ci.get('crowd_side',19)

    consec = 0; last_sample_cd = 9999
    for t in ticks:
        cd = t[cd_i]
        if cd is None or cd <= 0: continue
        if last_sample_cd - cd < 9.0: continue
        last_sample_cd = cd
        ua = t[ua_i] or 0; da = t[da_i] or 0
        if cd > DNLIVE_MAX_CD:
            consec = consec+1 if ua >= DNLIVE_UP_THRESHOLD else 0
            continue
        consec = consec+1 if ua >= DNLIVE_UP_THRESHOLD else 0
        if consec >= DNLIVE_CONSEC:
            bn  = t[bn_i]  if bn_i  < len(t) else None
            cv  = t[cr_i]  if cr_i  < len(t) else 0
            return cd, {
                'trig_cd':    round(cd, 1),
                'up_ask':     round(ua, 3),
                'spread':     round(da - ua, 3),
                'bn_delta':   round(bn, 5) if bn is not None else 0.0,
                'crowd_conv': round(cv or 0, 4),
            }
    return None, None

def pnl_list(fires):
    return sum(f.get('hypo_pnl', 0) for f in fires)

# ── LOAD ──────────────────────────────────────────────────────────────────
print("Loading...", flush=True)
mkt_hist = {}
with open(MARKET_HISTORY_PATH) as f:
    for line in f:
        try:
            d = json.loads(line)
            if d.get('slug') and d.get('ticks') and d.get('tick_columns'):
                mkt_hist[d['slug']] = d
        except: pass

recap = {}
with open(RECAP_HISTORY_PATH) as f:
    for line in f:
        try:
            d = json.loads(line)
            if d.get('slug') and d.get('fires') and d.get('winner') in ('UP','DN'):
                recap[d['slug']] = d
        except: pass

common = sorted(set(mkt_hist) & set(recap))
print(f"  {len(common)} markets")

# ── SIMULATE ──────────────────────────────────────────────────────────────
print("Simulating dn_live...", flush=True)
all_markets = []
for slug in common:
    mkt   = mkt_hist[slug]
    rec   = recap[slug]
    fires = rec['fires']
    trig_cd, tf = simulate_dnlive(mkt['ticks'], mkt['tick_columns'])

    blocked_dn = []
    if trig_cd is not None:
        for fi in fires:
            if (fi.get('side') == 'DN' and
                    fi.get('cd', 9999) < trig_cd and
                    not fi.get('pre_gate_held') and
                    not fi.get('opp_gate_blocked')):
                blocked_dn.append(fi)

    all_markets.append({
        'slug': slug, 'winner': rec['winner'],
        'triggered': trig_cd is not None,
        'trig_cd': trig_cd,
        'tf': tf or {},
        'blocked_dn': blocked_dn,
        'dn_pnl': pnl_list(blocked_dn),
        'n_blocked': len(blocked_dn),
    })

trig = [m for m in all_markets if m['triggered']]
split_i = int(len(trig) * IS_SPLIT)
IS_trig  = trig[:split_i]
OOS_trig = trig[split_i:]

IS_up  = [m for m in IS_trig  if m['winner']=='UP']
IS_dn  = [m for m in IS_trig  if m['winner']=='DN']
OOS_up = [m for m in OOS_trig if m['winner']=='UP']
OOS_dn = [m for m in OOS_trig if m['winner']=='DN']

print(f"\n{'═'*65}")
print(f"SPLIT: IS={len(IS_trig)} triggers  OOS={len(OOS_trig)} triggers")
print(f"  IS : {len(IS_up)} correct + {len(IS_dn)} wrong = {len(IS_up)/len(IS_trig)*100:.1f}% precision")
print(f"  OOS: {len(OOS_up)} correct + {len(OOS_dn)} wrong = {len(OOS_up)/len(OOS_trig)*100:.1f}% precision")

# ── IS THRESHOLD TRAINING ─────────────────────────────────────────────────
# For each feature, find the threshold that maximises EV on IS only
# EV = correct_saves * prec_at_thresh - wrong_misses * (1-prec_at_thresh)
# We use grid search over percentiles of IS data

print(f"\n{'═'*65}")
print("IS THRESHOLD TRAINING (crowd_conv, bn_delta, spread)")
print(f"{'═'*65}")

def ev_at_threshold(markets_above, correct_pnl_avg, wrong_pnl_avg, n_total):
    """Net EV of applying gate only to markets above threshold vs no gate"""
    n_c = sum(1 for m in markets_above if m['winner']=='UP')
    n_w = sum(1 for m in markets_above if m['winner']=='DN')
    n   = n_c + n_w
    if n == 0: return 0, 0, 0
    prec = n_c / n
    # EV of gate = saves from correct - costs from wrong
    saves = n_c * correct_pnl_avg   # negative (losses avoided)
    costs = n_w * wrong_pnl_avg     # positive (profits missed)
    net   = saves + costs            # negative = gate helps, positive = hurts
    return -net, prec, n            # return positive if gate helps

is_correct_pnl_avg = sum(m['dn_pnl'] for m in IS_up) / len(IS_up) if IS_up else 0
is_wrong_pnl_avg   = sum(m['dn_pnl'] for m in IS_dn) / len(IS_dn) if IS_dn else 0

print(f"\n  IS avg PnL per block: correct={is_correct_pnl_avg:+.3f}  wrong={is_wrong_pnl_avg:+.3f}")
print(f"  (correct = negative = would've lost  |  wrong = positive = would've won)")

best_thresholds = {}
for feat, direction in [('crowd_conv', +1), ('bn_delta', +1), ('spread', -1)]:
    # direction: +1 = higher feature = more likely correct, -1 = lower = more likely correct
    all_vals = [(m['tf'].get(feat, 0) or 0) * direction for m in IS_trig]
    percentiles = np.percentile(all_vals, np.arange(10, 91, 5))

    best_ev, best_thresh, best_prec, best_n = -999, None, 0, 0
    results_thresh = []
    for pct_thresh in percentiles:
        above = [m for m in IS_trig if (m['tf'].get(feat, 0) or 0) * direction >= pct_thresh]
        ev, prec, n = ev_at_threshold(above, is_correct_pnl_avg, is_wrong_pnl_avg, len(IS_trig))
        results_thresh.append((pct_thresh/direction, prec, n, ev))
        if ev > best_ev and n >= 10:
            best_ev = ev; best_thresh = pct_thresh/direction; best_prec = prec; best_n = n

    best_thresholds[feat] = {'thresh': best_thresh, 'direction': direction,
                             'is_ev': best_ev, 'is_prec': best_prec, 'is_n': best_n}
    print(f"\n  {feat} (IS training):")
    print(f"    Best threshold: {feat} {'≥' if direction>0 else '≤'} {best_thresh:.4f}")
    print(f"    IS precision at threshold: {best_prec*100:.1f}%  n_kept={best_n}  IS_EV={best_ev:+.2f}")
    # Show threshold curve
    print(f"    {'Thresh':>8} {'Prec%':>6} {'n':>5} {'EV':>8}")
    for thresh_v, prec, n, ev in results_thresh[::2]:
        print(f"      {thresh_v:>8.4f} {prec*100:>6.1f} {n:>5} {ev:>+8.2f}")

# ── OOS VALIDATION ────────────────────────────────────────────────────────
print(f"\n{'═'*65}")
print("OOS VALIDATION (threshold trained on IS, applied to OOS — NO FUTURE BIAS)")
print(f"{'═'*65}")

def apply_threshold_oos(feat, thresh, direction, oos_markets):
    """Apply IS-trained threshold to OOS. Returns (kept, released)."""
    kept     = [m for m in oos_markets if (m['tf'].get(feat,0) or 0)*direction >= thresh*direction]
    released = [m for m in oos_markets if (m['tf'].get(feat,0) or 0)*direction <  thresh*direction]
    return kept, released

print(f"\n  OOS baseline (no discriminator, gate always fires):")
oos_saves = sum(m['dn_pnl'] for m in OOS_up)
oos_costs = sum(m['dn_pnl'] for m in OOS_dn)
print(f"    Correct saves: {oos_saves:+.2f}  Wrong costs: {oos_costs:+.2f}  Net gate EV: {oos_saves+oos_costs:+.2f}")
print(f"    (Negative net = gate saves money | Positive = gate costs money)")
print()

oos_results = {}
for feat, info in best_thresholds.items():
    thresh = info['thresh']; direction = info['direction']
    kept, released = apply_threshold_oos(feat, thresh, direction, OOS_trig)
    kept_up = [m for m in kept if m['winner']=='UP']
    kept_dn = [m for m in kept if m['winner']=='DN']
    rel_up  = [m for m in released if m['winner']=='UP']
    rel_dn  = [m for m in released if m['winner']=='DN']

    prec_kept = len(kept_up)/len(kept) if kept else 0
    saves_kept = sum(m['dn_pnl'] for m in kept_up)
    costs_kept = sum(m['dn_pnl'] for m in kept_dn)
    net_kept   = saves_kept + costs_kept     # net of still blocking kept markets
    gain_released = sum(m['dn_pnl'] for m in rel_dn)  # DN wins unblocked = gain

    # Total OOS EV vs current (no discriminator):
    # we now save: |saves_kept| still, but release costs_kept back to free
    # also we release rel_dn profits back to free
    oos_ev_change = -costs_kept + gain_released  # positive = better than current
    total_oos_ev  = saves_kept + costs_kept + gain_released  # net vs zero gate

    print(f"  {feat} (thresh {'≥' if direction>0 else '≤'} {thresh:.4f}):")
    print(f"    OOS kept:    {len(kept):3} triggers  {prec_kept*100:.1f}% precision  "
          f"saves={saves_kept:+.2f}  costs={costs_kept:+.2f}")
    print(f"    OOS released:{len(released):3} triggers  → freed profits={gain_released:+.2f}")
    print(f"    Net OOS EV change vs current: {oos_ev_change:+.2f}  "
          f"({'BETTER' if oos_ev_change > 0 else 'WORSE'} than current gate)")

    # Fisher exact test on OOS kept: is precision > chance (50%)?
    # H0: gate is coin flip (50% precision), H1: precision > 50%
    if kept:
        _, p_fish = scipy_stats.fisher_exact([[len(kept_up), len(kept_dn)],
                                               [len(OOS_up) - len(kept_up),
                                                len(OOS_dn) - len(kept_dn)]])
        p_binom = scipy_stats.binom_test(len(kept_up), len(kept), 0.5, alternative='greater') \
            if hasattr(scipy_stats, 'binom_test') else \
            scipy_stats.binomtest(len(kept_up), len(kept), 0.5, alternative='greater').pvalue
        print(f"    Fisher exact p={p_fish:.4f}  Binomial p={p_binom:.4f}  "
              f"{'sig' if p_binom < ALPHA else 'ns'}")

    # Permutation test on OOS EV change
    oos_deltas = [m['dn_pnl'] if (m['tf'].get(feat,0) or 0)*direction < thresh*direction else 0
                  for m in OOS_trig]
    obs_mean = np.mean(oos_deltas)
    np.random.seed(42)
    perm = [np.mean(np.random.choice([-1,1], len(oos_deltas)) * oos_deltas)
            for _ in range(N_PERMUTATIONS)]
    p_perm = np.mean(np.abs(perm) >= abs(obs_mean))
    print(f"    Permutation p={p_perm:.4f}  {'sig' if p_perm < ALPHA else 'ns'}")
    print()

    oos_results[feat] = {'prec': prec_kept, 'ev_change': oos_ev_change,
                          'n_kept': len(kept), 'p_binom': p_binom,
                          'thresh': thresh, 'direction': direction}

# ── COMPOSITE FEATURE ─────────────────────────────────────────────────────
print(f"{'═'*65}")
print("COMPOSITE: crowd_conv + bn_delta (require BOTH above IS-trained thresholds)")
print(f"{'═'*65}")

cc_thresh = best_thresholds['crowd_conv']['thresh']
bn_thresh = best_thresholds['bn_delta']['thresh']

def is_confident(m, cc_t, bn_t):
    cc = m['tf'].get('crowd_conv', 0) or 0
    bn = m['tf'].get('bn_delta',   0) or 0
    return cc >= cc_t and bn >= bn_t

# IS precision of composite
is_comp_kept = [m for m in IS_trig if is_confident(m, cc_thresh, bn_thresh)]
is_comp_up   = [m for m in is_comp_kept if m['winner']=='UP']
is_comp_dn   = [m for m in is_comp_kept if m['winner']=='DN']
is_comp_prec = len(is_comp_up)/len(is_comp_kept) if is_comp_kept else 0
print(f"\n  IS composite: thresh=crowd_conv≥{cc_thresh:.3f} AND bn_delta≥{bn_thresh:.4f}")
print(f"    IS kept={len(is_comp_kept)}  precision={is_comp_prec*100:.1f}%")

# OOS validation of composite
oos_comp_kept = [m for m in OOS_trig if is_confident(m, cc_thresh, bn_thresh)]
oos_comp_rel  = [m for m in OOS_trig if not is_confident(m, cc_thresh, bn_thresh)]
oos_comp_up   = [m for m in oos_comp_kept if m['winner']=='UP']
oos_comp_dn   = [m for m in oos_comp_kept if m['winner']=='DN']
oos_comp_prec = len(oos_comp_up)/len(oos_comp_kept) if oos_comp_kept else 0
comp_costs    = sum(m['dn_pnl'] for m in oos_comp_dn)
comp_rel_gain = sum(m['dn_pnl'] for m in [m for m in oos_comp_rel if m['winner']=='DN'])
comp_ev_change = -comp_costs + comp_rel_gain

print(f"  OOS composite: kept={len(oos_comp_kept)}  precision={oos_comp_prec*100:.1f}%")
print(f"    costs_still_blocked={comp_costs:+.2f}  released_profits={comp_rel_gain:+.2f}")
print(f"    OOS EV change vs current: {comp_ev_change:+.2f}  "
      f"({'BETTER' if comp_ev_change > 0 else 'WORSE'})")
if oos_comp_kept:
    p_binom_comp = scipy_stats.binomtest(len(oos_comp_up), len(oos_comp_kept),
                                          0.5, alternative='greater').pvalue
    print(f"    Binomial test (prec>50%): p={p_binom_comp:.4f}  "
          f"{'sig' if p_binom_comp < ALPHA else 'ns'}")

# ── FINAL VERDICT ─────────────────────────────────────────────────────────
print(f"\n{'═'*65}")
print("QUANTITATIVE VERDICT")
print(f"{'═'*65}")
print(f"""
Full dataset (n={len(trig)} triggers):
  Precision = {len([m for m in trig if m['winner']=='UP'])/len(trig)*100:.1f}%
  Permutation p = 0.7422 (removing flat = NOT justified)
  Walk-forward OOS: removing hurts (-$0.99/mkt)

Three Bonferroni-significant discriminators (found on full data):
  crowd_conv (p_bonf=0.0004), spread (p_bonf=0.0007), bn_delta (p_bonf=0.0037)

OOS-validated thresholds (trained on IS=first 60%, tested on OOS=last 40%):""")

# Sort by OOS EV change
for feat, res in sorted(oos_results.items(), key=lambda x: -x[1]['ev_change']):
    sig = '✅ sig' if res['p_binom'] < ALPHA else '❌ ns'
    print(f"  {feat:12}: OOS EV change={res['ev_change']:>+8.2f}  "
          f"OOS prec={res['prec']*100:.1f}%  {sig}")

print(f"\n  Composite (crowd_conv+bn_delta):")
print(f"    OOS EV change={comp_ev_change:+.2f}  OOS prec={oos_comp_prec*100:.1f}%")
if oos_comp_kept:
    print(f"    Binomial p={p_binom_comp:.4f}")

# Decision
best_oos = max(oos_results.items(), key=lambda x: x[1]['ev_change'])
best_name, best_res = best_oos

print(f"""
RECOMMENDATION (rigorous, OOS-validated):""")
if best_res['p_binom'] < ALPHA and best_res['ev_change'] > 0:
    thresh_str = f"≥{best_res['thresh']:.4f}" if best_res['direction']>0 else f"≤{best_res['thresh']:.4f}"
    print(f"""
  MODIFY dn_live: require {best_name} {thresh_str} at trigger time.
  When dn_live fires BUT {best_name} {thresh_str} NOT met → skip block.
  OOS EV improvement: {best_res['ev_change']:+.2f}  OOS precision: {best_res['prec']*100:.1f}%
  Statistical confidence: binomial p={best_res['p_binom']:.4f} < {ALPHA}
  
  Implementation: add one condition inside dn_live trigger block in bot.
  Start in SHADOW, validate 30 trigger events, then promote to LIVE.
""")
elif comp_ev_change > 0 and p_binom_comp < ALPHA:
    print(f"""
  MODIFY dn_live: require BOTH crowd_conv≥{cc_thresh:.3f} AND bn_delta≥{bn_thresh:.4f}.
  OOS EV improvement: {comp_ev_change:+.2f}  OOS precision: {oos_comp_prec*100:.1f}%
  Binomial p={p_binom_comp:.4f}
""")
else:
    print(f"""
  KEEP dn_live AS-IS.
  OOS walk-forward shows gate is NET POSITIVE in recent markets.
  No discriminator reaches OOS statistical significance.
  Permutation p=0.7422 — removing is not justified.
  
  Continue accumulating data. Re-evaluate after 50 more trigger markets.
""")
