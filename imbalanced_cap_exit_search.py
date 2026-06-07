#!/usr/bin/env python3
"""
NEW SIGNAL: Imbalanced cap exit.
At a checkpoint cd_check (90/60/45/30), detect markets where:
  1. Side X has hit cap=6 (cap_reached_side)
  2. Other side Y has < N_other fires (e.g., <=4)
  3. X-side bid still has value (current_bid > value_floor)

When detected: sell X-side cap-6 positions at current bid.
Keep Y-side positions and any post-checkpoint fires normally.

Compare to baseline + sc6 alone + sc6e30. Must validate on TRAIN/TEST split.
"""
import json, sys, os, statistics
sys.path.insert(0, '/home/polybot/polymarket-bot')
os.chdir('/home/polybot/polymarket-bot/data')

DEPLOY_TS = 1778785939
all_recaps = {}
with open('/home/polybot/polymarket-bot/data/market_recap_history.jsonl') as f:
    for line in f:
        try:
            d = json.loads(line)
            if d.get('fires') and d.get('winner') and all('hypo_pnl' in f for f in d['fires']):
                all_recaps[d.get('slug')] = d
        except: pass

slugs_needed = set(all_recaps.keys())
tick_data = {}
print("Loading ticks...", file=sys.stderr)
import time
t0 = time.time()
with open('/home/polybot/polymarket-bot/data/market_history.jsonl') as f:
    for line in f:
        if len(tick_data) >= len(slugs_needed): break
        try:
            d = json.loads(line)
            s = d.get('slug')
            if s in slugs_needed and s not in tick_data:
                tick_data[s] = {'ticks': d.get('ticks', []), 'cols': d.get('tick_columns', [])}
        except: pass
print(f"  loaded {len(tick_data)} in {time.time()-t0:.1f}s", file=sys.stderr)

train = [s for s, r in all_recaps.items() if r.get('ts', 0) <= DEPLOY_TS]
test = [s for s, r in all_recaps.items() if r.get('ts', 0) > DEPLOY_TS]

def get_fires(s):
    return [{'cd': f.get('cd'), 'side': f.get('side'),
             'entry_price': f.get('entry_price'), 'hp': f.get('hypo_pnl', 0)}
            for f in all_recaps[s].get('fires', []) if f.get('cd') is not None]

def baseline(s): return sum(f['hp'] for f in get_fires(s))

def cap6_placed(s):
    """Returns the list of fires cap=6 places (chronological)."""
    fires = sorted(get_fires(s), key=lambda f: -f['cd'])
    up=0; dn=0; placed=[]
    for f in fires:
        if f['side']=='UP':
            if up>=6: continue
            up += 1
        else:
            if dn>=6: continue
            dn += 1
        placed.append(f)
    return placed

def cap6_pnl(s): return sum(f['hp'] for f in cap6_placed(s))

def bid_at(slug, target_cd):
    td = tick_data.get(slug)
    if not td or not td['cols']: return None, None
    cols = td['cols']; ticks = td['ticks']
    cd_i = cols.index('cd'); ub_i = cols.index('up_bid'); db_i = cols.index('dn_bid')
    best = None; best_d = 1e9
    for t in ticks:
        d = abs(t[cd_i] - target_cd)
        if d < best_d: best_d = d; best = t
    if best is None or best_d > 5: return None, None
    return best[ub_i], best[db_i]

def imbalanced_exit(s, cd_check, n_other_max, value_floor):
    """At cd_check:
       - Compute side counts so far
       - If one side hit 6 AND other has <= n_other_max fires AND that side's bid > value_floor
         → exit the cap-6 side at current bid
       - Other-side positions hold to settle
       - Return total $."""
    placed = cap6_placed(s)
    pre = [f for f in placed if f['cd'] > cd_check]
    post = [f for f in placed if f['cd'] <= cd_check]
    cur_ub, cur_db = bid_at(s, cd_check)
    if cur_ub is None or cur_db is None:
        return sum(f['hp'] for f in placed)  # no data, hold all

    # Counts at cd_check
    up_pre = sum(1 for f in pre if f['side']=='UP')
    dn_pre = sum(1 for f in pre if f['side']=='DN')

    exit_side = None
    exit_bid = None
    if up_pre >= 6 and dn_pre <= n_other_max and cur_ub > value_floor:
        exit_side = 'UP'; exit_bid = cur_ub
    elif dn_pre >= 6 and up_pre <= n_other_max and cur_db > value_floor:
        exit_side = 'DN'; exit_bid = cur_db

    if exit_side is None:
        # No exit, hold all to settle
        return sum(f['hp'] for f in placed)

    # Exit cap-6 side at bid
    winner = all_recaps[s]['winner']
    total = 0
    for f in pre:
        if f['side'] == exit_side:
            # Exit at bid
            e = f['entry_price']
            won = 1.0 if f['side'] == winner else 0.0
            unit_hp = won - e
            shares = f['hp'] / unit_hp if abs(unit_hp) > 0.001 else 0
            total += shares * (exit_bid - e)
        else:
            total += f['hp']  # hold to settle
    # Post-checkpoint fires hold to settle
    total += sum(f['hp'] for f in post)
    return total

def evalp(slugs, fn, **kw):
    deltas = [fn(s, **kw) - baseline(s) for s in slugs]
    n = len(deltas); cum = sum(deltas); mean = cum/n if n else 0
    return mean, cum

# Compute reference: cap=6 alone
b_cap6_train = sum(cap6_pnl(s) for s in train) - sum(baseline(s) for s in train)
b_cap6_test = sum(cap6_pnl(s) for s in test) - sum(baseline(s) for s in test)
print(f"\n=== REFERENCES ===")
print(f"  cap=6 alone:           TR ${b_cap6_train/len(train):+.2f}/mkt  TE ${b_cap6_test/len(test):+.2f}/mkt")
print()

# Sweep parameters
print(f"{'cd_check':>10s}{'n_other<=':>10s}{'value>':>10s}{'TR/mkt':>10s}{'TE/mkt':>10s}{'TR_n_trig':>12s}{'TE_n_trig':>12s}")
print("-" * 80)
for cd_check in [90, 60, 45, 30]:
    for n_other in [0, 1, 2, 3, 4, 5]:
        for value in [0.10, 0.20, 0.30, 0.40, 0.50]:
            # Apply imbalanced_exit on top of cap=6
            tr_pnls = [imbalanced_exit(s, cd_check, n_other, value) for s in train]
            te_pnls = [imbalanced_exit(s, cd_check, n_other, value) for s in test]
            tr_lift = sum(tr_pnls) - sum(baseline(s) for s in train)
            te_lift = sum(te_pnls) - sum(baseline(s) for s in test)
            tr_better_than_cap6 = tr_lift - b_cap6_train
            te_better_than_cap6 = te_lift - b_cap6_test
            # Count how many markets the exit triggered on
            tr_trig = sum(1 for s in train
                          if imbalanced_exit(s, cd_check, n_other, value) != cap6_pnl(s))
            te_trig = sum(1 for s in test
                          if imbalanced_exit(s, cd_check, n_other, value) != cap6_pnl(s))
            # Only show if improvement vs cap=6 alone
            if tr_better_than_cap6 > 30 or te_better_than_cap6 > 5:
                tr_per_mkt = tr_lift / len(train)
                te_per_mkt = te_lift / len(test)
                print(f"  {cd_check:>5d}    <={n_other:<4d}  >${value:.2f}    ${tr_per_mkt:+5.2f}    ${te_per_mkt:+5.2f}      {tr_trig:>5d}        {te_trig:>5d}")

# Specifically check market 1778836500
print()
print(f"=== Market 1778836500 specifically ===")
slug = 'btc-updown-5m-1778836500'
print(f"  baseline: ${baseline(slug):+.2f}")
print(f"  cap=6 alone: ${cap6_pnl(slug):+.2f}")
for cd_check in [90, 60, 45, 30]:
    for n_other in [4, 3, 2]:
        for value in [0.10, 0.20]:
            v = imbalanced_exit(slug, cd_check, n_other, value)
            triggered = "TRIG" if v != cap6_pnl(slug) else "    "
            print(f"  cd={cd_check} n_other<={n_other} val>${value:.2f}: ${v:+.2f}  {triggered}")
PYEOF