#!/usr/bin/env python3
"""
Test 3 honest options to fix stuck losers without killing big_wins:
1. Earlier checkpoint at cd=240 (side_skew + ae fingerprint)
2. Per-side fire cap (max N per side)
3. Real-time unrealized stop (cd-bucket trailing PnL)

All evaluated TRAIN→TEST (1108 train, 112 test) with no future info.
"""
import json, sys, os, statistics, math
sys.path.insert(0, '/home/polybot/polymarket-bot')
os.chdir('/home/polybot/polymarket-bot/data')

DEPLOY_TS = 1778785939

# Load
print("Loading recaps...", file=sys.stderr)
all_recaps = {}
with open('/home/polybot/polymarket-bot/data/market_recap_history.jsonl') as f:
    for line in f:
        try:
            d = json.loads(line)
            if d.get('fires') and d.get('winner') and all('hypo_pnl' in f for f in d['fires']):
                all_recaps[d.get('slug')] = d
        except: pass
train = [s for s, r in all_recaps.items() if r.get('ts', 0) <= DEPLOY_TS]
test  = [s for s, r in all_recaps.items() if r.get('ts', 0) >  DEPLOY_TS]
print(f"  TRAIN={len(train)}  TEST={len(test)}", file=sys.stderr)

def get_fires(slug):
    return [{'cd': f.get('cd'), 'side': f.get('side'),
             'entry_price': f.get('entry_price'), 'hp': f.get('hypo_pnl', 0)}
            for f in all_recaps[slug].get('fires', [])
            if f.get('cd') is not None and f.get('hypo_pnl') is not None]

def baseline(s): return sum(f['hp'] for f in get_fires(s))

# ============================================================
# Policy 1: cd=240 early checkpoint
# At cd=240, look at fires placed so far. If side-skew>X and ae<Y, refuse all future fires.
# ============================================================
def early_chkpt(slug, ckpt_cd, skew_thresh, ae_thresh, min_fires=3):
    """Refuse all fires placed at cd<ckpt_cd if early fires (cd>=ckpt_cd) show
       skew>skew_thresh AND ae<ae_thresh."""
    fires = get_fires(slug)
    early = [f for f in fires if f['cd'] >= ckpt_cd]
    if len(early) < min_fires:
        # Not enough early fires to evaluate — allow all
        return sum(f['hp'] for f in fires)
    up = sum(1 for f in early if f['side'] == 'UP')
    dn = sum(1 for f in early if f['side'] == 'DN')
    skew = abs(up - dn) / len(early)
    eps = [f['entry_price'] for f in early if f['entry_price']]
    ae = sum(eps)/len(eps) if eps else 0
    if skew >= skew_thresh and ae <= ae_thresh:
        # Refuse future fires
        return sum(f['hp'] for f in early)
    # else: allow all
    return sum(f['hp'] for f in fires)

# ============================================================
# Policy 2: Per-side fire cap
# Track fires per side; once cap hit, refuse more on that side
# ============================================================
def side_cap(slug, cap):
    """Process fires in cd-descending order (chronological).
    Once a side hits 'cap' fires, refuse all subsequent fires on that side."""
    fires = sorted(get_fires(slug), key=lambda f: -f['cd'])  # chronological (high cd first)
    up_count = 0; dn_count = 0
    pnl = 0
    for f in fires:
        if f['side'] == 'UP':
            if up_count >= cap: continue
            up_count += 1
        else:  # DN
            if dn_count >= cap: continue
            dn_count += 1
        pnl += f['hp']
    return pnl

# ============================================================
# Policy 3: Real-time unrealized stop (using actual bid prices from ticks)
# ============================================================
# This needs tick data
print("Loading tick data for policy 3...", file=sys.stderr)
tick_data = {}
slugs_needed = set(all_recaps.keys())
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
print(f"  loaded {len(tick_data)} tick files in {time.time()-t0:.1f}s", file=sys.stderr)

def unrealized_at_cd(slug, target_cd, fires_so_far):
    """Sum (current_bid - entry) for fires_so_far at the bid prices at target_cd."""
    td = tick_data.get(slug)
    if not td or not td['cols']: return None
    cols = td['cols']; ticks = td['ticks']
    cd_i = cols.index('cd')
    ub_i = cols.index('up_bid'); db_i = cols.index('dn_bid')
    best_t = None; best_d = 1e9
    for t in ticks:
        d = abs(t[cd_i] - target_cd)
        if d < best_d: best_d = d; best_t = t
    if best_t is None or best_d > 5: return None
    cur_ub = best_t[ub_i]; cur_db = best_t[db_i]
    if cur_ub is None or cur_db is None: return None
    total = 0
    for f in fires_so_far:
        bid = cur_ub if f['side'] == 'UP' else cur_db
        if bid is not None: total += (bid - (f['entry_price'] or 0))
    return total

def trailing_stop(slug, ckpt_cd, stop_threshold):
    """At ckpt_cd, compute unrealized using current bids. If u <= -stop_threshold,
       refuse all future fires (cd<ckpt_cd)."""
    fires = get_fires(slug)
    pre = [f for f in fires if f['cd'] >= ckpt_cd]
    u = unrealized_at_cd(slug, ckpt_cd, pre)
    if u is None:
        # No tick data, allow all
        return sum(f['hp'] for f in fires)
    if u <= -stop_threshold:
        # Refuse future fires
        return sum(f['hp'] for f in pre)
    return sum(f['hp'] for f in fires)

# ============================================================
# Eval helpers
# ============================================================
def evalp(slugs, fn, **kw):
    deltas = [fn(s, **kw) - baseline(s) for s in slugs]
    n = len(deltas)
    cum = sum(deltas); mean = cum/n if n else 0
    se = statistics.stdev(deltas)/math.sqrt(n) if n > 1 else 0
    t = mean/se if se > 0 else 0
    return mean, t, cum

def report_split(label, fn, **kw):
    tr_m, tr_t, tr_c = evalp(train, fn, **kw)
    te_m, te_t, te_c = evalp(test, fn, **kw)
    sig = " ✓" if tr_m > 0.3 and te_m > 0.5 else ""
    sig += " ★" if tr_m > 0.5 and te_m > 1.5 else ""
    return f"  {label:<58s}  TR ${tr_m:+5.2f}/mkt (t={tr_t:+.1f}) | TE ${te_m:+5.2f}/mkt (t={te_t:+.1f}){sig}"

# ============================================================
# Run tests
# ============================================================
print()
print("=" * 100)
print("POLICY 1: Earlier checkpoint at cd=240/220 (side-skew + ae fingerprint)")
print("=" * 100)
for ckpt in [220, 240, 260]:
    for skew in [0.60, 0.70, 0.80, 0.90, 1.0]:
        for ae in [0.20, 0.25, 0.30, 0.35]:
            label = f"early_chkpt cd>={ckpt} skew>={skew} ae<={ae}"
            print(report_split(label, early_chkpt, ckpt_cd=ckpt, skew_thresh=skew, ae_thresh=ae))

print()
print("=" * 100)
print("POLICY 2: Per-side fire cap")
print("=" * 100)
for cap in [4, 5, 6, 7, 8, 9, 10, 12, 15]:
    label = f"side_cap max={cap} per side"
    print(report_split(label, side_cap, cap=cap))

print()
print("=" * 100)
print("POLICY 3: Real-time unrealized stop at various checkpoints")
print("=" * 100)
for ckpt in [240, 200, 180, 150, 120]:
    for stop in [3.0, 5.0, 8.0, 10.0, 15.0]:
        label = f"trailing_stop @cd={ckpt} u<=-${stop}"
        print(report_split(label, trailing_stop, ckpt_cd=ckpt, stop_threshold=stop))

print()
print("Legend: ✓ = positive on both sets   ★ = test > +$1.50/mkt (great signal)")
