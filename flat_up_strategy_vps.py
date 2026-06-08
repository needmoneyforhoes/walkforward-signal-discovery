#!/usr/bin/env python3
"""
flat_up_strategy_vps.py
━━━━━━━━━━━━━━━━━━━━━━━
Run on VPS against the full market_history.jsonl (1GB+) to find
a quantitatively validated strategy for buying UP when UP ask
stays flat in the $0.43-$0.65 range early in the market.

Context: 351-market local file shows 73 flat-UP markets with
WR=43.8% baseline (EV-negative). BN is the one significant
separator (p=0.0009, Mann-Whitney). Need n≥200 flat-UP markets
for Bonferroni-corrected significance on compound signals.

Usage:
    python3 flat_up_strategy_vps.py \\
        --file ./market_history.jsonl \\
        --out flat_up_report.json
"""

import json, math, argparse, sys, time, os
from collections import defaultdict

try:
    from scipy import stats as st
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not installed — p-values will be approximate")

# ── Config ────────────────────────────────────────────────────────────────
FLAT_LO      = 0.43   # UP ask lower bound for "flat" classification
FLAT_HI      = 0.65   # UP ask upper bound
FLAT_MIN_TPS = 2      # must be flat at ≥N of T-200/T-150/T-120
TIMEPOINTS   = ['T-200', 'T-150', 'T-120', 'T-90']

# ── Helpers ───────────────────────────────────────────────────────────────
def wilson_ci(wins, n, z=1.96):
    if n == 0: return 0, 0
    p = wins / n
    lo = max(0, (p + z**2/(2*n) - z*math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n))
    hi = min(1, (p + z**2/(2*n) + z*math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n))
    return lo, hi

def binom_p(wins, n, p0):
    """One-sided binomial test: WR > p0"""
    if not HAS_SCIPY: 
        # Normal approx
        z = (wins/n - p0) / math.sqrt(p0*(1-p0)/n) if n > 0 else 0
        return 0.5 * (1 - math.erf(z / math.sqrt(2))) if z > 0 else 1.0
    return st.binomtest(wins, n, p0, alternative='greater').pvalue

def ev_per_share(wr, avg_entry):
    """EV per share at average entry price."""
    return wr * (1 - avg_entry) / avg_entry - (1 - wr)

def mannwhitney_p(a, b):
    if not HAS_SCIPY or not a or not b: return 1.0
    _, p = st.mannwhitneyu(a, b, alternative='two-sided')
    return p

def sig_stars(p):
    return '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ('~' if p<0.1 else ' ')))

# ── Progress display ──────────────────────────────────────────────────────
class Progress:
    """Terminal progress bar with ETA. Works over SSH, no curses needed."""
    BAR_W = 35

    def __init__(self, total, label=''):
        self.total   = total
        self.label   = label
        self.t0      = time.time()
        self.started = False

    def update(self, done):
        elapsed = time.time() - self.t0
        frac    = done / self.total if self.total else 1
        pct     = frac * 100

        if frac > 0:
            eta_s = elapsed / frac * (1 - frac)
            eta   = self._fmt(eta_s)
        else:
            eta = '--:--'

        elapsed_fmt = self._fmt(elapsed)
        filled  = int(self.BAR_W * frac)
        bar     = '█' * filled + '░' * (self.BAR_W - filled)
        speed   = done / elapsed if elapsed > 0 else 0

        line = (f"\r  {self.label}  [{bar}] {pct:5.1f}%  "
                f"{done:,}/{self.total:,}  "
                f"elapsed {elapsed_fmt}  ETA {eta}  "
                f"({speed:,.0f}/s)   ")
        sys.stdout.write(line)
        sys.stdout.flush()

    def done(self):
        elapsed = self._fmt(time.time() - self.t0)
        sys.stdout.write(f"\r  {self.label}  [{'█'*self.BAR_W}] 100.0%  "
                         f"{self.total:,}/{self.total:,}  elapsed {elapsed}  ✅\n")
        sys.stdout.flush()

    @staticmethod
    def _fmt(s):
        s = int(s)
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m:02d}m{sec:02d}s"

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    parser.add_argument('--out', default='flat_up_report.json')
    args = parser.parse_args()

    print(f"Loading {args.file} ...")
    # Count lines first for accurate ETA
    t_count = time.time()
    sys.stdout.write("  Counting lines ...")
    sys.stdout.flush()
    with open(args.file) as f:
        total_lines = sum(1 for l in f if l.strip())
    print(f"\r  Counting lines ... {total_lines:,} lines ({time.time()-t_count:.1f}s)")

    markets = []
    prog = Progress(total_lines, label='Loading markets')
    with open(args.file) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            try:
                markets.append(json.loads(line))
            except Exception:
                continue
            if i % 500 == 0:
                prog.update(i + 1)
    prog.done()
    print(f"  Loaded {len(markets):,} valid markets")

    # ── Step 1: classify each market ──────────────────────────────────────
    flat_pool = []   # markets where UP was flat at ≥FLAT_MIN_TPS timepoints

    print(f"\nStep 1/4 — Classifying markets ...")
    prog = Progress(len(markets), label='Classifying')
    for idx, m in enumerate(markets):
        if idx % 200 == 0:
            prog.update(idx + 1)
        winner = m.get('winner')
        if winner not in ('UP', 'DN'): continue
        sc   = m.get('skip_counterfactuals') or {}
        mw   = m.get('market_wide') or {}
        ticks = m.get('ticks') or []  # full tick array if available

        snaps = {}
        for tp in TIMEPOINTS:
            snap = sc.get(tp)
            if not snap: continue
            if winner == 'UP':
                up_ask = snap.get('winner_ask')
            else:
                lb = snap.get('loser_bid') or 0
                up_ask = round(1 - lb, 3) if lb else None
            if up_ask and 0 < up_ask < 1:
                snaps[tp] = {
                    'up_ask': up_ask,
                    'bn': snap.get('bn_delta', 0) or 0,
                    'crowd': snap.get('crowd_side'),
                    'conv': snap.get('crowd_conv') or 0,
                    'snipe_pnl': snap.get('snipe_pnl') or 0,
                }

        flat_tps = [tp for tp in TIMEPOINTS[:3]
                    if tp in snaps and FLAT_LO <= snaps[tp]['up_ask'] <= FLAT_HI]
        if len(flat_tps) < FLAT_MIN_TPS:
            continue

        # Entry snapshot = T-200, fallback T-150
        entry = snaps.get('T-200') or snaps.get('T-150')
        if not entry: continue

        # Tick-level features if available
        tick_features = {}
        if ticks:
            cols = m.get('tick_columns', [])
            try:
                ci_up  = cols.index('up_ask')  if 'up_ask'  in cols else None
                ci_bn  = cols.index('bn_delta_pct') if 'bn_delta_pct' in cols else None
                ci_cd  = cols.index('cd') if 'cd' in cols else None
                ci_ccs = cols.index('crowd_side') if 'crowd_side' in cols else None
                ci_ccv = cols.index('crowd_conviction') if 'crowd_conviction' in cols else None

                early_ticks = [t for t in ticks
                               if ci_cd is not None and t[ci_cd] >= 180]

                if early_ticks and ci_up is not None:
                    up_asks_early = [t[ci_up] for t in early_ticks
                                     if t[ci_up] is not None and 0 < t[ci_up] < 1]
                    if up_asks_early:
                        tick_features['up_ask_mean_early']  = sum(up_asks_early)/len(up_asks_early)
                        tick_features['up_ask_min_early']   = min(up_asks_early)
                        tick_features['up_ask_max_early']   = max(up_asks_early)
                        tick_features['up_ask_std_early']   = (
                            (sum((x - tick_features['up_ask_mean_early'])**2
                                 for x in up_asks_early) / len(up_asks_early))**0.5
                        )
                        # Time spent flat (in tick count)
                        flat_ticks = sum(1 for a in up_asks_early if FLAT_LO <= a <= FLAT_HI)
                        tick_features['pct_flat_early'] = flat_ticks / len(up_asks_early)

                if early_ticks and ci_bn is not None:
                    bns = [t[ci_bn] for t in early_ticks if t[ci_bn] is not None]
                    if bns:
                        tick_features['bn_mean_early'] = sum(bns)/len(bns)
                        tick_features['bn_trend'] = bns[-1] - bns[0] if len(bns)>1 else 0

            except (IndexError, ValueError):
                pass

        flat_pool.append({
            'slug':    m.get('slug'),
            'winner':  winner,
            'snaps':   snaps,
            'entry':   entry,
            'lc':      mw.get('lead_changes', 0) or 0,
            'bn_start':mw.get('bn_delta_start', 0) or 0,
            't5050':   mw.get('time_near_5050', 300) or 300,
            'flat_n':  len(flat_tps),
            'tick':    tick_features,
        })

    prog.done()
    n_up = sum(1 for r in flat_pool if r['winner']=='UP')
    n_dn = sum(1 for r in flat_pool if r['winner']=='DN')
    print(f"  Flat-UP pool: {len(flat_pool):,} markets  (UP={n_up}, DN={n_dn})")

    # ── Step 2: signal comparison UP vs DN ──────────────────────────────
    up_rows = [r for r in flat_pool if r['winner']=='UP']
    dn_rows = [r for r in flat_pool if r['winner']=='DN']

    print(f"\nStep 2/4 — Signal comparison (UP vs DN in flat zone) ...")
    signals = {
        'bn_T200':   lambda r: r['entry']['bn'],
        'conv_T200': lambda r: r['entry']['conv'],
        'lead_changes': lambda r: r['lc'],
        'time@50/50': lambda r: r['t5050'],
        'bn_start':  lambda r: r['bn_start'],
    }
    if flat_pool[0]['tick']:
        signals.update({
            'bn_mean_early': lambda r: r['tick'].get('bn_mean_early', 0),
            'bn_trend':      lambda r: r['tick'].get('bn_trend', 0),
            'pct_flat_early':lambda r: r['tick'].get('pct_flat_early', 0),
            'up_ask_std':    lambda r: r['tick'].get('up_ask_std_early', 0),
        })

    mw_results = {}
    for sig_name, sig_fn in signals.items():
        up_vals = [sig_fn(r) for r in up_rows if sig_fn(r) is not None]
        dn_vals = [sig_fn(r) for r in dn_rows if sig_fn(r) is not None]
        p = mannwhitney_p(up_vals, dn_vals)
        mw_results[sig_name] = p
        up_m = sum(up_vals)/len(up_vals) if up_vals else 0
        dn_m = sum(dn_vals)/len(dn_vals) if dn_vals else 0
        print(f"  {sig_name:<22} UP={up_m:+.4f}  DN={dn_m:+.4f}  p={p:.4f}{sig_stars(p)}")

    # ── Step 3: systematic threshold sweep ──────────────────────────────
    bn_thresholds      = [-0.05,-0.04,-0.03,-0.02,-0.01,0.0,0.005,0.010,0.015]
    conv_thresholds    = [0.50,0.52,0.54,0.55,0.57]
    lc_mins            = [0, 5, 10, 20]
    # bn_trend thresholds: how much does BN move during early phase?
    # UP markets: trend≈-0.0015, DN markets: trend≈-0.0337 (p=0.0001***)
    bn_trend_thrs      = [-0.05,-0.04,-0.03,-0.02,-0.01,0.0,0.005]

    # Count tests first for Bonferroni
    N_TESTS = 0
    for bt in bn_thresholds:
        for lt in [0,10]: N_TESTS += 1
    for ct in conv_thresholds: N_TESTS += 1
    for bt in bn_thresholds:
        for ct in conv_thresholds: N_TESTS += 1
    # tick-based: bn_mean_early and bn_trend sweeps (only if ticks present)
    N_TESTS += len(bn_thresholds)                  # bn_mean_early solo
    N_TESTS += len(bn_trend_thrs)                  # bn_trend solo
    N_TESTS += len(bn_trend_thrs) * len(bn_thresholds)  # bn_trend + bn_T200
    N_TESTS += len(bn_trend_thrs) * len(conv_thresholds) # bn_trend + conv

    bonf = 0.05 / max(N_TESTS, 1)
    print(f"\nStep 3/4 — Threshold sweep ({N_TESTS} combinations, Bonferroni p<{bonf:.4f}) ...")
    _sweep_prog = Progress(N_TESTS, label='Sweep  ')
    _sweep_done = [0]

    def run_test(name, cond_fn):
        _sweep_done[0] += 1
        if _sweep_done[0] % 3 == 0 or _sweep_done[0] == N_TESTS:
            _sweep_prog.update(_sweep_done[0])
        fires = [(r['winner']=='UP', r['entry']['up_ask'])
                 for r in flat_pool if cond_fn(r) and 0 < r['entry']['up_ask'] < 1]
        n = len(fires)
        if n < 10: return None
        wins = sum(1 for w,_ in fires if w)
        wr   = wins/n
        avg_e = sum(e for _,e in fires)/n
        tot_h = sum(((1-e)/e*5*e if w else -e*5) for w,e in fires)
        ev_sh = ev_per_share(wr, avg_e)
        p     = binom_p(wins, n, avg_e)
        ci_lo, ci_hi = wilson_ci(wins, n)
        return {'name':name,'n':n,'wins':wins,'wr':wr,'ci_lo':ci_lo,'ci_hi':ci_hi,
                'avg_e':avg_e,'tot_h':tot_h,'ev_sh':ev_sh,
                'ev_ci':ev_per_share(ci_lo, avg_e),'p':p}

    results = []
    for bt in bn_thresholds:
        for lt in [0, 10]:
            cond = (lambda r,b=bt,l=lt: r['entry']['bn']>=b and r['lc']>=l)
            res = run_test(f"BN≥{bt:+.3f}+lc≥{lt}", cond)
            if res: results.append(res)

    for ct in conv_thresholds:
        cond = (lambda r,c=ct: r['entry']['conv']>=c and r['entry']['crowd']=='UP')
        res = run_test(f"crowd=UP+conv≥{ct:.2f}", cond)
        if res: results.append(res)

    for bt in bn_thresholds:
        for ct in conv_thresholds:
            cond = (lambda r,b=bt,c=ct:
                    r['entry']['bn']>=b and r['entry']['conv']>=c and r['entry']['crowd']=='UP')
            res = run_test(f"BN≥{bt:+.3f}+conv≥{ct:.2f}", cond)
            if res: results.append(res)

    # Tick-based if available (VPS file has full tick arrays)
    has_ticks = any(r['tick'] for r in flat_pool)
    if has_ticks:
        for bt in bn_thresholds:
            cond = (lambda r,b=bt: r['tick'].get('bn_mean_early',0)>=b)
            res = run_test(f"bn_mean_early≥{bt:+.3f}", cond)
            if res: results.append(res)

        # bn_trend = BN direction during early ticks (p=0.0001***)
        # UP markets: bn_trend≈-0.0015 (flat), DN: -0.0337 (falling)
        # Strategy: buy UP only when BN not trending strongly negative
        for tt in bn_trend_thrs:
            cond = (lambda r,t=tt: r['tick'].get('bn_trend',0)>=t)
            res = run_test(f"bn_trend≥{tt:+.3f}", cond)
            if res: results.append(res)

        # bn_trend + bn_T200 compound
        for tt in bn_trend_thrs:
            for bt in bn_thresholds:
                cond = (lambda r,t=tt,b=bt:
                        r['tick'].get('bn_trend',0)>=t and r['entry']['bn']>=b)
                res = run_test(f"bn_trend≥{tt:+.3f}+bn_T200≥{bt:+.3f}", cond)
                if res: results.append(res)

        # bn_trend + conv compound
        for tt in bn_trend_thrs:
            for ct in conv_thresholds:
                cond = (lambda r,t=tt,c=ct:
                        r['tick'].get('bn_trend',0)>=t and
                        r['entry']['conv']>=c and r['entry']['crowd']=='UP')
                res = run_test(f"bn_trend≥{tt:+.3f}+conv≥{ct:.2f}", cond)
                if res: results.append(res)

    # BH correction
    sorted_res = sorted(results, key=lambda x: x['p'])
    n_all = len(sorted_res)
    for rank, r in enumerate(sorted_res, 1):
        r['bh']   = bool(r['p'] <= (rank/n_all*0.05))
        r['bonf'] = bool(r['p'] <  bonf)

    print(f"\n{'Condition':<38} {'n':>4} {'WR':>6} {'EV/sh':>7} {'p':>9} BF  BH  verdict")
    print("─"*85)
    for r in sorted_res[:30]:
        s=sig_stars(r['p']); b="✅" if r['bonf'] else "  "; bh="✅" if r['bh'] else "  "
        verdict=("✅DEPLOY?" if r['bonf'] and r['ev_sh']>0 else
                 ("🔬PROM?" if r['bh'] and r['ev_sh']>0 else
                  ("EV+" if r['ev_sh']>0 and r['ev_ci']>0 else "")))
        print(f"  {r['name']:<38} {r['n']:4d} {r['wr']:6.1%} {r['ev_sh']:+7.3f} {r['p']:9.4f}{s} {b} {bh}  {verdict}")

    _sweep_prog.done()

    # ── Step 4: survivors deep-dive ───────────────────────────────────
    print("\nStep 4/4 — Building report ...")
    survivors = [r for r in sorted_res if r['bonf'] and r['ev_sh'] > 0]
    promising = [r for r in sorted_res if r['bh'] and r['ev_sh'] > 0 and not r['bonf']]
    
    print(f"\n=== SURVIVORS (Bonferroni): {len(survivors)} ===")
    print(f"=== PROMISING (BH FDR):     {len(promising)} ===")

    for cat, group in [("BONFERRONI SURVIVORS", survivors), ("BH PROMISING", promising)]:
        if not group: continue
        print(f"\n{'─'*60}")
        print(f"  {cat}")
        print(f"{'─'*60}")
        for r in group[:5]:
            print(f"\n  📊 {r['name']}")
            print(f"     n={r['n']}  WR={r['wr']:.1%}  CI=[{r['ci_lo']:.1%},{r['ci_hi']:.1%}]")
            print(f"     EV/share={r['ev_sh']:+.4f}  EV@CI_lo={r['ev_ci']:+.4f}")
            print(f"     hypo_total=${r['tot_h']:+.2f}  p={r['p']:.5f}")
            print(f"     Min fires needed for LIVE: {max(0,30-r['n'])} more")

    # ── Step 5: output JSON ───────────────────────────────────────────
    def jsonify(v):
        """Convert any value to JSON-safe type."""
        if isinstance(v, bool):   return bool(v)
        if isinstance(v, float):  return round(v, 6)
        if isinstance(v, dict):   return {kk: jsonify(vv) for kk,vv in v.items()}
        if isinstance(v, list):   return [jsonify(x) for x in v]
        return v

    output = {
        'n_markets':            len(markets),
        'n_flat_pool':          len(flat_pool),
        'n_flat_up':            n_up,
        'n_flat_dn':            n_dn,
        'bonferroni_threshold': round(bonf, 6),
        'signal_pvalues':       {k: round(v, 6) for k,v in mw_results.items()},
        'survivors':            [jsonify(r) for r in survivors],
        'promising':            [jsonify(r) for r in promising],
        'all_results':          [jsonify(r) for r in sorted_res],
    }
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Report saved to {args.out}")
    print(f"   Survivors: {len(survivors)}  Promising: {len(promising)}")

if __name__ == '__main__':
    main()
