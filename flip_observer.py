#!/usr/bin/env python3
"""
FLIP GATE OBSERVER v4 — uses each market's ACTUAL timestamp for hour_utc/day_of_week.
Critical fix: previous version stamped every market with "now", inflating proba.
"""
import json, os, datetime

HISTORY_PATH = 'market_history.jsonl'
THRESHOLD = 0.70
REGIME_CUTOFF = '2026-05-01'

def parse_ts(ts_str):
    if not ts_str: return None
    try: return datetime.datetime.strptime(ts_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
    except: return None

def stats_block(label, probas):
    if not probas: print(f"\n[{label}] no data"); return
    n = len(probas); s = sorted(probas)
    pass_n = sum(1 for p in probas if p >= THRESHOLD)
    print(f"\n[{label}] n={n}")
    print(f"  min={min(probas):.3f}  p25={s[n//4]:.3f}  median={s[n//2]:.3f}  p75={s[3*n//4]:.3f}  max={max(probas):.3f}")
    print(f"  mean={sum(probas)/n:.3f}  >=0.70: {pass_n} ({pass_n/n*100:.1f}%)")

def main():
    try: import flip_gate_eval
    except ImportError: print("ERROR: flip_gate_eval not found"); return
    if not os.path.exists(HISTORY_PATH): print(f"ERROR: {HISTORY_PATH} not found"); return
    
    cutoff = datetime.datetime.strptime(REGIME_CUTOFF, '%Y-%m-%d')
    print(f"Regime cutoff: {cutoff} UTC")
    print(f"Using each market's TRUE ts for hour_utc/day_of_week\n")
    
    pre_probas, post_probas = [], []
    n_total = n_ok = n_error = n_no_ts = 0
    top_post = []
    
    with open(HISTORY_PATH) as f:
        for line in f:
            try: r = json.loads(line)
            except: continue
            n_total += 1
            ts = parse_ts(r.get('ts'))
            if ts is None:
                n_no_ts += 1
                continue
            ticks = r.get('ticks')
            if not ticks: continue
            
            # Pass market's ACTUAL ts so hour_utc/day_of_week are correct
            res = flip_gate_eval.evaluate(ticks, cd=200.0, current_time=ts)
            if not res['ok']:
                n_error += 1
                continue
            n_ok += 1
            p = res['flip_proba']
            if ts >= cutoff:
                post_probas.append(p)
                top_post.append((p, r.get('slug','?'), r.get('winner','?')))
            else:
                pre_probas.append(p)
    
    print(f"Total: {n_total}  ok: {n_ok}  errors: {n_error}  no_ts: {n_no_ts}")
    
    stats_block("ALL markets", pre_probas + post_probas)
    stats_block("PRE-May-1  (training regime)", pre_probas)
    stats_block("POST-May-1 (deployment regime)", post_probas)
    
    print(f"\n=== TOP 10 highest-proba POST-May-1 markets ===")
    top_post.sort(key=lambda x: -x[0])
    print(f"  {'proba':>6s}  {'winner':>6s}  slug")
    for p, slug, w in top_post[:10]:
        print(f"  {p:>5.3f}   {w:>6s}  {slug}")
    
    print(f"\n=== EXPECTED FROM TRAINING ===")
    print(f"  Training set (300 mkts): mean=0.44, pass=16%")
    print(f"  Test set     (129 mkts): mean=0.40, pass=16%")
    
    print(f"\n=== VERDICT ===")
    if not post_probas:
        print("  No post-May-1 data"); return
    
    post_mean = sum(post_probas)/len(post_probas)
    post_pass = sum(1 for p in post_probas if p>=THRESHOLD)/len(post_probas)*100
    
    if 0.30 <= post_mean <= 0.50 and 10 <= post_pass <= 25:
        print(f"  ✅ HEALTHY: post-May-1 mean={post_mean:.2f}, pass={post_pass:.0f}%")
        print(f"     Matches expected. Deploy at threshold {THRESHOLD}.")
    elif post_mean > 0.50:
        print(f"  ⚠️  HIGH: mean={post_mean:.2f}, pass={post_pass:.0f}%")
        print(f"     Expected 0.40/16%. Either real regime shift or feature drift.")
    elif post_mean < 0.20:
        print(f"  ⚠️  LOW: mean={post_mean:.2f}")
    else:
        print(f"  ?  borderline: mean={post_mean:.2f}, pass={post_pass:.0f}%")

if __name__ == "__main__":
    main()
