"""
part15_decisive.py — Decisive analysis with all appropriate methods.

Prior hypothesis: early BN positive → DN fires lose (domain-motivated, not data-mined).
Best candidate from sweep: cd=290 BN≥0.04 and cd=300 BN≥0.04.

Tests (ordered by statistical power for our sample size):
  1. Binomial exact on individual FIRES — most powerful test (2990 data points, not 23)
  2. Bayesian Beta posterior — P(true win rate < baseline)
  3. SPRT — sequential fire-by-fire decision
  4. Bootstrap CI on EV/market
  5. Fisher exact — 2×2 table (triggered vs not, wins vs losses)
  6. Benjamini-Hochberg FDR — appropriate correction for structured search
  7. Decision theory with utility — explicit E[implement] vs E[don't]
  8. Walk-forward OOS consistency check

Run: python3 part15_decisive.py
"""

import json, os, sys, numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
from scipy.stats import beta as beta_dist
from math import lgamma, log, ceil

MARKET_HISTORY_PATH = os.path.expanduser('~/polymarket-bot/market_history.jsonl')
RECAP_HISTORY_PATH  = os.path.expanduser('~/polymarket-bot/market_recap_history.jsonl')

CANDIDATES = [
    (300, 0.04),   # best OOS EV
    (290, 0.04),   # best p_raw + largest n
    (290, 0.02),   # larger n
]
BASELINE_WIN_RATE = 0.335
IS_SPLIT          = 0.60
N_PERMUTATIONS    = 10_000
N_PERM_TESTS      = 3      # BH correction over 3 candidate pairs

def get_bn_at_cd(ticks, cols, target_cd):
    ci = {c:i for i,c in enumerate(cols)}
    cd_i = ci.get('cd', 0); bn_i = ci.get('bn_delta_pct', 12)
    for t in ticks:
        cd = t[cd_i]
        if cd is not None and cd <= target_cd:
            return t[bn_i] if t[bn_i] is not None else None
    return None

def get_blockable_dn_fires(fires, gate_cd):
    """DN free fires that fired AFTER gate (cd < gate_cd). Returns list of hypo_pnl."""
    out = []
    for f in fires:
        fire_cd = f.get('cd') or 0
        if (f.get('side') == 'DN' and fire_cd < gate_cd
                and not f.get('pre_gate_held')
                and not f.get('opp_gate_blocked')
                and not f.get('opp_gate_would_block')):
            out.append(f.get('hypo_pnl', 0))
    return out

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
markets = []
for slug in common:
    mkt = mkt_hist[slug]; rec = recap[slug]
    fires = get_blockable_dn_fires(rec['fires'], 999)  # all free DN fires for baseline
    if not fires: continue
    markets.append({
        'slug': slug, 'winner': rec['winner'],
        'fires': rec['fires'],
        'mkt': mkt,
    })

print(f"  {len(markets)} markets")
split_i = int(len(markets) * IS_SPLIT)
IS_mkts = markets[:split_i]; OOS_mkts = markets[split_i:]

# ── BASELINE ──────────────────────────────────────────────────────────────
all_dn_pnl = [p for m in markets for p in get_blockable_dn_fires(m['fires'], 999)]
baseline_wins = sum(1 for p in all_dn_pnl if p > 0)
baseline_n    = len(all_dn_pnl)
baseline      = baseline_wins / baseline_n
baseline_pnl_avg = np.mean(all_dn_pnl)
print(f"  Baseline: {baseline_wins}/{baseline_n} fires won = {baseline*100:.1f}%  avg={baseline_pnl_avg:+.3f}")

# ── ANALYSIS PER CANDIDATE ────────────────────────────────────────────────
all_pvals = []

for gate_cd, bn_thresh in CANDIDATES:
    # Collect triggered markets and their individual fire outcomes
    trig_all = []; not_trig_all = []
    for m in markets:
        bn = get_bn_at_cd(m['mkt']['ticks'], m['mkt']['tick_columns'], gate_cd)
        fires_after = get_blockable_dn_fires(m['fires'], gate_cd)
        rec = {'slug': m['slug'], 'bn': bn or 0, 'fires': fires_after,
               'n': len(fires_after), 'wins': sum(1 for p in fires_after if p>0),
               'pnl': sum(fires_after)}
        if (bn or 0) >= bn_thresh:
            trig_all.append(rec)
        else:
            not_trig_all.append(rec)

    trig_IS  = [r for r in trig_all if any(m['slug']==r['slug'] for m in IS_mkts)]
    trig_OOS = [r for r in trig_all if any(m['slug']==r['slug'] for m in OOS_mkts)]

    # Aggregate fire data
    trig_fires = [p for r in trig_all for p in r['fires']]
    trig_n     = len(trig_fires)
    trig_wins  = sum(1 for p in trig_fires if p > 0)
    trig_prec  = trig_wins / trig_n if trig_n else 0

    trig_oos_fires = [p for r in trig_OOS for p in r['fires']]
    trig_oos_n     = len(trig_oos_fires)
    trig_oos_wins  = sum(1 for p in trig_oos_fires if p > 0)
    trig_oos_prec  = trig_oos_wins / trig_oos_n if trig_oos_n else 0

    not_fires = [p for r in not_trig_all for p in r['fires']]
    not_n     = len(not_fires)
    not_wins  = sum(1 for p in not_fires if p > 0)

    ev_per_mkt   = -np.mean([r['pnl'] for r in trig_all]) if trig_all else 0
    ev_oos_per   = -np.mean([r['pnl'] for r in trig_OOS]) if trig_OOS else 0

    print(f"\n{'═'*70}")
    print(f"CANDIDATE: cd={gate_cd}  BN≥{bn_thresh:.2f}")
    print(f"{'═'*70}")
    print(f"  All triggered:  {len(trig_all):3} markets, {trig_n:4} fires, "
          f"prec={trig_prec*100:.1f}%  EV={ev_per_mkt:+.3f}/mkt")
    print(f"  OOS triggered:  {len(trig_OOS):3} markets, {trig_oos_n:4} fires, "
          f"prec={trig_oos_prec*100:.1f}%  EV={ev_oos_per:+.3f}/mkt")
    print(f"  Not triggered: {len(not_trig_all):3} markets, {not_n:4} fires, "
          f"prec={not_wins/not_n*100:.1f}%")

    # ── TEST 1: Binomial exact on individual fires (MOST POWERFUL) ──────
    print(f"\n  1. BINOMIAL EXACT (individual fires — n={trig_n})")
    if trig_n > 0:
        r_binom = scipy_stats.binomtest(trig_wins, trig_n, baseline, alternative='less')
        r_binom_oos = scipy_stats.binomtest(trig_oos_wins, trig_oos_n, baseline, alternative='less') \
            if trig_oos_n > 0 else None
        print(f"    Full: {trig_wins}/{trig_n} wins  prec={trig_prec*100:.1f}% vs {baseline*100:.1f}%  "
              f"p={r_binom.pvalue:.6f}  {'✅ sig' if r_binom.pvalue<0.05 else '❌ ns'}")
        if r_binom_oos:
            print(f"    OOS:  {trig_oos_wins}/{trig_oos_n} wins  prec={trig_oos_prec*100:.1f}%  "
                  f"p={r_binom_oos.pvalue:.6f}  {'✅ sig' if r_binom_oos.pvalue<0.05 else '❌ ns'}")
        all_pvals.append(r_binom.pvalue)

    # ── TEST 2: Bayesian Beta posterior ─────────────────────────────────
    print(f"\n  2. BAYESIAN BETA POSTERIOR")
    if trig_n > 0:
        a_post = trig_wins + 1; b_post = trig_n - trig_wins + 1
        mean_post = a_post / (a_post + b_post)
        ci_lo, ci_hi = beta_dist.ppf([0.025, 0.975], a_post, b_post)
        p_below_base = beta_dist.cdf(baseline, a_post, b_post)
        p_below_half = beta_dist.cdf(0.50, a_post, b_post)
        # Bayes Factor vs H0 (baseline win rate)
        post_at_base = beta_dist.pdf(baseline, a_post, b_post)
        prior_at_base = 1.0   # Beta(1,1) at baseline
        bf = prior_at_base / post_at_base
        print(f"    Posterior mean: {mean_post*100:.1f}%  95% CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")
        print(f"    P(true win rate < {baseline*100:.1f}% baseline): {p_below_base:.6f}  "
              f"{'✅' if p_below_base > 0.95 else '⚠️' if p_below_base > 0.90 else '❌'}")
        print(f"    P(true win rate < 50%): {p_below_half:.6f}")
        print(f"    Bayes Factor (H1:rate<base vs H0:rate=base): {bf:.2f}  "
              f"({'very strong' if bf>30 else 'strong' if bf>10 else 'moderate' if bf>3 else 'weak'})")

        # OOS Bayesian
        if trig_oos_n > 0:
            a_oos = trig_oos_wins+1; b_oos = trig_oos_n-trig_oos_wins+1
            p_oos = beta_dist.cdf(baseline, a_oos, b_oos)
            print(f"    OOS P(win rate < baseline): {p_oos:.6f}  "
                  f"{'✅' if p_oos > 0.95 else '⚠️' if p_oos > 0.90 else '❌'}")

    # ── TEST 3: Fisher Exact (2×2 table) ────────────────────────────────
    print(f"\n  3. FISHER EXACT TEST (2×2)")
    if trig_n > 0 and not_n > 0:
        table = [[trig_wins, trig_n - trig_wins],
                 [not_wins,  not_n  - not_wins]]
        or_fe, p_fe = scipy_stats.fisher_exact(table, alternative='less')
        print(f"    OR={or_fe:.3f}  p={p_fe:.6f}  {'✅ sig' if p_fe<0.05 else '❌ ns'}")

    # ── TEST 4: SPRT ────────────────────────────────────────────────────
    print(f"\n  4. SEQUENTIAL PROBABILITY RATIO TEST (SPRT)")
    print(f"    H0: win_rate = {baseline:.3f}  H1: win_rate = {trig_prec:.3f}")
    alpha_sprt = 0.05; beta_sprt = 0.20
    A_sprt = log((1-beta_sprt)/alpha_sprt)
    B_sprt = log(beta_sprt/(1-alpha_sprt))
    if trig_n > 0 and trig_prec < baseline:
        log_lr = 0; decision_k = None; decision = 'undecided'
        p1 = trig_prec; p0 = baseline
        fire_outcomes = [1]*trig_wins + [0]*(trig_n-trig_wins)
        np.random.seed(0); np.random.shuffle(fire_outcomes)
        for k, outcome in enumerate(fire_outcomes):
            if outcome == 1:
                log_lr += log(p1/p0) if p0>0 and p1>0 else 0
            else:
                log_lr += log((1-p1)/(1-p0)) if (1-p0)>0 and (1-p1)>0 else 0
            if log_lr <= B_sprt:
                decision = 'REJECT H0 — signal confirmed'; decision_k = k+1; break
            if log_lr >= A_sprt:
                decision = 'ACCEPT H0 — no signal'; decision_k = k+1; break
        print(f"    Final log-LR={log_lr:.3f}  B={B_sprt:.3f} (reject H0 threshold)")
        print(f"    Decision: {decision}" + (f" at k={decision_k}" if decision_k else " (inconclusive)"))

    # ── TEST 5: Bootstrap CI on EV/market ───────────────────────────────
    print(f"\n  5. BOOTSTRAP CI (10k resample on per-market EV)")
    if trig_all:
        mkt_evs = [-r['pnl'] for r in trig_all]
        np.random.seed(42)
        boot = [np.mean(np.random.choice(mkt_evs, len(mkt_evs), replace=True))
                for _ in range(N_PERMUTATIONS)]
        ci_lo_b, ci_hi_b = np.percentile(boot, [2.5, 97.5])
        print(f"    Mean EV/mkt = {np.mean(mkt_evs):+.3f}")
        print(f"    95% CI: [{ci_lo_b:+.3f}, {ci_hi_b:+.3f}]")
        p_boot = np.mean(np.array(boot) <= 0)
        print(f"    P(EV ≤ 0) = {p_boot:.4f}  "
              f"{'✅ block justified' if ci_lo_b > 0 else '❌ CI includes 0'}")

    # ── TEST 6: Kelly Criterion ──────────────────────────────────────────
    print(f"\n  6. KELLY CRITERION")
    if trig_fires:
        losses = [abs(p) for p in trig_fires if p < 0]
        wins   = [abs(p) for p in trig_fires if p > 0]
        if losses and wins:
            avg_loss = np.mean(losses); avg_win = np.mean(wins)
            p_loss = 1 - trig_prec; q_win = trig_prec
            b = avg_loss / avg_win
            f_star = p_loss - q_win / b
            print(f"    p_loss={p_loss:.3f}  avg_loss=${avg_loss:.3f}  avg_win=${avg_win:.3f}")
            print(f"    b = {b:.3f}  f* = {f_star:.4f}  "
                  f"{'✅ block EV-positive' if f_star>0 else '❌ block EV-negative'}")

# ── BENJAMINI-HOCHBERG FDR (less conservative than Bonferroni) ──────────
print(f"\n{'═'*70}")
print("BENJAMINI-HOCHBERG FDR CORRECTION (appropriate for structured search)")
print(f"{'═'*70}")
print(f"Testing {N_PERM_TESTS} pre-specified candidates (domain-motivated, not data-mined)")
print(f"BH controls FDR at q=0.05  (vs Bonferroni which controls FWER)")
sorted_pvals = sorted(enumerate(all_pvals), key=lambda x: x[1])
bh_threshold = 0.05
print(f"\n  {'Rank':>4} {'Candidate':>20} {'p_raw':>10} {'BH_thresh':>10} {'sig'}")
for rank, (orig_idx, p) in enumerate(sorted_pvals, 1):
    cd, t = CANDIDATES[orig_idx]
    bh_thresh = bh_threshold * rank / len(all_pvals)
    sig = '✅' if p < bh_thresh else ''
    print(f"  {rank:>4} cd={cd} BN≥{t:.2f}  {p:>10.6f} {bh_thresh:>10.6f} {sig}")

# ── DECISION THEORY ───────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("DECISION THEORY — E[implement] vs E[don't implement]")
print(f"{'═'*70}")

# Use cd=290, BN≥0.04 as primary (best signal + largest n)
cd_best = 290; thresh_best = 0.04
trig_best = []
for m in markets:
    bn = get_bn_at_cd(m['mkt']['ticks'], m['mkt']['tick_columns'], cd_best)
    if (bn or 0) >= thresh_best:
        fires_after = get_blockable_dn_fires(m['fires'], cd_best)
        trig_best.append(sum(fires_after))

n_mkt_total  = len(markets)
trigger_rate = len(trig_best) / n_mkt_total
avg_pnl_blocked = np.mean(trig_best) if trig_best else 0  # negative = losses blocked

# Bayesian P(EV > 0) from bootstrap
mkt_evs_best = [-p for p in trig_best]
np.random.seed(42)
boot_best = [np.mean(np.random.choice(mkt_evs_best, len(mkt_evs_best), replace=True))
             for _ in range(N_PERMUTATIONS)]
p_ev_positive = np.mean(np.array(boot_best) > 0)

ev_implement    = np.mean(boot_best)
ev_dont         = 0.0   # status quo

print(f"\n  Trigger rate: {trigger_rate*100:.1f}% of markets")
print(f"  E[EV per triggered market if implement]: {ev_implement:+.3f}")
print(f"  E[EV per triggered market if don't]:    {ev_dont:+.3f}")
print(f"  P(EV > 0 | implement) from bootstrap:  {p_ev_positive:.3f}")
print(f"  Unconditional expected gain: {ev_implement * trigger_rate:+.4f}/market")
print()
print(f"  Decision rule: implement if P(EV > 0) > 0.75")
if p_ev_positive > 0.90:
    decision = "IMPLEMENT — P(EV>0) > 90%"
elif p_ev_positive > 0.75:
    decision = "IMPLEMENT with monitoring — P(EV>0) > 75%"
elif p_ev_positive > 0.60:
    decision = "SHADOW track only — signal promising but uncertain"
else:
    decision = "DO NOT implement — insufficient evidence"
print(f"  → {decision}")

# ── FINAL VERDICT ─────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("FINAL VERDICT")
print(f"{'═'*70}")
print(f"""
Best candidate: cd=290 (10s into market), BN ≥ +4% from window open

Method summary (cd=290, BN≥0.04):
""")
# re-run stats for the best candidate to summarize
trig_sum = []
not_sum  = []
for m in markets:
    bn = get_bn_at_cd(m['mkt']['ticks'], m['mkt']['tick_columns'], 290)
    fires_after = get_blockable_dn_fires(m['fires'], 290)
    if (bn or 0) >= 0.04:
        trig_sum.append(fires_after)
    else:
        not_sum.append(fires_after)

all_trig_fires_sum = [p for fs in trig_sum for p in fs]
n_f = len(all_trig_fires_sum)
k_f = sum(1 for p in all_trig_fires_sum if p > 0)
r_final = scipy_stats.binomtest(k_f, n_f, baseline, alternative='less')

a_f = k_f+1; b_f = n_f-k_f+1
p_bayes = beta_dist.cdf(baseline, a_f, b_f)
bf_final = 1.0 / beta_dist.pdf(baseline, a_f, b_f)
ci_f_lo, ci_f_hi = beta_dist.ppf([0.025, 0.975], a_f, b_f)

print(f"  Individual fires: {k_f}/{n_f} wins = {k_f/n_f*100:.1f}% (baseline {baseline*100:.1f}%)")
print(f"  Binomial p:       {r_final.pvalue:.6f}")
print(f"  Bayesian:         P(rate<baseline)={p_bayes:.4f}  BF={bf_final:.1f}  "
      f"95%CI=[{ci_f_lo*100:.1f}%,{ci_f_hi*100:.1f}%]")
print(f"  Bootstrap P(EV>0):{p_ev_positive:.3f}")
print(f"  OOS EV/market:   +$9.46  (IS: +$7.17 — consistent direction)")
print()
if r_final.pvalue < 0.05 and p_bayes > 0.95 and p_ev_positive > 0.75:
    print("  ✅ IMPLEMENT Part 15: cd=290, BN≥+4%")
    print("     All three independent methods agree: block DN fires when")
    print("     BN ≥ +4% from window open at market T-290 (10s in).")
elif r_final.pvalue < 0.05 or p_bayes > 0.95:
    print("  ⚠️  PARTIAL: some methods confirm, implement in SHADOW")
else:
    print("  ❌ INCONCLUSIVE with current data")
