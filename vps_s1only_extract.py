#!/usr/bin/env python3
"""
Run on VPS: python3 vps_s1only_extract.py
Extracts all S1only markets (bn200<-0.04 fires, bn120 does NOT confirm)
for offline statistical validation.
"""
import json, math
from collections import defaultdict

HISTORY = "/home/polybot/polymarket-bot/data/market_history.jsonl"
RECAP   = "/home/polybot/polymarket-bot/data/market_recap_history.jsonl"

def g_bn200(sc): 
    v = sc.get("T-200", {}).get("bn_delta")
    return v is not None and v < -0.040

def g_bn120(sc): 
    v = sc.get("T-120", {}).get("bn_delta")
    return v is not None and v < -0.030

def g_bn150(sc):
    v = sc.get("T-150", {}).get("bn_delta")
    return v  # raw value

# Load recap for outcome and PnL
recap_map = {}
try:
    with open(RECAP) as f:
        for line in f:
            d = json.loads(line)
            recap_map[d["slug"]] = d
except Exception as e:
    print(f"Recap load error: {e}")

# Extract S1only markets
s1only = []
all_gate_fired = []

with open(HISTORY) as f:
    for line in f:
        m = json.loads(line)
        sc = m.get("skip_counterfactuals", {})
        if not sc:
            continue
        
        bn200_fires = g_bn200(sc)
        bn120_fires = g_bn120(sc)
        bn150_val   = g_bn150(sc)
        
        # Entry depth (approximate from history if available)
        entry_depth = m.get("entry_dn_depth_avg")
        depth_fires = (entry_depth is not None and entry_depth > 534.0)
        
        slug    = m.get("slug", "")
        winner  = m.get("winner", "")
        
        rec = recap_map.get(slug, {})
        gc_up_pnl = rec.get("gc_up_pnl", None)
        s0 = rec.get("s0", None)
        s1 = rec.get("s1", None)
        s2 = rec.get("s2", None)
        
        row = {
            "slug": slug,
            "winner": winner,
            "bn200": sc.get("T-200", {}).get("bn_delta"),
            "bn120": sc.get("T-120", {}).get("bn_delta"),
            "bn150": bn150_val,
            "bn200_fires": bn200_fires,
            "bn120_fires": bn120_fires,
            "depth_fires": depth_fires,
            "entry_dn_depth": entry_depth,
            "gc_up_pnl": gc_up_pnl,
            "s0": s0, "s1": s1, "s2": s2,
            # crowd at T-200
            "crowd_200": sc.get("T-200", {}).get("crowd_side"),
            "conv_200":  sc.get("T-200", {}).get("crowd_conv"),
        }
        
        is_s1only = bn200_fires and not bn120_fires and not depth_fires
        is_any_gate = bn200_fires or bn120_fires or depth_fires
        
        if is_s1only:
            row["gate_type"] = "S1only"
            s1only.append(row)
        elif is_any_gate:
            row["gate_type"] = "S2_or_DEP"
            
        if is_any_gate:
            all_gate_fired.append(row)

print(f"Total markets scanned: loaded from {HISTORY}")
print(f"Gate-fired markets: {len(all_gate_fired)}")
print(f"S1only markets: {len(s1only)}")
print()

if s1only:
    ups   = [r for r in s1only if r["winner"]=="UP"]
    dns   = [r for r in s1only if r["winner"]=="DN"]
    print(f"S1only UP wins: {len(ups)}  DN wins: {len(dns)}")
    print(f"S1only UP rate: {len(ups)/len(s1only):.3f}")
    print()
    print("S1only market details:")
    for r in s1only:
        print(f"  {r['slug'][-10:]} winner={r['winner']} bn200={r['bn200']:>+.4f} "
              f"bn120={r['bn120']:>+.4f} bn150={r['bn150']:>+.4f if r['bn150'] else 'None':>8} "
              f"s0={r['s0']:>+8.2f if r['s0'] else 'None':>8} "
              f"s1={r['s1']:>+8.2f if r['s1'] else 'None':>8}")

# Save for upload
out = {
    "s1only": s1only,
    "all_gate_fired_count": len(all_gate_fired),
    "gate_summary": {
        "total": len(all_gate_fired),
        "S2_dn": sum(1 for r in all_gate_fired if r["gate_type"]=="S2_or_DEP" and r["winner"]=="DN"),
        "S2_up": sum(1 for r in all_gate_fired if r["gate_type"]=="S2_or_DEP" and r["winner"]=="UP"),
    }
}
with open("/home/polybot/s1only_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to /home/polybot/s1only_results.json")
print("Upload this file back to Claude for full statistical analysis.")
