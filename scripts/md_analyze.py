"""Summarise tensile MD runs: strain at the smoothed stress maximum (5-block running mean), the peak stress,
and the strain at which a nearest-neighbour bond first exceeds 1.9 / 2.0 A. usage: python md_analyze.py [glob]"""
import sys, glob, json, numpy as np
pat = sys.argv[1] if len(sys.argv) > 1 else '/home/user/graphene/md/*_s1.json'
rows = []
for fn in sorted(glob.glob(pat)):
    d = json.load(open(fn)); h = d['hist']
    if len(h) < 6: continue
    e = np.array([x['eps'] for x in h]); s = np.array([x['snn'] for x in h]); dm = np.array([x['dmax'] for x in h])
    sm = np.convolve(s, np.ones(5)/5, mode='valid'); em = e[2:-2]
    i = int(np.argmax(sm)); b19 = e[np.argmax(dm > 1.9)] if (dm > 1.9).any() else None; b20 = e[np.argmax(dm > 2.0)] if (dm > 2.0).any() else None
    r = dict(theta=d['theta'], T=d['T'], lam=d['lam'], eps_peak=float(em[i]), P_peak=float(sm[i]), eps_b19=b19, eps_b20=b20,
             Tmean=float(np.mean([x['T'] for x in h[:max(i, 1)]])), finished=d['fail'] is not None)
    rows.append(r)
    f = lambda x: f'{100*x:5.1f}' if x is not None else '  n/a'
    print(f"theta {d['theta']:2d} T {d['T']:5.0f} lam {d['lam']:<4g} eps_peak {f(r['eps_peak'])} P_peak {r['P_peak']:5.1f}  bond>1.9 {f(b19)}  bond>2.0 {f(b20)}  <T> {r['Tmean']:5.0f}")
json.dump(rows, open('/home/user/graphene/md_summary.json', 'w'), indent=1)
