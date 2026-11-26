"""Seed-averaged failure strain / peak 2D stress for the 48-atom v2 MD runs -> md_seedavg.json"""
import json, glob, numpy as np
rows = {}
for fn in glob.glob('/home/user/graphene/md/ft_omat_r2_s2_th*_s[0-9]_v2.json'):
    d = json.load(open(fn)); h = d['hist']
    if len(h) < 6: continue
    e = np.array([x['eps'] for x in h]); s = np.convolve([x['snn'] for x in h], np.ones(5)/5, 'valid'); i = np.argmax(s)
    rows.setdefault((d['theta'], d['lam'], d['T']), []).append((100*e[2:-2][i], s[i]))
out = []
for k in sorted(rows):
    v = np.array(rows[k]); m = v.mean(0); sd = v.std(0, ddof=1) if len(v) > 1 else [np.nan]*2
    print(f"th{k[0]:2d} lam{k[1]:<4g} T{k[2]:5.0f} n{len(v)} eps {m[0]:5.1f}±{sd[0]:.1f}  P {m[1]:5.1f}±{sd[1]:.1f}")
    out.append(dict(theta=k[0], lam=k[1], T=k[2], n=len(v), eps=float(m[0]), eps_sd=float(sd[0]), P=float(m[1]), P_sd=float(sd[1])))
json.dump(out, open('/home/user/graphene/md_seedavg.json', 'w'), indent=1)
