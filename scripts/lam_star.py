"""Failure-mode crossover lam*(theta): smallest biaxiality at which the K-point (Kekule) in-plane mode softens before
the stress maximum, i.e. Delta(lam) = eps_K(lam) - eps_m(lam) changes sign (linear interpolation between grid lams).
usage: python lam_star.py <model-file> [...]  -> prints table, writes lam_star_<seed tag>.json"""
import sys, glob, re, json, os
sys.path.insert(0, '/home/user/graphene')
from onsets import onsets
res = {}
for mf in sys.argv[1:]:
    tag = 'ft_' + mf.replace('/', '_'); seed = re.search(r'(r\d+_s\d+)', mf).group(1); res[seed] = {}
    for th in range(0, 31, 5):
        rows = []
        for fn in glob.glob(f'/home/user/graphene/phon/{tag}_th{th}_lam*.json') + glob.glob(f'/home/user/graphene/phon/{tag}_th{th}.json'):
            m = re.search(r'_lam([\d.]+)\.json$', fn); lam = float(m.group(1)) if m else 0.0
            o = onsets(fn); d = (o['eps_K'] if o['eps_K'] is not None else 0.30) - o['eps_m']
            rows.append((lam, o['eps_m'], o['eps_K'], d))
        if not rows: continue
        rows.sort(); ls = None
        for (l0, _, _, d0), (l1, _, _, d1) in zip(rows, rows[1:]):
            if d0 > 0 and d1 <= 0: ls = l0 + (l1-l0)*d0/(d0-d1); break
        res[seed][th] = dict(lam_star=ls, rows=rows)
        print(seed, f'theta {th:2d}', 'lam* =', f'{ls:.3f}' if ls else 'n/a', ' '.join(f'{l:g}:{100*d:+.1f}' for l, _, _, d in rows))
json.dump(res, open('/home/user/graphene/lam_star.json', 'w'), indent=1)
