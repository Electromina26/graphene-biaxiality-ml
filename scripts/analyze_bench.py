import json, glob, os, numpy as np
from scipy.interpolate import CubicSpline
S = json.load(open('/home/user/graphene_dft/summary.json'))
def ideal(eps, P):
    e = np.r_[0, eps]; p = np.r_[0, P]
    m = e <= e[np.argmax(p)] + 0.04
    cs = CubicSpline(e[m], p[m]); x = np.linspace(0, e[m].max(), 3000); y = cs(x); k = np.argmax(y)
    return x[k], y[k]
def summarize(fn):
    R = json.load(open(fn)); out = dict(model=R['model'], theta={})
    allerr = []
    for th in ['0', '10', '20', '30']:
        if th not in R['path']: continue
        p = R['path'][th]; e = np.array([r['eps'] for r in p]); P = np.array([r['P'] for r in p]); et = np.array([r['et'] for r in p])
        Y = np.polyfit(e[e <= 0.02], P[e <= 0.02], 1)[0]; nu = -et[0]/e[0]
        em, pm = ideal(e, P)
        sp = R['single'].get(th, [])
        err = [r['snn']-r['dft_snn'] for r in sp if r['eps'] <= 0.27]
        allerr += err
        fmax = max([r['fmax'] for r in sp if r['eps'] <= 0.27] or [np.nan])
        out['theta'][th] = dict(Y=Y, nu=nu, eps_m=em, P_m=pm, dft_eps_m=S['ideal'][th]['eps_m'], dft_P_m=S['ideal'][th]['P_m'],
                                stress_rmse=float(np.sqrt(np.mean(np.square(err)))) if err else None, fmax_at_dft=fmax)
    out['stress_rmse_all'] = float(np.sqrt(np.mean(np.square(allerr)))) if allerr else None
    return out
if __name__ == '__main__':
    rows = []
    for f in sorted(glob.glob('/home/user/graphene/bench/*.json')):
        if not os.path.exists(f.replace('.json', '.done')): continue
        s = summarize(f); rows.append(s)
        t = s['theta']
        print(f"{s['model']:20s} Y0={t['0']['Y']:6.1f} nu={t['0']['nu']:.3f} "
              + ' '.join(f"em{th}={100*t[th]['eps_m']:5.1f}/{t[th]['P_m']:5.1f}" for th in t)
              + f"  rmse={s['stress_rmse_all']:.2f}")
    print('DFT  Y=316.2 nu=0.19 ' + ' '.join(f"em{th}={100*S['ideal'][th]['eps_m']:5.1f}/{S['ideal'][th]['P_m']:5.1f}" for th in ['0','10','20','30']))
    json.dump(rows, open('/home/user/graphene/bench_summary.json', 'w'), indent=1)
