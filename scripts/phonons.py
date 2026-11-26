"""Phonon stability of graphene along uniaxial-stress paths with an MLIP (phonopy, finite displacements).
For each loading angle theta and strain eps: relax the primitive cell under uniaxial stress
(mlip_uniaxial.solve_point), build a 6x6 supercell, compute force constants, and find the lowest
frequency on a dense q mesh (and at K, M). Output: phon/<model>_th<theta>.json
usage: python phonons.py <model> <theta> <eps_list>"""
import sys, json, os, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/graphene')
import mlip_uniaxial as mu
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
THz2cm = 33.35641

from phonon_reanalyze import fc, analyze
def phonon_scan(calc, cell, frac, N=6, mesh=36):
    """corrected K points (2/3,1/3),(1/3,2/3); in-plane / out-of-plane minima (see phonon_reanalyze.py)"""
    return analyze(fc(calc, cell, frac, N), mesh)

if __name__ == '__main__':
    model, th = sys.argv[1], float(sys.argv[2]); epss = [float(x) for x in sys.argv[3].split(',')]
    lam = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0; mu.LAM = lam
    calc = mu.make_calc(model); mu.AREF = mu.equilibrium_a(calc)
    os.makedirs('/home/user/graphene/phon', exist_ok=True)
    tag = model.replace('ft:', 'ft_').replace('/', '_')
    out = f'/home/user/graphene/phon/{tag}_th{th:g}' + (f'_lam{lam:g}' if lam else '') + '.json'
    res = json.load(open(out)) if os.path.exists(out) else []
    done = {round(r['eps'], 5) for r in res}
    frac = np.array([[0, 0, .5], [1/3, 1/3, .5]]); et, g, prev = 0., 0., None
    for eps in epss:
        gt = et*eps/prev if prev else (lam-0.17)*eps; gg = g*eps/prev if prev else 0.
        r, frac = mu.solve_point(calc, th, eps, (gt, gg), frac); et, g, prev = r['et'], r['g'], eps
        if round(eps, 5) in done: continue
        U, n, t = mu.U_of(np.radians(th), eps, et, g); cell = (U @ mu.ref_cell(mu.AREF).T).T
        p = phonon_scan(calc, cell, frac)
        res.append(dict(theta=th, lam=lam, eps=eps, et=et, g=g, P=r['P'], cell=cell.tolist(), frac=frac.tolist(), **p))
        json.dump(res, open(out, 'w'), indent=1, default=float)
        print(model, th, eps, round(r['P'], 2), 'IP', round(p['min_inplane'], 1), np.round(p['q_inplane'], 3), 'OP', round(p['min_outplane'], 1), 'K', np.round(p['K'][:2], 1), "K'", np.round(p['Kp'][:2], 1), flush=True)
