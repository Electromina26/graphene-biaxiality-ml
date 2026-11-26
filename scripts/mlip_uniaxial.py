"""Uniaxial-stress loading of graphene with an ASE calculator (same protocol as the DFT driver).
For each applied strain eps along n(theta): find et, g with sigma_tt = sigma_nt = 0, atoms relaxed.
Also single-point stresses at the DFT geometries.
usage: python mlip_uniaxial.py <model> <outfile.json>"""
import sys, json, warnings, numpy as np
warnings.filterwarnings('ignore')
from ase import Atoms
from ase.optimize import BFGS
A0, LZ = 2.4669, 12.0
EPS = [0.005,0.01,0.02,0.03,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30]
THETAS = [0,10,20,30]
DFT = '/home/user/graphene_dft/dft/run/th{}/results.json'

def make_calc(name):
    if name.startswith('mace-'):
        from mace.calculators import mace_mp
        return mace_mp(model=name[5:], default_dtype='float64', device='cpu')
    if name.startswith('ft:'):           # fine-tuned MACE model file
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=name[3:], default_dtype='float64', device='cpu')
    if name == 'chgnet':
        from chgnet.model.dynamics import CHGNetCalculator
        return CHGNetCalculator(use_device='cpu')
    if name == '7net-0':
        from sevenn.calculator import SevenNetCalculator
        return SevenNetCalculator('7net-0', device='cpu')
    if name == '7net-mf-ompa':
        from sevenn.calculator import SevenNetCalculator
        return SevenNetCalculator('7net-mf-ompa', modal='mpa', device='cpu')
    if name == 'mattersim-1M':
        from mattersim.forcefield import MatterSimCalculator
        return MatterSimCalculator(load_path='MatterSim-v1.0.0-1M.pth', device='cpu')
    if name == 'mattersim-5M':
        from mattersim.forcefield import MatterSimCalculator
        return MatterSimCalculator(load_path='MatterSim-v1.0.0-5M.pth', device='cpu')
    raise ValueError(name)

def U_of(th, eps, et, g):
    n = np.array([np.cos(th), np.sin(th)]); t = np.array([-np.sin(th), np.cos(th)])
    U2 = np.eye(2) + eps*np.outer(n, n) + et*np.outer(t, t) + g*(np.outer(n, t) + np.outer(t, n))
    U = np.eye(3); U[:2, :2] = U2; return U, n, t

def ref_cell(a=None):
    a = A0 if a is None else a
    return np.array([[a, 0, 0], [a/2, a*np.sqrt(3)/2, 0], [0, 0, LZ]])

LAM = 0.0     # stress biaxiality: target sigma_tt = LAM*sigma_nn (0 = uniaxial stress)
AREF = None   # model's own equilibrium lattice constant (set in main)
def equilibrium_a(calc):
    from scipy.optimize import brentq
    def s(a):
        at = Atoms('C2', cell=ref_cell(a), scaled_positions=[[0, 0, .5], [1/3, 1/3, .5]], pbc=True); at.calc = calc
        return stress2d(at)[0, 0]
    return brentq(s, 2.35, 2.60, xtol=1e-7)

def stress2d(atoms):
    s = atoms.get_stress(voigt=False)        # eV/A^3, ASE sign: positive = tensile? ASE: stress = -virial/V (tension positive)
    return s*LZ*16.0217663                   # N/m  (1 eV/A^2 = 16.02 N/m)

def relax(atoms, fmax=5e-4):
    BFGS(atoms, logfile=None).run(fmax=fmax, steps=300)

def solve_point(calc, th_deg, eps, guess, frac):
    th = np.radians(th_deg); sym = abs(th_deg % 30) < 1e-9
    A = ref_cell(AREF); et, g = guess
    def resid(et, g, frac):
        U, n, t = U_of(th, eps, et, g); cell = (U @ A.T).T
        at = Atoms('C2', cell=cell, scaled_positions=frac, pbc=True); at.calc = calc
        relax(at)
        s = stress2d(at); n3 = np.r_[n, 0]; t3 = np.r_[t, 0]
        return n3@s@n3, t3@s@t3, n3@s@t3, at
    hist = []
    for it in range(30):
        snn, stt, snt, at = resid(et, g, frac); frac = at.get_scaled_positions()
        r = stt - LAM*snn
        hist.append((et, g, r, snt))
        if abs(r) < 0.005 and (sym or abs(snt) < 0.005): break
        if len(hist) >= 2 and abs(hist[-1][0]-hist[-2][0]) > 1e-8:
            d = (hist[-1][2]-hist[-2][2])/(hist[-1][0]-hist[-2][0]); d = d if d > 20 else 350.
        else: d = 350.
        et = et - r/d
        if not sym:
            if len(hist) >= 2 and abs(hist[-1][1]-hist[-2][1]) > 1e-8:
                dg = (hist[-1][3]-hist[-2][3])/(hist[-1][1]-hist[-2][1]); dg = dg if dg > 20 else 290.
            else: dg = 290.
            g = g - snt/dg
    pos = at.get_positions(); cell = at.cell.array
    # bond lengths (minimum image)
    d = []
    for i, j in [(0, 0), (-1, 0), (0, -1)]:
        v = pos[1] - pos[0] + i*cell[0] + j*cell[1]; d.append(np.linalg.norm(v))
    return dict(theta=th_deg, lam=LAM, eps=eps, et=et, g=g, snn=snn, stt=stt, snt=snt, P=snn*(1+et),
                E=at.get_potential_energy(), bonds=sorted(d), iters=len(hist)), frac

if __name__ == '__main__':
    name, out = sys.argv[1], sys.argv[2]
    calc = make_calc(name)
    AREF = equilibrium_a(calc); print(name, 'a0 =', AREF, flush=True)
    res = dict(model=name, a0=AREF, path={}, single={})
    for th in THETAS:
        frac = np.array([[0, 0, 0.5], [1/3, 1/3, 0.5]]); et, g = 0.0, 0.0; rows = []; prev = None
        for eps in EPS:
            gt = et*eps/prev if prev else -0.17*eps; gg = g*eps/prev if prev else 0.0
            r, frac = solve_point(calc, th, eps, (gt, gg), frac)
            et, g, prev = r['et'], r['g'], eps; rows.append(r)
            print(name, th, eps, round(r['P'], 3), round(r['et'], 5), r['iters'], flush=True)
        res['path'][th] = rows
        # single points at DFT relaxed geometries
        sp = []
        for r in json.load(open(DFT.format(th))):
            at = Atoms('C2', positions=r['pos'], cell=r['cell'], pbc=True); at.calc = calc
            s = stress2d(at); n = np.array([np.cos(np.radians(th)), np.sin(np.radians(th)), 0]); t = np.array([-n[1], n[0], 0])
            f = np.abs(at.get_forces()).max()
            sp.append(dict(eps=r['eps'], snn=n@s@n, stt=t@s@t, snt=n@s@t, fmax=f, E=at.get_potential_energy(),
                           dft_snn=r['snn'], dft_E=r['E']))
        res['single'][th] = sp
        json.dump(res, open(out, 'w'), indent=1, default=float)
