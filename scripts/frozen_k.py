"""Frozen-phonon validation of the K-point (Kekule) in-plane mode, DFT vs fine-tuned MLIP.
For a stored MLIP-relaxed strained primitive cell (phon/..._v2.json, given theta, eps):
  1. MLIP phonopy eigenvector of the lowest in-plane mode at K = (2/3, 1/3) (60-degree basis).
  2. sqrt3 x sqrt3 R30 supercell A1 = a1 + a2, A2 = -a1 + 2 a2 (commensurate with K: K.A = integers).
  3. Displace u_j = Re[e_s exp(2 pi i K.r_j)] (r_j = full position, phonopy convention), scaled to max in-plane amplitude A; cell held fixed.
  4. E(A) with QE (PBE, same settings as the training set) and with the MLIP; curvature -> frequency.
usage: python frozen_k.py <theta> <eps> [lam] ; output frozen/<tag>.json"""
import sys, os, json, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/graphene')
import mlip_uniaxial as mu, qe_sp
from phonon_reanalyze import fc
from ase import Atoms
from ase.build import make_supercell
MODEL = os.environ.get('FKMODEL', 'ft:/home/user/graphene/ft/ft_omat_s2.model')
AMPS = [0.0, 0.03, 0.06, 0.09]
th, eps = float(sys.argv[1]), float(sys.argv[2]); lam = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
src = f'/home/user/graphene/phon/' + MODEL.replace('ft:', 'ft_').replace('/', '_') + f'_th{th:g}' + (f'_lam{lam:g}' if lam else '') + '_v2.json'
if not os.path.exists(src): src = src.replace('_v2.json', '.json')
r = [x for x in json.load(open(src)) if abs(x['eps']-eps) < 1e-6][0]
cell, frac = np.array(r['cell']), np.array(r['frac'])
calc = mu.make_calc(MODEL)
ph = fc(calc, cell, frac)
K = np.array([2/3, 1/3, 0])
ph.run_qpoints([K], with_eigenvectors=True); d = ph.get_qpoints_dict()
f = d['frequencies'][0]*33.35641; ev = d['eigenvectors'][0]
zw = np.abs(ev[2, :])**2+np.abs(ev[5, :])**2
m = [i for i in range(6) if zw[i] < .5][0]                  # lowest in-plane mode at K
e = ev[:, m].reshape(2, 3)
prim = Atoms('C2', cell=cell, scaled_positions=frac, pbc=True)
P = np.array([[1, 1, 0], [-1, 2, 0], [0, 0, 1]])
sc = make_supercell(prim, P)
# lattice translation R_j (fractional, primitive basis) and sublattice index s of each supercell atom
fr = np.linalg.solve(cell.T, sc.positions.T).T
u = np.zeros((len(sc), 3))
for j in range(len(sc)):
    s = int(np.argmin([np.linalg.norm(((fr[j]-frac[k]+.5) % 1)-.5) for k in range(2)]))
    R = np.round(fr[j]-frac[s])
    u[j] = np.real(e[s]*np.exp(2j*np.pi*K @ (R+frac[s])))     # phonopy convention: phase of the full atomic position
u[:, 2] = 0; u /= np.abs(u[:, :2]).max()
print(f'theta {th} eps {eps} lam {lam}: MLIP K in-plane mode {f[m]:.1f} cm-1; supercell {len(sc)} atoms', flush=True)
os.makedirs('/home/user/graphene/frozen', exist_ok=True)
tag = f'th{th:g}_e{eps:g}' + (f'_lam{lam:g}' if lam else '')
out = f'/home/user/graphene/frozen/{tag}.json'
res = json.load(open(out)) if os.path.exists(out) else dict(theta=th, eps=eps, lam=lam, mlip_freq=float(f[m]), pts=[])
done = {p['A'] for p in res['pts']}
for A in AMPS:
    if A in done: continue
    at = sc.copy(); at.positions += A*u
    at.calc = calc; Em = at.get_potential_energy()
    Ed, F, S, k = qe_sp.run(at, '/home/user/graphene/frozenwork', f'{tag}_A{A}')
    res['pts'].append(dict(A=A, E_mlip=float(Em), E_dft=float(Ed)))
    json.dump(res, open(out, 'w'), indent=1)
    print(A, Em, Ed, flush=True)
