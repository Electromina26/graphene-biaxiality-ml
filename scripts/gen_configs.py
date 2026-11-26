"""Generate DFT training configurations for fine-tuning: strained graphene supercells
along uniaxial-stress paths (any loading angle), with thermal-like rattling (in- and out-of-plane)
and K-point (Kekule-type) soft-mode distortions. Writes configs/<id>.json (positions, cell, meta)."""
import json, os, numpy as np
from ase import Atoms
rng = np.random.default_rng(2026)
A0, LZ = 2.4669, 12.0
OUT = '/home/user/graphene/configs'; os.makedirs(OUT, exist_ok=True)
D = {t: json.load(open(f'/home/user/graphene_dft/dft/run/th{t}/results.json')) for t in (0, 10, 20, 30)}

def path_et_g(th, eps):
    """transverse strain and shear along the DFT uniaxial-stress path, interpolated in theta."""
    ets = []; gs = []
    for t in (0, 10, 20, 30):
        e = np.array([r['eps'] for r in D[t]]); o = np.argsort(e)
        ets.append(np.interp(eps, e[o], np.array([r['et'] for r in D[t]])[o]))
        if t in (10, 20):
            gs.append(np.interp(eps, e[o], np.array([r['g'] for r in D[t]])[o])/np.sin(np.radians(6*t)))
    et = np.interp(th, (0, 10, 20, 30), ets)
    g = np.mean(gs)*np.sin(np.radians(6*th))
    return et, g

def U_of(th, eps, et, g):
    th = np.radians(th); n = np.array([np.cos(th), np.sin(th)]); t = np.array([-np.sin(th), np.cos(th)])
    U = np.eye(3); U[:2, :2] = np.eye(2)+eps*np.outer(n, n)+et*np.outer(t, t)+g*(np.outer(n, t)+np.outer(t, n)); return U

def prim(th, eps, et, g):
    A = np.array([[A0, 0, 0], [A0/2, A0*np.sqrt(3)/2, 0], [0, 0, LZ]])
    cell = (U_of(th, eps, et, g) @ A.T).T
    return Atoms('C2', cell=cell, scaled_positions=[[0, 0, .5], [1/3, 1/3, .5]], pbc=True)

def kekule(at3, amp, prim_cell):
    """displacement pattern with wavevector K (commensurate with the sqrt3 x sqrt3 R30 cell):
    random in-plane polarisation on each sublattice, amplitude amp (A)."""
    rec = np.linalg.inv(prim_cell).T[:2, :2]*2*np.pi      # primitive reciprocal vectors
    K = (2*rec[0]+rec[1])/3
    pos = at3.positions[:, :2]; u = np.zeros((len(at3), 3))
    for sub in range(2):
        idx = np.array([i for i in range(len(at3)) if np.isclose(((at3.positions[i,:2]-at3.positions[0,:2]) @ np.linalg.inv(prim_cell[:2,:2]) % 1), 0, atol=1e-3).all() == (sub == 0)])
        pol = rng.normal(size=2)+1j*rng.normal(size=2); pol /= np.linalg.norm(pol)
        ph = np.exp(1j*(pos[idx] @ K))
        u[idx, :2] = np.real(np.outer(ph, pol))
    u *= amp/np.abs(u).max(); return u

def write(i, at, meta):
    json.dump(dict(cell=at.cell.array.tolist(), positions=at.positions.tolist(), meta=meta),
              open(f'{OUT}/{i:04d}.json', 'w'))

i = 0
# group B: 2x2 supercells, random angle/strain, rattled (in-plane + out-of-plane)
for k in range(120):
    th = rng.uniform(0, 30); eps = rng.uniform(0, 0.30)
    et, g = path_et_g(th, eps); et += rng.normal(0, 0.008); g += rng.normal(0, 0.004)
    at = prim(th, eps, et, g)*(2, 2, 1)
    s_ip = rng.choice([0.02, 0.05, 0.08, 0.12]); s_z = rng.choice([0.0, 0.05, 0.15, 0.3])
    d = rng.normal(0, s_ip, (len(at), 3)); d[:, 2] = rng.normal(0, s_z, len(at)) if s_z > 0 else 0
    at.positions += d
    write(i, at, dict(group='B', theta=th, eps=eps, et=et, g=g, s_ip=s_ip, s_z=s_z)); i += 1
# group C: sqrt3 x sqrt3 R30 supercells (K folds to Gamma) near/after instability,
# Kekule (K-point) distortions + small rattle
from ase.build import make_supercell
for k in range(70):
    th = rng.uniform(0, 30); eps = rng.uniform(0.10, 0.30)
    et, g = path_et_g(th, eps); et += rng.normal(0, 0.005); g += rng.normal(0, 0.003)
    p = prim(th, eps, et, g)
    at = make_supercell(p, [[2, 1, 0], [-1, 1, 0], [0, 0, 1]])
    amp = rng.choice([0.03, 0.06, 0.10, 0.15])
    at.positions += kekule(at, amp, p.cell.array) + rng.normal(0, 0.015, (len(at), 3))
    write(i, at, dict(group='C', theta=th, eps=eps, et=et, g=g, kekule=amp)); i += 1
print('written', i)
