"""Active-learning DFT configurations (round 1), filling gaps found in validation:
 D: primitive 2-atom cells under biaxial strain (transverse strain from contraction up to equibiaxial).
 E: sqrt3 x sqrt3 R30 cells (A1 = a1+a2, A2 = -a1+2a2; commensurate with K = (2/3,1/3)) carrying
    in-plane K-point (Kekule-type) displacements, on uniaxial-stress and biaxial strain states.
Writes configs/<id>.json with ids from 0190 upward."""
import json, os, numpy as np
from ase import Atoms
from ase.build import make_supercell
rng = np.random.default_rng(31)
A0, LZ = 2.4669, 12.0
OUT = '/home/user/graphene/configs'
D = {t: json.load(open(f'/home/user/graphene_dft/dft/run/th{t}/results.json')) for t in (0, 10, 20, 30)}

def et_uniax(th, eps):
    ets = []
    for t in (0, 10, 20, 30):
        e = np.array([r['eps'] for r in D[t]]); o = np.argsort(e)
        ets.append(np.interp(eps, e[o], np.array([r['et'] for r in D[t]])[o]))
    return np.interp(th, (0, 10, 20, 30), ets)

def prim(th, eps, et):
    t = np.radians(th); n = np.array([np.cos(t), np.sin(t)]); tt = np.array([-n[1], n[0]])
    U = np.eye(3); U[:2, :2] += eps*np.outer(n, n)+et*np.outer(tt, tt)
    A = np.array([[A0, 0, 0], [A0/2, A0*np.sqrt(3)/2, 0], [0, 0, LZ]])
    return Atoms('C2', cell=(U @ A.T).T, scaled_positions=[[0, 0, .5], [1/3, 1/3, .5]], pbc=True)

def kekule_sc(p, amp):
    """sqrt3 supercell of primitive p with a random in-plane K-point displacement pattern of max amplitude amp."""
    P = np.array([[1, 1, 0], [-1, 2, 0], [0, 0, 1]]); sc = make_supercell(p, P)
    cell = p.cell.array; frac0 = p.get_scaled_positions()
    fr = np.linalg.solve(cell.T, sc.positions.T).T; K = np.array([2/3, 1/3, 0])
    pol = [(rng.normal(size=2)+1j*rng.normal(size=2)) for _ in range(2)]
    u = np.zeros((len(sc), 3))
    for j in range(len(sc)):
        s = int(np.argmin([np.linalg.norm(((fr[j]-frac0[k]+.5) % 1)-.5) for k in range(2)]))
        R = np.round(fr[j]-frac0[s]); u[j, :2] = np.real(pol[s]*np.exp(2j*np.pi*K @ R))
    sc.positions += u*amp/np.abs(u[:, :2]).max()
    return sc

def write(i, at, meta):
    json.dump(dict(cell=at.cell.array.tolist(), positions=at.positions.tolist(), meta=meta), open(f'{OUT}/{i:04d}.json', 'w'))

if __name__ == '__main__':
    i = 190
    for k in range(50):                         # D: biaxial strain, primitive
        th = rng.uniform(0, 30); eps = rng.uniform(0.04, 0.26)
        w = rng.uniform(0, 1); et = (1-w)*et_uniax(th, eps)+w*eps     # from uniaxial-stress contraction to equibiaxial
        write(i, prim(th, eps, et), dict(group='D', theta=th, eps=eps, et=et, w=w)); i += 1
    for k in range(60):                         # E: Kekule-distorted sqrt3 cells
        th = rng.uniform(0, 30); eps = rng.uniform(0.10, 0.28)
        w = rng.choice([0, 0, 0.3, 0.6, 1.0]) if eps < 0.24 else 0
        et = (1-w)*et_uniax(th, eps)+w*eps
        amp = rng.choice([0.03, 0.06, 0.10, 0.15])
        sc = kekule_sc(prim(th, eps, et), amp); sc.positions += rng.normal(0, 0.01, (len(sc), 3))
        write(i, sc, dict(group='E', theta=th, eps=eps, et=et, w=w, kekule=amp)); i += 1
    print('written up to', i-1)
