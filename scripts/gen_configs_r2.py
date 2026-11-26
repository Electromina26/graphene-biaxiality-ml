"""Active-learning round 2 (group F): targeted at the failure-mode crossover (biaxiality lam 0.7-1, near the stress peak),
where DFT frozen-phonon checks showed the round-1 model K mode ~2-4 meV (per 6-atom cell at 0.03 A) too soft.
 F1: MLIP(r1)-relaxed primitive cells from the lam scans (theta 0, 30; lam .75-.95; eps .15-.22) -> sqrt3 cell with random
     in-plane K-pattern polarisation (phase convention exp(iK.r), full position), amplitude .02-.08 A, + 0.01 A rattle.
 F2: same construction on strain states at random theta (0-30), eps .15-.22, transverse strain mixed toward equibiaxial (w .6-1).
The frozen-phonon validation structures (frozen/) are NOT included. ids from 0300."""
import json, glob, numpy as np
from ase import Atoms
from ase.build import make_supercell
import gen_configs_al as al      # reuses prim(), et_uniax()
rng = np.random.default_rng(47)
OUT = '/home/user/graphene/configs'
P = np.array([[1, 1, 0], [-1, 2, 0], [0, 0, 1]]); K = np.array([2/3, 1/3, 0])

def kpattern(p, amp):
    sc = make_supercell(p, P); cell = p.cell.array; frac0 = p.get_scaled_positions()
    fr = np.linalg.solve(cell.T, sc.positions.T).T
    pol = [(rng.normal(size=2)+1j*rng.normal(size=2)) for _ in range(2)]; u = np.zeros((len(sc), 3))
    for j in range(len(sc)):
        s = int(np.argmin([np.linalg.norm(((fr[j]-frac0[k]+.5) % 1)-.5) for k in range(2)]))
        u[j, :2] = np.real(pol[s]*np.exp(2j*np.pi*K @ fr[j]))
    sc.positions += u*amp/np.abs(u[:, :2]).max() + rng.normal(0, 0.01, (len(sc), 3)); return sc

VALID = {(0, 0.19, 0.85), (0, 0.19, 0.9), (0, 0.185, 0.95), (30, 0.17, 0.95), (30, 0.18, 0.9), (30, 0.175, 0.85),
         (0, 0.14, 1.0), (0, 0.16, 1.0)}          # frozen-phonon DFT validation states
i = 300; rows = []
for fn in sorted(glob.glob('/home/user/graphene/phon/ft__home_user_graphene_ft_ft_omat_r1_s2.model_th*_lam0.*.json')):
    for r in json.load(open(fn)):
        held = (round(r['theta']), round(r['eps'], 3), r['lam']) in VALID      # keep validation states out of training
        if 0.75 <= r['lam'] <= 0.95 and 0.15 <= r['eps'] <= 0.22 and not held: rows.append(r)
pick = rng.choice(len(rows), size=min(26, len(rows)), replace=False)
for k in pick:
    r = rows[k]; p = Atoms('C2', cell=r['cell'], scaled_positions=r['frac'], pbc=True); amp = rng.uniform(0.02, 0.08)
    sc = kpattern(p, amp)
    json.dump(dict(cell=sc.cell.array.tolist(), positions=sc.positions.tolist(),
                   meta=dict(group='F', theta=r['theta'], eps=r['eps'], lam=r['lam'], kekule=amp, src='r1-lam-scan')), open(f'{OUT}/{i:04d}.json', 'w')); i += 1
for k in range(14):
    th = rng.uniform(0, 30); eps = rng.uniform(0.15, 0.22); w = rng.uniform(0.6, 1.0)
    et = (1-w)*al.et_uniax(th, eps)+w*eps; amp = rng.uniform(0.02, 0.08)
    sc = kpattern(al.prim(th, eps, et), amp)
    json.dump(dict(cell=sc.cell.array.tolist(), positions=sc.positions.tolist(),
                   meta=dict(group='F', theta=th, eps=eps, et=et, w=w, kekule=amp, src='strain')), open(f'{OUT}/{i:04d}.json', 'w')); i += 1
print('written 0300 ..', i-1, 'from', len(rows), 'candidate scan states')
