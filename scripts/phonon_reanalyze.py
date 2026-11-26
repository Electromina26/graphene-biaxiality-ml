"""Recompute phonons from stored relaxed geometries (phon/<tag>_th*.json) with corrected
high-symmetry points and in-plane / out-of-plane mode classification.
For the 60-degree basis a1=(a,0), a2=(a/2, a*sqrt3/2): K = (2/3,1/3), K' = (1/3,2/3), M = (1/2,0),(0,1/2),(1/2,1/2).
Out-of-plane (flexural) branches are decoupled from in-plane ones in a flat sheet; under uniaxial stress the
ZA branch for q perpendicular to the load is marginal (omega ~ kappa q^2), so the instability criterion uses
in-plane modes and, separately, out-of-plane modes with a tolerance.
usage: python phonon_reanalyze.py <model> <json files...>"""
import sys, json, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/graphene')
import mlip_uniaxial as mu
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
THz2cm = 33.35641
HS = dict(K=[2/3, 1/3, 0], Kp=[1/3, 2/3, 0], M1=[.5, 0, 0], M2=[0, .5, 0], M3=[.5, .5, 0])

def fc(calc, cell, frac, N=6):
    u = PhonopyAtoms(symbols=['C', 'C'], cell=cell, scaled_positions=frac)
    ph = Phonopy(u, supercell_matrix=np.diag([N, N, 1]), primitive_matrix=np.eye(3))   # keep OUR basis (default 'auto' re-standardises the cell)
    ph.generate_displacements(distance=0.01, is_plusminus=True)
    F = []
    for sc in ph.supercells_with_displacements:
        at = Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True); at.calc = calc
        F.append(at.get_forces())
    ph.forces = F; ph.produce_force_constants(); ph.symmetrize_force_constants(); return ph

def modes(ph, qs):
    ph.run_qpoints(qs, with_eigenvectors=True); d = ph.get_qpoints_dict()
    f = np.array(d['frequencies'])*THz2cm; ev = np.array(d['eigenvectors'])       # (nq, 6, 6) columns = modes
    zw = np.sum(np.abs(ev[:, [2, 5], :])**2, axis=1)                               # out-of-plane weight per mode
    return f, zw

def analyze(ph, mesh=36):
    q = np.array([[i/mesh, j/mesh, 0] for i in range(mesh) for j in range(mesh)])
    f, zw = modes(ph, q)
    qn = np.linalg.norm(((q+.5) % 1)-.5, axis=1); far = qn > 1.5/mesh
    ip = np.where((zw < .5) & far[:, None], f, np.inf); op = np.where((zw >= .5) & far[:, None], f, np.inf)
    i1 = np.unravel_index(np.argmin(ip), ip.shape); i2 = np.unravel_index(np.argmin(op), op.shape)
    out = dict(min_inplane=float(ip[i1]), q_inplane=q[i1[0]].tolist(), min_outplane=float(op[i2]), q_outplane=q[i2[0]].tolist())
    for k, qq in HS.items():
        fk, zk = modes(ph, [qq]); out[k] = fk[0].tolist(); out[k+'_z'] = zk[0].tolist()
    return out

if __name__ == '__main__':
    calc = mu.make_calc(sys.argv[1])
    for fn in sys.argv[2:]:
        R = json.load(open(fn))
        for r in R:
            r.update(analyze(fc(calc, np.array(r['cell']), np.array(r['frac']))))
            print(fn[-12:], r['eps'], 'IP %.1f' % r['min_inplane'], np.round(r['q_inplane'], 3), 'OP %.1f' % r['min_outplane'],
                  'K', np.round(r['K'][:2], 1), "K'", np.round(r['Kp'][:2], 1), flush=True)
        json.dump(R, open(fn.replace('.json', '_v2.json'), 'w'), indent=1, default=float)
