"""Finite-temperature tensile MD with the fine-tuned MACE potential.
Orthorhombic graphene supercell (nx x ny rectangular 4-atom cells; x = zigzag, y = armchair). nx multiple of 3 so the
K point is commensurate (the Kekule mode can develop). Load along x (theta = 0, ZZ) or y (theta = 30, AC).
Langevin thermostat; the load axis is strained at a constant engineering rate; the transverse axis follows a
Berendsen-type control of the biaxiality: sigma_tt -> lam * sigma_nn (lam = 0: uniaxial stress).
Failure = first time the time-averaged load stress drops 15 % below its running maximum, or a C-C distance > 2.0 A
(nearest-neighbour bonds only). Output md/<tag>.json with (eps, sigma_nn, sigma_tt, T, max bond) and failure strain.
usage: python md_strain.py <model> <theta 0|30> <T K> <lam> [seed] [rate per ps] [nx ny]"""
import sys, os, json, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/graphene')
import mlip_uniaxial as mu
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.neighborlist import neighbor_list

model, th, T, lam = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1
rate = float(sys.argv[6]) if len(sys.argv) > 6 else 0.05          # strain per ps (5e10 s^-1)
nx, ny = (int(sys.argv[7]), int(sys.argv[8])) if len(sys.argv) > 8 else (6, 2)
A0, LZ, DT = 2.4669, 12.0, 1.0                                       # A, A, fs
EPS0 = 0.10                                                          # start from a pre-strained state (affine), then ramp
rng = np.random.default_rng(seed)

def sheet():
    a, b = A0, A0*np.sqrt(3)
    base = np.array([[0, 0], [a/2, b/6], [a/2, b/2], [0, 2*b/3]])    # 4-atom rectangular cell (zigzag along x)
    pos = [[i*a+p[0], j*b+p[1], LZ/2] for i in range(nx) for j in range(ny) for p in base]
    return Atoms(f'C{len(pos)}', positions=pos, cell=[nx*a, ny*b, LZ], pbc=True)

at = sheet(); n_ax, t_ax = (0, 1) if th == 0 else (1, 0)
L0 = at.cell.lengths()[:2].copy()
nu = 0.17                                                            # initial transverse guess for the affine start
s = np.ones(2); s[n_ax] = 1+EPS0; s[t_ax] = 1+EPS0*(lam-nu)/(1-lam*nu)                     # linear-elastic transverse strain for sigma_tt = lam sigma_nn
at.set_cell(np.diag([L0[0]*s[0], L0[1]*s[1], LZ]), scale_atoms=True)
at.positions[:, 2] += rng.normal(0, 0.02, len(at))                   # break the flat symmetry (allow rippling)
from mace.calculators import MACECalculator
at.calc = MACECalculator(model_paths=model.split(':', 1)[1], default_dtype='float32', device='cpu')   # float32 for MD speed
MaxwellBoltzmannDistribution(at, temperature_K=T, rng=rng); Stationary(at)
dyn = Langevin(at, DT*units.fs, temperature_K=T, friction=0.01/units.fs, rng=rng)

def stress2d():                                                       # N/m, positive = tension
    st = at.get_stress(voigt=False, include_ideal_gas=True)     # ASE: positive in tension (checked against DFT protocol)
    return st[n_ax, n_ax]*LZ*16.0217663, st[t_ax, t_ax]*LZ*16.0217663

NB = 20; tauP = 0.02                                                 # barostat every NB steps; gain (1/(N/m))
de = rate*DT*1e-3*NB                                                 # strain increment per block
tag = f"{os.path.basename(model.split(':')[-1]).replace('.model', '')}_th{th}_T{T:g}_lam{lam:g}_s{seed}_v2" + (f'_{nx}x{ny}' if (nx, ny) != (6, 2) else '') + (f'_r{rate:g}' if rate != 0.05 else '')
os.makedirs('/home/user/graphene/md', exist_ok=True); out = f'/home/user/graphene/md/{tag}.json'
hist = []; eps = EPS0; smax = -1e9; fail = None
i0, j0 = neighbor_list('ij', at, 1.8)                                 # bonded pairs fixed from the start
keep = i0 < j0; i0, j0 = i0[keep], j0[keep]
NEQ = int(os.environ.get('MD_NEQ', 100))                             # equilibration blocks (100 x 20 fs = 2 ps) at EPS0, no ramp
for blk in range(NEQ):
    acc = np.zeros(2)
    for k in range(NB): dyn.run(1); acc += stress2d()
    snn, stt = acc/NB; c = at.cell.array.copy(); f = np.ones(3)
    f[t_ax] = 1 + np.clip(-tauP*1e-2*(stt-lam*snn), -2e-3, 2e-3); at.set_cell(np.diag(np.diag(c)*f), scale_atoms=True)
print(f'equilibrated: T {at.get_temperature():.0f} snn {snn:.2f} stt {stt:.2f}', flush=True)
traj = f'/home/user/graphene/md/{tag}_traj.xyz'
if os.path.exists(traj): os.remove(traj)
for blk in range(int((0.40-EPS0)/de)+1):
    acc = np.zeros(2)
    for k in range(NB):
        dyn.run(1); acc += stress2d()
    snn, stt = acc/NB
    c = at.cell.array.copy(); f = np.ones(3)
    f[n_ax] = (1+eps+de)/(1+eps)                                     # ramp the load axis
    f[t_ax] = 1 + np.clip(-tauP*1e-2*(stt-lam*snn), -2e-3, 2e-3)    # transverse control toward stt = lam*snn
    at.set_cell(np.diag(np.diag(c)*f), scale_atoms=True); eps += de
    dmax = float(at.get_distances(i0, j0, mic=True).max()) if False else float(np.max(np.linalg.norm(
        at.get_distances(i0, j0, mic=True, vector=True), axis=-1)))
    Tinst = at.get_temperature()
    hist.append(dict(eps=eps, snn=snn, stt=stt, T=Tinst, dmax=dmax))
    if blk % 10 == 0 and fail is None:                                # snapshots for DFT spot checks
        snap = at.copy(); snap.info.update(eps=eps, snn=snn, T=Tinst); snap.write(traj, append=True)
    if len(hist) >= 5:
        sm = np.mean([h['snn'] for h in hist[-5:]]); smax = max(smax, sm)
        if fail is None and (sm < 0.85*smax or dmax > 2.0):
            fail = dict(eps=hist[-5]['eps'], smax=smax, reason='stress drop' if sm < 0.85*smax else 'bond>2A')
    if blk % 25 == 0 or fail:
        json.dump(dict(theta=th, T=T, lam=lam, seed=seed, rate=rate, nx=nx, ny=ny, natoms=len(at), hist=hist, fail=fail),
                  open(out, 'w'))
        print(f'eps {eps:.4f} snn {snn:6.2f} stt {stt:6.2f} T {Tinst:5.0f} dmax {dmax:.3f}', flush=True)
    if fail and eps > fail['eps']+0.02: break
at.write(f'/home/user/graphene/md/{tag}_final.xyz')
print('FAIL', fail, flush=True)
