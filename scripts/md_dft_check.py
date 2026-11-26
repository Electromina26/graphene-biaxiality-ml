"""DFT spot checks of the fine-tuned MACE models on finite-temperature MD snapshots (48-atom strained sheets).
Selects snapshots from md/ft_omat_r2_s2_<run>_traj.xyz (closest to a target fraction of the run's peak strain), runs QE single
points (same protocol as training, qe_sp.py) and evaluates both r2 committee models in float64.
Resumable: results appended to md_dft_check.json.  usage: python md_dft_check.py "<tag:frac ...>"
run = e.g. th0_T900_lam0_s1_v2_3x2 ; frac = eps/eps_peak (e.g. 0.95)."""
import sys, os, json, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/graphene')
import qe_sp                                        # MLIP evaluation done separately (md_dft_eval.py) to save memory
from ase.io import read
OUT = '/home/user/graphene/md_dft_check.json'
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def peak(run):                                       # strain at the 5-block smoothed stress maximum of that run
    h = json.load(open(f'/home/user/graphene/md/ft_omat_r2_s2_{run}.json'))['hist']
    e = np.array([x['eps'] for x in h]); s = np.convolve([x['snn'] for x in h], np.ones(5)/5, 'valid'); return e[2:-2][np.argmax(s)]
for job in sys.argv[1].split():
    tag, frac = job.split(':'); key = f'{tag}_f{frac}'
    if key in res: continue
    tr = read(f'/home/user/graphene/md/ft_omat_r2_s2_{tag}_traj.xyz', ':')
    tgt = float(frac)*peak(tag); i = int(np.argmin([abs(a.info['eps']-tgt) for a in tr])); at = tr[i]
    E, F, S, k = qe_sp.run(at, '/home/user/graphene/mdcheck', key)
    r = dict(tag=tag, frac=float(frac), eps=at.info['eps'], T_inst=at.info['T'], natoms=len(at), kmesh=k,
             cell=at.cell.array.tolist(), positions=at.positions.tolist(), E_dft=E, F_dft=F.tolist(), S_dft=S.tolist())
    res[key] = r; json.dump(res, open(OUT, 'w'))
    print(key, f"eps {r['eps']:.3f} T {r['T_inst']:.0f}", flush=True)
