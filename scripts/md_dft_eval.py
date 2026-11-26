"""Evaluate the r2 committee on the DFT-checked MD snapshots (md_dft_check.json): force RMSE/MAE, energy error per atom
(after removing the mean offset over snapshots), 2D stress error (N/m). usage: python md_dft_eval.py"""
import sys, json, numpy as np, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/user/graphene')
import mlip_uniaxial as mu
from ase import Atoms
R = json.load(open('/home/user/graphene/md_dft_check.json')); LZ = 12.0; c2 = LZ*16.0217663
for s in ('s1', 's2'):
    calc = mu.make_calc(f'ft:/home/user/graphene/ft/ft_omat_r2_{s}.model'); dE = []
    for k, r in R.items():
        at = Atoms(f"C{r['natoms']}", cell=r['cell'], positions=r['positions'], pbc=True); at.calc = calc
        F = np.array(r['F_dft']); Fm = at.get_forces(); Sm = at.get_stress(voigt=False); S = np.array(r['S_dft'])
        r[f'E_{s}'] = float(at.get_potential_energy()); r[f'Frmse_{s}'] = float(np.sqrt(np.mean((Fm-F)**2)))
        r[f'dS_{s}'] = [float((Sm[i, i]-S[i, i])*c2) for i in (0, 1)]; r[f'S2d_dft'] = [float(S[i, i]*c2) for i in (0, 1)]
        dE.append((r[f'E_{s}']-r['E_dft'])/r['natoms'])
    off = np.mean(dE); sh = json.load(open('/home/user/graphene/eref.json'))['shift']
    for (k, r), d in zip(R.items(), dE):
        r[f'dEat_{s}'] = 1000*(d-off)
        print(s, f"{k:28s} eps {r['eps']:.3f} T {r['T_inst']:5.0f}  Frmse {1000*r['Frmse_'+s]:5.0f} meV/A  dE {1000*(d-off):+5.2f} meV/at"
              f"  dS(nn,tt) {r['dS_'+s][0]:+5.2f} {r['dS_'+s][1]:+5.2f} N/m (DFT {r['S2d_dft'][0]:.1f}, {r['S2d_dft'][1]:.1f})")
    print(s, f'absolute energy error (ML - DFT, after the training reference shift): mean {1000*(off+sh):+.2f}  std {1000*np.std(dE):.2f} meV/atom;  force RMSE (all) {1000*np.sqrt(np.mean([r["Frmse_"+s]**2 for r in R.values()])):.0f} meV/A')
json.dump(R, open('/home/user/graphene/md_dft_eval.json', 'w'))
