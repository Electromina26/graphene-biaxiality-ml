"""Single-point QE (PBE, SSSP PAW, 50/400 Ry, MV 0.01 Ry) on an ASE Atoms object.
Returns energy (eV), forces (eV/A), stress (eV/A^3, ASE sign: tension positive)."""
import numpy as np, subprocess, os, re
LOW = bool(os.environ.get('QE_LOWMEM'))   # large cells: wavefunctions to disk, smaller Davidson/mixing workspaces (same numerics)
PW = os.environ.get('PW', '/home/user/graphene_dft/dft/qe/bin/pw.x')
PSEUDO = '/home/user/graphene_dft/dft/pseudo'
RY = 13.605693122994; BOHR = 0.529177210903
def kmesh(cell, dens=24*2.4669):
    """k-points so that k_i * |a_i| ~ dens (24 for the primitive cell)."""
    return [max(1, int(round(dens/np.linalg.norm(cell[i])))) for i in range(2)]
def run(atoms, wd, label, ec=50.0, mpi=1):
    os.makedirs(wd, exist_ok=True)
    k = kmesh(atoms.cell.array)
    inp = f"""&control
 calculation='scf', prefix='{label}', pseudo_dir='{PSEUDO}', outdir='{wd}/tmp_{label}',
 tstress=.true., tprnfor=.true., disk_io='{"high" if LOW else "none"}'
/
&system
 ibrav=0, nat={len(atoms)}, ntyp=1, ecutwfc={ec}, ecutrho={8*ec}, occupations='smearing', smearing='mv', degauss=0.01
/
&electrons
 conv_thr={1e-9*len(atoms)/2:.2e}, mixing_beta=0.4{", mixing_ndim=4, diago_david_ndim=2" if LOW else ""}
/
ATOMIC_SPECIES
C 12.011 C.UPF
CELL_PARAMETERS angstrom
""" + ''.join(f" {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n" for v in atoms.cell.array) + \
"ATOMIC_POSITIONS angstrom\n" + ''.join(f"C {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n" for p in atoms.positions) + \
f"K_POINTS automatic\n{k[0]} {k[1]} 1 0 0 0\n"
    fi = f'{wd}/{label}.in'; fo = f'{wd}/{label}.out'; open(fi, 'w').write(inp)
    cmd = ([ 'mpirun', '--allow-run-as-root', '-np', str(mpi)] if mpi > 1 else []) + [PW, '-in', fi]
    with open(fo, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS='1'))
    txt = open(fo).read()
    subprocess.run(['rm', '-rf', f'{wd}/tmp_{label}'])
    if 'JOB DONE' not in txt or 'convergence NOT achieved' in txt: raise RuntimeError('pw failed ' + fo)
    E = float(re.findall(r'!\s+total energy\s+=\s+(-?\d+\.\d+)', txt)[-1])*RY
    fl = txt.split('Forces acting on atoms')[1].split('\n')
    F = np.array([[float(x) for x in l.split('=')[1].split()[:3]] for l in fl if 'atom' in l and 'force =' in l][:len(atoms)])*RY/BOHR
    blk = txt.split('total   stress')[-1].split('\n')[1:4]
    S_kbar = np.array([[float(x) for x in l.split()[3:6]] for l in blk])   # kbar, QE: positive = compressive? (pressure sign)
    stress = -S_kbar*0.1/160.21766208     # GPa -> eV/A^3 ; sign flipped to tension-positive
    return E, F, stress, k
