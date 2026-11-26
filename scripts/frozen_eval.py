"""Re-evaluate an MLIP on the exact frozen-phonon structures already computed with DFT (frozenwork/*.in).
Reports relative energies E(A)-E(0) in meV per supercell for DFT and the model.
usage: python frozen_eval.py <model-spec> [tag-glob]"""
import sys, glob, os, re, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/graphene')
import mlip_uniaxial as mu
from ase.io import read
calc = mu.make_calc(sys.argv[1]); pat = sys.argv[2] if len(sys.argv) > 2 else '*'
out = {}
for fn in sorted(glob.glob(f'/home/user/graphene/frozen/{pat}.json')):
    tag = os.path.basename(fn)[:-5]; R = json.load(open(fn)); rows = []
    for p in sorted(R['pts'], key=lambda p: p['A']):
        at = read(f'/home/user/graphene/frozenwork/{tag}_A{p["A"]}.in', format='espresso-in'); at.calc = calc
        rows.append([p['A'], p['E_dft'], at.get_potential_energy()])
    e0d, e0m = rows[0][1], rows[0][2]
    out[tag] = [dict(A=a, dE_dft=1000*(d-e0d), dE_mlip=1000*(m-e0m)) for a, d, m in rows]
    print(tag, ' '.join(f'A={a}: DFT {1000*(d-e0d):+.1f} / ML {1000*(m-e0m):+.1f}' for a, d, m in rows[1:]), flush=True)
json.dump(out, open('/home/user/graphene/frozen_eval_' + re.sub(r'\W+', '_', sys.argv[1])[-40:] + '.json', 'w'), indent=1)
