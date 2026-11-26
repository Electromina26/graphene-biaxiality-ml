"""Assemble DFT data into extxyz for MACE fine-tuning.
train.xyz: groups B, C (85%) + primitive uniaxial-stress paths theta = 0, 10, 30 (paper-2 DFT)
test.xyz : groups B, C (15%) + full theta = 20 path (held-out loading direction)"""
import json, glob, numpy as np
from ase import Atoms
from ase.io import write
LZ = 12.0; NM = 16.0217663
rng = np.random.default_rng(7)
SH = json.load(open('/home/user/graphene/eref.json'))['shift']   # per-atom offset QE -> foundation-model energy scale
def mk(pos, cell, E, F, S, tag):
    at = Atoms(f'C{len(pos)}', positions=pos, cell=cell, pbc=True)
    at.info['REF_energy'] = float(E) - SH*len(pos); at.arrays['REF_forces'] = np.array(F, float)
    at.info['REF_stress'] = np.array(S, float).reshape(3, 3); at.info['config_type'] = tag
    return at
train, test = [], []
for f in sorted(glob.glob('/home/user/graphene/dft/*.json')):
    d = json.load(open(f))
    if np.abs(d['forces']).max() > 30: continue            # drop 3 extreme short-bond configs (d_min < 1.13 A)
    at = mk(d['positions'], d['cell'], d['energy'], d['forces'], d['stress'], {'F': 'Fk'}.get(d['meta']['group'], d['meta']['group']))   # 'F'/'T' would be parsed as booleans by extxyz
    (test if rng.random() < 0.15 else train).append(at)
for th in (0, 10, 20, 30):
    for r in json.load(open(f'/home/user/graphene_dft/dft/run/th{th}/results.json')):
        if r['eps'] > 0.285: continue                     # exclude post-rupture states
        t = np.radians(th); n = np.array([np.cos(t), np.sin(t), 0]); tt = np.array([-n[1], n[0], 0])
        s = (r['snn']*np.outer(n, n) + r['stt']*np.outer(tt, tt) + r['snt']*(np.outer(n, tt)+np.outer(tt, n)))/(LZ*NM)
        at = mk(r['pos'], r['cell'], r['E'], np.zeros((2, 3)), s, f'path{th}')
        (test if th == 20 else train).append(at)
write('/home/user/graphene/train.xyz', train, format='extxyz')
write('/home/user/graphene/test.xyz', test, format='extxyz')
print('train', len(train), 'test', len(test))
