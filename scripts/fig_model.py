"""Fig 1 (model & methods schematic):
(a) honeycomb lattice: primitive cell (a1, a2), 4-atom rectangular cell, 6x2 MD supercell, ZZ / AC axes, loading angle theta
    and the stress state sigma_nn, sigma_tt = lam*sigma_nn;
(b) Brillouin zone (Gamma, M, K, K') and the sqrt3 x sqrt3 reduced zone onto which K folds (Kekule order parameter);
(c) the DFT frozen-phonon Kekule distortion (equibiaxial 16 %, A = 0.09 A): displacement arrows, bonds coloured by d - d0;
(d) a finite-temperature MD snapshot (ZZ, 900 K, uniaxial stress, near the stress peak): bonds coloured by length, atoms by z."""
import numpy as np, matplotlib, glob
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.collections import LineCollection
from ase.io import read
from ase.neighborlist import neighbor_list
import sys; sys.path.insert(0, '/home/user/graphene'); import origin_style as O; O.apply(8.5)
A0 = 2.4669; b = A0*np.sqrt(3)
fig = plt.figure(figsize=(7.4, 7.0)); gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.12)
axa, axb, axc, axd = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
def arrow(ax, p, q, c='k', lw=1.2, ms=8, **k): ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=ms, color=c, lw=lw, **k))
# (a) lattice
base = np.array([[0, 0], [A0/2, b/6], [A0/2, b/2], [0, 2*b/3]])
P = np.array([[i*A0+p[0], j*b+p[1]] for i in range(-1, 8) for j in range(-1, 4) for p in base])
P = P[(P[:, 0] > -1.3) & (P[:, 0] < 16.0) & (P[:, 1] > -0.6) & (P[:, 1] < 2*b+1.0)]
i, j = np.where(np.triu(np.linalg.norm(P[:, None]-P[None], axis=-1) < 1.6, 1))
axa.add_collection(LineCollection([[P[a], P[c]] for a, c in zip(i, j)], colors='#9aa5b1', lw=0.9))
axa.scatter(*P.T, s=9, c='#34495e', zorder=3)
axa.add_patch(Polygon([[0, 0], [6*A0, 0], [6*A0, 2*b], [0, 2*b]], fill=True, fc='#f5b04122', ec='#e67e22', lw=1.4, ls='--'))
axa.add_patch(Polygon([[0, 0], [A0, 0], [A0, b], [0, b]], fill=False, ec='#c0392b', lw=1.4))
o = np.array([A0/2, b/2]); a1 = np.array([A0, 0]); a2 = np.array([A0/2, b/2])
arrow(axa, o, o+a1, c='#1f5aa6'); arrow(axa, o, o+a2, c='#1f5aa6')
axa.text(*(o+a1+[0.1, -0.55]), r'$\mathbf{a}_1$', color='#1f5aa6'); axa.text(*(o+a2+[0.15, 0.05]), r'$\mathbf{a}_2$', color='#1f5aa6')
axa.text(-1.2, -1.5, '— 4-atom rectangular cell', color='#c0392b', fontsize=7.5)
axa.text(-1.2, -2.4, '-- MD supercell 6×2 (48 C atoms)', color='#e67e22', fontsize=7.5)
c0 = np.array([10.0, -6.6]); arrow(axa, c0, c0+[2.6, 0], lw=1.4); arrow(axa, c0, c0+[0, 2.6], lw=1.4)
axa.text(*(c0+[2.8, -0.25]), 'x: ZZ\n(θ=0°)', fontsize=7.5, va='center'); axa.text(*(c0+[-1.6, 2.9]), 'y: AC (θ=30°)', fontsize=7.5)
th = np.radians(18); n = np.array([np.cos(th), np.sin(th)]); t = np.array([-np.sin(th), np.cos(th)]); c1 = np.array([3.6, -6.2])
arrow(axa, c1, c1+2.6*n, c='#c0392b', lw=2.0, ms=11); arrow(axa, c1, c1-2.6*n, c='#c0392b', lw=2.0, ms=11)
arrow(axa, c1, c1+1.5*t, c='#7a4fa3', lw=1.6); arrow(axa, c1, c1-1.5*t, c='#7a4fa3', lw=1.6)
axa.text(*(c1+2.7*n+[0, -0.1]), r'$\sigma_{nn}$', color='#c0392b', fontsize=9)
axa.text(*(c1+1.6*t+[-1.9, 0]), r'$\sigma_{tt}=\lambda\sigma_{nn}$', color='#7a4fa3', fontsize=8.5)
axa.plot([c1[0], c1[0]+2.2], [c1[1], c1[1]], 'k:', lw=0.8); axa.text(*(c1+[1.9, 0.12]), 'θ', fontsize=9)
axa.set_xlim(-1.5, 16.2); axa.set_ylim(-9.3, 10.0); axa.set_aspect('equal'); axa.axis('off')
axa.set_title('(a) lattice, cells and loading', loc='left', fontsize=9, fontweight='bold')
# (b) Brillouin zone
Kr = 4*np.pi/(3*A0); hexK = np.array([[Kr*np.cos(np.radians(30+60*k)), Kr*np.sin(np.radians(30+60*k))] for k in range(6)])
axb.add_patch(Polygon(hexK, fill=True, fc='#1f5aa611', ec='#1f5aa6', lw=1.4))
Kr3 = Kr/np.sqrt(3); hex3 = np.array([[Kr3*np.cos(np.radians(60*k)), Kr3*np.sin(np.radians(60*k))] for k in range(6)])
axb.add_patch(Polygon(hex3, fill=True, fc='#e67e2222', ec='#e67e22', lw=1.2, ls='--'))
M = np.array([0, Kr*np.sqrt(3)/2])
for p, s, c in ((np.zeros(2), 'Γ', 'k'), (hexK[0], 'K', '#c0392b'), (hexK[1], "K'", '#c0392b'), (M*np.array([1, 1])@np.array([[np.cos(np.pi/3), np.sin(np.pi/3)], [-np.sin(np.pi/3), np.cos(np.pi/3)]]).T*0+[Kr*np.sqrt(3)/2*np.cos(np.radians(0)), 0], 'M', '#1f5aa6')):
    axb.plot(*p, 'o', color=c, ms=4.5, zorder=4); axb.text(*(p+([0.07, 0.07] if s != 'Γ' else [-0.25, -0.22])), s, color=c, fontsize=10)
axb.plot([0, hexK[0][0], Kr*np.sqrt(3)/2, 0], [0, hexK[0][1], 0, 0], color='#555', lw=0.9)
arrow(axb, hexK[0]*0.93, hexK[0]*0.08, c='#c0392b', lw=1.0, ms=7, connectionstyle='arc3,rad=0.35')
axb.text(-1.75, -2.25, 'dashed: reduced zone of the √3×√3 cell:\nK, K′ fold onto Γ → Kekulé mode\ncaptured in 6-atom DFT cells', fontsize=7.2, color='#e67e22')
axb.text(0.3, 1.5, 'phonons: 6×6 supercell,\nfinite displacements', fontsize=7.2)
axb.set_xlim(-1.8, 1.9); axb.set_ylim(-2.75, 1.95); axb.set_aspect('equal'); axb.axis('off')
axb.set_title('(b) Brillouin zone and K-point folding', loc='left', fontsize=9, fontweight='bold')
# (c) Kekule distortion from the DFT frozen-phonon structures
s0 = read('/home/user/graphene/frozenwork/th0_e0.16_lam1_A0.0.in', format='espresso-in')
s9 = read('/home/user/graphene/frozenwork/th0_e0.16_lam1_A0.09.in', format='espresso-in')
u = s9.positions-s0.positions; u -= u.mean(0)
r0 = s0.repeat((3, 3, 1)); r9 = s9.repeat((3, 3, 1)); U = np.tile(u, (9, 1))
i, j, D = neighbor_list('ijd', r9, 1.9); i0, j0, D0 = neighbor_list('ijd', r0, 1.9); k = i < j
d0 = np.median(D0); seg = [[r9.positions[a, :2], r9.positions[a, :2]+(r9.get_distance(a, c, mic=True, vector=True))[:2]] for a, c in zip(i[k], j[k])]
lc = LineCollection(seg, cmap='coolwarm', lw=2.6); lc.set_array(D[k]-d0); lc.set_clim(-0.06, 0.06); axc.add_collection(lc)
axc.scatter(*r9.positions[:, :2].T, s=10, c='#2c3e50', zorder=3)
for p, d in zip(r9.positions[:, :2], U[:, :2]): arrow(axc, p, p+d*14, c='#27ae60', lw=1.1, ms=6)
cb = fig.colorbar(lc, ax=axc, fraction=0.04, pad=0.01); cb.set_label('d − d$_0$ (Å)', fontsize=7.5); cb.ax.tick_params(labelsize=6.5)
axc.text(0.0, -0.03, 'equibiaxial ε = 16 %, A = 0.09 Å (arrows ×14)\nDFT: ΔE(0.06 Å) = −25.7 meV/cell → unstable',
         transform=axc.transAxes, fontsize=6.8, va='top')
axc.set_aspect('equal'); axc.autoscale(); axc.axis('off')
axc.set_title('(c) Kekulé (K-point) soft mode, DFT', loc='left', fontsize=9, fontweight='bold')
# (d) MD snapshot
tr = read('/home/user/graphene/md/ft_omat_r2_s2_th0_T900_lam0_s1_v2_traj.xyz', ':'); sn = min(tr, key=lambda a: abs(a.info['eps']-0.185))
sr = sn.repeat((1, 2, 1)); i, j, D, V = neighbor_list('ijdD', sr, 2.1); k = i < j
seg = [[sr.positions[a, :2], sr.positions[a, :2]+v[:2]] for a, v in zip(i[k], V[k])]
lc = LineCollection(seg, cmap='viridis', lw=2.0); lc.set_array(D[k]); lc.set_clim(1.40, 1.85); axd.add_collection(lc)
z = sr.positions[:, 2]-sr.positions[:, 2].mean()
sc = axd.scatter(*sr.positions[:, :2].T, s=14, c=z, cmap='PuOr', vmin=-0.4, vmax=0.4, zorder=3, edgecolors='k', linewidths=0.2)
cb = fig.colorbar(lc, ax=axd, fraction=0.04, pad=0.01); cb.set_label('bond length (Å)', fontsize=7.5); cb.ax.tick_params(labelsize=6.5)
c = sr.cell.array; axd.add_patch(Polygon([[0, 0], c[0, :2], c[0, :2]+c[1, :2], c[1, :2]], fill=False, ec='#e67e22', lw=1, ls='--'))
axd.text(0.0, -0.03, f"ZZ uniaxial, 900 K, ε = {100*sn.info['eps']:.1f} %, σ = {sn.info['snn']:.1f} N/m\natoms coloured by height z (±0.4 Å); ×2 in y",
         transform=axd.transAxes, fontsize=6.8, va='top')
axd.set_aspect('equal'); axd.autoscale(); axd.axis('off')
axd.set_title('(d) MD snapshot at 900 K', loc='left', fontsize=9, fontweight='bold')
O.save(fig, 'fig_model', bbox_inches='tight')
