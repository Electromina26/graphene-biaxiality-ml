"""Fig: active learning of the Kekule (K-point) energy landscape, validated on 8 held-out DFT frozen-phonon states (Origin style).
(a), (b) frozen-phonon energy dE(A) for an unstable (equibiaxial 16 %) and a stable near-crossover state (ZZ, lam 0.85, eps 19 %):
DFT vs zero-shot MACE-OMAT-0 and fine-tuning rounds 0, 1, 2;
(c) parity of dE (A = 0.03 and 0.06 A) for all states, with MAE and number of wrong stability verdicts (sign of dE at 0.03 A)."""
import json, sys, numpy as np
sys.path.insert(0, '/home/user/graphene'); import origin_style as O; import matplotlib.pyplot as plt
O.apply(8.5)
P = '/home/user/graphene/frozen_eval_'
M = [('mace_medium_omat_0', 'Zero-shot MACE-OMAT-0', '#FF8000', 'v'), ('t_home_user_graphene_ft_ft_omat_s2_model', 'FT round 0', '#808080', 's'),
     ('ome_user_graphene_ft_ft_omat_r1_s2_model', 'FT round 1 (AL)', '#0000FF', '^'), ('ome_user_graphene_ft_ft_omat_r2_s2_model', 'FT round 2 (AL)', '#FF0000', 'o')]
D = {k: json.load(open(P+k+'.json')) for k, *_ in M}
states = list(D[M[-1][0]])
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.85), constrained_layout=True)
def name(s):
    th, e, l = s.split('_'); return f"{'ZZ' if th == 'th0' else 'AC'}, λ = {l[3:]}, ε = {100*float(e[1:]):.1f} %"
C = {}
for a, st, s in ((ax[0], 'th0_e0.16_lam1', 'a'), (ax[1], 'th0_e0.19_lam0.85', 'b')):
    r = D[M[-1][0]][st]; A = [p['A'] for p in r]
    a.plot(A, [p['dE_dft'] for p in r], 'k*-', ms=10, lw=2, label='DFT (PBE)', zorder=5)
    C[f'{st}_A'] = A; C[f'{st}_DFT'] = [p['dE_dft'] for p in r]
    for k, lab, c, m in M:
        a.plot(A, [p['dE_mlip'] for p in D[k][st]], m+'--', color=c, mfc=c if lab.endswith('2 (AL)') else 'w', lw=1.1, label=lab); C[f'{st}_{lab}'] = [p['dE_mlip'] for p in D[k][st]]
    a.axhline(0, color='k', lw=0.6, ls=':'); a.set_xlabel('Kekulé amplitude A (Å)'); a.set_xlim(-0.004, 0.094); a.set_xticks([0, 0.03, 0.06, 0.09])
    a.text(0.04, 0.86, ('equibiaxial, ε = 16 %\nDFT: unstable' if s == 'a' else 'ZZ, λ = 0.85, ε = 19 %\nDFT: stable'), transform=a.transAxes, fontsize=7.2, va='top')
    O.panel(a, s)
O.csvout('fig_al_ab', C)
ax[0].set_ylabel('ΔE (meV per 6-atom cell)'); ax[0].set_ylim(-100, 300); ax[1].set_ylim(-40, 150)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='outside lower center', ncol=5, fontsize=7)
CC = {'model': [], 'MAE_meV': [], 'wrong_verdicts_of_8': []}
for k, lab, c, m in M:
    x = [p['dE_dft'] for s in states for p in D[k][s] if p['A'] in (0.03, 0.06)]; y = [p['dE_mlip'] for s in states for p in D[k][s] if p['A'] in (0.03, 0.06)]
    mae = np.mean(np.abs(np.array(x)-y))
    wrong = sum(np.sign([p for p in D[k][s] if p['A'] == 0.03][0]['dE_dft']) != np.sign([p for p in D[k][s] if p['A'] == 0.03][0]['dE_mlip']) for s in states)
    ax[2].plot(x, y, m, color=c, mfc='w' if k != M[-1][0] else c, ms=4.5)
    ax[2].text(0.97, 0.95-0.075*(3-M.index((k, lab, c, m))), f'MAE {mae:.1f} meV, {wrong}/8 wrong', color=c, transform=ax[2].transAxes, ha='right', va='top', fontsize=6.8)
    CC[f'{lab}_dE_DFT'] = x; CC[f'{lab}_dE_ML'] = y; CC['model'].append(lab); CC['MAE_meV'].append(mae); CC['wrong_verdicts_of_8'].append(int(wrong)); print(lab, round(mae, 1), wrong)
O.csvout('fig_al_c', CC)
ax[2].plot([-60, 200], [-60, 200], 'k-', lw=0.8); ax[2].set_xlim(-30, 40); ax[2].set_ylim(-60, 125)
ax[2].set_xlabel('ΔE$_{DFT}$ (meV)'); ax[2].set_ylabel('ΔE$_{ML}$ (meV)')
ax[2].axhline(0, color='k', lw=0.5, ls=':'); ax[2].axvline(0, color='k', lw=0.5, ls=':'); O.panel(ax[2], 'c')
ax[2].text(0.96, 0.05, '8 held-out states\nA = 0.03, 0.06 Å', transform=ax[2].transAxes, fontsize=6.8, ha='right')
O.save(fig, 'fig_al')
