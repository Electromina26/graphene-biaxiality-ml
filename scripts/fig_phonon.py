"""Fig: phonons of strained graphene (MACE-FT r2_s2, 6x6 supercell, finite displacements).
(a) dispersions along Gamma-M-K-Gamma-K' for: unstrained, ZZ uniaxial eps=0.22, equibiaxial eps=0.12 and 0.145
    (imaginary frequencies plotted as negative);
(b) lowest K-point frequency vs strain for theta = 0 / 30 and lam = 0, 0.9, 1: the Kekule (K-point) mode softens
    to zero before the stress peak only when the load is close to equibiaxial."""
import json, sys, numpy as np, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/user/graphene')
import origin_style as O; import matplotlib.pyplot as plt
import mlip_uniaxial as mu
from phonon_reanalyze import fc, THz2cm
O.apply(8.5); CA = {}
P = '/home/user/graphene/phon/ft__home_user_graphene_ft_ft_omat_r2_s2.model_'
calc = mu.make_calc('ft:/home/user/graphene/ft/ft_omat_r2_s2.model'); a0 = mu.equilibrium_a(calc)
def rec(f, e): return min(json.load(open(P+f+'.json')), key=lambda r: abs(r['eps']-e))
frac0 = [[0, 0, .5], [1/3, 1/3, .5]]
states = [('unstrained', np.array([[a0, 0, 0], [a0/2, a0*np.sqrt(3)/2, 0], [0, 0, 12.]]), frac0, '#000000'),
          ('ZZ uniaxial, ε = 22 %', *[np.array(rec('th0', .22)[k]) for k in ('cell', 'frac')], '#0000FF'),
          ('equibiaxial, ε = 12 %', *[np.array(rec('th0_lam1', .12)[k]) for k in ('cell', 'frac')], '#FF8000'),
          ('equibiaxial, ε = 14.5 %', *[np.array(rec('th0_lam1', .145)[k]) for k in ('cell', 'frac')], '#FF0000')]
nodes = [('Γ', [0, 0, 0]), ('M', [.5, 0, 0]), ('K', [2/3, 1/3, 0]), ('Γ', [0, 0, 0]), ("K'", [1/3, 2/3, 0])]
LS = json.load(open('/home/user/graphene/lam_star.json'))['r2_s2']
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.25, 1]))
for lab, cell, frac, c in states:
    ph = fc(calc, cell, frac); rec_ = np.linalg.inv(cell).T*2*np.pi
    X, F, ticks, x0 = [], [], [0.], 0.
    for (_, q1), (_, q2) in zip(nodes[:-1], nodes[1:]):
        q = np.linspace(q1, q2, 41); ph.run_qpoints(q); f = ph.get_qpoints_dict()['frequencies']*THz2cm
        L = np.linalg.norm((np.array(q2)-q1)@rec_); X.append(x0+np.linspace(0, L, 41)); F.append(f); x0 += L; ticks.append(x0)
    X, F = np.concatenate(X), np.concatenate(F)
    for b in range(F.shape[1]): ax[0].plot(X/ticks[-1], F[:, b], color=c, lw=1.1, label=lab if b == 0 else None)
    CA[f'{lab}_x'] = X/ticks[-1]
    for b in range(F.shape[1]): CA[f'{lab}_branch{b+1}_cm-1'] = F[:, b]
    print(lab, 'min', F.min().round(1), 'K', np.round(F[np.argmin(abs(X-ticks[2]))], 1), flush=True)
    tk = np.array(ticks)/ticks[-1]
O.csvout('fig_phonon_a', CA | {'tick_pos': tk, 'tick_label': [n for n, _ in nodes]})
ax[0].tick_params(axis='x', which='minor', bottom=False, top=False)
for t in tk: ax[0].axvline(t, color='k', lw=0.5)
ax[0].axhline(0, color='k', lw=0.6, ls=':'); ax[0].set_xticks(tk); ax[0].set_xticklabels([n for n, _ in nodes]); ax[0].set_xlim(0, 1)
ax[0].set_ylabel('Frequency (cm$^{-1}$)'); ax[0].set_ylim(-420, 1900)
ax[0].legend(frameon=False, fontsize=6.6, loc='upper center', ncol=2, columnspacing=1.0, handlelength=1.4)
ax[0].annotate('Kekulé mode\nat K → 0', xy=(tk[2], 60), xytext=(tk[2]+0.07, -250), fontsize=7, color='#c0392b',
               arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.8))
O.panel(ax[0], 'a', y=0.80)
CB = {}
for f, lab, c, m in (('th0', 'ZZ, λ = 0', '#0000FF', 'o'), ('th30', 'AC, λ = 0', '#FF0000', 's'), ('th0_lam0.9', 'ZZ, λ = 0.9', '#0000FF', 'o'),
                     ('th30_lam0.9', 'AC, λ = 0.9', '#FF0000', 's'), ('th0_lam1', 'λ = 1', '#008000', '^')):
    d = json.load(open(P+f+'.json')); e = [100*r['eps'] for r in d]; k = [min(r['K'][0], r['Kp'][0]) for r in d]
    op = '0.9' not in f and 'lam1' not in f; CB[f'{f}_strain_pct'] = e; CB[f'{f}_wK_cm-1'] = k
    ax[1].plot(e, k, m+('-' if op else '--' if '0.9' in f else ':'), ms=3.2, lw=1.1, color=c, mfc='w' if op else c, label=lab)
    th, lam = ('30' if 'th30' in f else '0'), (0.9 if '0.9' in f else 1.0 if 'lam1' in f else 0.0)
    em = [r[1] for r in LS[th]['rows'] if abs(r[0]-lam) < 1e-6][0]; ax[1].plot(100*em, 690, 'v', ms=6, color=c, mec='k', mew=0.5); CB[f'{f}_eps_m_pct'] = [100*em]
ax[1].axhline(0, color='k', lw=0.6); O.csvout('fig_phonon_b', CB)
ax[1].set_xlabel('Strain ε$_{nn}$ (%)'); ax[1].set_ylabel('Lowest K/K′ frequency (cm$^{-1}$)')
ax[1].set_xlim(11.5, 29.5); ax[1].set_ylim(-800, 820)
ax[1].text(28.9, 600, '▼ stress peak ε$_m$', fontsize=6.8, ha='right'); ax[1].legend(frameon=False, fontsize=7, loc='lower left')
O.panel(ax[1], 'b')
O.save(fig, 'fig_phonon')
