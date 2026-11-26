"""Fig: finite-temperature tensile MD (48-atom cells, MACE-FT r2, 0.05 /ps), Origin style.
(a) stress-strain curves, ZZ and AC uniaxial stress, 10-1200 K (seed 1, 5-block running mean);
(b) peak 2D stress vs T (seed mean +- sd) for lam = 0, 0.9, 1 with experiments at room temperature;
(c) failure strain (strain at peak stress) vs T."""
import json, sys, numpy as np
sys.path.insert(0, '/home/user/graphene'); import origin_style as O
import matplotlib.pyplot as plt
O.apply(8.5)
D = '/home/user/graphene/md/ft_omat_r2_s2_'
S = json.load(open('/home/user/graphene/md_seedavg.json'))
fig, ax = plt.subplots(1, 3, figsize=(7.4, 3.2), constrained_layout=True)
Ts = [10, 300, 600, 900, 1200]; TC = dict(zip(Ts, ['#000000', '#0000FF', '#008000', '#FF8000', '#FF0000']))
C = {}
for th, ls in ((0, '-'), (30, '--')):
    for T in Ts:
        h = json.load(open(f'{D}th{th}_T{T}_lam0_s1_v2.json'))['hist']
        e = np.array([x['eps'] for x in h]); s = np.convolve([x['snn'] for x in h], np.ones(5)/5, 'valid')
        ax[0].plot(100*e[2:-2], s, ls, color=TC[T], lw=1.2, label=f'{T} K' if th == 0 else None)
        C[f'{"ZZ" if th == 0 else "AC"}_{T}K_strain_pct'] = 100*e[2:-2]; C[f'{"ZZ" if th == 0 else "AC"}_{T}K_stress_Nm'] = s
O.csvout('fig_md_a', C)
ax[0].set_xlabel('Strain ε (%)'); ax[0].set_ylabel('Stress σ$_{nn}$ (N/m)'); ax[0].set_xlim(10, 28); ax[0].set_ylim(20, 40)
l1 = ax[0].legend(loc='lower center', bbox_to_anchor=(0.42, 0.0), ncol=2, fontsize=7, handlelength=1.4, columnspacing=0.8)
ax[0].text(0.04, 0.84, 'λ = 0\nsolid: ZZ\ndashed: AC', transform=ax[0].transAxes, fontsize=7, va='top')
O.panel(ax[0], 'a')
sty = [(0, 0, 'ZZ, λ = 0', '#0000FF', 'o', 'w'), (30, 0, 'AC, λ = 0', '#FF0000', 's', 'w'), (0, 0.9, 'ZZ, λ = 0.9', '#0000FF', 'o', None),
       (30, 0.9, 'AC, λ = 0.9', '#FF0000', 's', None), (0, 1, 'λ = 1 (equibiaxial)', '#008000', '^', None)]
Cb = {'T_K': Ts}
for th, lam, lab, c, m, mf in sty:
    r = sorted([x for x in S if x['theta'] == th and abs(x['lam']-lam) < 1e-6], key=lambda x: x['T'])
    T = [x['T'] for x in r]; sd = lambda k: [0 if np.isnan(x[k]) else x[k] for x in r]
    kw = dict(fmt=m+('-' if lam == 0 else '--' if lam == 0.9 else ':'), color=c, mfc=mf or c, ms=4.5, lw=1.1, capsize=2.2)
    ax[1].errorbar(T, [x['P'] for x in r], yerr=sd('P_sd'), label=lab, **kw)
    ax[2].errorbar(T, [x['eps'] for x in r], yerr=sd('eps_sd'), **kw)
    k = lab.replace(' ', '').replace('(equibiaxial)', '')
    Cb |= {f'{k}_T': T, f'{k}_Ppeak': [x['P'] for x in r], f'{k}_Psd': sd('P_sd'), f'{k}_epsf': [x['eps'] for x in r], f'{k}_epssd': sd('eps_sd')}
O.csvout('fig_md_bc', Cb | {'exp_T': [300, 300], 'exp_P_Nm': [42, 110*0.335], 'exp_err': [4, None], 'exp_ref': ['Lee2008', 'Jaddi2024']})
ax[1].errorbar([270], [42], yerr=[4], fmt='*', ms=10, color='k', capsize=2.5, label='Exp., nanoindentation [Lee 2008]')
ax[1].plot([335], [110*0.335], 'h', ms=7, color='#800080', zorder=5, label='Exp., on-chip tensile [Jaddi 2024]')
ax[1].set_xlabel('Temperature (K)'); ax[1].set_ylabel('Peak stress (N/m)'); ax[1].set_ylim(25, 47)
ax[2].set_xlabel('Temperature (K)'); ax[2].set_ylabel('Failure strain ε$_f$ (%)'); ax[2].set_ylim(11, 27.5)
for a, s in zip(ax[1:], 'bc'): a.set_xticks([0, 300, 600, 900, 1200]); a.set_xlim(-80, 1280); O.panel(a, s, x=0.96, ha='right')
h, l = ax[1].get_legend_handles_labels(); want = ['ZZ, λ = 0', 'λ = 1 (equibiaxial)', 'AC, λ = 0', 'Exp., nanoindentation [Lee 2008]', 'ZZ, λ = 0.9', 'Exp., on-chip tensile [Jaddi 2024]', 'AC, λ = 0.9']; h = [h[l.index(w)] for w in want]; l = want; fig.legend(h, l, loc='outside lower center', ncol=4, fontsize=7, columnspacing=1.2)
O.save(fig, 'fig_md')
