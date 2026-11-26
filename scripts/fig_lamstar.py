"""Fig: failure-mode crossover map (Origin style). (a) Delta = eps_K - eps_m vs lam for theta = 0, 15, 30 (r2_s2);
(b) lam*(theta) for both committee models with the DFT frozen-phonon brackets."""
import json, numpy as np, sys
sys.path.insert(0, '/home/user/graphene'); import origin_style as O
import matplotlib.pyplot as plt
O.apply(9)
d = json.load(open('/home/user/graphene/lam_star.json'))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
cols = {'0': ('#0000FF', 'o'), '15': ('#008000', '^'), '30': ('#FF0000', 's')}
C = {}
for th, (c, m) in cols.items():
    rows = d['r2_s2'][th]['rows']; l = [r[0] for r in rows]; D = [100*r[3] for r in rows]
    ax[0].plot(l, D, m+'-', color=c, mfc='white', label=f'θ = {th}°' + (' (ZZ)' if th == '0' else ' (AC)' if th == '30' else ''))
    C[f'lam_th{th}'] = l; C[f'Delta_pct_th{th}'] = D
O.csvout('fig_lamstar_a', C)
ax[0].axhline(0, color='k', lw=0.8, ls='--'); ax[0].axhspan(0, 12, color='#0000FF', alpha=0.05, lw=0); ax[0].axhspan(-12, 0, color='#FF0000', alpha=0.05, lw=0)
ax[0].set_xlabel('Stress biaxiality λ = σ$_{tt}$/σ$_{nn}$'); ax[0].set_ylabel('Δ = ε$_K$ − ε$_m$ (%)')
ax[0].set_xlim(0.45, 1.03); ax[0].set_ylim(-11.5, 10); ax[0].legend(loc='lower left')
ax[0].text(0.50, 8.6, 'Elastic limit first', fontsize=8, color='#00008B', va='top', style='italic')
ax[0].text(0.50, -1.2, 'Kekulé softening first', fontsize=8, color='#8B0000', va='top', style='italic')
O.panel(ax[0], 'a', x=0.96, ha='right')
C = {}
for s, m, c in (('r2_s1', 's', '#000000'), ('r2_s2', 'o', '#0000FF')):
    th = sorted(int(t) for t, v in d[s].items() if v['lam_star']); ls = [d[s][str(t)]['lam_star'] for t in th]
    ax[1].plot(th, ls, m+'-', color=c, mfc='white' if s == 'r2_s1' else c, ms=6 if s == 'r2_s1' else 5, label=f'MACE-FT, model {s[-1]}')
    C[f'theta_{s}'] = th; C[f'lamstar_{s}'] = ls
O.csvout('fig_lamstar_b', C | {'theta_DFT': [0, 30], 'DFT_low': [0.85, 0.90], 'DFT_high': [0.90, 0.95]})
for t, lo, hi in ((0, 0.85, 0.90), (30, 0.90, 0.95)):
    ax[1].errorbar(t, (lo+hi)/2, yerr=(hi-lo)/2, fmt='none', ecolor='#FF8000', elinewidth=7, alpha=0.55, capsize=0)
ax[1].plot([], [], 's', color='#FF8000', alpha=0.6, ms=8, label='DFT frozen-phonon bracket')
ax[1].set_xlabel('Loading angle θ (deg)'); ax[1].set_ylabel('Crossover biaxiality λ*'); ax[1].set_xticks(range(0, 31, 10))
ax[1].set_xlim(-3, 33); ax[1].set_ylim(0.83, 0.96); ax[1].legend(loc='lower right')
ax[1].set_xticklabels(['0 (ZZ)', '10', '20', '30 (AC)'])
O.panel(ax[1], 'b')
O.save(fig, 'fig_lamstar')
