"""Fig: foundation-model benchmark and fine-tuning / active learning (Origin style).
(a) uniaxial stress-strain (ZZ, AC): DFT vs zero-shot MACE-OMAT-0 vs fine-tuned MACE (r2) and two other foundation models;
(b) stress RMSE vs DFT on the DFT-relaxed path structures (log scale), all models incl. the three fine-tuning rounds;
(c) ideal strains eps_m (ZZ, AC) with the DFT values;
(d) Young modulus vs ZZ-AC anisotropy of the ideal strain."""
import json, sys, numpy as np
sys.path.insert(0, '/home/user/graphene_dft'); sys.path.insert(0, '/home/user/graphene')
import origin_style as O; import matplotlib.pyplot as plt
from analyze import analyze
O.apply(8.5)
B = {r['model']: r for r in json.load(open('/home/user/graphene/bench_summary.json'))}
DR = json.load(open('/home/user/graphene/dft_ref_same_protocol.json'))
FT = lambda n: f'ft:/home/user/graphene/ft/ft_omat_{n}.model'
order = ['mace-small', 'mace-medium', 'mace-medium-0b3', 'mace-medium-mpa-0', 'chgnet', '7net-0', '7net-mf-ompa', 'mattersim-1M',
         'mattersim-5M', 'mace-medium-omat-0', FT('s1'), FT('r1_s1'), FT('r2_s1'), FT('r2_s2')]
label = {'mace-small': 'MACE-MP-0 S', 'mace-medium': 'MACE-MP-0 M', 'mace-medium-0b3': 'MACE-MP-0b3', 'mace-medium-mpa-0': 'MACE-MPA-0',
         'mace-medium-omat-0': 'MACE-OMAT-0', 'chgnet': 'CHGNet', '7net-0': 'SevenNet-0', '7net-mf-ompa': 'SevenNet-MF-ompa',
         'mattersim-1M': 'MatterSim-1M', 'mattersim-5M': 'MatterSim-5M', FT('s1'): 'FT round 0', FT('r1_s1'): 'FT round 1 (AL)',
         FT('r2_s1'): 'FT round 2 (AL), m1', FT('r2_s2'): 'FT round 2 (AL), m2'}
order = [m for m in order if m in B]; isft = lambda m: m.startswith('ft:')
fig, ax = plt.subplots(2, 2, figsize=(7.4, 6.4), constrained_layout=True); ax = ax.ravel()
# (a)
CA = {}
for th, ls in [(0, '-'), (30, '--')]:
    D = analyze(th); m = D['eps'] <= 0.28; tag = 'ZZ' if th == 0 else 'AC'
    ax[0].plot(D['eps'][m]*100, D['Pn'][m], 'k'+ls, lw=2.4, label='DFT (PBE)' if th == 0 else None)
    CA[f'DFT_{tag}_strain'] = D['eps'][m]*100; CA[f'DFT_{tag}_P'] = D['Pn'][m]
    for mod, c in [('mace-medium', '#808080'), ('mattersim-5M', '#008000'), ('mace-medium-omat-0', '#FF8000'), (FT('r2_s2'), '#FF0000')]:
        fn = {FT('r2_s2'): 'ft-omat-r2-s2'}.get(mod, mod)
        R = json.load(open(f'/home/user/graphene/bench/{fn}.json'))['path'][str(th)]
        ax[0].plot([r['eps']*100 for r in R], [r['P'] for r in R], ls, color=c, lw=1.3, label=(label[mod] if not isft(mod) else 'MACE-FT (this work)') if th == 0 else None)
        CA[f'{label[mod]}_{tag}_strain'] = [r['eps']*100 for r in R]; CA[f'{label[mod]}_{tag}_P'] = [r['P'] for r in R]
O.csvout('fig_bench_a', CA)
ax[0].set_xlabel('Strain ε (%)'); ax[0].set_ylabel('Nominal stress P (N/m)'); ax[0].set_xlim(0, 30); ax[0].set_ylim(0, 45)
ax[0].legend(loc='lower right', fontsize=7); ax[0].text(0.04, 0.80, 'solid: ZZ\ndashed: AC', transform=ax[0].transAxes, fontsize=7.5, va='top')
O.panel(ax[0], 'a')
# (b) stress RMSE
y = np.arange(len(order)); r = [B[m]['stress_rmse_all'] for m in order]
ax[1].barh(y, r, color=['#FF0000' if isft(m) else '#FF8000' if m == 'mace-medium-omat-0' else '#7F7F7F' for m in order], height=0.65, ec='k', lw=0.6)
for i, v in enumerate(r): ax[1].text(v*1.08, i, f'{v:.2f}', va='center', fontsize=6.8)
ax[1].set_yticks(y); ax[1].set_yticklabels([label[m] for m in order], fontsize=7); ax[1].tick_params(axis='y', which='minor', left=False, right=False)
ax[1].set_xscale('log'); ax[1].set_xlim(0.04, 12); ax[1].set_xlabel('Stress RMSE vs DFT (N/m)'); ax[1].invert_yaxis()
O.csvout('fig_bench_b', {'model': [label[m] for m in order], 'stress_RMSE_Nm': r})
O.panel(ax[1], 'b', x=0.96, ha='right')
# (c) ideal strains
x = np.arange(len(order)); w = 0.4
def em(mod, th): v = B[mod]['theta'][th]['eps_m']*100; return np.nan if v >= 29.9 else v
zz = [em(m, '0') for m in order]; ac = [em(m, '30') for m in order]
ax[2].bar(x-w/2, zz, w, color='#0000FF', ec='k', lw=0.5, label='ZZ'); ax[2].bar(x+w/2, ac, w, color='#FF0000', ec='k', lw=0.5, label='AC')
for i, v in enumerate(zz):
    if np.isnan(v): ax[2].bar(i-w/2, 30, w, color='none', ec='#0000FF', lw=0.8, hatch='////'); ax[2].text(i-w/2, 30.6, '>30', ha='center', fontsize=6, color='#0000FF', rotation=90)
ax[2].axhline(DR['0']['eps_m']*100, color='#0000FF', ls='--', lw=1.2, label='DFT ZZ'); ax[2].axhline(DR['30']['eps_m']*100, color='#FF0000', ls='--', lw=1.2, label='DFT AC')
ax[2].set_xticks(x); ax[2].set_xticklabels([label[m] for m in order], rotation=60, ha='right', fontsize=6.8)
ax[2].tick_params(axis='x', which='minor', bottom=False, top=False); ax[2].set_xlim(-0.7, len(order)-0.3)
ax[2].set_ylabel('Ideal strain ε$_m$ (%)'); ax[2].set_ylim(0, 37); ax[2].legend(loc='upper left', ncol=4, fontsize=7, columnspacing=1.0)
O.csvout('fig_bench_c', {'model': [label[m] for m in order], 'eps_m_ZZ_pct': zz, 'eps_m_AC_pct': ac,
                          'DFT_ZZ': [DR['0']['eps_m']*100], 'DFT_AC': [DR['30']['eps_m']*100]})
O.panel(ax[2], 'c', x=0.96, ha='right')
# (d) Y vs anisotropy
CD = {'model': [], 'Y_Nm': [], 'aniso_pct': [], 'capped': []}
for m in order:
    Y = B[m]['theta']['0']['Y']; cap = B[m]['theta']['0']['eps_m'] >= 0.299
    a = (min(B[m]['theta']['0']['eps_m'], 0.30)-B[m]['theta']['30']['eps_m'])*100
    if isft(m): ax[3].plot(Y, a, 'D', ms=6, color='#FF0000', mec='k', mew=0.5)
    else:
        ax[3].plot(Y, a, '^' if cap else 'o', ms=6, mfc='w', mew=1.1, color='#FF8000' if m == 'mace-medium-omat-0' else '#404040')
        ax[3].annotate(label[m], (Y, a), fontsize=6.2, xytext={'7net-mf-ompa': (4, -9), 'mattersim-5M': (-10, -10), 'mace-medium-omat-0': (4, 3)}.get(m, (4, 2)), textcoords='offset points')
    CD['model'].append(label[m]); CD['Y_Nm'].append(Y); CD['aniso_pct'].append(a); CD['capped'].append(int(cap))
O.csvout('fig_bench_d', CD | {'DFT_Y': [DR['0']['Y']], 'DFT_aniso': [(DR['0']['eps_m']-DR['30']['eps_m'])*100]})
ax[3].plot(DR['0']['Y'], (DR['0']['eps_m']-DR['30']['eps_m'])*100, '*', ms=15, color='k', zorder=5)
ax[3].axvline(DR['0']['Y'], color='k', lw=0.7, ls=':'); ax[3].axhline((DR['0']['eps_m']-DR['30']['eps_m'])*100, color='k', lw=0.7, ls=':')
ax[3].plot([], [], '*', ms=11, color='k', label='DFT (PBE)'); ax[3].plot([], [], 'D', ms=6, color='#FF0000', mec='k', mew=0.5, label='MACE-FT rounds 0–2')
ax[3].plot([], [], 'o', mfc='w', color='#404040', label='Foundation models'); ax[3].plot([], [], '^', mfc='w', color='#404040', label='lower bound (ε$_m^{ZZ}$ > 30 %)')
ax[3].set_xlabel('Young modulus Y (N/m)'); ax[3].set_ylabel('ε$_m^{ZZ}$ − ε$_m^{AC}$ (%)'); ax[3].set_xlim(215, 352); ax[3].set_ylim(-8, 15)
ax[3].legend(loc='lower right', fontsize=6.8); O.panel(ax[3], 'd')
O.save(fig, 'fig_bench')
