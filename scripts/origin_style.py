"""OriginPro-like publication style for all figures of Paper 3.
Arial-metric sans font (Liberation Sans), inward major+minor ticks on all four sides, 1.1 pt frame,
no grid, bold panel letters; export vector PDF + EPS and 600 dpi PNG/TIFF; CSV of the plotted data
(fig/data/<name>_<panel>.csv) so every panel can be re-plotted directly in OriginPro."""
import logging; logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import os, csv, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
FIG = '/home/user/graphene/fig'
ORIGIN = ['#0000FF', '#FF0000', '#008000', '#FF00FF', '#000080', '#808000', '#800080', '#008080', '#FF8000']   # Origin default palette

def apply(size=9):
    plt.rcParams.update({
        'font.family': ['Liberation Sans', 'DejaVu Sans'], 'font.size': size, 'mathtext.fontset': 'custom',
        'mathtext.rm': 'Liberation Sans', 'mathtext.it': 'Liberation Sans:italic', 'mathtext.bf': 'Liberation Sans:bold',
        'axes.linewidth': 1.1, 'axes.labelsize': size+0.5, 'axes.titlesize': size+0.5, 'axes.labelpad': 3,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'xtick.major.size': 4.5, 'ytick.major.size': 4.5, 'xtick.minor.size': 2.5, 'ytick.minor.size': 2.5,
        'xtick.major.width': 1.0, 'ytick.major.width': 1.0, 'xtick.minor.width': 0.8, 'ytick.minor.width': 0.8,
        'xtick.labelsize': size, 'ytick.labelsize': size, 'xtick.major.pad': 4, 'ytick.major.pad': 3,
        'lines.linewidth': 1.4, 'lines.markersize': 5, 'lines.markeredgewidth': 0.9, 'errorbar.capsize': 2.5,
        'legend.frameon': False, 'legend.fontsize': size-1, 'legend.handlelength': 1.8, 'legend.borderaxespad': 0.4,
        'axes.grid': False, 'savefig.dpi': 600, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def panel(ax, s, x=0.03, y=0.96, ha='left', **k):
    ax.text(x, y, f'({s})', transform=ax.transAxes, fontweight='bold', fontsize=plt.rcParams['font.size']+1.5, va='top', ha=ha, **k)

def minor(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator(2)); ax.yaxis.set_minor_locator(AutoMinorLocator(2))

def save(fig, name, **k):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIG}/{name}.{ext}', dpi=600, **k)
    fig.savefig(f'{FIG}/{name}.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'}, **k)

def csvout(name, cols):
    """cols: dict header -> 1D sequence (ragged allowed, padded with blanks)."""
    os.makedirs(f'{FIG}/data', exist_ok=True)
    keys = list(cols); n = max(len(v) for v in cols.values())
    with open(f'{FIG}/data/{name}.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(keys)
        for i in range(n):
            w.writerow([('' if i >= len(cols[c]) or cols[c][i] is None or (isinstance(cols[c][i], float) and np.isnan(cols[c][i]))
                         else f'{cols[c][i]:.6g}' if isinstance(cols[c][i], (float, np.floating)) else cols[c][i]) for c in keys])
