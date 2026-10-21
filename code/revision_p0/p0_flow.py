"""P0 revision: sample-flow figure (fig15), layer-specific availability."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path.cwd() / 'IBS' / 'figs'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})

fig, axes = plt.subplots(
    2, 1, figsize=(7.2, 8.4),
    gridspec_kw={'height_ratios': [1.08, 0.92]},
)
for ax in axes:
    ax.set_xlim(0, 10)
    ax.axis('off')


def box(ax, x, y, w, h, text, fc='#eef3fb', ec='#4a6fa5', fs=9, weight='normal'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05',
                                linewidth=1.1, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, linespacing=1.35, fontweight=weight)


def arrow(ax, xy1, xy2, color='#4a6fa5'):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle='-|>', mutation_scale=11,
                                 linewidth=1.1, color=color,
                                 shrinkA=2, shrinkB=2))


# Panel A: Cell 2020
ax = axes[0]
ax.set_ylim(0, 6.4)
ax.set_title('A  Cell 2020 discovery cohort', fontsize=11, fontweight='bold', loc='left', pad=4)
box(ax, 0.7, 5.25, 8.6, 0.95,
    'Cell 2020 cohort\n464 labeled samples / 75 subjects\n'
    'IBS-D 169 | IBS-C 142 | Healthy 153',
    fc='#e8eef8', ec='#365f91', fs=8.8, weight='bold')
arrow(ax, (5.0, 5.25), (5.0, 4.72))

ax.add_patch(FancyBboxPatch((0.2, 2.28), 9.6, 2.32,
                            boxstyle='round,pad=0.04', linewidth=1.0,
                            edgecolor='#8aa4c8', facecolor='#fbfcfe'))
ax.text(5.0, 4.38, 'Layer-specific availability (all layers: 75 subjects)',
        ha='center', va='center', fontsize=8.4, color='#27476e',
        fontweight='bold')

cards = [
    (0.45, 3.55, 'Genus / species\n457 / 75  → Bristol 441 / 67'),
    (5.10, 3.55, 'KO / KEGG\n459 / 75  → Bristol 437 / 67'),
    (0.45, 2.95, 'NMR metabolome\n445 / 75  → Bristol 424 / 67'),
    (5.10, 2.95, 'Bile acids\n447 / 75  → Bristol 426 / 67'),
    (0.45, 2.35, 'Tryptophan\n274 / 75  → Bristol 259 / 67'),
    (5.10, 2.35, 'NMR + genus\n440 / 75  → Bristol 424 / 67'),
]
for x, y, text in cards:
    box(ax, x, y, 4.45, 0.52, text, fc='#ffffff', ec='#a9bfdc', fs=8.2)

ax.plot([5.0, 5.0], [2.28, 2.02], color='#4a6fa5', lw=1.1)
ax.plot([2.35, 7.65], [2.02, 2.02], color='#4a6fa5', lw=1.1)
arrow(ax, (2.35, 2.02), (2.35, 1.86))
arrow(ax, (7.65, 2.02), (7.65, 1.86))
box(ax, 0.2, 0.62, 4.70, 1.18,
    'Bristol regression tasks\nNMR 424 / 67; combined 424 / 67\n'
    'common complete subset 255 / 67',
    fc='#eef8ef', ec='#4f8a5b', fs=8.1)
box(ax, 5.10, 0.62, 4.70, 1.18,
    'Classification tasks\nlayer-specific N; all 75 subjects\n'
    'IBS-SSS: 250 / 51',
    fc='#eef8ef', ec='#4f8a5b', fs=8.1)
ax.text(5.0, 0.18,
        'Missing from 464 labeled samples: genus/species 7; KO/KEGG 5; NMR 19; Bristol 23.\n'
        'Layer intersections are not identical (genus and KO/KEGG overlap = 453 samples).',
        ha='center', va='center', fontsize=7.2, color='#555', linespacing=1.3)

# Panel B: Jacobs 2023
ax = axes[1]
ax.set_ylim(0, 6.2)
ax.set_title('B  Jacobs 2023 external validation', fontsize=11, fontweight='bold', loc='left', pad=4)
box(ax, 1.1, 5.05, 8.8, 0.95,
    'Jacobs 2023 cohort\n368 samples\n'
    'Control 139 (BH=Normal) | C 89 | D 84 | M 37 | Unspecified 19',
    fc='#e8eef8', ec='#365f91', fs=8.5, weight='bold')
box(ax, 1.1, 3.55, 4.8, 0.95,
    'Primary external evaluation\n210 samples\nC 89 | D 84 | M 37',
    fc='#eef3fb', ec='#4a6fa5', fs=8.6)
box(ax, 6.2, 3.62, 3.6, 0.82,
    'Excluded from primary\nControl 139 (BH=Normal)\nUnspecified 19',
    fc='#fdf1e7', ec='#c47a3d', fs=7.6)
box(ax, 1.1, 2.18, 4.8, 0.88,
    'D vs C sensitivity\n173 samples\nC 89 | D 84',
    fc='#eef3fb', ec='#4a6fa5', fs=8.6)
box(ax, 1.1, 0.62, 8.8, 1.05,
    'Zero-shot transfer: Cell 2020 (NMR n=424) → Jacobs 2023\n'
    'No target labels used for fitting or feature selection',
    fc='#eef8ef', ec='#4f8a5b', fs=8.2)
arrow(ax, (3.5, 5.05), (3.5, 4.50))
arrow(ax, (3.5, 3.55), (3.5, 3.06))
arrow(ax, (8.0, 5.05), (8.0, 4.44))
ax.text(5.0, 0.18,
        'Primary endpoint D vs C+M (n=210); sensitivity D vs C (n=173).',
        ha='center', va='center', fontsize=7.4, color='#555')

fig.tight_layout(h_pad=1.2)
fig.savefig(OUT / 'fig15_sample_flow.png', dpi=300, bbox_inches='tight')
print(OUT / 'fig15_sample_flow.png')
