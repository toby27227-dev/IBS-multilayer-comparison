"""构建多组学联合特征矩阵：属水平 CLR + NMR 代谢组 log1p。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'IBS'))
import IBS

gen = pd.read_csv(ROOT / 'IBS/gmrepo/abundance_genus_clinical.csv', index_col=0)
nmr = pd.read_csv(ROOT / 'IBS/cellomics/nmr.csv', index_col=0)
idx = gen.index.intersection(nmr.index)

G = gen.loc[idx].values.astype(float)
N = nmr.loc[idx].values.astype(float)
pseudo = IBS.resolve_pseudo(G, IBS.PSEUDOCOUNT)
Gc = IBS.clr_transform(G, pseudo, IBS.CLR_METHOD)

Gc = pd.DataFrame(Gc, index=idx, columns=[f'g_{c}' for c in gen.columns])
Nl = pd.DataFrame(np.log1p(np.clip(N, 0, None)), index=idx,
                  columns=[f'm_{c}' for c in nmr.columns])
C = pd.concat([Gc, Nl], axis=1)
C.to_csv(Path(__file__).resolve().parent / 'combined_genus_nmr.csv')
print('combined:', C.shape, '| runs:', len(idx))
