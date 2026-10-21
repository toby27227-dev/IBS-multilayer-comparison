# -*- coding: utf-8 -*-
"""点标注贪心放置：以渲染 bbox 为准避免文字-文字、文字-散点重叠。"""
import numpy as np


def place_labels(ax, items, offsets=None, fontsize=7.5, marker_s=42, pad=1.0,
                 verbose=False):
    offsets = offsets or {}
    fig = ax.figure
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    texts = {}
    for t, x, y in items:
        texts[t] = ax.annotate(t, (x, y), textcoords='offset points',
                               xytext=offsets.get(t, (8, 4)),
                               fontsize=fontsize, zorder=4,
                               bbox=dict(facecolor='white', edgecolor='none',
                                         pad=0.4))
    pts = {t: ax.transData.transform((x, y)) for t, x, y in items}
    rad = ren.points_to_pixels(np.sqrt(marker_s) / 2 + pad)
    ax_bb = ax.get_window_extent(ren)

    def bbox(t):
        return texts[t].get_window_extent(ren)

    def rect_ov(a, b):
        return not (a.x1 <= b.x0 or b.x1 <= a.x0 or
                    a.y1 <= b.y0 or b.y1 <= a.y0)

    def point_gap(t, k):
        bx = bbox(t)
        px, py = pts[k]
        return np.hypot(min(max(px, bx.x0), bx.x1) - px,
                        min(max(py, bx.y0), bx.y1) - py)

    cands = [(dx, dy) for dx in range(-72, 33, 8)
             for dy in range(-24, 25, 6)]
    cands.sort(key=lambda c: abs(c[0]) + abs(c[1]))
    order = sorted(texts, key=lambda k: -sum(
        np.hypot(pts[k][0] - q[0], pts[k][1] - q[1]) < 40
        for q in pts.values()))
    for _ in range(3):
        for k in order:
            for dx, dy in cands:
                texts[k].set_position((dx, dy))
                bx = bbox(k)
                if not (ax_bb.x0 - 6 <= bx.x0 and bx.x1 <= ax_bb.x1 + 6
                        and bx.y0 >= ax_bb.y0 - 6 and bx.y1 <= ax_bb.y1 + 6):
                    continue
                if any(rect_ov(bx, bbox(kk)) for kk in texts if kk != k):
                    continue
                if any(point_gap(k, taxon) <
                       (rad * 0.7 if taxon == k else rad) for taxon in pts):
                    continue
                break
    if verbose:
        resid = [f'{a}|{b}' for i, a in enumerate(texts)
                 for b in list(texts)[i + 1:] if rect_ov(bbox(a), bbox(b))]
        resid += [f'{k}~{t}' for k in texts for t in pts
                  if point_gap(k, t) < (rad * 0.7 if t == k else rad)]
        print('label overlaps:', resid if resid else 'none')
    return texts
