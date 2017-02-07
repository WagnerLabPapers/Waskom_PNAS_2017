import os.path as op
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
from seaborn.external.husl import husl_to_rgb
from PIL import Image
from cStringIO import StringIO

import moss
import lyman


def set_style():
    """Consistent style for plots."""
    sns.set(style="ticks", context="paper", font_scale=.9,
            rc={"xtick.major.size": 3, "ytick.major.size": 3,
                "xtick.major.width": 1, "ytick.major.width": 1,
                "xtick.major.pad": 3.5, "ytick.major.pad": 3.5,
                "axes.linewidth": 1, "lines.linewidth": 1})


def savefig(f, fname):

    fstem = op.join("figures", op.basename(fname.strip(".py")))

    # Save to PDF
    f.savefig(fstem + ".pdf", dpi=150)

    # Save to TIFF
    # Uses a trip through PIL -- native saving in matplotlib was producing
    # bad-looking .tiff files. I have no idea why; it used to work.
    buffer = StringIO()
    f.savefig(buffer, format="png", dpi=300)
    buffer.seek(0)
    im = Image.open(buffer)
    im.save(fstem + ".tiff")


def points_to_lines(ax, w=.8, **kws):
    """Replace the central tendency glyph from a pointplot with a line."""
    for col in ax.collections:
        for (x, y), fc in zip(col.get_offsets(), col.get_facecolors()):
            ax.plot([x - w / 2, x + w / 2], [y, y], color=fc, **kws)
        col.remove()


def get_colormap(exp, as_cmap=True):
    """Get experiment-specific diverging colormaps."""
    lums = np.linspace(50, 99, 128)
    sats = np.linspace(80, 20, 128)
    assert exp in ["dots", "sticks", "rest"]
    h1 = 240 if exp == "dots" else 160
    h2 = 20
    lut = ([husl_to_rgb(h1, s, l) for s, l in zip(sats, lums)] +
           [husl_to_rgb(h2, s, l) for s, l in zip(sats, lums)][::-1])
    if as_cmap:
        return mpl.colors.ListedColormap(lut)
    return lut


def get_ifs_view(subj, hemi):
    """Return mlab.view parameters to center IFS in the window."""
    views = dict(pc07=dict(lh=(170, 80, 130, [0, 45, 20]),
                           rh=(10, 80, 130, [0, 50, 20])),
                 pc08=dict(lh=(170, 85, 150, [0, 50, 20]),
                           rh=(10, 80, 130, [0, 55, 20])),
                 pc11=dict(lh=(170, 80, 140, [0, 55, 30]),
                           rh=(10, 80, 130, [0, 50, 35])),
                 pc12=dict(lh=(170, 80, 155, [0, 50, 30]),
                           rh=(10, 80, 130, [0, 50, 35])),
                 pc13=dict(lh=(170, 80, 140, [0, 50, 20]),
                           rh=(10, 80, 125, [0, 60, 25])),
                 pc14=dict(lh=(170, 80, 140, [0, 50, 35]),
                           rh=(10, 80, 135, [0, 50, 35])),
                 pc15=dict(lh=(170, 80, 135, [0, 70, 30]),
                           rh=(10, 80, 145, [0, 55, 25])),
                 pc16=dict(lh=(170, 80, 145, [0, 60, 20]),
                           rh=(10, 80, 145, [0, 55, 17])),
                 pc17=dict(lh=(170, 75, 145, [0, 60, 23]),
                           rh=(10, 75, 135, [0, 50, 23])),
                 pc18=dict(lh=(170, 80, 170, [0, 55, 23]),
                           rh=(10, 80, 170, [0, 65, 23])),
                 pc19=dict(lh=(170, 80, 130, [0, 55, 25]),
                           rh=(10, 80, 130, [0, 55, 25])),
                 pc22=dict(lh=(170, 80, 130, [0, 40, 25]),
                           rh=(10, 80, 120, [0, 45, 25])),
                 pc23=dict(lh=(170, 80, 150, [0, 55, 15]),
                           rh=(10, 80, 150, [0, 55, 15])),
                 pc24=dict(lh=(170, 80, 150, [0, 60, 35]),
                           rh=(10, 80, 140, [0, 55, 33])),
                 ti01=dict(lh=(170, 80, 135, [0, 60, 22]),
                           rh=(10, 80, 130, [0, 55, 15])),
                 ti03=dict(lh=(170, 80, 135, [0, 50, 20]),
                           rh=(10, 80, 145, [0, 55, 25])),
                 ti05=dict(lh=(170, 80, 145, [0, 50, 35]),
                           rh=(10, 80, 145, [0, 50, 30])),
                 ti06=dict(lh=(170, 80, 140, [0, 50, 35]),
                           rh=(10, 80, 130, [0, 47, 33])),
                 ti07=dict(lh=(170, 80, 150, [0, 68, 27]),
                           rh=(10, 80, 155, [0, 65, 20])),
                 ti08=dict(lh=(170, 80, 150, [0, 60, 25]),
                           rh=(10, 80, 145, [0, 60, 20])),
                )

    return views[subj][hemi]


def get_subject_order(exp):

    subjects = lyman.determine_subjects([exp + "_subjects"])
    accs = pd.Series(index=subjects, dtype=np.float)
    for subj in subjects:
        fname = "decoding_analysis/{}_{}_ifs.pkz".format(subj, exp)
        accs.ix[subj] = moss.load_pkl(fname).acc
    return list(accs.sort(inplace=False, ascending=False).index)
