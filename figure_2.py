import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from mayavi import mlab
from surfer import Brain

import moss
from lyman.tools.plotting import crop

from surfutils import roi_to_surf
from plotutils import (set_style, savefig,
                       get_colormap, get_ifs_view, get_subject_order)


def setup_figure():

    f = plt.figure(figsize=(6, 2.15))

    brain_gs = plt.GridSpec(2, 2, .03, .18, .52, .99, .05, .05)
    brain_axes = [f.add_subplot(gs) for gs in brain_gs]
    brain_axes = np.reshape(brain_axes, (2, 2))

    hist_gs = plt.GridSpec(2, 1, .53, .20, .65, .975)
    hist_axes = [f.add_subplot(gs) for gs in hist_gs]

    cmap_gs = plt.GridSpec(1, 2, .05, .07, .50, .14, .05, .05)
    cmap_axes = [f.add_subplot(gs) for gs in cmap_gs]

    clust_gs = plt.GridSpec(1, 1, .75, .20, .98, .98)
    clust_ax = f.add_subplot(clust_gs[0])

    f.text(.003, .92, "A", size=12)
    f.text(.68, .92, "B", size=12)

    return f, brain_axes, hist_axes, cmap_axes, clust_ax


def plot_brains(subjects, axes):

    for subj, subj_axes in zip(subjects, axes):

        exp = dict(pc="dots", ti="sticks")[subj[:2]]

        data_fname = "roi_cache/{}_{}_ifs.npz".format(subj, exp)
        with np.load(data_fname) as dobj:
            vox_ijk = dobj["vox_ijk"]

        res_fname = "decoding_analysis/{}_{}_ifs.pkz".format(subj, exp)
        res = moss.load_pkl(res_fname)
        prefs = res.prefs

        surf_vals = roi_to_surf(exp, subj, prefs, vox_ijk)

        lut = get_colormap(exp, False)

        for hemi, ax in zip(["lh", "rh"], subj_axes):

            b = Brain(subj, hemi, "inflated", background="white",
                      cortex=("binary", -4, 8, False),
                      size=(1000, 600))

            b.add_data(surf_vals.ix[hemi].fillna(-11).values,
                       colormap=lut, colorbar=False,
                       thresh=-10, min=-1.75, max=1.75)

            mlab.view(*get_ifs_view(subj, hemi))
            img = crop(b.screenshot())
            ax.imshow(img, rasterized=True)
            ax.set(xticks=[], yticks=[])
            b.close()


def plot_hists(subjects, axes, label_last=1, ymax=350):

    bins = np.linspace(-2, 2, 20)

    for subj, ax in zip(subjects, axes):

        exp = dict(pc="dots", ti="sticks")[subj[:2]]

        res_fname = "decoding_analysis/{}_{}_ifs.pkz".format(subj, exp)
        res = moss.load_pkl(res_fname)
        prefs = res.prefs
        pctiles = res.pref_pctiles

        cmap = get_colormap(exp)

        plot_prefs = [prefs[pctiles < 10],
                      prefs[pctiles > 90],
                      prefs[(pctiles >= 10) & (pctiles <= 90)]]

        ax.hist(plot_prefs, histtype="barstacked", rwidth=1,
                color=[cmap(.01), cmap(.99), ".9"], bins=bins)

        ax.set(xlim=(-2, 2),
               xticks=[-2, -1, 0, 1, 2],
               xticklabels=[],
               yticks=[],
               ylim=(0, ymax))

    for ax in axes[-label_last:]:
        ax.set_xlabel("Context\npreference", labelpad=2, fontsize=7)
        ax.set(xticklabels=[-2, -1, 0, 1, 2])

    for ax in axes:
        sns.despine(ax=ax, left=True)


def plot_colorbars(f, axes):

    dots, sticks = get_colormap("dots"), get_colormap("sticks")

    xx = np.arange(200).reshape(1, 200)

    axes[0].imshow(xx, rasterized=True, aspect="auto", cmap=dots)
    axes[1].imshow(xx, rasterized=True, aspect="auto", cmap=sticks)

    kws = dict(size=7, ha="center")
    f.text(.08, .015, "Motion", **kws)
    f.text(.24, .015, "Color", **kws)
    f.text(.32, .015, "Orientation", **kws)
    f.text(.48, .015, "Color", **kws)

    plt.setp(axes, xticks=[], yticks=[])


def add_compass(ax, hemi, color="w"):

    if hemi == "lh":
        x = .88
        xlabels = "A", "P"
    elif hemi == "rh":
        x = .12
        xlabels = "P", "A"
    else:
        raise ValueError(hemi)
    y = .20
    xw = .05
    yw = .08

    arrowprops = dict(arrowstyle="<->, head_width=.12, head_length=.12",
                      shrinkA=0, shrinkB=0, linewidth=.4,
                      edgecolor=color)

    ax.annotate("", xy=(x - xw, y), xytext=(x + xw, y),
                xycoords="axes fraction", textcoords="axes fraction",
                color=color, arrowprops=arrowprops)
    ax.annotate("", xy=(x, y - yw), xytext=(x, y + yw),
                xycoords="axes fraction", textcoords="axes fraction",
                color=color, arrowprops=arrowprops)

    xw *= 1.65
    yw *= 1.65

    if x < 0:
        x = 1 + x
    xpos = (x - xw, y), (x + xw, y)
    for (x_, y_), s_ in zip(xpos, xlabels):
        ax.text(x_, y_, s_, size=5, color=color,
                transform=ax.transAxes,
                ha="center", va="center")

    ypos = (x, y - yw), (x, y + yw)
    ylabels = "V", "D"
    for (x_, y_), s_ in zip(ypos, ylabels):
        ax.text(x_, y_, s_, size=5, color=color,
                transform=ax.transAxes,
                ha="center", va="center")


def plot_cluster_error(ax):

    res_ftemp = "spatial_analysis/{}_{}_ifs.pkz"
    for exp in ["dots", "sticks"]:

        subjects = get_subject_order(exp)
        color = get_colormap(exp, as_cmap=False)[20]

        errs = []
        for subj in subjects:

            res = moss.load_pkl(res_ftemp.format(subj, exp))
            x = res.steps

            norm = res.null.mean()
            errs.append(res.real / norm)

        errs = np.vstack(errs)
        mean = errs.mean(axis=0)
        ax.plot(x, mean, color=color, lw=2)
        sem = stats.sem(errs, axis=0)
        ax.fill_between(x, mean - sem, mean + sem, alpha=.2, color=color)


    ax.axhline(y=1, lw=1, dashes=[5, 2],
               color=".5", zorder=0,
               xmin=.02, xmax=.98)

    ax.set(xlim=(0, 42),
           ylim=(.55, 1.45),
           yticks=[.6, .8, 1, 1.2, 1.4],
           xticks=[0, 10, 20, 30, 40],
           xlabel="Neighborhood radius (mm)",
           ylabel="Normalized error")

    sns.despine(ax=ax, trim=True)
            

if __name__ == "__main__":

    set_style()
    f, brain_axes, hist_axes, cbar_axes, clust_ax = setup_figure()

    subjects = ["pc11", "ti05"]

    plot_brains(subjects, brain_axes)
    plot_hists(subjects, hist_axes)
    plot_colorbars(f, cbar_axes)
    plot_cluster_error(clust_ax)

    add_compass(brain_axes[-1, 0], "lh")
    add_compass(brain_axes[-1, 1], "rh")

    savefig(f, __file__)
