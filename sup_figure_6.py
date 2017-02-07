import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import moss

from plotutils import set_style, savefig, get_subject_order


def setup_figure():

    f = plt.figure(figsize=(7, 5))

    mat_grid = plt.GridSpec(2, 6, .07, .52, .98, .95, .15, .20)
    mat_axes = [f.add_subplot(spec) for spec in mat_grid]
    sticks_axes, rest_axes = np.array_split(mat_axes, 2)

    scatter_grid = plt.GridSpec(1, 6, .07, .30, .98, .49, .15, .05)
    scatter_axes = [f.add_subplot(spec) for spec in scatter_grid]

    kde_grid = plt.GridSpec(1, 6, .07, .07, .98, .21, .15, .05)
    kde_axes = [f.add_subplot(spec) for spec in kde_grid]

    cbar_ax = f.add_axes([.04, .62, .015, .26])

    return f, sticks_axes, rest_axes, scatter_axes, kde_axes, cbar_ax


def plot_corrmats(subjects, axes, exp):

    for subj, ax in zip(subjects, axes):

        fname = "correlation_analysis/{}_{}_ifs.pkz".format(subj, exp)
        corrmat = moss.load_pkl(fname).corrmat

        ax.imshow(corrmat - np.eye(len(corrmat)),
                  cmap="RdBu_r", vmin=-.15, vmax=.15,
                  rasterized=True)

        ax.set(xticks=[], yticks=[])
        sns.despine(ax=ax, left=True, bottom=True)


def plot_scatters(subjects, axes):

    ftemp = "correlation_analysis/{}_{}_ifs.pkz"
    for subj, ax in zip(subjects, axes):

        sticks = moss.load_pkl(ftemp.format(subj, "sticks")).corrmat
        rest = moss.load_pkl(ftemp.format(subj, "rest")).corrmat

        triu = np.triu_indices_from(rest, 1)

        ax.scatter(sticks[triu], rest[triu], s=3, linewidth=.2,
                   color=".6", edgecolor="w",
                   rasterized=True)

        ax.plot([-.2, .8], [-.2, .8], lw=1, dashes=(5, 2), color=".3")

    plt.setp(axes,
             xlim=(-.25, .8), ylim=(-.25, .8),
             xticks=np.linspace(-.2, .8, 6),
             yticks=np.linspace(-.2, .8, 6),
             aspect="equal")
    plt.setp(axes[1:], yticklabels=[])
    for ax in axes:
        sns.despine(ax=ax, trim=True)
        plt.setp(ax.get_xticklabels(), size=6)
        plt.setp(ax.get_yticklabels(), size=6)


def plot_kdes(subjects, axes):

    ftemp = "correlation_analysis/{}_{}_ifs.pkz"
    for subj, ax in zip(subjects, axes):

        sticks = moss.load_pkl(ftemp.format(subj, "sticks")).corrmat
        rest = moss.load_pkl(ftemp.format(subj, "rest")).corrmat

        triu = np.triu_indices_from(rest, 1)

        sns.kdeplot(sticks[triu], color=".15",
                    label="residual", ax=ax)
        sns.kdeplot(rest[triu], color=".45", dashes=[4, 1],
                    label="resting", ax=ax)

    plt.setp(axes,
             xlim=(-.25, .8), ylim=(0, 17),
             xticks=np.linspace(-.2, .8, 6),
             yticks=[])

    for ax in axes:
        sns.despine(ax=ax, left=True, trim=True)
        plt.setp(ax.get_xticklabels(), size=6)
        plt.setp(ax.get_yticklabels(), size=6)

    axes[0].legend(bbox_to_anchor=(1.2, .8))
    for ax in axes[1:]:
        ax.legend_ = None


def plot_colorbar(ax):

    xx = np.linspace(1, 0, 200)[:, np.newaxis]
    ax.imshow(xx, cmap="RdBu_r", aspect="auto", rasterized=True)
    ax.set(xticks=[], yticks=[])
    ax.text(.75, -20, .15, ha="right", va="top", size=7)
    ax.text(.75, 220, -.15, ha="right", va="bottom", size=7)


if __name__ == "__main__":

    set_style()
    f, sticks_axes, rest_axes, scatter_axes, kde_axes, cbar_ax = setup_figure()

    subjects = get_subject_order("sticks")

    plot_corrmats(subjects, sticks_axes, "sticks")
    plot_corrmats(subjects, rest_axes, "rest")
    plot_scatters(subjects, scatter_axes)
    plot_kdes(subjects, kde_axes)
    plot_colorbar(cbar_ax)

    f.text(.525, .960, "Spontaneous correlations (residual)",
           size=9, ha="center")
    f.text(.525, .725, "Spontaneous correlations (resting)",
           size=9, ha="center")
    f.text(.525, .01, "Timeseries correlation (r)",
           size=8, ha="center")
    f.text(.525, .24, "Residual correlation (r)",
           size=8, ha="center")
    f.text(.02, .40, "Resting correlation (r)",
           size=8, va="center", rotation=90)
    f.text(.02, .13, "Density (a.u.)",
           size=8, va="center", rotation=90)
    f.text(.01, .75, "Timeseries correlation (r)",
           size=8, va="center", rotation=90)

    f.text(.01, .96, "A", size=12)
    f.text(.01, .52, "B", size=12)
    f.text(.01, .21, "C", size=12)

    savefig(f, __file__)
