import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import moss

from plotutils import set_style, savefig, get_subject_order


def setup_figure():

    f = plt.figure(figsize=(7, 4.65))

    dots_grid = plt.GridSpec(2, 7, .08, .45, .98, .94, .15, .15)
    dots_axes = [f.add_subplot(spec) for spec in dots_grid]

    sticks_grid = plt.GridSpec(1, 6, .12, .09, .92, .34, .15, .1)
    sticks_axes = [f.add_subplot(spec) for spec in sticks_grid]

    return f, dots_axes, sticks_axes


def plot_prediction_curves(subjects, axes, exp):

    res_ftemp = "spatial_analysis/{}_{}_ifs.pkz"
    for subj, ax in zip(subjects, axes):

        res = moss.load_pkl(res_ftemp.format(subj, exp))
        x = res.steps

        norm = res.null.mean()
        real = res.real / norm
        pint = res.pint / norm

        ax.plot(x, real, "o-", color=".15",
                ms=2.5, clip_on=False)
        ax.fill_between(x, *pint, color=".4", alpha=.3)

        cross_x, cross_y = res.intersect
        cross_y /= norm

        ax.plot([cross_x, cross_x], [0, cross_y],
                lw=.8, dashes=[3, 1], color=".5", zorder=0)

        ax.set(xlim=(0, 40), ylim=(0, 2),
               xticks=np.linspace(0, 40, 5),
               yticks=[0, 1, 2],
               yticklabels=[0, 1, 2])

        sns.despine(ax=ax)

    ylabel = "Normalized error"
    plt.setp(axes[1:7], yticklabels=[])
    axes[0].set(ylabel=ylabel)

    if exp == "dots":
        plt.setp(axes[8:], yticklabels=[])
        plt.setp(axes[:7], xticklabels=[])
        axes[7].set_ylabel(ylabel)


if __name__ == "__main__":

    set_style()
    f, dots_axes, sticks_axes = setup_figure()

    dots_subjects = get_subject_order("dots")
    sticks_subjects = get_subject_order("sticks")

    plot_prediction_curves(dots_subjects, dots_axes, "dots")
    plot_prediction_curves(sticks_subjects, sticks_axes, "sticks")

    f.text(.525, .015, "Neighborhood radius (mm)", size=10, ha="center")
    f.text(.525, .96, "Experiment 1", size=11, ha="center")
    f.text(.525, .36, "Experiment 2", size=11, ha="center")

    savefig(f, __file__)
