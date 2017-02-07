import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import moss

from plotutils import set_style, savefig, get_subject_order


def setup_figure():

    f = plt.figure(figsize=(7, 6))

    dots_grid = plt.GridSpec(2, 7, .1, .56, .98, .94, .1, .15)
    dots_axes = [f.add_subplot(spec) for spec in dots_grid]

    sticks_grid = plt.GridSpec(1, 6, .18, .31, .92, .48, .1, .1)
    sticks_axes = [f.add_subplot(spec) for spec in sticks_grid]

    rest_grid = plt.GridSpec(1, 6, .18, .07, .92, .24, .1, .1)
    rest_axes = [f.add_subplot(spec) for spec in rest_grid]

    return f, dots_axes, sticks_axes, rest_axes


def plot_distance_corrs(subjects, axes, exp):

    for subj, ax in zip(subjects, axes):

        res_fname = "correlation_analysis/{}_{}_ifs.pkz".format(subj, exp)
        res = moss.load_pkl(res_fname)
        x = res.distance_thresh

        for dim, color, marker in zip(["3D", "2D"], [".5", ".2"], ["x", "+"]):
            same, diff = res.corr_distance[dim].T
            ax.plot(x, same - diff, "o-", color=color, ms=3, clip_on=False)

            sig = res.corr_distance_pctiles[dim] > 95
            stary = -.005 if exp == "dots" else -.0025
            ax.plot(x[sig], np.ones(sig.sum()) * stary,
                    marker=marker, ls="", mew=.35, mec=".2", ms=3)

        ylim = (-.01, .08) if exp == "dots" else (-.005, .04)
        yticks = np.array([0, .01, .02, .03, .04])
        yticks =  yticks * 2 if exp == "dots" else yticks
        ax.set(xlim=(-2, 42), ylim=ylim, yticks=yticks)
        sns.despine(ax=ax, trim=True)

    ylabel = "Subnetwork strength\n($r_{\mathrm{same}} - r_{\mathrm{diff}}$)"
    plt.setp(axes[1:7], yticklabels=[])
    axes[0].set_ylabel(ylabel)

    if exp == "dots":
        plt.setp(axes[8:], yticklabels=[])
        plt.setp(axes[:7], xticklabels=[])
        axes[7].set_ylabel(ylabel)


if __name__ == "__main__":

    set_style()
    f, dots_axes, sticks_axes, rest_axes = setup_figure()

    dots_subjects = get_subject_order("dots")
    sticks_subjects = get_subject_order("sticks")

    plot_distance_corrs(dots_subjects, dots_axes, "dots")
    plot_distance_corrs(sticks_subjects, sticks_axes, "sticks")
    plot_distance_corrs(sticks_subjects, rest_axes, "rest")

    f.text(.525, .01, "Distance threshold (mm)", size=10, ha="center")
    f.text(.525, .95, "Experiment 1 (residuals)", size=11, ha="center")
    f.text(.525, .50, "Experiment 2 (residuals)", size=11, ha="center")
    f.text(.525, .25, "Experiment 2 (resting)", size=11, ha="center")

    dots_axes[0].text(10, .038, "2D Distance", color=".2", size=7)
    dots_axes[0].text(10, .005, "3D Distance", color=".5", size=7)

    savefig(f, __file__)
