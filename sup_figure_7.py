import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import moss

from plotutils import set_style, savefig, get_subject_order


def setup_figure():

    f, axes = plt.subplots(1, 6, figsize=(6, 2))

    return f, axes


def plot_time_corrs(subjects, axes):

    x = np.arange(1, 5)
    palette = [".2", ".5"]

    for subj, ax in zip(subjects, axes):

        res_fname = "correlation_analysis/{}_rest_ifs.pkz".format(subj)
        res = moss.load_pkl(res_fname)

        for line, color in zip(res.corr_times.T, palette):
            ax.plot(x, line, "o-", color=color, ms=3, clip_on=False)

        sig = res.corr_times_pctiles > 95
        ax.plot(x[sig], np.ones(sig.sum()) * .0025,
                marker=(6, 2, 0), ls="", mew=.35, mec=".2", ms=3)

        ax.set(xticks=x, xlim=(.6, 4.4), ylim=(0, .07))
        sns.despine(ax=ax, trim=True)

    plt.setp(axes[1:], yticklabels=[])
    axes[0].set_ylabel("Correlation (r)")


if __name__ == "__main__":

    set_style()
    f, axes = setup_figure()

    subjects = get_subject_order("sticks")

    plot_time_corrs(subjects, axes)

    axes[0].text(2.5, .041, "Same context", color=".2", size=7, ha="center")
    axes[0].text(2.9, .014, "Different\ncontext", color=".5", size=7, ha="center")
    f.text(.55, .03, "Resting scan number", size=10, ha="center")

    f.tight_layout()
    f.subplots_adjust(bottom=.2)

    savefig(f, __file__)
