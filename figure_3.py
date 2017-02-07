import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import moss

from plotutils import savefig, get_colormap


def setup_figure():

    f = plt.figure(figsize=(3.5, 3))

    mds_axes = [
        f.add_axes([.09, .56, .30, .33]),
        f.add_axes([.39, .56, .30, .33]),
        f.add_axes([.69, .56, .30, .33]),
        ]

    point_axes = [
        f.add_axes([.14, .10, .20, .45]),
        f.add_axes([.44, .10, .20, .45]),
        f.add_axes([.74, .10, .20, .45]),
        ]

    f.text(.24, .91, "Experiment 1\n(Residual)", size=8, ha="center")
    f.text(.54, .91, "Experiment 2\n(Residual)", size=8, ha="center")
    f.text(.84, .91, "Experiment 2\n(Resting)", size=8, ha="center")

    f.text(.01, .93, "A", size=12)
    f.text(.01, .56, "B", size=12)

    return f, mds_axes, point_axes


def plot_mds(subjects, experiments, axes):

    for subj, exp, ax in zip(subjects, experiments, axes):

        res_fname = "correlation_analysis/{}_{}_ifs.pkz".format(subj, exp)
        res = moss.load_pkl(res_fname)
        sorter = np.argsort(np.abs(res.prefs))

        x_, y_ = res.mds_coords.T.dot(res.prefs)
        t = np.arctan2(y_, x_)
        rot = [[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]]
        x, y = np.dot(rot, res.mds_coords[sorter].T)

        cmap = get_colormap(exp)

        ax.scatter(x, y, c=res.prefs[sorter],
                   cmap=cmap, vmin=-1.75, vmax=1.75,
                   s=8, linewidth=0)

        ax.set(xlim=(-.9, .9), ylim=(-.9, .9), aspect="equal")
        ax.set_axis_off()


def plot_points(df, axes):

    for exp, ax in zip(["dots", "sticks", "rest"], axes):

        exp_df = pd.melt(df.query("exp == @exp"),
                         "subj", ["within", "between"], "test", "corr")

        sns.pointplot(x="test", y="corr", hue="test", data=exp_df,
                      dodge=.5, join=False, ci=95,
                      palette=[".15", ".5"], ax=ax)
        plt.setp(ax.lines, linewidth=2)
        
        sns.pointplot(x="test", y="corr", hue="subj", data=exp_df,
                      palette=[".75"], scale=.75, ax=ax)
        plt.setp(ax.collections[:], facecolor="w", zorder=20)

        ax.legend_ = None
        ax.set(ylabel="",
               xlabel="",
               xticks=[-.1, 1.1],
               xticklabels=["Same\ncontext", "Different\ncontext"])

    axes[0].set(ylim=(0, .105), ylabel="Timeseries correlation (r)")
    axes[1].set(ylim=(0, .0525))
    axes[2].set(ylim=(0, .0525))

    for ax in axes:
        sns.despine(ax=ax, trim=True)

if __name__ == "__main__":

    sns.set(style="ticks", context="paper", font_scale=.9,
            rc={"xtick.major.size": 3, "ytick.major.size": 3,
                "xtick.major.width": 1, "ytick.major.width": 1,
                "xtick.major.pad": 3.5, "ytick.major.pad": 3.5,
                "axes.linewidth": 1, "lines.linewidth": 1})

    f, mds_axes, point_axes = setup_figure()

    plot_mds(["pc13", "ti06", "ti06"],
             ["dots", "sticks", "rest"],
             mds_axes)

    corr_df = pd.read_csv("data/correlation_results.csv")
    plot_points(corr_df, point_axes)

    savefig(f, __file__)
