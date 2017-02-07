import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mayavi import mlab
from surfer import Brain

from lyman.tools.plotting import crop

from plotutils import set_style, savefig, points_to_lines


def setup_figure():

    f = plt.figure(figsize=(3.5, 2.25))

    brain_axes = [
        f.add_axes([.01, .50, .40, .49]),
        f.add_axes([.01, .01, .40, .49]),
        ]

    swarm_axes = [
        f.add_axes([.55, .10, .22, .83]),
        f.add_axes([.77, .10, .22, .83])
        ]

    f.text(.01, .93, "A", size=12)
    f.text(.42, .93, "B", size=12)

    return f, brain_axes, swarm_axes


def plot_brains(axes, palette):

    lat_ax, med_ax = axes
    lat_color, med_color = palette

    b = Brain("fsaverage", "lh", "pial",
              background="white", size=(1200, 1200))

    b.add_morphometry("curv", grayscale=True, min=-.5, max=.5,
                      colormap="Greys", colorbar=False)

    b.add_label("roi_labels/lh.ifs.label", alpha=.9, color="#feb308")
    b.add_label("roi_labels/lh.mfc.label", alpha=.9, color="#cf6275")

    mlab.view(160, 70)
    lat_ax.imshow(crop(b.screenshot()), rasterized=True)

    mlab.view(15, 90)
    med_ax.imshow(crop(b.screenshot()),  rasterized=True)

    b.close()

    for ax in axes:
        ax.set_axis_off()


def plot_swarms(df, axes, palette):

    for exp, ax in zip(["dots", "sticks"], axes):

        exp_df = df.query("experiment == @exp")

        ax.axhline(.5, .1, .9, dashes=(5, 2), color=".6")        
        ax.set(ylim=(.4, .9), yticks=[.4, .5, .6, .7, .8, .9])

        sns.pointplot(x="roi", y="acc", data=exp_df,
                      palette=palette, join=False, ci=None, ax=ax)
        points_to_lines(ax, lw=3)

        sns.swarmplot(x="roi", y="acc", data=exp_df, size=4,
                      color=".85", # facecolor="none",
                      linewidth=1, edgecolor=".4", ax=ax)

        ax.set(xlabel="", ylabel="", xticklabels=["IFS", "MFC"])

    ax_l, ax_r = axes
    ax_l.set(ylabel="Decoding accuracy")
    ax_r.set(yticks=[])

    ax_l.text(.5, .91, "Experiment 1", ha="center", va="center", size=7.5)
    ax_r.text(.5, .91, "Experiment 2", ha="center", va="center", size=7.5)

    sns.despine(ax=ax_l, trim=True)
    sns.despine(ax=ax_r, left=True, trim=True)


if __name__ == "__main__":

    set_style()
    f, brain_axes, swarm_axes = setup_figure()
    palette = ["#feb308", "#cf6275"]

    plot_brains(brain_axes, palette)

    acc_df = pd.read_csv("data/decoding_results.csv")
    plot_swarms(acc_df, swarm_axes, palette)

    savefig(f, __file__)
