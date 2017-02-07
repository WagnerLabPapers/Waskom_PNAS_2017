import numpy as np
import matplotlib.pyplot as plt

from plotutils import set_style, savefig, get_colormap, get_subject_order

from figure_2 import plot_brains, plot_hists


def setup_figure():

    f = plt.figure(figsize=(7, 2.8))

    brain_gs = plt.GridSpec(3, 4, .13, .13, .87, .99, .05, .05)
    brain_axes = [f.add_subplot(gs) for gs in brain_gs]
    brain_axes = np.vstack(np.array_split(brain_axes, 6))

    hist_gs = plt.GridSpec(3, 2, .01, .15, .99, .99, 7, .08)
    hist_axes = [f.add_subplot(gs) for gs in hist_gs]

    cbar_ax = f.add_axes([.35, .07, .3, .04])

    return f, brain_axes, hist_axes, cbar_ax


def plot_colorbar(f, ax):

    cmap = get_colormap("sticks")

    xx = np.arange(200).reshape(1, 200)

    ax.imshow(xx, rasterized=True, aspect="auto", cmap=cmap)

    kws = dict(size=7, ha="center")
    f.text(.35, .03, "Orientation", **kws)
    f.text(.65, .03, "Color", **kws)

    ax.set(xticks=[], yticks=[])


if __name__ == "__main__":

    set_style()
    f, brain_axes, hist_axes, cbar_ax = setup_figure()

    subjects = get_subject_order("sticks")

    plot_brains(subjects, brain_axes)
    plot_hists(subjects, hist_axes, 2, 450)
    plot_colorbar(f, cbar_ax)

    savefig(f, __file__)
