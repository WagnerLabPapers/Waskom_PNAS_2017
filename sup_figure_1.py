import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from plotutils import set_style, savefig


def poisson_disc_sample(array_radius, fixation_radius, radius, candidates):
    """Find positions using poisson-disc sampling."""
    # See http://bost.ocks.org/mike/algorithms/
    rs = np.random.RandomState(15)
    uniform = rs.uniform
    randint = rs.randint

    # Start at a fixed point we know will work
    start = 0, array_radius / 2
    samples = [start]
    queue = [start]

    while queue:

        # Pick a sample to expand from
        s_idx = randint(len(queue))
        s_x, s_y = queue[s_idx]

        for i in xrange(candidates):

            # Generate a candidate from this sample
            a = uniform(0, 2 * np.pi)
            r = uniform(radius, 2 * radius)
            x, y = s_x + r * np.cos(a), s_y + r * np.sin(a)

            # Check the three conditions to accept the candidate
            in_array = np.sqrt(x ** 2 + y ** 2) < array_radius
            in_ring = np.all(cdist(samples, [(x, y)]) > radius)
            in_fixation = np.sqrt(x ** 2 + y ** 2) < fixation_radius

            if in_array and in_ring and not in_fixation:
                # Accept the candidate
                samples.append((x, y))
                queue.append((x, y))
                break

        if (i + 1) == candidates:
            # We've exhausted the particular sample
            queue.pop(s_idx)

    # Remove first sample to give space around the fix point
    samples = np.array(samples)[1:]

    return samples


def plot_dots_design(f):

    # --- Define the axes positions

    lefts = np.linspace(.03, .41, 4)
    bottoms = np.linspace(.68, .53, 4)

    ratio = f.get_figwidth() / f.get_figheight()
    width = .22
    height = width * ratio

    # --- Initialize the frames

    axes = []
    for left, bottom in zip(lefts, bottoms):
        ax = f.add_axes([left, bottom, width, height], axisbg=".3")
        ax.set(xticks=[], yticks=[], xlim=(-1, 1), ylim=(-1, 1))
        for _, spine in ax.spines.items():
            spine.set(linewidth=2, color="w")
        axes.append(ax)

    # --- Add cue frame

    for ax in axes[2:]:
        colors = ["w", "k", "w", ".3"]
        for i, x in enumerate(np.linspace(.1, .2, 4)):
            c = colors[i]
            w = 2 * (1 - x)
            rect = plt.Rectangle((x - 1, x - 1), w, w,
                                 linewidth=0, facecolor=c)
            ax.add_artist(rect)

    # --- Add stimulus dots

    rs = np.random.RandomState(0)

    # Plot the dots themselves
    xy = poisson_disc_sample(1.5, .15, .19, 50)
    m = np.abs(xy).max(axis=1) < .73
    xy = xy[m]
    xx, yy = xy.T
    r, g = "#ED8A45", "#00BD3B"
    c = rs.choice([r, g], p=[.75, .25], size=m.sum())
    ax.scatter(xx, yy, c=c, zorder=2, marker="s", s=8, linewidth=0)

    # Plot arrows indicating dot motion
    t = rs.uniform(0, np.pi * 2, m.sum())
    t[:m.sum() // 3] = np.pi / 2
    t = rs.permutation(t)
    xy_d = .06 * np.c_[np.cos(t), np.sin(t)]
    for xy_i, xy_d_i, c_i in zip(xy, xy_d, c):
        x_i, y_i = xy_i
        x_d_i, y_d_i = xy_d_i
        ax.arrow(x_i, y_i, x_d_i, y_d_i, facecolor=c_i,
                 head_width=.03, head_length=.02, edgecolor=c_i,
                 overhang=.5)

    # --- Add fixation point

    fix_colors = ["k", "w", "w", "w"]
    for ax, c in zip(axes, fix_colors):
        ax.scatter(0, 0, c=c, s=15, linewidth=0, zorder=2)

    # --- Add timing info
    times = ["2 - 10s", ".5s", "0 or 1s", "0 or 2s"]
    for time, ax in zip(times, axes):
        ax.text(-.9, -1.15, time, size=8)


def plot_sticks_design(f):

    # --- Define axes positions

    lefts = np.linspace(.05, .36, 3)
    bottoms = np.linspace(.15, .05, 3)

    ratio = f.get_figwidth() / f.get_figheight()
    width = .24
    height = width * ratio

    # --- Initialize the axes frames

    axes = []
    for left, bottom in zip(lefts, bottoms):
        ax = f.add_axes([left, bottom, width, height], axisbg=".3")
        ax.set(xticks=[], yticks=[], xlim=(-1, 1), ylim=(-1, 1))
        for _, spine in ax.spines.items():
            spine.set(linewidth=2, color="w")
        axes.append(ax)

    # --- Add the cue polygon

    cue = mpl.patches.CirclePolygon((0, 0), .1, 5,
                                    color=".5", linewidth=0)
    ax.add_artist(cue)

    # --- Add the stimulus

    points = poisson_disc_sample(.9, .15, .1, 30)
    
    # Define markers to show left and right tilted sticks
    markers = dict(left=[(-1, 1), (1, -1)], right=[(-1, -1), (1, 1)])

    # Assign colors to show red and green sticks
    rgb_colors = np.array(["#A7D2A2", "#F2BB97"])
    
    # Assign orientation and color values for each stick
    rs = np.random.RandomState(0)
    stick_ori = rs.choice(["left", "right"], len(points), p=[.7, .3])
    stick_colors = rgb_colors[rs.choice([0, 1], len(points), p=[.7, .3])]

    for ori in ["left", "right"]:

        # Draw the subset of sticks with this orientation
        ori_points = points[stick_ori == ori]
        ax.scatter(*ori_points.T, s=8, linewidth=.75,
                   edgecolor=stick_colors[stick_ori == ori],
                   marker=markers[ori])

    # --- Add the fixation point

    fix_colors = ["k", "w", "w", "w"]
    for ax, c in zip(axes, fix_colors):
        ax.scatter(0, 0, c=c, s=15, linewidth=0, zorder=2)

    # --- Add timing info
    times = [".72 - 7.2s", ".72s", "< 2.88s"]
    for time, ax in zip(times, axes):
        ax.text(-.9, -1.15, time, size=8)


def plot_behavior(f):

    # --- Set up the axes
    grid = plt.GridSpec(2, 2, .71, .05, .99, .85, .5, .5)
    axes = [f.add_subplot(spec) for spec in grid]

    # --- Plot Experiment 1 behavior

    dots_behav = pd.read_csv("data/dots_behav.csv")

    sns.pointplot(x="context", y="rt", hue="subj",
                  ci=None, palette=[".5"], order=["motion", "color"],
                  data=dots_behav, ax=axes[0])

    sns.pointplot(x="context", y="correct", hue="subj",
                  ci=None, palette=[".5"], order=["motion", "color"],
                  data=dots_behav, ax=axes[1])

    # --- Plot Exeperiment 2 behavior

    sticks_behav = pd.read_csv("data/sticks_behav.csv")

    sns.pointplot(x="context", y="rt", hue="subj",
                  ci=None, palette=[".3"], order=["ori", "hue"],
                  data=sticks_behav.query("context_diff == 'easy'"),
                  ax=axes[2])

    sns.pointplot(x="context", y="rt", hue="subj",
                  ci=None, palette=[".6"], order=["ori", "hue"],
                  data=sticks_behav.query("context_diff == 'hard'"),
                  ax=axes[2])

    sns.pointplot(x="context", y="correct", hue="subj",
                  ci=None, palette=[".3"], order=["ori", "hue"],
                  data=sticks_behav.query("context_diff == 'easy'"),
                  ax=axes[3])

    sns.pointplot(x="context", y="correct", hue="subj",
                  ci=None, palette=[".6"], order=["ori", "hue"],
                  data=sticks_behav.query("context_diff == 'hard'"),
                  ax=axes[3])

    # --- Tweak axes attributes

    for ax in axes[::2]:
        ax.set(ylim=(.6, 1.8), yticks=np.linspace(.6, 1.8, 5))
        ax.set_ylabel("Reaction time (s)", labelpad=3)

    for ax in axes[1::2]:
        ax.set(ylim=(.5, 1))
        ax.set_ylabel("P(correct)", labelpad=1)

    for ax in axes[:2]:
        ax.set(xticklabels=["Motion", "Color"])

    for ax in axes[2:]:
        ax.set(xticklabels=["Orient", "Color"])

    for ax in axes:
        plt.setp(ax.collections, sizes=[5], clip_on=False)
        ax.legend_ = None
        ax.set(xlabel="", xlim=(-.2, 1.2))
        sns.despine(ax=ax, trim=True)


if __name__ == "__main__":

    set_style()
    f = plt.figure(figsize=(6.5, 5))

    plot_dots_design(f)
    plot_sticks_design(f)
    plot_behavior(f)

    f.text(.5, .95, "Experiment 1", size=12, ha="center")
    f.text(.5, .45, "Experiment 2", size=12, ha="center")

    f.text(.01, .95, "A", size=12)
    f.text(.01, .45, "B", size=12)
    f.text(.64, .85, "C", size=12)
    f.text(.64, .37, "D", size=12)

    savefig(f, __file__)
