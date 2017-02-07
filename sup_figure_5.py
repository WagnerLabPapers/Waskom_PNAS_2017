import matplotlib.pyplot as plt

from plotutils import set_style, savefig, get_subject_order
from figure_3 import plot_mds


def setup_figure():

    f = plt.figure(figsize=(7, 4.5))

    dots_grid = plt.GridSpec(2, 7, .02, .51, .98, .94, 0, 0)
    dots_axes = [f.add_subplot(spec) for spec in dots_grid]

    sticks_grid = plt.GridSpec(1, 6, .09, .26, .91, .47, 0, 0)
    sticks_axes = [f.add_subplot(spec) for spec in sticks_grid]

    rest_grid = plt.GridSpec(1, 6, .09, .01, .91, .22, 0, 0)
    rest_axes = [f.add_subplot(spec) for spec in rest_grid]

    return f, dots_axes, sticks_axes, rest_axes


if __name__ == "__main__":

    set_style()
    f, dots_axes, sticks_axes, rest_axes = setup_figure()

    dots_subjects = get_subject_order("dots")
    dots_experiment = ["dots"] * len(dots_subjects)
    plot_mds(dots_subjects, dots_experiment, dots_axes)

    sticks_subjects = get_subject_order("sticks")
    sticks_experiment = ["sticks"] * len(sticks_subjects)
    plot_mds(sticks_subjects, sticks_experiment, sticks_axes)

    rest_experiment = ["rest"] * len(sticks_subjects)
    plot_mds(sticks_subjects, rest_experiment, rest_axes)

    f.text(.5, .95, "Experiment 1 (Residual)", ha="center", size=10)
    f.text(.5, .48, "Experiment 2 (Residual)", ha="center", size=10)
    f.text(.5, .23, "Experiment 2 (Resting)", ha="center", size=10)

    savefig(f, __file__)
