import sys
import os
import os.path as op

import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error

import moss


def prediction_curve(dmat, vals, steps, radius):
    """Return MSE from predicting values from neighbors at radial steps."""
    # Set null distances (greater than some threshold) to 0.
    # Not in general a great idea, but fine here because we don't
    # do anything with identity edges, and sums will be faster
    # if we don't have to worry about nans
    dmat = np.nan_to_num(dmat)

    error_vals = []
    for step in steps:
        neighbors = (np.abs(dmat - step) < radius).astype(np.float)
        neighbors /= neighbors.sum(axis=1, keepdims=True)
        predicted = neighbors.dot(vals)
        m = ~np.isnan(predicted)
        error_vals.append(mean_squared_error(vals[m], predicted[m]))
    return np.array(error_vals)


def null_curves(dmat, vals, steps, radius, seed=None, n=100):
    """Shuffle vals and estimate null prediction curves."""
    rs = np.random.RandomState(seed)
    null_dist = []
    for _ in range(n):
        shuffled_vals = rs.permutation(vals)
        curve = prediction_curve(dmat, shuffled_vals, steps, radius)
        null_dist.append(curve)
    return np.array(null_dist)


def find_intersection(x, real, null):
    """Find the point at which the real error crosses the null error."""
    xx = np.linspace(x.min(), x.max(), 100)
    rreal = interp1d(x, real)(xx)
    nnull = interp1d(x, null.mean(axis=0))(xx)
    cross_point = np.argmin(rreal < nnull)
    cross_x = xx[cross_point]
    cross_y = rreal[cross_point]
    return cross_x, cross_y


if __name__ == "__main__":

    try:
        _, subj, exp, roi = sys.argv
    except ValueError:
        sys.exit("Usages: spatial_analysis.py <subj> <exp> <roi>")
    
    # Ensure that the output exists
    if not op.exists("spatial_analysis"):
        os.mkdir("spatial_analysis")

    # Load the context preference data
    fname = "decoding_analysis/{}_{}_{}.pkz".format(subj, exp, roi)
    prefs = moss.load_pkl(fname).prefs

    # Load the distance matrix
    fname = "roi_cache/{}_{}_{}.npz".format(subj, exp, roi)
    with np.load(fname) as dobj:
        dmat = dobj["dmat2d"]

    # Define the steps and radius
    steps = np.arange(2, 42, 2)
    radius = 2

    # Compute the real curve
    real = prediction_curve(dmat, prefs, steps, radius)

    # Compute the null curves
    seed = sum(map(ord, subj + "_spatial"))
    null = null_curves(dmat, prefs, steps, radius, seed)

    # Compute the PI of the null model
    pint = np.percentile(null, 5, axis=0), np.percentile(null, 95, axis=0)

    # Find the intersection between the real and null curves
    intersect = find_intersection(steps, real, null)

    # Save the results
    res = moss.Results(real=real,
                       null=null,
                       pint=pint,
                       steps=steps,
                       radius=radius,
                       intersect=intersect)

    fname = "spatial_analysis/{}_{}_{}.pkz".format(subj, exp, roi)
    moss.save_pkl(fname, res)
