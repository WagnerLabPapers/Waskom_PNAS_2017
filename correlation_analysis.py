import sys
import os
import os.path as op
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance
from statsmodels.regression.linear_model import OLS
from sklearn.manifold import MDS

import lyman
import moss
from moss import glm

from decoding_analysis import dots_design, sticks_design


PROJECT = lyman.gather_project_info()


def confound_design(exp, subj, run, hpf_kernel):
    """Build a matrix of confound variables."""
    analysis_dir = PROJECT["analysis_dir"]

    if exp == "dots":
        tr = 1
    elif exp == "sticks":
        tr = .72

    # Load in the confound information
    fstem = op.join(analysis_dir, exp, subj,
                    "preproc", "run_{}".format(run))
    motion = (pd.read_csv(op.join(fstem, "realignment_params.csv"))
                .filter(regex="rot|trans")
                .apply(stats.zscore))
    nuisance = (pd.read_csv(op.join(fstem, "nuisance_variables.csv"))
                  .filter(regex="wm_")
                  .apply(stats.zscore))
    artifacts = (pd.read_csv(op.join(fstem, "artifacts.csv"))
                   .any(axis=1))

    # Upsample dots data
    if exp == "dots":
        motion = pd.DataFrame(moss.upsample(motion, 2))
        nuisance = pd.DataFrame(moss.upsample(nuisance, 2))
        new_tps = np.arange(0, 229.5, .5)
        artifacts = artifacts.reindex(new_tps).ffill().reset_index(drop=True)

    # Build this portion of the design matrix
    confounds = pd.concat([motion, nuisance], axis=1)
    dmat = glm.DesignMatrix(confounds=confounds,
                            artifacts=artifacts,
                            hpf_kernel=hpf_kernel,
                            tr=tr)

    return dmat.design_matrix


def regress_task(exp, subj, data):
    """Fit a model of the task and return the residuals."""
    if exp == "dots":
        ntp = 459
        design, _ = dots_design(subj)
        design.loc[:, "duration"] = 0
        hpf_kernel = glm.fsl_highpass_matrix(ntp, 128, 1)

    elif exp == "sticks":
        ntp = 515
        design, _ = sticks_design(subj)
        design.loc[:, "duration"] = 0
        design.loc[:, "onset"] = (design["onset"] / .72).round()
        hpf_kernel = glm.fsl_highpass_matrix(ntp, 178, 1)

    conditions = design["condition"].unique()

    # Upsample the dots data to match the design resolution
    if exp == "dots":
        run_data = np.split(data, 12)
        data = np.concatenate([moss.upsample(d, 2) for d in run_data])

    # Make a design matrix for each run and then concatenate
    Xs = []
    for run, run_df in design.groupby("run"):

        Xrun = glm.DesignMatrix(run_df,
                                glm.FIR(tr=1, nbasis=24, offset=-2),
                                condition_names=conditions,
                                hpf_kernel=hpf_kernel,
                                ntp=ntp, tr=1, oversampling=1).design_matrix

        # Regress confounds out of the design matrix
        confounds = confound_design(exp, subj, run, hpf_kernel)
        assert len(confounds) == len(Xrun)
        confounds.index = Xrun.index
        Xrun = OLS(Xrun, confounds).fit().resid

        Xs.append(Xrun)
    X = pd.concat(Xs)

    resid = OLS(data, X).fit().resid.values
    return resid


def timeseries_correlation(data, n_runs, use_runs=None):
    """Compute correlation matrices, within run, then average."""
    if use_runs is None:
        use_runs = np.arange(n_runs)
    corrmats = []
    for run, run_data in enumerate(np.split(data, n_runs)):
        if run in use_runs:
            run_corrmat = np.corrcoef(run_data.T)
            corrmats.append(run_corrmat)
    assert len(corrmats) == len(use_runs)
    corrmat = np.mean(corrmats, axis=0)
    return corrmat


def tail_correlations(corrmat, tails, mask=None):
    """Compute the mean within and between tail correlations."""
    assert corrmat.shape == (len(tails), len(tails))
    if mask is not None:
        assert corrmat.shape == mask.shape

    def is_within(pair):
        a, b = pair
        return a == b

    # Identify cells in the matrix that represent within or between pairs
    pairs = product(tails, tails)
    wmat = np.reshape(map(is_within, pairs), corrmat.shape)
    bmat = ~wmat

    # Possibly exclude cells with the mask
    if mask is not None:
        wmat &= mask
        bmat &= mask

    # Remove the diagonal from the within matrix
    wmat[np.diag_indices_from(wmat)] = False

    # Average the correlations for within and between pairs
    corr_w = corrmat[wmat].mean()
    corr_b = corrmat[bmat].mean()

    return corr_w, corr_b


def permutation_test(corrmat, tails, mask=None, n=100, seed=None):
    """Permute tail assignments to generate null distribution."""
    rs = np.random.RandomState(seed)
    corrs_real = tail_correlations(corrmat, tails, mask)
    corrs_null = []
    for _ in xrange(n):
        perm_tails = rs.permutation(tails)
        corrs_null.append(tail_correlations(corrmat, perm_tails, mask))
    diff_real = np.subtract(*corrs_real)
    diff_null = np.subtract(*zip(*corrs_null))
    pctile = stats.percentileofscore(diff_null, diff_real)
    return pctile


def mds_variance_explained(corrmat, mds_coords):
    """Determine how much variance is explained by projection onto MDS coords."""
    orig_dist = (1 - corrmat)[np.triu_indices_from(corrmat, 1)]
    mds_dist = distance.pdist(mds_coords)
    r, _ = stats.pearsonr(orig_dist, mds_dist)
    return r ** 2


if __name__ == "__main__":

    try:
        _, subj, exp, roi = sys.argv
    except ValueError:
        sys.exit("Usages: decoding_analysis.py <subj> <exp> <roi>")

    # Ensure that the output exists
    if not op.exists("correlation_analysis"):
        os.mkdir("correlation_analysis")

    # Load the data
    data_fname = "roi_cache/{}_{}_{}.npz".format(subj, exp, roi)
    dobj = np.load(data_fname)
    data = dobj["ts_data"]
    dmats = {"2D": dobj["dmat2d"], "3D": dobj["dmat3d"]}

    # Load the outputs of the decoding analysis
    task_exp = "sticks" if exp == "rest" else exp
    res_fname = "decoding_analysis/{}_{}_{}.pkz".format(subj, task_exp, roi)
    res = moss.load_pkl(res_fname)

    # Remove voxels that were excluded from the decoding analysis
    data = data[:, res.good_voxels]

    # Pull out the tail voxels
    tail_mask = res.tails.astype(np.bool)
    data = data[:, tail_mask]
    tails = res.tails[tail_mask]
    prefs = res.prefs[tail_mask]

    # Regress the task-related effects out of experiment data
    if exp != "rest":
        data = regress_task(exp, subj, data)

    # Compute the timeseries correlation
    n_runs = dict(rest=8, dots=12, sticks=12)[exp]
    corrmat = timeseries_correlation(data, n_runs)

    # Compute the within and between tail correlations
    tail_corrs = tail_correlations(corrmat, tails)

    # Permute the tail assignments to get a null distribution
    seed = sum(map(ord, subj + "_tail_corrs"))
    corr_pctile = permutation_test(corrmat, tails, seed=seed)

    # Compute the tail correlations excluding proximate voxels
    corr_distance = {}
    corr_distance_pctiles = {}
    distance_thresholds = np.arange(0, 42, 4)

    for dim in ["2D", "3D"]:

        # Select out voxels from the appropriate distance matrix
        dmat = dmats[dim]
        dmat = dmat[res.good_voxels][:, res.good_voxels]
        dmat = dmat[tail_mask][:, tail_mask]

        dim_distance = []
        dim_distance_pctiles = []

        for thresh in distance_thresholds:
            # Apply a mask that only uses edges above the threshold
            # The mask is created this way because edges > 50mm are
            # nan, and the test will return False for both comparisons
            mask = ~(dmat < thresh)
            dim_distance.append(tail_correlations(corrmat, tails, mask))
            dim_distance_pctiles.append(permutation_test(corrmat, tails, mask,
                                                         seed=seed))
        corr_distance[dim] = np.array(dim_distance)
        corr_distance_pctiles[dim] = np.array(dim_distance_pctiles)
        assert tail_corrs == tuple(corr_distance[dim][0])
        assert corr_pctile == corr_distance_pctiles[dim][0]

    # Compute the tail correlations over time in the resting-state
    if exp == "rest":
        corr_times = []
        corr_times_pctiles = []
        # Use runs over each scan, averaged over sessions
        run_indices = zip(np.arange(4), np.arange(4, 8))
        for use_runs in run_indices:
            time_corrmat = timeseries_correlation(data, n_runs, use_runs)
            corr_times.append(tail_correlations(time_corrmat, tails))
            corr_times_pctiles.append(permutation_test(time_corrmat, tails,
                                                       seed=seed))
        corr_times = np.array(corr_times)
        corr_times_pctiles = np.array(corr_times_pctiles)
    else:
        corr_times = None
        corr_times_pctiles = None

    # Use MDS to reduce the dimensionality of the correlation matrix
    seed = sum(map(ord, subj + "_mds"))
    mds = MDS(n_init=50, random_state=seed, dissimilarity="precomputed")
    mds.fit(1 - corrmat)
    mds_varexp = mds_variance_explained(corrmat, mds.embedding_)

    # Save out the results
    res = moss.Results(tails=tails,
                       prefs=prefs,
                       corrmat=corrmat,
                       tail_corrs=tail_corrs,
                       corr_pctile=corr_pctile,
                       distance_thresh=distance_thresholds,
                       corr_distance=corr_distance,
                       corr_distance_pctiles=corr_distance_pctiles,
                       corr_times=corr_times,
                       corr_times_pctiles=corr_times_pctiles,
                       mds_params=mds.get_params(),
                       mds_coords=mds.embedding_,
                       mds_varexp=mds_varexp)
    fname = "correlation_analysis/{}_{}_{}.pkz".format(subj, exp, roi)
    moss.save_pkl(fname, res)
