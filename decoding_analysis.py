from __future__ import division, print_function
import sys
import os
import os.path as op
from copy import copy

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from statsmodels.regression.linear_model import OLS

import lyman
import moss
from moss import glm
import moss.design as md


PROJECT = lyman.gather_project_info()


class Dataset(object):

    def __init__(self, exp, subj, data):
        """Initialize the object and build the data matrix."""
        assert exp in {"dots", "sticks"}
        self.exp = exp
        self.subj = subj
        self.data = data

        self.hpf_kernel = self._get_hpf_kernel()
        self.deconvolve()

    def _get_hpf_kernel(self):
        """Cache the highpass filter matrix, which is expensive to build."""
        if self.exp == "dots":
            return glm.fsl_highpass_matrix(230, 128, 2)
        elif self.exp == "sticks":
            return glm.fsl_highpass_matrix(515, 128, .72)

    def design_info(self):
        """Return a design DataFrame for the relevant experiment."""
        if self.exp == "dots":
            return dots_design(self.subj)
        elif self.exp == "sticks":
            return sticks_design(self.subj)

    def signal_confound_design(self, run):
        """Build a matrix of signal confound variables."""
        analysis_dir = PROJECT["analysis_dir"]

        if self.exp == "dots":
            tr = 2
        elif self.exp == "sticks":
            tr = .72

        fstem = op.join(analysis_dir, self.exp, self.subj,
                        "preproc", "run_{}".format(run))

        motion = (pd.read_csv(op.join(fstem, "realignment_params.csv"))
                    .filter(regex="rot|trans")
                    .apply(stats.zscore))
        nuisance = (pd.read_csv(op.join(fstem, "nuisance_variables.csv"))
                      .filter(regex="wm_")
                      .apply(stats.zscore))
        artifacts = (pd.read_csv(op.join(fstem, "artifacts.csv"))
                       .any(axis=1))

        confounds = pd.concat([motion, nuisance], axis=1)
        dmat = glm.DesignMatrix(confounds=confounds,
                                artifacts=artifacts,
                                hpf_kernel=self.hpf_kernel,
                                tr=tr)

        return dmat.design_matrix

    def deconvolve(self):
        """Fit a model for each run to get condition amplitude estimates."""
        beta_list = []

        # Set experiment-specific variables
        if self.exp == "dots":
            tr = 2
            ntp = 230
            condition_names = ["context", "trial_type", "cue"]
            context_map = dict(motion=0, color=1)
        elif self.exp == "sticks":
            tr = .72
            ntp = 515
            condition_names = ["context", "cue", "ori_diff", "hue_diff"]
            context_map=dict(ori=0, hue=1)

        # Initialize the data and design
        design, info = self.design_info()
        all_conditions = design.condition.sort(inplace=False).unique()
        run_data = np.split(self.data, 12, axis=0)
        hrf_model = glm.GammaDifferenceHRF(temporal_deriv=True, tr=tr)

        # Keep track of voxels with nonzero variance
        assert np.unique([d.shape[0] for d in run_data]).size == 1
        n_voxels = run_data[0].shape[1]
        good_voxels = np.ones(n_voxels, np.bool)

        for run, run_design in design.groupby("run"):

            # Build the design matrix
            dmat = glm.DesignMatrix(run_design, hrf_model, ntp,
                                    hpf_kernel=self.hpf_kernel,
                                    condition_names=all_conditions,
                                    tr=tr)

            # Set up the regression variables
            X = dmat.design_matrix
            Y = pd.DataFrame(run_data[run - 1], index=X.index)

            # Regress signal confounds out of the design matrix
            # (They have already been removed from the data)
            signal_confounds = self.signal_confound_design(run)
            X = OLS(X, signal_confounds).fit().resid

            # Fit the experiment model and extract the condition betas
            betas = OLS(Y, X).fit().params.ix[info.ix[run].index]
            beta_list.append(betas)

            # Identify bad voxels
            good_voxels &= (Y.var(axis=0) > 0).values

        # Reformat the condition information by each variable
        conditions = pd.DataFrame((info.index
                                   .get_level_values("condition")
                                   .str
                                   .split("-")
                                   .tolist()), columns=condition_names)

        # Build the relevant objects for classification
        samples = conditions.index
        runs = pd.Series(info.index.get_level_values("run"), index=samples)
        betas = pd.concat(beta_list, ignore_index=True)
        rt = pd.Series(info["rt"].values, index=samples)
        y = conditions.context.map(context_map)

        # Remove null or single observation samples
        use = pd.Series(info["count"].values > 1, index=samples)
        runs, betas, rt, y = runs[use], betas[use], rt[use], y[use]

        # Assign instance attributes
        self.design = design
        self.betas = betas
        self.runs = runs
        self.rt = rt
        self.y = y
        self.good_voxels = good_voxels

    @property
    def X(self):

        # Scale the data across features
        X = stats.zscore(np.asarray(self.betas))

        # Remove zero-variance features
        X = X[:, self.good_voxels]
        assert not np.isnan(X).any()

        # Regress out behavioral confounds
        rt = np.asarray(self.rt)
        m, s = np.nanmean(rt), np.nanstd(rt)
        rt = np.nan_to_num((rt - m) / s)
        X = OLS(X, rt).fit().resid

        return X

    @property
    def X_df(self):

        return pd.DataFrame(self.X)

    def train_test_split(self, test_run):

        # Split and scale the decoding matrix
        X = np.asarray(self.betas)
        train_X, test_X = self.split_and_zscore(X, test_run)

        # Split the class labels
        train_y = np.asarray(self.y.loc[self.runs != test_run])
        test_y = np.asarray(self.y.loc[self.runs == test_run])

        # Split and scale the confound data
        train_rt, test_rt = self.split_and_zscore(self.rt, test_run)
        train_rt, test_rt = np.nan_to_num(train_rt), np.nan_to_num(test_rt)

        # Regress the behavioral confounds out of the data
        rt_beta = OLS(train_X, train_rt).fit().params
        train_X = train_X - np.outer(train_rt, rt_beta)
        test_X = test_X - np.outer(test_rt, rt_beta)

        assert len(train_X) == len(train_y)
        assert len(test_X) == len(test_y)

        return (train_X, train_y), (test_X, test_y)

    def split_and_zscore(self, data, test_run):

        # Enforse type and size of the data
        data = np.asarray(data)
        if data.ndim == 1:
            data = np.expand_dims(data, 1)

        # Identify training and test samples
        train = np.asarray(self.runs != test_run)
        test = np.asarray(self.runs == test_run)

        train_data = data[train]
        test_data = data[test]

        # Compute the mean and standard deviation of the training set
        m, s = np.nanmean(train_data), np.nanstd(train_data)

        # Scale the training and test set
        train_data = (train_data - m) / s
        test_data = (test_data - m) / s

        return train_data, test_data


def dots_design(subj):
    """Deconvolution information for the dots experiment."""
    data_dir = PROJECT["data_dir"]
    data_fname = op.join(data_dir, "dots_data.csv")
    df = pd.read_csv(data_fname).query("subj == @subj")

    # Add in the main conditions that will represent decoding samples
    df.loc[:, "condition"] = (df.context + "-" +
                              df.trial_type + "-" +
                              df.cue.astype(str))

    cond = md.build_condition_ev(df, "cue_onset", "condition", "model_dur")
    parts = [cond]

    # Add in regressors to control for RT confounds
    for part, part_df in df.groupby("context"):
        name_stem = part

        rt = md.build_parametric_ev(part_df.query("stim and answered"),
                                    "cue_onset",
                                    name_stem + "_rt",
                                    "rt",
                                    "model_dur")
        parts.append(rt)

    design = pd.concat(parts)

    # Compute the cell-wise trial_counts and mean rt
    info = condition_info(df)

    return design, info


def sticks_design(subj):
    """Deconvolution information for the sticks experiment."""
    data_dir = PROJECT["data_dir"]
    data_fname = op.join(data_dir, "sticks_data.csv")
    df = pd.read_csv(data_fname).query("subj == @subj")

    # Add in the main conditions that will represent decoding samples
    df.loc[:, "condition"] = (df.context + "-" +
                              df.cue_idx.astype(str) + "-" +
                              df.ori_diff + "-" +
                              df.hue_diff)
    trial_dur = df.rt.mean()

    cond = md.build_condition_ev(df, "stim_onset", "condition", trial_dur)
    parts = [cond]

    # Add in regressors to control for RT confounds
    for part, part_df in df.groupby(["context", "context_diff"]):
        name_stem = "_".join(part)

        rt = md.build_parametric_ev(part_df.query("answered"),
                                    "stim_onset",
                                    name_stem + "_rt",
                                    "rt",
                                    trial_dur)

        parts.append(rt)

    design = pd.concat(parts)

    # Compute the cell-wise trial_counts and mean rt
    info = condition_info(df)

    return design, info


def condition_info(df):

    runs = df.run.sort(inplace=False).unique()
    conditions = df.condition.sort(inplace=False).unique()
    idx = pd.MultiIndex.from_product([runs, conditions],
                                     names=["run", "condition"])
    counts = df.groupby("run").condition.value_counts().reindex(idx)
    rt = df.groupby(["run", "condition"]).rt.mean().reindex(idx)
    return pd.concat([counts, rt], keys=["count", "rt"], axis=1)


def decode(model, ds):

    scores = []
    weights = []

    for run in ds.runs.unique():
        splits = ds.train_test_split(run)
        (train_X, train_y), (test_X, test_y) = splits
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        scores.append(np.mean(test_y == pred_y))
        weights.append(len(test_y))

    return np.average(scores, weights=weights)


def permute_labels(ds, rs=None):
    """Return a copy of the dataset with target labels permuted within run."""
    if rs is None:
        rs = np.random.RandomState()
    ds = copy(ds)
    ds.y = ds.y.groupby(ds.runs).transform(rs.permutation)
    return ds


def compute_prefs(model, ds, cov=None):
    """Invert the decoding model to get voxel preferences."""
    if cov is None:
        cov = np.cov(ds.X.T)
    weights = model.fit(ds.X, ds.y).coef_.squeeze()
    return cov.dot(weights)


def percentile_score(null, real):
    """Vectorized function for computing percentile of score."""
    if np.isscalar(real):
        return stats.percentileofscore(null, real)

    percentiles = []
    assert len(null) == len(real)
    for null_i, real_i in zip(null, real):
        percentiles.append(stats.percentileofscore(null_i, real_i, "mean"))
    assert len(percentiles) == len(real)

    return np.array(percentiles)


if __name__ == "__main__":

    try:
        _, subj, exp, roi = sys.argv
    except ValueError:
        sys.exit("Usages: decoding_analysis.py <subj> <exp> <roi>")

    # Ensure that the output exists
    if not op.exists("decoding_analysis"):
        os.mkdir("decoding_analysis")

    # Load the data
    data_fname = "roi_cache/{}_{}_{}.npz".format(subj, exp, roi)
    data = np.load(data_fname)["ts_data"]

    # Build the dataset
    ds = Dataset(exp, subj, data)

    # Compute the feature covariance
    cov = np.cov(ds.X.T)

    # Obtain the real decoding accuracy and preferences
    model = LogisticRegression()
    acc = decode(model, ds)
    prefs = compute_prefs(model, ds, cov)

    # Obtain null distributions for accuracy and prefs
    null_accs = []
    null_prefs = []
    seed = sum(map(ord, subj + exp + "decoding"))
    rs = np.random.RandomState(seed)
    for _ in xrange(100):
        ds_perm = permute_labels(ds, rs)
        null_accs.append(decode(model, ds_perm))
        null_prefs.append(compute_prefs(model, ds_perm, cov))

    # Evaluate the results relative to null distributions
    null_accs = np.array(null_accs)
    null_prefs = np.array(null_prefs).T
    acc_pctile = percentile_score(null_accs, acc)
    pref_pctiles = percentile_score(null_prefs, prefs)

    # Identify tails of the preference distribution
    tails = np.zeros_like(prefs, np.int)
    tails[pref_pctiles < 10] = -1
    tails[pref_pctiles > 90] = 1

    # Save the results
    res = moss.Results(acc=acc,
                       null=null_accs,
                       acc_pctile=acc_pctile,
                       chance=null_accs.mean(),
                       prefs=prefs,
                       tails=tails,
                       pref_pctiles=pref_pctiles,
                       good_voxels=ds.good_voxels)

    fname = "decoding_analysis/{}_{}_{}.pkz".format(subj, exp, roi)
    moss.save_pkl(fname, res)
