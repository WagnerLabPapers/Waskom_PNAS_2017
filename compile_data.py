import numpy as np
import pandas as pd

import moss
import lyman


if __name__ == "__main__":

    subjects = dict(dots=lyman.determine_subjects(["dots_subjects"]),
                    sticks=lyman.determine_subjects(["sticks_subjects"]))

    # ----- Combine decoding results

    rois = ["ifs", "mfc"]

    decoding_data = {}
    for exp, subj_list in subjects.iteritems():

        # Set up the dataframe for this experiment
        idx = pd.MultiIndex.from_product([subj_list, rois],
                                         names=["subj", "roi"])
        exp_df = pd.DataFrame(index=idx,
                              columns=["acc", "chance", "pctile"],
                              dtype=np.float)

        # Load the data for each subject/roi
        for subj, roi in idx:
            fname = "decoding_analysis/{}_{}_{}.pkz".format(subj, exp, roi)
            res = moss.load_pkl(fname)
            exp_df.ix[subj, roi] = [res.acc, res.chance, res.acc_pctile]
        decoding_data[exp] = exp_df

    # Combine across experiments and save
    decoding_df = pd.concat(decoding_data, names=["experiment"]).reset_index()
    decoding_df.to_csv("data/decoding_results.csv", index=False)

    # ----- Combine correlation results

    subjects["rest"] = subjects["sticks"]
    corr_dfs = {}
    for exp in ["dots", "sticks", "rest"]:

        subj_list = subjects[exp]
        exp_data = []

        for subj in subj_list:
            fname = "correlation_analysis/{}_{}_ifs.pkz".format(subj, exp)
            exp_data.append(moss.load_pkl(fname).tail_corrs)

        corr_dfs[exp] = pd.DataFrame(exp_data,
                                     index=subj_list,
                                     columns=["within", "between"])

    corr_df = pd.concat(corr_dfs, names=["exp", "subj"]).reset_index()
    corr_df.to_csv("data/correlation_results.csv", index=False)
