"""
We want to take 7 subjects, and load their stimulus projections from GSN (computed via load_gsn_results.py) using flag compute_pcs.
If we have n=20 PCs for each subject, we will have 7 * 20 = 140 projections (200 x 140 matrix).
For the remaining participant, load their voxel data and separate into trials.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
import itertools
from sklearn.linear_model import LinearRegression

d_uid_to_sess = {'cvn7009': '20231202-ST001',
                    'cvn7012': '20231202-ST001',
                    'cvn7002': '20231203-ST001',
                    'cvn7011': '20231204-ST001',
                    'cvn7007': '20231204-ST001',
                    'cvn7006': '20231215-ST001',
                    'cvn7013': '20240508-ST001',
                    'cvn7016': '20240530-ST001',}

def fit_and_evaluate(U_j, train_trials, test_trials):
    """
    Fits a linear regression model without an intercept to the training data using the normal equation,
    reconstructs the training data using the learned coefficients, and evaluates the model on test data
    by computing the R^2 score.

    Parameters:
        U_j (ndarray): The feature matrix (independent variables) used for fitting the model.
        train_trials (ndarray): The target values for training (dependent variables).
        test_trials (ndarray): The target values for testing.

    Returns:
        score_test (float): The R^2 score of the model on the test data, multiplied by 100.
    """
    # Reshape train_trials if necessary (assuming train_trials is a vector)
    if train_trials.ndim == 1:
        train_trials = train_trials.reshape(-1, 1)
    if test_trials.ndim == 1:
        test_trials = test_trials.reshape(-1, 1)

    # Fit the model using the normal equation
    # U_j.T * U_j * coef_ = U_j.T * train_trials
    coef_ = np.linalg.solve(U_j.T @ U_j, U_j.T @ train_trials)

    # Reconstruct the training data using the learned coefficients
    recons = U_j @ coef_

    # Calculate the R^2 score for the test data
    total_variance = np.sum((test_trials - np.mean(test_trials))**2)
    residual_variance = np.sum((test_trials - recons)**2)
    score_test = (1 - residual_variance / total_variance) * 100

    return score_test

SUBJECTDIR = '/Users/gt/Library/CloudStorage/GoogleDrive-gretatu@mit.edu/My Drive/Research2020/Sentence7T/FMRI_FSAVERAGE/'
GSNOUTPUTS = '/Users/gt/Library/CloudStorage/GoogleDrive-gretatu@mit.edu/My Drive/Research2020/Sentence7T/GSN_outputs/'
PROJOUTPUTDIR = '/Users/gt/Library/CloudStorage/GoogleDrive-gretatu@mit.edu/My Drive/Research2020/Sentence7T/GSN_outputs/outputs/'
OUTPUTDIR = os.path.join(GSNOUTPUTS, 'gsn_generalization')
if not os.path.exists(OUTPUTDIR):
    os.makedirs(OUTPUTDIR)

save = True

flip_sign = True # Load the sign-flipped stimulus projections
unit_norm_projs = True # If True, normalize the stimulus projections to have unit norm per subject
filter_out_low_ncsnr = 0.4 # if not None, filter out test voxels with ncsnr < filter_out_low_ncsnr
parc = 3
if parc in [1, 2, 3]:
    parc_col = 'external_parc'
elif parc in [4, 5]:
    parc_col = 'parc_lang'
uids = ['cvn7009', 'cvn7012', 'cvn7002', 'cvn7011', 'cvn7007', 'cvn7006', 'cvn7013', 'cvn7016']

# For all combinations of 7 train subjects, 1 test subject
train_uid_combs = list(itertools.combinations(uids, 7))

scores_train_over_uids = []
scores_test_over_uids = []
nc_over_uids = []
for train_uids in train_uid_combs:
    test_uid = list(set(uids) - set(train_uids))
    print(f'Train: {train_uids}, Test: {test_uid}')

    # For each train subject, load their stimulus projections (200 x 20)
    train_projs = []
    for train_uid in train_uids:
        fname = f'{train_uid}_lh_{parc}_proj_20_flip-{flip_sign}.pkl'
        projfn = os.path.join(PROJOUTPUTDIR, fname)
        proj = pickle.load(open(projfn, 'rb'))
        if unit_norm_projs: # Normalize the stimulus projections to have unit norm
            proj = proj / np.linalg.norm(proj) # now this chunk of 200x20 is unit norm
        train_projs.append(proj)

    # Get a matrix of all train projections (200 x 140). Stack them
    train_projs = np.hstack(train_projs)
    assert train_projs.shape == (proj.shape[0], len(train_uids) * proj.shape[1])

    # Run SVD on train_projs
    U, S, VT = np.linalg.svd(train_projs, full_matrices=False)
    V = VT.T # python returns the transpose

    # # scatter of S
    # plt.figure()
    # plt.scatter(range(1, len(S) + 1), S)
    # plt.xlabel('PC')
    # plt.ylabel('Singular Value')
    # plt.show()

    # For the test subject, load their voxel data
    assert len(test_uid) == 1
    test_uid = test_uid[0]
    test_sess = d_uid_to_sess[test_uid]
    datafn_test = os.path.join(SUBJECTDIR, f'{test_sess}-{test_uid}', 'GLMestimatesingletrialoutputs', 'extracted_voxs',
                          f'lh_baseline200_{test_sess}-{test_uid}_extracted-voxs-{parc_col}_parcs-{parc}.pkl')
    d_test = pickle.load(open(datafn_test, 'rb'))
    data_test = d_test['betas_parc_3d'] # (vertices, stimuli, trials)

    # Load the GSN pickle to get the signal mean (mnS)
    gsnfn_test = os.path.join(GSNOUTPUTS, f'{test_uid}_lh_{parc}_gsn.pkl')
    if parc in [1, 2, 3]:
        # lh_baseline200_20240508-ST001-cvn7013_extracted-voxs-external_parc_parcs-1_permute-False.pkl
        gsnfn_test = os.path.join(GSNOUTPUTS, f'lh_baseline200_{test_sess}-{test_uid}_extracted-voxs-{parc_col}_parcs-{parc}_permute-False.pkl')
    elif parc in [4, 5]: # lh_baseline200_20240508-ST001-cvn7013_extracted-voxs-parc_lang_parcs-5_gsn_outputs.pkl
        gsnfn_test = os.path.join(GSNOUTPUTS, f'lh_baseline200_{test_sess}-{test_uid}_extracted-voxs-{parc_col}_parcs-{parc}_gsn_outputs.pkl')

    results_test = pickle.load(open(gsnfn_test, 'rb'))

    # mnS contains the signal mean per voxel. subtract from data_test
    mnS_reshaped = results_test['mnS'].reshape(-1, 1, 1)  # reshaping to (vertices, 1, 1)
    data_test_subtracted = data_test - mnS_reshaped

    ncsnr = results_test['ncsnr']
    # Filter out voxels with low ncsnr
    if filter_out_low_ncsnr is not None:
        data_test_subtracted2 = data_test_subtracted[ncsnr > filter_out_low_ncsnr, :, :]
        print(f'Filtering out {data_test_subtracted.shape[0] - data_test_subtracted2.shape[0]} voxels with ncsnr < {filter_out_low_ncsnr}')
        ncsnr2 = ncsnr[ncsnr > filter_out_low_ncsnr]

        # Visualize the ncsnr of the voxels, pre and post filtering
        # plt.figure()
        # plt.hist(ncsnr, bins=50, alpha=0.5, label='Pre-Filtering')
        # plt.hist(ncsnr2, bins=50, alpha=0.5, label='Post-Filtering')
        # plt.xlabel('Noise Ceiling SNR')
        # plt.ylabel('Frequency')
        # plt.legend()
        # plt.show()

        data_test_subtracted = data_test_subtracted2
        ncsnr = ncsnr2

    # Compute the nc based of off ncsnr: nc = ncsnr^2 / (ncsnr^2 + 1/n) * 100 where n is the number of trials in the test, i.e., 2
    n = 2
    nc = (ncsnr ** 2) / (ncsnr ** 2 + 1/n) * 100 # This gets the nc of each voxel (post filtering)
    nc_mean_over_voxs = np.mean(nc)
    nc_over_uids.append(nc_mean_over_voxs)

    # # Separate the data_test_subtracted into test_train_trials (trials 1 and 2) and test_test_trials (trial 3)
    # test_uid_train_trials = data_test_subtracted[:, :, :2]
    # test_uid_train_trials = np.mean(test_uid_train_trials, axis=2)     # Average over the two trials
    # test_uid_test_trials = data_test_subtracted[:, :, 2]

    # Separate the data_test_subtracted into test_train_trials (trials 1) and test_test_trials (trial 2 and 3). Iterate over combinations of trials
    train_trials = [0, 1, 2]
    test_trials = [[1, 2], [0, 2], [0, 1]]

    scores_train_over_trial_combos = []
    scores_test_over_trial_combos = []
    for train_trial, test_trial in zip(train_trials, test_trials):
        test_uid_train_trials = data_test_subtracted[:, :, train_trial]
        test_uid_test_trials = data_test_subtracted[:, :, test_trial]
        # Average over the two test trials
        test_uid_test_trials = np.mean(test_uid_test_trials, axis=2)

        # Run linear regression on the train trials

        # 1 trial data, do SVD. Get the U matrix. Iterate over the number of PCs.


        scores_train = []
        scores_test = []
        for j in range(0, 20):
            # Take out the jth PC from U
            U_j = U[:, :j+1] # design matrix is 200 by j

            # reg = LinearRegression(fit_intercept=False).fit(U_j, test_uid_train_trials.T)
            # recons = U_j @ reg.coef_.T

            # score_test = reg.score(U_j, test_uid_test_trials.T) * 100
            # scores_test.append(score_test)
            #
            # score_train = reg.score(U_j, test_uid_train_trials.T) * 100
            # scores_train.append(score_train)
            #
            # print(f'PCs: {j+1}, Score: {score_test}')
            #
            # scores_train_over_trial_combos.append(scores_train)
            # scores_test_over_trial_combos.append(scores_test)

            # Instead of running regression on all voxels, run on each voxel separately
            scores_train_vox = []
            scores_test_vox = []
            for i in range(test_uid_train_trials.shape[0]):
                reg = LinearRegression(fit_intercept=False).fit(U_j, test_uid_train_trials[i, :])
                score_test = reg.score(U_j, test_uid_test_trials[i, :]) * 100
                scores_test_vox.append(score_test)
                score_train = reg.score(U_j, test_uid_train_trials[i, :]) * 100
                scores_train_vox.append(score_train)

                # Use the function: fit_and_evaluate
                score_test2 = fit_and_evaluate(U_j, test_uid_train_trials[i, :], test_uid_test_trials[i, :])
                assert np.isclose(score_test, score_test2)

            scores_train.append(np.mean(scores_train_vox))
            scores_test.append(np.mean(scores_test_vox))

        scores_train_over_trial_combos.append(scores_train)
        scores_test_over_trial_combos.append(scores_test)

    plt.figure()
    for i, (scores_train, scores_test) in enumerate(zip(scores_train_over_trial_combos, scores_test_over_trial_combos)):
        plt.plot(range(1, 21), scores_train, label=f'Train Trial {train_trials[i]}')
        plt.plot(range(1, 21), scores_test, label=f'Test Trial {test_trials[i]}')
    # Add a horizontal line for the noise ceiling
    plt.axhline(y=nc_mean_over_voxs, color='black', linestyle='--', label='NC')
    plt.xlabel('Number of PCs')
    plt.xticks(range(1, 21))
    plt.ylabel('R^2')
    plt.title(f'Parc {parc}, test UID {test_uid}, filter: {filter_out_low_ncsnr}, norm: {unit_norm_projs}')
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUTDIR, f'train_test_R2_{test_uid}_lh_{parc}_flip-{flip_sign}_filter-{filter_out_low_ncsnr}_norm-{unit_norm_projs}.png'))
    plt.show()

    # Average over the 3 trials and store across people
    scores_train_over_uids.append(np.mean(scores_train_over_trial_combos, axis=0))
    scores_test_over_uids.append(np.mean(scores_test_over_trial_combos, axis=0))

# Plot across uids
d_uid_color = {
    'cvn7009': 'red',
    'cvn7012': 'blue',
    'cvn7002': 'green',
    'cvn7011': 'purple',
    'cvn7007': 'orange',
    'cvn7006': 'brown',
    'cvn7013': 'pink',
    'cvn7016': 'gray'
}

plt.figure()
for i, (scores_train, scores_test) in enumerate(zip(scores_train_over_uids, scores_test_over_uids)):
    test_uid = list(set(uids) - set(train_uid_combs[i]))
    plt.plot(range(1, 21), scores_train, label=f'Train UID {train_uid_combs[i]}', alpha=0.2, color=d_uid_color[test_uid[0]])
    plt.plot(range(1, 21), scores_test, label=f'Test UID {test_uid}', alpha=0.2, color=d_uid_color[test_uid[0]], linestyle='dotted')
    # Add a horizontal line for the noise ceiling, in each subjects color
    plt.axhline(y=nc_over_uids[i], color=d_uid_color[test_uid[0]], linestyle='--', label='NC')

plt.plot(range(1, 21), np.mean(scores_train_over_uids, axis=0), label='Train Grand Mean', color='black')
plt.plot(range(1, 21), np.mean(scores_test_over_uids, axis=0), label='Test Grand Mean', color='black', linestyle='dotted')
plt.axhline(y=np.mean(nc_over_uids), color='black', linestyle='--', label='NC Grand Mean')
plt.xlabel('Number of PCs')
plt.xticks(range(1, 21))
plt.ylabel('R^2')
plt.title(f'Parc {parc}, filter: {filter_out_low_ncsnr}, norm: {unit_norm_projs}')
# plt.legend()
if save:
    plt.savefig(os.path.join(OUTPUTDIR, f'train_test_R2_across_uids_lh_{parc}_flip-{flip_sign}_filter-{filter_out_low_ncsnr}_norm-{unit_norm_projs}.png'))
plt.show()







print('Done!')






    # ## test plot to viz the subtraction
    # vertex_index = 6
    # num_trials = 3
    # fig, axes = plt.subplots(1, num_trials, figsize=(5 * num_trials, 5))
    # # Check if only one trial is there to handle subplot indexing
    # if num_trials == 1:
    #     axes = [axes]
    # for i in range(num_trials):
    #     # Scatter plot for each trial comparing pre and post-subtraction
    #     axes[i].scatter(data_test[vertex_index, :, i], data_test_subtracted[vertex_index, :, i], alpha=0.5)
    #     axes[i].set_title(f'Vertex {vertex_index} Trial {i}')
    #     axes[i].set_xlabel('Pre-Subtraction Values')
    #     axes[i].set_ylabel('Post-Subtraction Values')
    #     axes[i].plot([-10, 10], [-10, 10], 'k--')
    # plt.tight_layout()
    # plt.show()