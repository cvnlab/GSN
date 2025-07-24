"""
Do held-out subject analysis of number of PCs needed to explain the variance in the test data.
Do SVD on one trial, and test on the other two trials.
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

save = False

use_repwise_gsn = True
shuffle_reps = True
n_trials_for_test = 1 # 2 or 1
filter_out_low_ncsnr = 0.45 # if not None, filter out test voxels with ncsnr < filter_out_low_ncsnr
flip_sign = True
parc = 4
if parc in [1, 2, 3]:
    parc_col = 'external_parc'
elif parc in [4, 5]:
    parc_col = 'parc_lang'
elif parc in [24]:
    parc_col = 'parc_glasser'
uids = ['cvn7009', 'cvn7012', 'cvn7002', 'cvn7011', 'cvn7007', 'cvn7006', 'cvn7013', 'cvn7016']


scores_train_over_uids = []
scores_test_over_uids = []
nc_over_uids = []
for uid in uids:
    print(f'uid: {uid}')

    # For the subject, load their voxel data
    sess = d_uid_to_sess[uid]
    datafn = os.path.join(SUBJECTDIR, f'{sess}-{uid}', 'GLMestimatesingletrialoutputs', 'extracted_voxs',
                          f'lh_baseline200_{sess}-{uid}_extracted-voxs-{parc_col}_parcs-{parc}.pkl')
    if shuffle_reps:
        datafn = datafn.replace('.pkl', '_shuffled-reps.pkl')

    d = pickle.load(open(datafn, 'rb'))
    data = d['betas_parc_3d'] # (vertices, stimuli, trials)

    # Load the GSN pickle to get the signal mean (mnS)
    if parc in [1, 2, 3, 24]:
        # lh_baseline200_20240508-ST001-cvn7013_extracted-voxs-external_parc_parcs-1_permute-False.pkl
        gsnfn = os.path.join(GSNOUTPUTS, f'lh_baseline200_{sess}-{uid}_extracted-voxs-{parc_col}_parcs-{parc}_permute-False.pkl')
    elif parc in [4, 5]: # lh_baseline200_20240508-ST001-cvn7013_extracted-voxs-parc_lang_parcs-5_gsn_outputs.pkl
        gsnfn = os.path.join(GSNOUTPUTS, f'lh_baseline200_{sess}-{uid}_extracted-voxs-{parc_col}_parcs-{parc}_gsn_outputs.pkl')

    results = pickle.load(open(gsnfn, 'rb'))

    # mnS contains the signal mean per voxel. subtract from data_test
    mnS_reshaped = results['mnS'].reshape(-1, 1, 1)  # reshaping to (vertices, 1, 1)
    data_subtracted = data - mnS_reshaped

    ncsnr = results['ncsnr']
    # Filter out voxels with low ncsnr
    if filter_out_low_ncsnr is not None:
        data_subtracted2 = data_subtracted[ncsnr > filter_out_low_ncsnr, :, :]
        print(f'Filtering out {data_subtracted.shape[0] - data_subtracted2.shape[0]} voxels with ncsnr < {filter_out_low_ncsnr}')
        ncsnr = ncsnr[ncsnr > filter_out_low_ncsnr]
        data_subtracted = data_subtracted2

    # Compute the nc based of off ncsnr: nc = ncsnr^2 / (ncsnr^2 + 1/n) * 100 where n is the number of trials in the test, i.e., 2
    n = 2
    nc = (ncsnr ** 2) / (ncsnr ** 2 + 1/n) * 100 # This gets the nc of each voxel (post filtering)
    nc_mean_over_voxs = np.mean(nc)
    nc_over_uids.append(nc_mean_over_voxs)

    # Separate the data_test_subtracted into test_train_trials (trials 1) and test_test_trials (trial 2 and 3). Iterate over combinations of trials
    if n_trials_for_test == 2:
        train_trials = [0, 1, 2]
        test_trials = [[1, 2], [0, 2], [0, 1]]
    elif n_trials_for_test == 1:
        train_trials = [[0, 1], [0, 2], [1, 2]]
        test_trials = [2, 1, 0]

    scores_train_over_trial_combos = []
    scores_test_over_trial_combos = []
    for train_trial, test_trial in zip(train_trials, test_trials):

        uid_train_trials = data_subtracted[:, :, train_trial]  # (vertices, stimuli)
        uid_test_trials = data_subtracted[:, :, test_trial]  # (vertices, stimuli, 2)
        # Average over the two test trials
        if n_trials_for_test == 2:
            uid_test_trials = np.mean(uid_test_trials, axis=2)  # (vertices, stimuli)
        elif n_trials_for_test == 1:  # instead average over the train trials
            uid_train_trials = np.mean(uid_train_trials, axis=2)  # (vertices, stimuli)

        if use_repwise_gsn:
            # We want to use the projections for the train data, based on two trials (n_trials_for_test = 1)
            assert n_trials_for_test == 1
            # Load the subjects stimulus projections (200 x 20)
            projfname = f'{uid}_lh_{parc}_proj_20_flip-{flip_sign}_reps-{train_trial[0]}-{train_trial[1]}.pkl'
            projfn = os.path.join(PROJOUTPUTDIR, projfname)
            proj = pickle.load(open(projfn, 'rb'))
            U = proj # (200, 20)
        else:
            # Just use SVD on the data
            # Run linear regression on the train trials: 1 trial data, do SVD. Get the U matrix. Iterate over the number of PCs.
            # We want to do SVD on stimuli x vertices, so we need to transpose the data
            U, S, Vt = np.linalg.svd(uid_train_trials.T, full_matrices=False)

        # Just average over the voxels in the test data
        # uid_test_trials = np.mean(uid_test_trials, axis=0)  # (stimuli) # make it (1, stimuli) for the function
        # uid_test_trials = uid_test_trials.reshape(1, -1)
        n_vox = uid_test_trials.shape[0]

        # Also average over the voxels in the train data
        # uid_train_trials = np.mean(uid_train_trials, axis=0)  # (stimuli) # make it (1, stimuli) for the function
        # uid_train_trials = uid_train_trials.reshape(1, -1)

        scores_train = []
        scores_test = []
        for j in range(0, 20):
            # Take out the jth PC from U
            U_j = U[:, :j+1] # design matrix is 200 by j

            # Instead of running regression on all voxels, run on each voxel separately
            scores_train_vox = []
            scores_test_vox = []
            for i in range(n_vox):
                # reg = LinearRegression(fit_intercept=False).fit(U_j, uid_train_trials[i, :])
                # score_test = reg.score(U_j, uid_test_trials[i, :]) * 100
                # # scores_test_vox.append(score_test)
                # score_train = reg.score(U_j, uid_train_trials[i, :]) * 100
                # # scores_train_vox.append(score_train)

                # Use the function: fit_and_evaluate
                score_test = fit_and_evaluate(U_j=U_j, train_trials=uid_train_trials[i, :], test_trials=uid_test_trials[i, :])
                scores_test_vox.append(score_test)
                score_train = fit_and_evaluate(U_j=U_j, train_trials=uid_train_trials[i, :], test_trials=uid_train_trials[i, :])
                scores_train_vox.append(score_train)
                # assert np.isclose(score_test, score_test2)

            # Append mean over voxels
            # scores_train.append(np.mean(scores_train_vox))
            # scores_test.append(np.mean(scores_test_vox))

            # Append all voxels
            scores_train.append(scores_train_vox)
            scores_test.append(scores_test_vox)

        scores_train_over_trial_combos.append(scores_train)
        scores_test_over_trial_combos.append(scores_test)

    # Flatten out the lists scores_train_over_trial_combos and scores_test_over_trial_combos
    # There are three rep combos, and then 20 PCs

    plt.figure()
    for i, (scores_train, scores_test) in enumerate(zip(scores_train_over_trial_combos, scores_test_over_trial_combos)):
        # scores_train now has 20 lists, each with n_vox elements. so we need to find the voxel within each of the 20 overall lists
        for v in range(n_vox):
            # plt.plot(range(1, 21), [scores_train[j][v] for j in range(20)], label=f'Train Trial {train_trials[i]}')
            plt.plot(range(1, 21), [scores_test[j][v] for j in range(20)], label=f'Test Trial {test_trials[i]}',
                     linewidth=0.5, alpha=0.3)
        # plt.plot(range(1, 21), scores_train, label=f'Train Trial {train_trials[i]}')
        # plt.plot(range(1, 21), scores_test, label=f'Test Trial {test_trials[i]}')
    # Add a horizontal line for the noise ceiling
    plt.axhline(y=nc_mean_over_voxs, color='black', linestyle='--', label='NC')
    plt.xlabel('Number of PCs')
    plt.xticks(range(1, 21))
    plt.ylabel('R^2')
    plt.title(f'Parc {parc}, test UID {uid}, filter: {filter_out_low_ncsnr}, '
              f'n_trials_for_test: {n_trials_for_test}, shuffle_reps: {shuffle_reps}, repwise: {use_repwise_gsn}',
                fontsize=10)
    # plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUTDIR, f'train_test_R2_{uid}_lh_{parc}_filter-{filter_out_low_ncsnr}_n_trials_for_test-{n_trials_for_test}_shuffle_reps-{shuffle_reps}_repwise-{use_repwise_gsn}.png'))
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
    plt.plot(range(1, 21), scores_train, label=f'Train split of data UID {uids[i]}', alpha=0.2, color=d_uid_color[uids[i]], linestyle='solid')
    plt.plot(range(1, 21), scores_test, label=f'Test split of data UID {uids[i]}', alpha=0.2, color=d_uid_color[uids[i]], linestyle='dotted')
    # Add a horizontal line for the noise ceiling, in each subjects color
    plt.axhline(y=nc_over_uids[i], color=d_uid_color[uids[i]], linestyle='--', label=f'NC UID {uids[i]}')

plt.plot(range(1, 21), np.mean(scores_train_over_uids, axis=0), label='Train Grand Mean', color='black')
plt.plot(range(1, 21), np.mean(scores_test_over_uids, axis=0), label='Test Grand Mean', color='black', linestyle='dotted')
plt.axhline(y=np.mean(nc_over_uids), color='black', linestyle='--', label='NC Grand Mean')
plt.xlabel('Number of PCs')
plt.xticks(range(1, 21))
plt.ylabel('R^2')
plt.title(f'Parc {parc}, filter: {filter_out_low_ncsnr}, n_trials_for_test: {n_trials_for_test}, shuffle_reps: {shuffle_reps}, repwise: {use_repwise_gsn}',
            fontsize=10)
# plt.legend()
if save:
    plt.savefig(os.path.join(OUTPUTDIR, f'train_test_R2_across_uids_lh_{parc}_filter-{filter_out_low_ncsnr}_n_trials_for_test-{n_trials_for_test}_shuffle_reps-{shuffle_reps}_repwise-{use_repwise_gsn}.png'))
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