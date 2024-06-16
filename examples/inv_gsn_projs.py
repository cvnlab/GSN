import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
import pandas as pd

PLOTDIR = '/Users/gt/Library/CloudStorage/GoogleDrive-gretatu@mit.edu/My Drive/Research2020/Sentence7T/GSN_outputs/plots/'
PROJOUTPUTDIR = '/Users/gt/Library/CloudStorage/GoogleDrive-gretatu@mit.edu/My Drive/Research2020/Sentence7T/GSN_outputs/outputs/'
files = glob.glob(f'{PROJOUTPUTDIR}*.pkl')

# Load GSN results into a dict with keys as the filenames
results = {}
for f in files:
    fname = f.split('/')[-1].split('.')[0]
    results[fname] = pickle.load(open(f, 'rb'))

# Load a stimset from a random subject
stimset_fname = '/Users/gt/Library/CloudStorage/GoogleDrive-gretatu@mit.edu/My Drive/Research2020/Sentence7T/FMRI_FSAVERAGE/20231202-ST001-cvn7009/GLMestimatesingletrialoutputs/extracted_voxs/lh_baseline200_20231202-ST001-cvn7009_extracted-voxs-external_parc_parcs-1.pkl'
d_stimset = pickle.load(open(stimset_fname, 'rb'))
# for each of stimset_rep{1,2,3}_ordered, check that the item_id col is the same
assert np.all(d_stimset['stimset_rep1_ordered']['item_id'].values == d_stimset['stimset_rep2_ordered']['item_id'].values)
assert np.all(d_stimset['stimset_rep1_ordered']['item_id'].values == d_stimset['stimset_rep3_ordered']['item_id'].values)
stimset = d_stimset['stimset_rep1_ordered']


# Define the linguistic properties (features) of interest
feats_of_interest = [
    'surprisal-gpt2-xl_raw_mean',
    'surprisal-5gram_raw_mean',
    'surprisal-pcfg_raw_mean',
    'rating_gram_mean',
    'rating_sense_mean',
    #
    'rating_others_thoughts_mean',
    'rating_physical_mean',
     'rating_places_mean',
    #
    'rating_valence_mean',
    'rating_arousal_mean',
    #
    'rating_imageability_mean',
    #
    'rating_frequency_mean',
    'rating_conversational_mean',

    'mean_cos_sim_to_clusters',
]


#
# # For each GSN result, correlate the linguistic properties with the first principal component
# for f, result in results.items():
#     # Correlate the linguistic properties with the first principal component
#     fname = f.split('/')[-1].split('.')[0]
#
#     for feat in feats_of_interest:
#         r = np.corrcoef(result[:, 0], stimset[feat])[0, 1]
#         print(f'{fname} {feat} corr: {r:.2f}')

# Compute the cosine between the projections of projection 1 for all subjects, across ROIs, and make a gigantic heatmap
uid_order = ['cvn7009', 'cvn7012', 'cvn7002', 'cvn7011', 'cvn7007', 'cvn7006', 'cvn7013', 'cvn7016']
parc_order = [1, 2, 3, 4, 5]

# proj_ns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
#
# # Get the projections for all subjects
# projs = []
# # Loop across uids, and then parcs
# labels = []
# for proj_n in proj_ns:
#     for uid in uid_order:
#         for parc in parc_order:
#             # e.g., cvn7012_lh_4_proj_20
#             results_key = f'{uid}_lh_{parc}_proj_20'
#             result = results[results_key]
#             proj = result[:, proj_n]
#             projs.append(proj)
#             labels.append(f'{proj_n}_{uid}_{parc}')
#
# # Normalize the projections
# projs_norm = [proj / np.linalg.norm(proj) for proj in projs]
#
# # Compute the cosine similarity between all pairs of projections
# cos_sim = np.zeros((len(projs), len(projs)))
#
# for i, proj1 in enumerate(projs_norm):
#     for j, proj2 in enumerate(projs_norm):
#         cos_sim[i, j] = np.dot(proj1, proj2)
#
#
# plt.figure(figsize=(35, 35))
# ax = plt.imshow(abs(cos_sim), cmap='RdBu_r', vmin=-1, vmax=1)
# plt.xticks(np.arange(0.5, len(projs), 1), labels, rotation=90,
#               horizontalalignment='center', fontsize=3)
# plt.yticks(np.arange(0.5, len(projs), 1), labels, rotation=0, horizontalalignment='right',
#            fontsize=3)
# plt.colorbar(ax, shrink=0.5)
# plt.tight_layout()
# # save as pdf
# plt.savefig(f'{PLOTDIR}cosine_similarity_proj_{np.min(proj_ns)}-{np.max(proj_ns)}.pdf')
# plt.show()

# For a given uid, correlate their projections with each other
for uid in uid_order:
    projs = []
    for parc in parc_order:
        results_key = f'{uid}_lh_{parc}_proj_20'
        result = results[results_key] # 200 x 20

        df = pd.DataFrame(result).set_index(stimset.index)
        # concat with the stimset
        df = pd.concat([df, stimset[feats_of_interest]], axis=1)

        # Correlate the projections with each other
        corrs = df.corr()
        plt.figure(figsize=(10, 8))
        plt.imshow(corrs, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='none')
        plt.colorbar()
        plt.title(f'{uid} {parc}')
        plt.show()



# For each top projection, average across subjects for a given parc
# and correlate with linguistic properties
proj_n = 2
d_avg_projs = {}
for parc in parc_order:
    projs_parc = []
    for uid in uid_order:
        results_key = f'{uid}_lh_{parc}_proj_20'
        result = results[results_key]
        proj = result[:, proj_n] # top
        projs_parc.append(proj)
    avg_proj = np.mean(projs_parc, axis=0)
    d_avg_projs[f'{parc}'] = avg_proj

# For each linguistic property, correlate with the average projection. Plot as bars
for feat in feats_of_interest:
    for parc in parc_order:
        r = np.corrcoef(d_avg_projs[f'{parc}'], stimset[feat])[0, 1]
        print(f'{feat} {parc} corr: {r:.2f}')

# Make figure with bars, for each roi and linguistic property. As subplots
fig, axs = plt.subplots(4, 4, figsize=(12, 8), constrained_layout=True)
axs = axs.flatten()
for i, feat in enumerate(feats_of_interest):
    for parc in parc_order:
        r = np.corrcoef(d_avg_projs[f'{parc}'], stimset[feat])[0, 1]
        axs[i].bar(parc, r)
        axs[i].set_title(feat)
        axs[i].set_xlabel('ROI')
        axs[i].set_ylabel('Correlation')
        # fix limits
        axs[i].set_ylim(-0.5, 0.7)
plt.show()

# For each parcel, plot the correlation across a range of projections across all features
proj_ns = np.arange(0, 20)
d_projs = {}
for parc in parc_order:
    projs_parc = []
    for proj_n in proj_ns:
        for uid in uid_order:
            results_key = f'{uid}_lh_{parc}_proj_20'
            result = results[results_key]
            proj = result[:, proj_n]
            projs_parc.append(proj)

        d_projs[f'{parc}_{proj_n}'] = np.mean(projs_parc, axis=0)

# Plot the correlations, for each roi, make separate plot, and show correlations across all linguistic properties across projections
for parc in parc_order:
    fig, axs = plt.subplots(4, 4, figsize=(12, 8), constrained_layout=True)
    axs = axs.flatten()
    for i, feat in enumerate(feats_of_interest):
        for proj_n in proj_ns:
            r = np.corrcoef(d_projs[f'{parc}_{proj_n}'], stimset[feat])[0, 1]
            axs[i].bar(proj_n, r)
            axs[i].set_title(feat)
            axs[i].set_xlabel('Projection')
            axs[i].set_ylabel('Correlation')
            # fix limits
            axs[i].set_ylim(-0.5, 0.7)
    plt.suptitle(f'ROI {parc}', fontsize=20)
    plt.show()

# Do this per subject, don't average across subjects
d_projs = {}
for uid in uid_order:
    for parc in parc_order:
        projs_parc = []
        for proj_n in proj_ns:
            results_key = f'{uid}_lh_{parc}_proj_20'
            result = results[results_key]
            proj = result[:, proj_n]
            projs_parc.append(proj)

            d_projs[f'{uid}_{parc}_{proj_n}'] = np.mean(projs_parc, axis=0)

# Plot the correlations, for each roi, make separate plot, and show correlations across all linguistic properties across projections
for uid in uid_order:
    for parc in parc_order:
        fig, axs = plt.subplots(4, 4, figsize=(12, 8), constrained_layout=True)
        axs = axs.flatten()
        for i, feat in enumerate(feats_of_interest):
            for proj_n in proj_ns:
                r = np.corrcoef(d_projs[f'{uid}_{parc}_{proj_n}'], stimset[feat])[0, 1]
                axs[i].bar(proj_n, r)
                axs[i].set_title(feat)
                axs[i].set_xlabel('Projection')
                axs[i].set_ylabel('Correlation')
                # fix limits
                axs[i].set_ylim(-0.5, 0.7)
        plt.suptitle(f'{uid} ROI {parc}', fontsize=20)
        plt.show()

