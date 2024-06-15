import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob

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
# feats_of_interest = [
#     'surprisal-gpt2-xl_raw_mean',
#     # 'rating_gram_mean',
#     'rating_sense_mean',
#     #
#     # 'rating_others_thoughts_mean',
#     # 'rating_physical_mean',
#     # 'rating_places_mean',
#     #
#     'rating_valence_mean',
#     # 'rating_arousal_mean',
#     #
#     # 'rating_imageability_mean',
#     #
#     # 'rating_frequency_mean',
#     # 'rating_conversational_mean',
# ]
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

proj_ns = [0, 1, 2]

# Get the projections for all subjects
projs = []
# parcs = []
# uids = []
# for f, result in results.items():
#     # Get the parc and uid
#     parc = f.split('/')[-1].split('_')[2]
#     uid = f.split('/')[-1].split('_')[0]
#     uids.append(uid)
#     parcs.append(parc)
#
#     # Get the first projection
#     proj = result[:, proj_n]
#     projs.append(proj)

# Loop across uids, and then parcs
labels = []
for proj_n in proj_ns:
    for uid in uid_order:
        for parc in parc_order:
            # e.g., cvn7012_lh_4_proj_20
            results_key = f'{uid}_lh_{parc}_proj_20'
            result = results[results_key]
            proj = result[:, proj_n]
            projs.append(proj)
            labels.append(f'{proj_n}_{uid}_{parc}')

# Normalize the projections
projs_norm = [proj / np.linalg.norm(proj) for proj in projs]

# Compute the cosine similarity between all pairs of projections
cos_sim = np.zeros((len(projs), len(projs)))

for i, proj1 in enumerate(projs_norm):
    for j, proj2 in enumerate(projs_norm):
        cos_sim[i, j] = np.dot(proj1, proj2)


plt.figure(figsize=(25, 25))
ax = plt.imshow(abs(cos_sim), cmap='RdBu_r', vmin=-1, vmax=1)
plt.xticks(np.arange(0.5, len(projs), 1), labels, rotation=90,
           # center the ticks
              horizontalalignment='center',)
plt.yticks(np.arange(0.5, len(projs), 1), labels, rotation=0, horizontalalignment='right')
plt.colorbar(ax)
plt.tight_layout()
# save as pdf
plt.savefig(f'{PLOTDIR}cosine_similarity_proj_{"_".join([str(n) for n in proj_ns])}.pdf')
plt.show()




# normalize to unit and compute dot prod

