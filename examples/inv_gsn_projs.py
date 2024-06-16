import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
import pandas as pd
import argparse

"""
Investigate the PC projections from GSN (computed via load_gsn_results.py) using flag compute_pcs.
"""

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
]

def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't'):
        return True
    elif v.lower() in ('false', 'f'):
        return False
    else:
        import argparse
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(raw_args=None):

    # DIRECTORIES
    parser = argparse.ArgumentParser()
    parser.add_argument("--PLOTDIR", type=str,
                        default='/Users/gt/Library/CloudStorage/GoogleDrive-gretatu@mit.edu/My Drive/Research2020/Sentence7T/GSN_outputs/plots/',
                        help='Directory to save plots')
    parser.add_argument("--PROJOUTPUTDIR", type=str,
                        default='/Users/gt/Library/CloudStorage/GoogleDrive-gretatu@mit.edu/My Drive/Research2020/Sentence7T/GSN_outputs/outputs/',
                        help='Directory to save GSN outputs')
    parser.add_argument("--stimset_fname", type=str,
                        default='/Users/gt/Library/CloudStorage/GoogleDrive-gretatu@mit.edu/My Drive/Research2020/Sentence7T/FMRI_FSAVERAGE/20231202-ST001-cvn7009/GLMestimatesingletrialoutputs/extracted_voxs/lh_baseline200_20231202-ST001-cvn7009_extracted-voxs-external_parc_parcs-1.pkl',
                        help='Path to a stimulus set (shared across people, all items are sorted')
    parser.add_argument("--flip_sign", type=str2bool, default=True,
                        help='Whether to flip the sign of the projection, if the mean is negative')
    parser.add_argument("--large_heatmap", type=str2bool, default=False,
                        help='Whether to make a large heatmap of the cosine similarity between all projections')
    parser.add_argument("--save", type=str2bool, default=True,
                        help='Whether to save the results')
    args = parser.parse_args(raw_args)

    files = glob.glob(f'{args.PROJOUTPUTDIR}*flip-{args.flip_sign}.pkl')
    print(f'Loading {len(files)} files: {files}')

    # Load GSN results into a dict with keys as the filenames
    results = {}
    for f in files:
        fname = f.split('/')[-1].split('.')[0]
        # 'cvn7012_lh_4_proj_20_flip-True', drop the flip-True
        fname_save = '_'.join(fname.split('_')[:-1])
        results[fname_save] = pickle.load(open(f, 'rb')) # 200 stimuli x 20 projections

    # Load a stimset from a random subject
    d_stimset = pickle.load(open(args.stimset_fname, 'rb'))
    # for each of stimset_rep{1,2,3}_ordered, check that the item_id col is the same
    assert np.all(d_stimset['stimset_rep1_ordered']['item_id'].values == d_stimset['stimset_rep2_ordered']['item_id'].values)
    assert np.all(d_stimset['stimset_rep1_ordered']['item_id'].values == d_stimset['stimset_rep3_ordered']['item_id'].values)
    stimset = d_stimset['stimset_rep1_ordered']



    # Compute the cosine between the projections of projection 1 for all subjects, across ROIs, and make a gigantic heatmap
    uid_order = ['cvn7009', 'cvn7012', 'cvn7002', 'cvn7011', 'cvn7007', 'cvn7006', 'cvn7013', 'cvn7016']
    parc_order = [1, 2, 3, 4, 5]
    proj_ns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    # Get the projections for all subjects
    projs = []
    # Loop across uids, and then parcs
    labels = []
    item_ids = []
    for proj_n in proj_ns:
        for uid in uid_order:
            for parc in parc_order:
                # e.g., cvn7012_lh_4_proj_20
                results_key = f'{uid}_lh_{parc}_proj_20'
                result = results[results_key]
                proj = result[:, proj_n]
                projs.append(proj)
                labels.append(f'{proj_n}_{uid}_{parc}')
                # Item_ids are 1-200
                item_ids.append(np.arange(1, 201))

    # Normalize the projections
    projs_norm = [proj / np.linalg.norm(proj) for proj in projs]

    # Compute the cosine similarity between all pairs of projections
    cos_sim = np.zeros((len(projs), len(projs)))

    for i, proj1 in enumerate(projs_norm):
        for j, proj2 in enumerate(projs_norm):
            cos_sim[i, j] = np.dot(proj1, proj2)

    # Make a huge heatmap
    if args.large_heatmap:
        plt.figure(figsize=(35, 35))
        ax = plt.imshow(cos_sim, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.xticks(np.arange(0.5, len(projs), 1), labels, rotation=90,
                      horizontalalignment='center', fontsize=3)
        plt.yticks(np.arange(0.5, len(projs), 1), labels, rotation=0, horizontalalignment='right',
                   fontsize=3)
        plt.colorbar(ax, shrink=0.5)
        plt.tight_layout()
        if args.save:
            plt.savefig(f'{args.PLOTDIR}cosine_similarity_proj_{np.min(proj_ns)}-{np.max(proj_ns)}_flip-{args.flip_sign}.pdf')
        plt.show()

    # Load the projections into a dataframe. For each subject, we have 20 projections for each ROI (of length 200)
    # We can use projs and labels from the previous section. Split labels into proj_n, uid, parc, and item_id
    proj_ns_label = [int(label.split('_')[0]) for label in labels]
    uids = [label.split('_')[1] for label in labels]
    parcs = [int(label.split('_')[2]) for label in labels]

    # Make a long dataframe. Have columns for projection, proj_n, uid, parc, item_id. Projs is a list of lists. Fold it into a single list and repeat the other columns
    projs_flat = [proj for proj_list in projs for proj in proj_list]
    item_ids_flat = [item_id for item_id_list in item_ids for item_id in item_id_list]
    proj_ns_flat = [proj_n for proj_n in proj_ns_label for _ in range(len(projs[0]))]
    uids_flat = [uid for uid in uids for _ in range(len(projs[0]))]
    parcs_flat = [parc for parc in parcs for _ in range(len(projs[0]))]
    df = pd.DataFrame({'proj': projs_flat, 'proj_n': proj_ns_flat, 'uid': uids_flat, 'parc': parcs_flat, 'item_id': item_ids_flat})
    assert (8 * 5 * 20 * 200) == df.shape[0]

    # Merge with the stimset on item_id
    df_merged = pd.merge(df, stimset, on='item_id')

    # Average across subjects for each parc and projection and item_id
    df_avg_parc = df_merged.groupby(['parc', 'proj_n', 'item_id']).mean().reset_index()

    # Correlate the average projection with the linguistic properties, features of interest
    df_avg_parc_filtered = df_avg_parc[feats_of_interest + ['proj', 'proj_n', 'parc']]
    # For each proj_n and parc, correlate with the linguistic properties. Make heatmap
    for proj_n in proj_ns:
        for parc in parc_order:
            df_parc_proj = df_avg_parc_filtered[(df_avg_parc_filtered['proj_n'] == proj_n) & (df_avg_parc_filtered['parc'] == parc)]
            # Drop the proj_n and parc columns
            df_parc_proj = df_parc_proj.drop(['proj_n', 'parc'], axis=1)
            corrs = df_parc_proj.corr()
            plt.figure(figsize=(10, 8))
            plt.imshow(corrs, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='none')
            plt.xticks(np.arange(len(df_parc_proj.columns)), df_parc_proj.columns,
                       rotation=90)  # Adjusted tick positions
            plt.yticks(np.arange(len(df_parc_proj.columns)), df_parc_proj.columns)
            plt.colorbar()
            plt.title(f'proj {proj_n} parc {parc} (average across n={df_merged["uid"].nunique()} subjects)')
            plt.tight_layout()
            if args.save:
                plt.savefig(f'{args.PLOTDIR}proj_{proj_n}_parc_{parc}_corr_flip-{args.flip_sign}.pdf',
                            dpi=300)
            plt.show()

    # For a given parc, make subplots according to the correlation with each linguistic property. Let each subplot
    # have the correlations with each projection
    lims = [-0.7, 0.7]
    for parc in parc_order:
        df_parc = df_avg_parc_filtered[df_avg_parc_filtered['parc'] == parc]
        fig, axs = plt.subplots(4, 4, figsize=(12, 8), constrained_layout=True)
        axs = axs.flatten()
        for i, feat in enumerate(feats_of_interest):
            for proj_n in proj_ns:
                df_parc_proj = df_parc[df_parc['proj_n'] == proj_n]
                corrs = df_parc_proj.corr()
                r = corrs.loc['proj', feat]
                axs[i].bar(proj_n, r)
                axs[i].set_title(feat)
                axs[i].set_xlabel('Projection')
                axs[i].set_ylabel('Correlation')
                # fix limits
                axs[i].set_ylim(lims)
        plt.suptitle(f'ROI {parc} (average across n={df_merged["uid"].nunique()} subjects)', fontsize=14)
        plt.tight_layout()
        if args.save:
            plt.savefig(f'{args.PLOTDIR}parc_{parc}_corr_flip-{args.flip_sign}.pdf',
                        dpi=300)
        plt.show()



if __name__ == '__main__':
    main()


## DRAFTED
    # # For a given uid, correlate their projections with each other
    # for uid in uid_order:
    #     projs = []
    #     for parc in parc_order:
    #         results_key = f'{uid}_lh_{parc}_proj_20'
    #         result = results[results_key] # 200 x 20
    #
    #         df = pd.DataFrame(result).set_index(stimset.index)
    #         # concat with the stimset
    #         df = pd.concat([df, stimset[feats_of_interest]], axis=1)
    #
    #         # Correlate the projections with each other
    #         corrs = df.corr()
    #         plt.figure(figsize=(10, 8))
    #         plt.imshow(corrs, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='none')
    #         plt.colorbar()
    #         plt.title(f'{uid} {parc}')
    #         plt.show()
