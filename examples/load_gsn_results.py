"""
Load GSN results and make plots

% stimuli x vertices     x     vertices x 10


data = vertices x 200 x 3  (PSC)


mnS   vertices x 1

mnN

cSb

cNb


% columns of v have the PC
[u,s,v] = svd(cSb,0);


mean(data,3)   vertices x 200

D = mean(data,3) - mnS   (compute data RELATIVE to the mean of the signal gaussian) noise is zero mean

D (vertices x 200)

D' * v(:,1:20) ==> done!


8 subjects x 5 ROIs x 200 sentences x 20   ===>   "stimulus projections onto the GSN's signal PCs for the first 20 pcs"

"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
from gsn.convert_covariance_to_correlation import convert_covariance_to_correlation

d_uid_to_sess = {'cvn7009': '20231202-ST001',
                    'cvn7012': '20231202-ST001',
                    'cvn7002': '20231203-ST001',
                    'cvn7011': '20231204-ST001',
                    'cvn7007': '20231204-ST001',
                    'cvn7006': '20231215-ST001',
                    'cvn7013': '20240508-ST001',
                    'cvn7016': '20240530-ST001',}

SUBJECTDIR = '/Users/gt/Library/CloudStorage/GoogleDrive-gretatu@mit.edu/My Drive/Research2020/Sentence7T/FMRI_FSAVERAGE/'
GSNOUTPUTS = '/Users/gt/Library/CloudStorage/GoogleDrive-gretatu@mit.edu/My Drive/Research2020/Sentence7T/GSN_outputs/'
# One dir down from GSNOUTPUTS
PLOTDIR = os.path.join(GSNOUTPUTS, 'plots')
if not os.path.exists(PLOTDIR):
    os.makedirs(PLOTDIR)
OUTPUTDIR = os.path.join(GSNOUTPUTS, 'outputs')
if not os.path.exists(OUTPUTDIR):
    os.makedirs(OUTPUTDIR)

save = True

make_plots = False
compute_pcs = True # compute SVD and project onto the stimulus data
flip_sign = True # If the mean of a column of V is negative, flip the sign of that column
permute = False
parc = 24
if permute:
    files = glob.glob(f'{GSNOUTPUTS}*parcs-{parc}*permute-{permute}*.pkl')
else:
    files = glob.glob(f'{GSNOUTPUTS}*parcs-{parc}*.pkl')
    files = [f for f in files if not 'permute-True' in f]

if parc in [1, 2, 3]:
    parc_col = 'external_parc'
elif parc in [24]:
    parc_col = 'parc_glasser'
elif parc in [4, 5]:
    parc_col = 'parc_lang'


for f in files:

    # Load GSN pickle
    results = pickle.load(open(f, 'rb'))
    uid = f.split('/')[-1].split('-')[2].split('_')[0]
    hemi = f.split('/')[-1].split('_')[0]
    parc = f.split('/')[-1].split('_')[5].split('-')[-1]

    if compute_pcs:
        # Compute SVD on cSb
        U_Sb, S_Sb, vT_Sb = np.linalg.svd(results['cSb'], full_matrices=False) # In this case we do not worry about mean-centering the data

        # Now load the corresponding data
        sess = d_uid_to_sess[uid]
        datafn = os.path.join(SUBJECTDIR, f'{sess}-{uid}', 'GLMestimatesingletrialoutputs', 'extracted_voxs',
                                f'{hemi}_baseline200_{sess}-{uid}_extracted-voxs-{parc_col}_parcs-{parc}.pkl')
        d = pickle.load(open(datafn, 'rb'))
        data = d['betas_parc_3d']

        # Mean across trials (3rd dimension)
        data_mean = np.mean(data, axis=2)

        D = data_mean - results['mnS'].reshape(-1, 1) # Compute data RELATIVE to the mean of the signal gaussian (noise is zero mean)

        # Project data onto the PCs
        n_pcs = 20
        # proj = np.dot(D.T, vT_Sb.T[:, :n_pcs]) # same as np.dot(D.T, vT_Sb.T)[:, :n_pcs]
        V = vT_Sb.T # python returns the transpose of the V matrix. Transpose back to get the correct V
        # cols of V are voxel PCs. unit length, new basis

        # visualize the two matrices, different subplots, same color limits
        # fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        # ax[0].imshow(U_Sb, aspect='auto', interpolation='none', cmap='bwr', vmin=-0.1, vmax=0.1)
        # ax[0].set_title('U_Sb')
        # ax[1].imshow(V, aspect='auto', interpolation='none', cmap='bwr', vmin=-0.1, vmax=0.1)
        # ax[1].set_title('V')
        # plt.suptitle(f'uid: {uid}, hemi: {hemi}, parc: {parc}')
        # plt.show()


        # Check column means of V
        if flip_sign:
            for i in range(V.shape[1]):
                if np.mean(V[:, i]) < 0:
                    V[:, i] = -V[:, i]

        proj = np.dot(D.T, V[:, :n_pcs])

        # proj_rank1 = np.dot(D.T, V[:, 0])


        # Take stimset_rep{1,2,3}_ordered and assert that item_ids are 1-200
        stimset_rep1_ordered = d['stimset_rep1_ordered']
        stimset_rep2_ordered = d['stimset_rep2_ordered']
        stimset_rep3_ordered = d['stimset_rep3_ordered']
        assert np.all(stimset_rep1_ordered['item_id'] == np.arange(1, 201))
        assert np.all(stimset_rep2_ordered['item_id'] == np.arange(1, 201))
        assert np.all(stimset_rep3_ordered['item_id'] == np.arange(1, 201))
        print(f'Item IDs are 1-200 for {uid}')

        # Save the projections in the output directory
        projfn = os.path.join(OUTPUTDIR, f'{uid}_{hemi}_{parc}_proj_{n_pcs}_flip-{flip_sign}.pkl')
        if save:
            pickle.dump(proj, open(projfn, 'wb'))
            print(f'Saved projections to {projfn}')



    # Visualize noise ceiling SNR
    if make_plots:
        plt.figure()
        plt.hist(results['ncsnr'])
        plt.xlabel('Noise ceiling SNR')
        plt.ylabel('Frequency')
        plt.title(f'uid: {uid}, hemi: {hemi}, parc: {parc}')
        if save:
            plt.savefig(os.path.join(PLOTDIR, f'ncsnr_{uid}_{hemi}_{parc}.png'))
        plt.show()

        # Visualize covariance estimates
        plt.figure(figsize=(10, 8))

        # Set range for color limits
        rng = [-1, 1]
        cmap = 'bwr'
        plot_corr = True

        if plot_corr:
            rng = [-1, 1]
            # Add another variabnle to results, convert covariance to correlation
            cN_corr, bad = convert_covariance_to_correlation(m=results['cN'])
            cNb_corr, bad = convert_covariance_to_correlation(m=results['cNb'])
            cS_corr, bad = convert_covariance_to_correlation(m=results['cS'])
            cSb_corr, bad = convert_covariance_to_correlation(m=results['cSb'])

            # Look at diagonal values, print unique values
            print(f'cN_corr diag: {np.unique(np.diag(cN_corr))}')
            print(f'cNb_corr diag: {np.unique(np.diag(cNb_corr))}')
            print(f'cS_corr diag: {np.unique(np.diag(cS_corr))}')
            print(f'cSb_corr diag: {np.unique(np.diag(cSb_corr))}')


        # Noise covariance estimate
        plt.figure(figsize=(20, 16))
        plt.subplot(2, 2, 1)
        if plot_corr:
            plt.imshow(cN_corr, vmin=rng[0], vmax=rng[1], aspect='auto', interpolation='none', cmap=cmap)
        else:
            plt.imshow(results['cN'], vmin=rng[0], vmax=rng[1], aspect='auto', interpolation='none')
        plt.colorbar()
        plt.title('cN')
        plt.axis('tight')

        # Final noise covariance estimate
        plt.subplot(2, 2, 2)
        if plot_corr:
            plt.imshow(cNb_corr, vmin=rng[0], vmax=rng[1], aspect='auto', interpolation='none', cmap=cmap)
        else:
            plt.imshow(results['cNb'], vmin=rng[0], vmax=rng[1], aspect='auto', interpolation='none')
        plt.colorbar()
        plt.title('cNb')
        plt.axis('tight')

        # Signal covariance estimate
        plt.subplot(2, 2, 3)
        if plot_corr:
            plt.imshow(cS_corr, vmin=rng[0], vmax=rng[1], aspect='auto', interpolation='none', cmap=cmap)
        else:
            plt.imshow(results['cS'], vmin=rng[0], vmax=rng[1], aspect='auto', interpolation='none')
        plt.colorbar()
        plt.title('cS')
        plt.axis('tight')

        # Final signal covariance estimate
        plt.subplot(2, 2, 4)
        if plot_corr:
            plt.imshow(cSb_corr, vmin=rng[0], vmax=rng[1], aspect='auto', interpolation='none', cmap=cmap)
        else:
            plt.imshow(results['cSb'], vmin=rng[0], vmax=rng[1], aspect='auto', interpolation='none')
        plt.colorbar()
        plt.title('cSb')
        plt.axis('tight')

        plt.suptitle(f'uid: {uid}, hemi: {hemi}, parc: {parc}, plot_corr: {plot_corr}')
        if save:
            plt.savefig(os.path.join(PLOTDIR, f'matrices_plot_corr_{plot_corr}_{uid}_{hemi}_{parc}.png'))
        plt.show()

        # Comparing raw and final covariance estimates
        print(np.corrcoef(results['cN'].reshape(-1), results['cNb'].reshape(-1))[0, 1])
        print(np.corrcoef(results['cS'].reshape(-1), results['cSb'].reshape(-1))[0, 1])

        def ed_fun(x):
            return np.sum(x) ** 2 / np.sum(x ** 2)


        # Convert raw output Csb and Cnb
        U_Sb, S_Sb, vT_Sb = np.linalg.svd(results['cSb'], full_matrices=False)
        U_Nb, S_Nb, vT_Nb = np.linalg.svd(results['cNb'], full_matrices=False)

        # Convert correlation matrices (every voxel is now treated as having equal weight in the PCA)
        uSb_corr, bad = convert_covariance_to_correlation(results['cSb'])
        uNb_corr, bad = convert_covariance_to_correlation(results['cNb'])
        uSb, sSb, vSb = np.linalg.svd(uSb_corr, full_matrices=False)
        uNb, sNb, vNb = np.linalg.svd(uNb_corr, full_matrices=False)

        # Plot the spectra of S_Sb and S_Nb
        top_n_dim = 100

        plt.figure(figsize=(10, 8))
        plt.subplot(1, 2, 1)
        plt.plot(S_Sb[:top_n_dim], 'r-', label='Signal')
        plt.plot(S_Nb[:top_n_dim], 'k-', label='Noise')
        # log y-axis
        plt.yscale('log')
        plt.title(f'Signal ED = {ed_fun(S_Sb):.1f}, Noise ED = {ed_fun(S_Nb):.1f}')
        plt.xlabel('Dimension')
        plt.ylabel('Eigenvalue')
        plt.legend()

        # Plot the spectra of sS and sN
        plt.subplot(1, 2, 2)
        plt.plot(sSb[:top_n_dim], 'r-', label='Signal')
        plt.plot(sNb[:top_n_dim], 'k-', label='Noise')
        # log y-axis
        plt.yscale('log')
        plt.title(f'Signal ED = {ed_fun(sSb):.1f}, Noise ED = {ed_fun(sNb):.1f}')
        plt.xlabel('Dimension')
        plt.ylabel('Eigenvalue')
        plt.legend()

        plt.suptitle(f'uid: {uid}, hemi: {hemi}, parc: {parc}')
        if save:
            plt.savefig(os.path.join(PLOTDIR, f'spectra_{uid}_{hemi}_{parc}.png'))
        plt.show()

        # Cumulative variance explained plot (for both covariance and correlation matrices)
        plt.figure(figsize=(10, 8))
        plt.plot(np.cumsum(S_Sb) / np.sum(S_Sb), 'r-', label='Signal')
        plt.plot(np.cumsum(S_Nb) / np.sum(S_Nb), 'k-', label='Noise')
        plt.title('Covariance')
        plt.xlabel('Dimension')
        plt.ylabel('Cumulative variance explained')
        plt.legend()
        if save:
            plt.savefig(os.path.join(PLOTDIR, f'cumvar_cov_{uid}_{hemi}_{parc}.png'))
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.plot(np.cumsum(sSb) / np.sum(sSb), 'r-', label='Signal')
        plt.plot(np.cumsum(sNb) / np.sum(sNb), 'k-', label='Noise')
        plt.title('Correlation')
        plt.xlabel('Dimension')
        plt.ylabel('Cumulative variance explained')
        plt.legend()
        if save:
            plt.savefig(os.path.join(PLOTDIR, f'cumvar_corr_{uid}_{hemi}_{parc}.png'))
        plt.show()

