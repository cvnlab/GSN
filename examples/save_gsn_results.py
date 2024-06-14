import os
from os.path import join, exists, split
import sys

sys.path.append(f'{os.getcwd()}/../')
from gsn.perform_gsn import perform_gsn
from gsn.convert_covariance_to_correlation import convert_covariance_to_correlation

import numpy as np
import matplotlib.pyplot as plt
import pickle

make_plots = False # if False, just save GSN outputs
save_outputs = True

d_uid_to_sess = {'cvn7009': '20231202-ST001',
                    'cvn7012': '20231202-ST001',
                    'cvn7002': '20231203-ST001',
                    'cvn7011': '20231204-ST001',
                    'cvn7007': '20231204-ST001',
                    'cvn7006': '20231215-ST001',
                    'cvn7013': '20240508-ST001',
                    'cvn7016': '20240530-ST001',}


# Get path to the directory to which GSN was installed
homedir = split(os.getcwd())[0]

# Create directory for saving outputs from example 1
# outputdir = join(homedir, 'examples', 'example1outputs')
outputdir = '/nese/mit/group/evlab/u/gretatu/Sentence7T/GSN_outputs/'
# SUBJECTDIR = '/Users/gt/Library/CloudStorage/GoogleDrive-gretatu@mit.edu/My Drive/Research2020/Sentence7T/FMRI_FSAVERAGE/'
SUBJECTDIR = '/nese/mit/group/evlab/u/gretatu/Sentence7T/FMRI_FSAVERAGE/'

uids = ['cvn7009', 'cvn7012', 'cvn7002', 'cvn7011', 'cvn7007', 'cvn7006', 'cvn7013', 'cvn7016']
uids = ['cvn7012']
hemis = ['rh']
parcs = [7]

for uid in uids:
    sess = d_uid_to_sess[uid]
    for hemi in hemis:
        for parc in parcs:
            print(f'uid: {uid}, sess: {sess}, hemi: {hemi}, parc: {parc}')

            fname = f'{hemi}_baseline200_{sess}-{uid}_extracted-voxs-parc_lang_parcs-{parc}.pkl'
            datafn = join(SUBJECTDIR, f'{sess}-{uid}', 'GLMestimatesingletrialoutputs', 'extracted_voxs', 'extracted_voxs', fname) # quirk of scp

            print(f'loading data from {datafn}')
            print(f'directory to save example1 outputs:\n\t{outputdir}\n')

            # Load pickle
            d = pickle.load(open(datafn, 'rb'))
            X = d['betas_parc_3d']


            # Perform GSN.
            results = perform_gsn(X, {'wantshrinkage': True})

            if save_outputs:
                # Save GSN outputs
                outputfn = join(outputdir, f'{hemi}_baseline200_{sess}-{uid}_extracted-voxs-parc_lang_parcs-{parc}_gsn_outputs.pkl')
                print(f'saving GSN outputs to {outputfn}')
                pickle.dump(results, open(outputfn, 'wb'))

            if make_plots:

                # Output various results from GSN
                print(results['shrinklevelN'])
                print(results['shrinklevelD'])

                # Visualize noise ceiling SNR
                plt.figure()
                plt.hist(results['ncsnr'])
                plt.xlabel('Noise ceiling SNR')
                plt.ylabel('Frequency')
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

                # Noise covariance estimate
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

                plt.show()

                # Comparing raw and final covariance estimates
                print(np.corrcoef(results['cN'].reshape(-1), results['cNb'].reshape(-1))[0,1])
                print(np.corrcoef(results['cS'].reshape(-1), results['cSb'].reshape(-1))[0,1])

                # Perform PCA on covariance matrices
                # def convert_covariance_to_correlation(cov):
                #     d = np.sqrt(np.diag(cov))
                #     corr = cov / np.outer(d, d)
                #     np.fill_diagonal(corr, 1)
                #     return corr

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

                plt.subplot(1, 2, 1)
                plt.plot(S_Sb[:top_n_dim], 'r-', label='Signal')
                plt.plot(S_Nb[:top_n_dim], 'k-', label='Noise')
                # log y-axis
                plt.yscale('log')
                plt.title(f'Signal ED = {ed_fun(sSb):.1f}, Noise ED = {ed_fun(sNb):.1f}')
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

                plt.show()


                # plt.figure(figsize=(12, 5))
                # # Eigenspectra plot
                # plt.subplot(1, 2, 1)
                # plt.plot(sS, 'r-', label='Signal')
                # plt.plot(sN, 'k-', label='Noise')
                # plt.title(f'Signal ED = {ed_fun(sS):.1f}, Noise ED = {ed_fun(sN):.1f}')
                # plt.xlabel('Dimension')
                # plt.ylabel('Eigenvalue')
                # plt.legend()
                #
                # # Cumulative variance explained plot
                # plt.subplot(1, 2, 2)
                # plt.plot(np.cumsum(sS) / np.sum(sS) * 100, 'r-', label='Signal')
                # plt.plot(np.cumsum(sN) / np.sum(sN) * 100, 'k-', label='Noise')
                # plt.xlabel('Dimension')
                # plt.ylabel('Cumulative variance explained (%)')
                #
                # plt.tight_layout()
                # plt.show()
