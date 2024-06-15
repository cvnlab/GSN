"""
Load GSN results and make plots
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
from gsn.convert_covariance_to_correlation import convert_covariance_to_correlation

SUBJECTDIR = '/nese/mit/group/evlab/u/gretatu/Sentence7T/FMRI_FSAVERAGE/'

GSNOUTPUTS = '/Users/gt/Library/CloudStorage/GoogleDrive-gretatu@mit.edu/My Drive/Research2020/Sentence7T/GSN_outputs/'
# One dir down from GSNOUTPUTS
SAVEDIR = os.path.join(GSNOUTPUTS, 'plots')
if not os.path.exists(SAVEDIR):
    os.makedirs(SAVEDIR)

save = True

permute = False
parc = 3
if permute:
    files = glob.glob(f'{GSNOUTPUTS}*parcs-{parc}*permute-{permute}*.pkl')
else:
    files = glob.glob(f'{GSNOUTPUTS}*parcs-{parc}*.pkl')



for f in files:

    # Load pickle
    results = pickle.load(open(f, 'rb'))
    uid = f.split('/')[-1].split('-')[2].split('_')[0]
    hemi = f.split('/')[-1].split('_')[0]
    parc = f.split('/')[-1].split('_')[5].split('-')[-1]

    # Visualize noise ceiling SNR
    plt.figure()
    plt.hist(results['ncsnr'])
    plt.xlabel('Noise ceiling SNR')
    plt.ylabel('Frequency')
    plt.title(f'uid: {uid}, hemi: {hemi}, parc: {parc}')
    if save:
        plt.savefig(os.path.join(SAVEDIR, f'ncsnr_{uid}_{hemi}_{parc}.png'))
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
        plt.savefig(os.path.join(SAVEDIR, f'matrices_plot_corr_{plot_corr}_{uid}_{hemi}_{parc}.png'))
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
        plt.savefig(os.path.join(SAVEDIR, f'spectra_{uid}_{hemi}_{parc}.png'))
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
        plt.savefig(os.path.join(SAVEDIR, f'cumvar_cov_{uid}_{hemi}_{parc}.png'))
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(np.cumsum(sSb) / np.sum(sSb), 'r-', label='Signal')
    plt.plot(np.cumsum(sNb) / np.sum(sNb), 'k-', label='Noise')
    plt.title('Correlation')
    plt.xlabel('Dimension')
    plt.ylabel('Cumulative variance explained')
    plt.legend()
    if save:
        plt.savefig(os.path.join(SAVEDIR, f'cumvar_corr_{uid}_{hemi}_{parc}.png'))
    plt.show()

