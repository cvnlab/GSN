import os
from os.path import join, exists, split
import sys

sys.path.append(f'{os.getcwd()}/../')
from gsn.perform_gsn import perform_gsn
from gsn.convert_covariance_to_correlation import convert_covariance_to_correlation

import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

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

# # outputdir = join(homedir, 'examples', 'example1outputs')
# outputdir = '/nese/mit/group/evlab/u/gretatu/Sentence7T/GSN_outputs/'
# # SUBJECTDIR = '/Users/gt/Library/CloudStorage/GoogleDrive-gretatu@mit.edu/My Drive/Research2020/Sentence7T/FMRI_FSAVERAGE/'
# SUBJECTDIR = '/nese/mit/group/evlab/u/gretatu/Sentence7T/FMRI_FSAVERAGE/'

def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        print(f'Already a boolean: {v}')
        return v
    if v.lower() in ('true', 't'):
        # print(f'String arg - True: {v}')
        return True
    elif v.lower() in ('false', 'f'):
        # print(f'String arg - False: {v}')
        return False
    else:
        print(f'String arg - {v}')
        import argparse
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(raw_args=None):

    # DIRECTORIES
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputdir", type=str,
                        default='/nese/mit/group/evlab/u/gretatu/Sentence7T/GSN_outputs/')
    parser.add_argument("--SUBJECTDIR", type=str,
                        default='/nese/mit/group/evlab/u/gretatu/Sentence7T/')
    parser.add_argument("--uid", type=str, default='cvn7012')
    parser.add_argument("--hemi", type=str, default='lh')
    parser.add_argument("--parc_col", type=str, default='external_parc')
    parser.add_argument("--parc", type=int, default=4)
    parser.add_argument("--permute", type=str2bool, default=False)
    parser.add_argument("--random_seed", type=int, default=0)
    args = parser.parse_args(raw_args)

    sess = d_uid_to_sess[args.uid]

    savestr = f'{args.hemi}_baseline200_{sess}-{args.uid}_extracted-voxs-{args.parc_col}_parcs-{args.parc}_permute-{args.permute}'

    print(f'uid: {args.uid}, sess: {sess}, hemi: {args.hemi}, parc_col: {args.parc_col}, parc: {args.parc} permute: {args.permute}')

    fname = f'{args.hemi}_baseline200_{sess}-{args.uid}_extracted-voxs-{args.parc_col}_parcs-{args.parc}.pkl'
    # datafn = join(args.SUBJECTDIR, f'{args.sess}-{args.uid}', 'GLMestimatesingletrialoutputs', 'extracted_voxs', 'extracted_voxs', fname) # quirk of scp
    datafn = join(args.SUBJECTDIR, 'extracted_voxs', fname)


    print(f'loading data from {datafn}')
    print(f'directory to save example1 outputs:\n\t{args.outputdir}\n')

    # datafn = '/Users/gt/Library/CloudStorage/GoogleDrive-gretatu@mit.edu/My Drive/Research2020/Sentence7T/FMRI_FSAVERAGE/20231202-ST001-cvn7012/GLMestimatesingletrialoutputs/extracted_voxs/lh_baseline200_20231202-ST001-cvn7012_extracted-voxs-parc_lang_parcs-4.pkl'

    # Load pickle
    d = pickle.load(open(datafn, 'rb'))
    X = d['betas_parc_3d']

    if args.permute:
        np.random.seed(args.random_seed)
        # X is (vertices, stimuli, trials). Keep vertices the same and permute stimuli across trials.
        # Collapse last two dimensions and permute within that dimension
        X = X.reshape(X.shape[0], -1)
        # X = X[:, np.random.permutation(X.shape[1])]
        # # Reshape back to original shape
        # X = X.reshape(d['betas_parc_3d'].shape)

        # For each vertex, permute the stimuli across trials
        for i in range(X.shape[0]):
            X[i] = X[i, np.random.permutation(X.shape[1])]
        X = X.reshape(d['betas_parc_3d'].shape)

        print(f'Permuting X!')


    # Perform GSN.
    results = perform_gsn(X, {'wantshrinkage': True})

    if save_outputs:
        # Save GSN outputs
        # args.outputdir = '/Users/gt/Library/CloudStorage/GoogleDrive-gretatu@mit.edu/My Drive/Research2020/Sentence7T/GSN_outputs/'
        outputfn = join(args.outputdir, f'{savestr}.pkl')
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


if __name__ == '__main__':
    main()