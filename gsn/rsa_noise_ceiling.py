import numpy as np
import scipy
import itertools
import warnings
from gsn.utilities import nanreplace
from gsn.calc_cod import calc_cod
from gsn.calc_shrunken_covariance import calc_shrunken_covariance
from gsn.construct_nearest_psd_covariance import construct_nearest_psd_covariance
import scipy.stats as stats
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

def rsa_noise_ceiling(data, opt = None):
    """
    Use the GSN (generative modeling of signal and noise) method to estimate
    an RSA noise ceiling.

    Parameters:
    data (numpy.ndarray): An array of shape (voxels, conditions, trials). This indicates
        the measured responses to different conditions on distinct trials. The number of
        trials must be at least 2.
    opt (dict, optional): A dictionary with the following optional fields:
        wantverbose (bool, optional): Whether to print status statements. Default: True.
        rdmfun (function, optional): A function that constructs an RDM. Specifically, the
            function should accept as input a data matrix (e.g., voxels x conditions) and
            output an RDM with some dimensionality (can be a column vector, 2D matrix, etc.).
            Default: A function that calculates dissimilarity as 1-r, and then extracts the
            lower triangle (excluding the diagonal) as a column vector.
        comparefun (function, optional): A function that quantifies the similarity of two RDMs.
            Specifically, the function should accept as input two RDMs (in the format returned
            by `rdmfun`) and output a scalar. Default: `np.corrcoef`
        wantfig (int or str, optional):
            0 means do not make a figure.
            1 means plot a figure in a new figure window.
            A string value means write a figure to filename prefix A (e.g., '/path/to/figure'
            will result in '/path/to/figure.png' being written).
            Default: 1.
        ncsims (int, optional): The number of Monte Carlo simulations to run for the purposes
            of estimating the RSA noise ceiling. The final answer is computed as the median
            across simulation results. Default: 50.
        ncconds (int, optional): The number of conditions to simulate in the Monte Carlo
            simulations. In theory, the RSA noise ceiling estimate should be invariant to the
            number of conditions simulated, but the higher the number, the more stable/accurate
            the results. Default: 50.
        nctrials (int, optional): The number of trials to target for the RSA noise ceiling
            estimate. For example, setting `nctrials` to 10 will result in the calculation of
            a noise ceiling estimate for the case in which responses are averaged across 10
            trials per condition. Default: data.shape[2].
        splitmode (int, optional): Controls the way in which trials are divided for the data
            reliability calculation.
            0 means use only the maximum split-half number of trials. In the case of an odd
            number of trials T, use floor(T/2).
            1 means use numbers of trials increasing by a factor of 2 starting at 1 and
            including the maximum split-half number of trials. For example, if there are 10
            trials total, use [1 2 4 5].
            2 means use the maximum split-half number of trials as well as half of that number
            (rounding down if necessary). For example, if there are 11 trials total, use [2 5].
            The primary value of using multiple numbers of trials (e.g., options 1 and 2) is to
            provide greater insight for the figure inspection that is created. However, in terms
            of accuracy of RSA noise ceiling estimates, option 0 should be fine (and will result
            in faster execution).
            Default: 0.
        scs (list, optional): Controls the way in which the posthoc scaling factor is
            determined. Specifically, `scs` is:
            A list where each element is a non-negative value. This specifies the
            specific scale factors to evaluate. There is a trade-off between speed of
            execution and the discretization/precision of the results.
            Default: np.arange(0, 2.1, 0.1).
        simchunk (int, optional): The chunk size for the data-splitting and model-based
            simulations. Default is 50, which indicates to perform 50 simulations for each
            case, and then increment in steps of 50 if necessary to achieve `simthresh`.
            Must be 2 or greater.
        simthresh (float, optional): The value for the robustness metric that must be
            exceeded in order to halt the data-splitting simulations. The lower this number,
            the faster the execution time, but the less accurate the results. Default: 10.
        maxsimnum (int, optional): The maximum number of simulations to perform for the
            data-splitting simulations. Default: 1000.
        random_seed (int, optional): Random seed for reproducibility. Default: 42.

    Notes:
    - If `comparefun` ever returns NaN, we automatically replace these cases with 0.
      This is a convenient workaround for degenerate cases that might arise, e.g., cases
      where the signal is generated as all zero.
    - For splits of the empirical data, it might be the case that exhaustive combinations
      might be faster than the random-sampling approach. Thus, if all splits (as controlled
      by `splitmode`) can be exhaustively done in less than or equal to `opt.simchunk`
      iterations, then we will compute the exhaustive (and therefore exact) solution,
      instead of the random-sampling approach.

    Returns:
    nc (float): A scalar with the noise ceiling estimate.
    ncdist (numpy.ndarray): 1 x ncsims with the result of each Monte Carlo simulation.
        Note that `nc` is simply the median of `ncdist`.
    results (dict): A dictionary with additional details:
        mnN - the estimated mean of the noise (1 x voxels)
        cN - the raw estimated covariance of the noise (voxels x voxels)
        cNb - the final estimated covariance after biconvex optimization
        shrinklevelN - shrinkage level chosen for cN
        shrinklevelD - shrinkage level chosen for the estimated data covariance
        mnS - the estimated mean of the signal (1 x voxels)
        cS - the raw estimated covariance of the signal (voxels x voxels)
        cSb - the final estimated covariance after biconvex optimization
        sc - the post-hoc scaling factor for cSb that was selected and used
             for the purposes of the RSA simulations
        splitr - the data split-half reliability value that was obtained for the
                 largest trial number that was evaluated. (We return the median
                 result across the simulations that were conducted.)
        ncsnr - the 'noise ceiling SNR' estimate for each voxel (1 x voxels).
                This is, for each voxel, the std dev of the estimated signal
                distribution divided by the std dev of the estimated noise
                distribution. Note that this is computed on the raw
                estimated covariances. Also, note that we apply positive
                rectification (to prevent non-sensical negative ncsnr values).
        numiters - the number of iterations used in the biconvex optimization.
                0 means the first estimate was already positive semi-definite.

    History:
    - 2025/01/15 - add random seed control for reproducibility
    - 2024/09/11 - misc. bug fixes
    - 2024/08/24 - add results['numiters']
    - 2024/01/05:
        (1) Major change to use the biconvex optimization procedure --
            we now have cSb and cNb as the final estimates;
        (2) cSb no longer has the scaling baked in and instead we
            create a separate temporary variable cSb_rsa;
        (3) Remove the rapprox output.

    Example:
    data = np.tile(2 * np.random.randn(100, 40), (1, 1, 4)) + np.random.randn(100, 40, 4)
    nc, ncdist, results = rsa_noise_ceiling(data, {'splitmode': 1})

    Internal options (not for general use):
    shrinklevels (list, optional): Like the input to calc_shrunken_covariance.py.
        Default: [].
    mode (int, optional):
        0 means use the data-reliability method.
        1 means use no scaling.
        2 means scale to match the un-regularized average variance.
        Default: 0.
    """

    # Set random seed for reproducibility
    if opt is None:
        opt = {}
    np.random.seed(opt.get('random_seed', 42))

    # Set default values for opt
    opt.setdefault('wantverbose', 1)
    opt.setdefault('rdmfun', lambda d: pdist(d.T, 'correlation'))
    opt.setdefault('comparefun', lambda x, y: stats.pearsonr(x, y)[0])
    opt.setdefault('wantfig', 1)
    opt.setdefault('ncsims', 50)
    opt.setdefault('ncconds', 50)
    opt.setdefault('nctrials', data.shape[2])
    opt.setdefault('splitmode', 0)
    opt.setdefault('scs', np.arange(0, 2.1, 0.1))
    opt.setdefault('simchunk', 50)
    opt.setdefault('simthresh', 10)
    opt.setdefault('maxsimnum', 1000)
    opt.setdefault('shrinklevels', np.linspace(0,1,51))
    opt.setdefault('mode', 0)

    # calc
    nvox = data.shape[0]
    ncond = data.shape[1]
    ntrial = data.shape[2]

    # deal with massaging inputs and sanity checks
    assert ntrial >= 2, "Number of trials must be at least 2."
    opt['scs'] = np.unique(opt['scs'])
    assert np.all(opt['scs'] >= 0), "All elements in 'scs' should be non-negative."
    assert opt['simchunk'] >= 2, "simchunk must be 2 or greater."

    # ESTIMATION OF COVARIANCES

    # estimate noise covariance
    if opt['wantverbose']:
        print('Estimating noise covariance...', end='')
    mnN, cN, shrinklevelN, nllN = calc_shrunken_covariance(np.transpose(data, (2, 0, 1)),
                                                         5, opt['shrinklevels'], 1)
    if opt['wantverbose']:
        print('done.')

    # estimate data covariance
    if opt['wantverbose']:
        print('Estimating data covariance...', end='')
    mnD, cD, shrinklevelD, nllD = calc_shrunken_covariance(np.mean(data, axis=2).T,
                                                         5, opt['shrinklevels'], 1)
    if opt['wantverbose']:
        print('done.')

    # estimate signal covariance
    if opt['wantverbose']:
        print('Estimating signal covariance...', end='')
    mnS = mnD - mnN
    cS = cD - cN / ntrial
    if opt['wantverbose']:
        print('done.')

    # prepare some outputs
    sd_noise = np.sqrt(np.maximum(np.diag(cN), 0))   # std of the noise (1 x voxels)
    sd_signal = np.sqrt(np.maximum(np.diag(cS), 0))  # std of the signal (1 x voxels)
    ncsnr = np.divide(sd_signal, sd_noise, out=np.zeros_like(sd_signal), where=sd_noise!=0)

    # BICONVEX OPTIMIZATION
    if opt['wantverbose']:
        print('Performing biconvex optimization...', end='')

    # init
    cNb = cN
    cSb_old = cS
    cNb_old = cN
    numiters = 0  # numiters == 0 means the first estimate was already PSD

    while True:
        # calculate new estimate of cSb
        temp = cD - cNb / ntrial
        cSb, _ = construct_nearest_psd_covariance(temp)

        # calculate new estimate of cNb
        temp = (ncond * (ntrial - 1) * ntrial**2) / (ncond * ntrial**2 * (ntrial - 1) + ncond - 1) * cN \
               + (ncond - 1) / (ncond * ntrial**2 * (ntrial - 1) + ncond - 1) * ntrial * (cD - cSb)
        cNb, _ = construct_nearest_psd_covariance(temp)

        # check deltas
        cScheck = np.corrcoef(cSb_old.flatten(), cSb.flatten())[0, 1]
        cNcheck = np.corrcoef(cNb_old.flatten(), cNb.flatten())[0, 1]

        # convergence?
        if cScheck > 0.999 and cNcheck > 0.999:
            break

        # update
        numiters += 1
        cSb_old = cSb
        cNb_old = cNb

    if opt['wantverbose']:
        print('done.')

    # POST SCALING
    if opt['mode'] == 1:
        # do nothing
        sc = 1
        cSb_rsa = cSb
        splitr = []
    elif opt['mode'] == 2:
        # scale the nearest approximation to match the average variance
        # observed in the original estimate of the signal covariance
        sc = np.maximum(np.mean(np.diag(cS)), 0) / np.mean(np.diag(cSb))  # ensure non-negative scaling
        cSb_rsa, _ = construct_nearest_psd_covariance(cSb * sc)  # scaling and run through construct_nearest_psdcovariance
        splitr = []
    elif opt['mode'] == 0:
        # calculate the number of trials to put into the two splits
        if opt['splitmode'] == 0:
            splitnums = [np.floor(ntrial / 2)]
        elif opt['splitmode'] == 1:
            splitnums = np.unique([2**i for i in range(int(np.floor(np.log2(ntrial / 2))))] + [np.floor(ntrial / 2)])
        elif opt['splitmode'] == 2:
            splitnums = [np.floor(np.floor(ntrial / 2) / 2), np.floor(ntrial / 2)]
            splitnums = [num for num in splitnums if num > 0]

        # calculate data split reliability
        if opt['wantverbose']:
            print('Calculating data split reliability...', end='')

        # first, we need to figure out if we can do exhaustive combinations
        doexhaustive = True
        combolist = {}
        validmatrix = {}
        for nn in range(len(splitnums) - 1, -1, -1):
            # if the dimensionality seems too large, just get out
            if scipy.special.comb(ntrial, splitnums[nn]) > 2 * opt['simchunk']:
                doexhaustive = False
                break

            # calculate the full set of possibilities
            combolist[nn] = list(itertools.combinations(range(ntrial), int(splitnums[nn])))
            ncomb = len(combolist[nn])

            # figure out pairs of splits that are mutually exclusive
            validmatrix[nn] = np.zeros((ncomb, ncomb), dtype=int)
            for r in range(ncomb):
                for c in range(ncomb):
                    if c <= r:  # only the upper triangle as potentially valid
                        continue
                    if not set(combolist[nn][r]).intersection(set(combolist[nn][c])):
                        validmatrix[nn][r, c] = 1

            # if the number of combinations to process is more than opt.simchunk, just give up
            if np.sum(validmatrix[nn]) > opt['simchunk']:
                doexhaustive = False
                break

        # if it looks like we can do it exhaustively, do it!
        if doexhaustive:
            if opt['wantverbose']:
                print('doing exhaustive set of combinations...', end='')
            datasplitr = np.zeros((len(splitnums),1))
            for nn in range(len(splitnums)):
                ncomb = len(combolist[nn])
                temp = []
                for r in range(ncomb):
                    for c in range(ncomb):
                        if validmatrix[nn][r, c]:
                            temp.append(nanreplace(opt['comparefun'](opt['rdmfun'](np.mean(data[:, :, combolist[nn][r]], axis=2)),
                                                                    opt['rdmfun'](np.mean(data[:, :, combolist[nn][c]], axis=2)))))
                datasplitr[nn] = np.median(temp)
            splitr = datasplitr[-1]  # result for the "most trials" data split case

        # otherwise, do the random-sampling approach
        else:
            iicur = 1
            iimax = opt['simchunk']
            datasplitr = np.zeros((len(splitnums), iimax))
            while True:
                for nn in range(len(splitnums)):
                    for si in range(iicur, iimax):
                        temp = np.random.permutation(ntrial)
                        datasplitr[nn, si] = nanreplace(opt['comparefun'](opt['rdmfun'](np.mean(data[:, :, temp[:splitnums[nn]]], axis=2)),
                                                                        opt['rdmfun'](np.mean(data[:, :, temp[splitnums[nn]:splitnums[nn] * 2]], axis=2))))
                robustness = np.mean(np.abs(np.median(datasplitr, axis=1)) / (np.iqr(datasplitr, axis=1) / 2 / np.sqrt(datasplitr.shape[1])))
                if robustness > opt['simthresh']:
                    break
                iicur = iimax
                iimax += opt['simchunk']
                if iimax > opt['maxsimnum']:
                    break
                datasplitr.resize((len(splitnums), iimax))

            splitr = np.median(datasplitr[-1])  # median result for the "most trials" data split case

        if opt['wantverbose']:
            print('done.')

        # calculate model-based split reliability
        if opt['wantverbose']:
            print('Calculating model split reliability...', end='')
        iicur = 1  # current sim number
        iimax = opt['simchunk']  # current targeted max
        modelsplitr = np.zeros((len(opt['scs']), len(splitnums), iimax))

        # precompute
        tempcS = np.zeros((cSb.shape[0], cSb.shape[1], len(opt['scs'])))
        for sci in range(len(opt['scs'])):
            tempcS[:, :, sci], _ = construct_nearest_psd_covariance(cSb * opt['scs'][sci])

        while True:
            robustness = np.zeros(len(opt['scs']))
            for sci in range(len(opt['scs'])):
                for nn in range(len(splitnums)):
                    for si in range(iicur, iimax):
                        signal = np.random.multivariate_normal(mnS.squeeze(), tempcS[:, :, sci], opt['ncconds'])  # cond x voxels
                        noise = np.random.multivariate_normal(mnN.squeeze(), cNb / splitnums[nn], opt['ncconds'] * 2)  # 2*cond x voxels
                        measurement1 = signal + noise[:opt['ncconds'], :]  # cond x voxels
                        measurement2 = signal + noise[opt['ncconds']:, :]  # cond x voxels
                        modelsplitr[sci, nn, si] = nanreplace(opt['comparefun'](opt['rdmfun'](measurement1.T),
                                                                            opt['rdmfun'](measurement2.T)))

                temp = modelsplitr[sci, :, :].reshape(-1, modelsplitr.shape[2])
                robustness[sci] = np.mean(np.abs(np.median(temp, axis=1)) / (scipy.stats.iqr(temp, axis=1) / 2 / np.sqrt(temp.shape[1])))

            robustness = np.mean(robustness)
            if robustness > opt['simthresh']:
                break
            iicur = iimax + 1
            iimax += opt['simchunk']
            if iimax > opt['maxsimnum']:
                break
            modelsplitr.resize((len(opt['scs']), len(splitnums), iimax))

        if opt['wantverbose']:
            print('done.')

        # calculate R^2 between model-based results and the data results and find the max
        if opt['wantverbose']:
            print('Finding best model...', end='')
        # Assuming calccod is a function that calculates coefficient of determination
        R2s = calc_cod(np.median(modelsplitr, axis=2), np.tile(np.median(datasplitr, axis=1), (len(opt['scs']), 1)), 1, None, 0)
        bestii = np.argmax(R2s)
        sc = opt['scs'][bestii]
        if opt['wantverbose']:
            print('done.')

        # impose scaling and run it through construct_nearest_psdcovariance
        cSb_rsa, _ = construct_nearest_psd_covariance(cSb * sc)

        # Warn if data split r is higher than all of the model split r
        temp = np.median(modelsplitr[:, -1, :], axis=1)
        if splitr > np.max(temp):
            warnings.warn('The empirical data split r seems to be out of the range of the model. '
                        'Something may be wrong; results may be inaccurate. Consider increasing the <scs> input.')

        # Sanity check on the smoothness of the R2 results
        if len(R2s) >= 4:
            smoothed_R2s = np.convolve(R2s, [1/3, 1/3, 1/3], mode='valid')
            temp_R2 = calc_cod(smoothed_R2s, R2s[1:-1])  # Assuming calccod is implemented
            if temp_R2 < 90:
                warnings.warn(f'The R2 values appear to be non-smooth (smooth function explains only {temp_R2:.1f}% variance). '
                            'Something may be wrong; results may be inaccurate. Consider increasing simchunk, simthresh, and/or maxsimnum.')

    # Performing Monte Carlo simulations for RSA noise ceiling
    if opt['wantverbose']:
        print('Performing Monte Carlo simulations...', end='')
    ncdist = np.zeros(opt['ncsims'])
    for rr in range(opt['ncsims']):
        signal = np.random.multivariate_normal(mnS.squeeze(), cSb_rsa, opt['ncconds'])
        noise = np.random.multivariate_normal(mnN.squeeze(), cNb / opt['nctrials'], opt['ncconds'])
        measurement = signal + noise
        ncdist[rr] = nanreplace(opt['comparefun'](opt['rdmfun'](signal.T), opt['rdmfun'](measurement.T)))

    if opt['wantverbose']:
        print('done.')

    # Finish up
    # Compute median across simulations
    nc = np.median(ncdist)

    # Prepare additional outputs
    results = {
        'mnN': mnN,
        'cN': cN,
        'cNb': cNb,
        'shrinklevelN': shrinklevelN,
        'shrinklevelD': shrinklevelD,
        'mnS': mnS,
        'cS': cS,
        'cSb': cSb,
        'sc': sc,
        'splitr': splitr,
        'ncsnr': ncsnr,
        'numiters': numiters
    }

    # MAKE A FIGURE
    if opt['mode'] == 0 and opt['wantfig'] != 0:
        if opt['wantverbose']:
            print('Creating figure...')

        fig = plt.figure(figsize=(26, 20))  # Adjusted size for Python
        gs = fig.add_gridspec(4, 6)

        # Mean of Signal
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.hist(mnS, bins=30)  # Assuming 30 bins for histogram
        ax1.set_ylabel('Frequency')
        ax1.set_title('Mean of Signal')

        # Covariance of Signal (Raw)
        ax2 = fig.add_subplot(gs[0, 2:4])
        mx = np.max(np.abs(cS)) if np.max(np.abs(cS)) != 0 else 1
        im2 = ax2.imshow(cS, aspect='equal', cmap='viridis', norm=Normalize(vmin=-mx, vmax=mx))
        fig.colorbar(im2, ax=ax2)
        ax2.set_title('Covariance of Signal (Raw)')

        # Optimized and scaled
        ax3 = fig.add_subplot(gs[0, 4:6])
        mx = np.max(np.abs(cSb_rsa)) if np.max(np.abs(cSb_rsa)) != 0 else 1
        im3 = ax3.imshow(cSb_rsa, aspect='equal', cmap='viridis', norm=Normalize(vmin=-mx, vmax=mx))
        fig.colorbar(im3, ax=ax3)
        ax3.set_title('Optimized and scaled')

        # Mean of Noise
        ax4 = fig.add_subplot(gs[1, 0:2])
        ax4.hist(mnN, bins=30)  # Assuming 30 bins for histogram
        ax4.set_ylabel('Frequency')
        ax4.set_title('Mean of Noise')

        # Covariance of Noise (Optimized)
        ax5 = fig.add_subplot(gs[1, 2:4])
        mx = np.max(np.abs(cNb)) if np.max(np.abs(cNb)) != 0 else 1
        im5 = ax5.imshow(cNb, aspect='equal', cmap='viridis', norm=Normalize(vmin=-mx, vmax=mx))
        fig.colorbar(im5, ax=ax5)
        ax5.set_title('Covariance of Noise (Optimized)')

        # Noise ceiling SNR
        ax6 = fig.add_subplot(gs[1, 4:6])
        ax6.hist(ncsnr, bins=30)  # Assuming 30 bins for histogram
        ax6.set_ylabel('Frequency')
        ax6.set_title('Noise ceiling SNR')

        # Model and Data Similarity (spanning multiple cells)
        ax7 = fig.add_subplot(gs[2:4, 1:4])  # Spans rows 2-4 and columns 1-4
        cmap0 = get_cmap('viridis')(np.linspace(0, 1, len(opt['scs'])))
        for sci in range(len(opt['scs'])):
            md0 = np.median(modelsplitr[sci, :, :], axis=1)
            se0 = scipy.stats.iqr(modelsplitr[sci, :, :], axis=1) / 2 / np.sqrt(modelsplitr.shape[2])
            ax7.errorbar(splitnums, md0, yerr=se0, fmt='o-', color=cmap0[sci], linewidth=1)

            lw0 = 3 if opt['scs'][sci] == sc else 1
            mark0 = 'o' if opt['scs'][sci] == sc else 'x'
            ax7.plot(splitnums, md0, 'r' + mark0 + '-', linewidth=lw0)

        md0 = np.median(datasplitr, axis=1)
        sd0 = scipy.stats.iqr(datasplitr, axis=1) / 2
        se0 = sd0 / np.sqrt(datasplitr.shape[1])
        ax7.errorbar(splitnums, md0, yerr=sd0, fmt='k-', linewidth=1)
        ax7.errorbar(splitnums, md0, yerr=se0, fmt='k-', linewidth=3)
        ax7.plot(splitnums, md0, 'kd-', linewidth=3)
        ax7.set_xlim([np.min(splitnums) - 1, np.max(splitnums) + 1])
        ax7.set_xlabel('Number of trials in each split')
        ax7.set_ylabel('Similarity (comparefun output)')
        
        if doexhaustive:
            ax7.set_title(f"Data (ALL sims); Model ({modelsplitr.shape[2]} sims); splitr={splitr[0]:.3f}")
        else:
            ax7.set_title(f"Data ({datasplitr.shape[1]} sims); Model ({modelsplitr.shape[2]} sims); splitr={splitr[0]:.3f}")
  
        # Scaling factor and R2 (spanning multiple cells)
        ax8 = fig.add_subplot(gs[2:4, 4:6])  # Spans rows 2-4 and columns 4-6
        ax8.plot(opt['scs'], R2s, 'ro-')
        ax8.axvline(x=sc, color='k', linestyle='-', linewidth=3)
        ax8.set_xlabel('Scaling factor')
        ax8.set_ylabel('R^2 between model and data (%)')
        ax8.set_title(f'sc={sc:.2f}, nc={nc:.3f} +/- {scipy.stats.iqr(ncdist)/2/np.sqrt(len(ncdist)):.3f}')

        # Saving the figure
        if opt['wantfig'] != 1:
            plt.savefig(opt['wantfig'])

        if opt['wantverbose']:
            print('done.')

    return nc, ncdist, results
