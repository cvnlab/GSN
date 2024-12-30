import numpy as np
from gsn.perform_gsn import perform_gsn
from gsn.calc_cod import calc_cod
from gsn.utilities import zerodiv
from gsn.convert_covariance_to_correlation import convert_covariance_to_correlation

def gsn_denoise(data, V=None, opt=None):
    """
    Main entry point for denoising.
    
    Args:
        data: shape (nunits, nconds, ntrials)
            The measured responses to different conditions on distinct trials.
            The number of trials must be at least 2 in some scenarios.
        V: basis selection mode or basis matrix
            0 means perform GSN and use eigenvectors of signal covariance (cSb)
            1 means perform GSN and use eigenvectors of signal covariance transformed by inverse noise covariance
            2 means perform GSN and use eigenvectors of noise covariance (cNb)
            3 means naive PCA (eigenvectors of trial-averaged data covariance)
            4 means use randomly generated orthonormal basis
            B means use user-supplied basis B (nunits x D where D >= 1, columns unit-length and orthogonal)
            Default: 0
        opt: dict with optional fields
            cv_mode: int
                0 means cross-validation using n-1 (train) / 1 (test) splits
                1 means cross-validation using 1 (train) / n-1 (test) splits
                -1 means magnitude thresholding based on component magnitudes
                Default: 0
            cv_threshold_per: str
                'population' or 'unit', specifying whether to use unit-wise or population thresholding
                Default: 'unit'
            cv_thresholds: array
                Thresholds to evaluate in cross-validation (positive integers)
                Default: 1:D where D is maximum dimensions
            cv_scoring_fn: callable
                Function to compute denoiser performance
                Default: negative_mse_columns
            mag_type: int
                0 means use eigenvalues (V must be 0,1,2,3)
                1 means use signal variance from data
                Default: 0
            mag_frac: float
                Fraction of maximum magnitude for thresholding
                Default: 0.01
            mag_mode: int
                0 means use smallest number of contiguous dimensions from left
                1 means use all dimensions that survive threshold
                Default: 0
            denoisingtype: int
                0 means trial-averaged denoising
                1 means single-trial denoising
                Default: 0

    Returns:
        dict with fields:
            denoiser: (nunits x nunits) denoising matrix
            cv_scores: (length(cv_thresholds) x ntrials x nunits) CV performance scores
            best_threshold: scalar or array indicating optimal threshold(s)
            denoiseddata: 
                If denoisingtype=0: (nunits x nconds) trial-averaged denoised data
                If denoisingtype=1: (nunits x nconds x ntrials) single-trial denoised data
            fullbasis: (nunits x dims) full basis matrix
            signalsubspace: (nunits x dims) final basis functions for denoising, or None
            dimreduce: (dims x nconds) or (dims x nconds x ntrials) reduced dimension data, or None
            mags: array of component magnitudes, or None
            dimsretained: scalar integer indicating number of dimensions retained, or None
    """

    # 1) Check for infinite or NaN data => some tests want an AssertionError.
    assert np.isfinite(data).all(), "Data contains infinite or NaN values."

    nunits, nconds, ntrials = data.shape

    # 2) If we have fewer than 2 trials, raise an error
    if ntrials < 2:
        raise ValueError("Data must have at least 2 trials.")

    # 2b) Check for minimum number of conditions
    assert nconds >= 2, "Data must have at least 2 conditions to estimate covariance."

    # 3) If V is None => treat it as 0
    if V is None:
        V = 0

    # 4) Prepare default opts
    if opt is None:
        opt = {}
        
    # Validate cv_threshold_per before setting defaults
    if 'cv_threshold_per' in opt:
        if opt['cv_threshold_per'] not in ['unit', 'population']:
            raise KeyError("cv_threshold_per must be 'unit' or 'population'")
            
    # Now set defaults
    opt.setdefault('cv_scoring_fn', negative_mse_columns)
    opt.setdefault('cv_mode', 0)
    opt.setdefault('cv_threshold_per', 'unit')
    opt.setdefault('mag_type', 0)
    opt.setdefault('mag_frac', 0.01)
    opt.setdefault('mag_mode', 0)
    opt.setdefault('denoisingtype', 0)  # Default to trial-averaged denoising

    gsn_results = None

    # 5) If V is an integer => glean basis from GSN results
    if isinstance(V, int):
        if V not in [0, 1, 2, 3, 4]:
            raise ValueError("V must be in [0..4] (int) or a 2D numpy array.")
        gsn_results = perform_gsn(data)
        cSb = gsn_results['cSb']
        cNb = gsn_results['cNb']

        # Helper for pseudo-inversion (in case cNb is singular)
        def inv_or_pinv(mat):
            return np.linalg.pinv(mat)

        if V == 0:
            # Just eigen-decompose cSb
            evals, evecs = np.linalg.eigh(cSb)
            basis = np.fliplr(evecs)
            mags = np.abs(np.flip(evals))  # Store magnitudes for later
        elif V == 1:
            cNb_inv = inv_or_pinv(cNb)
            transformed_cov = cNb_inv @ cSb
            evals, evecs = np.linalg.eigh(transformed_cov)
            basis = np.fliplr(evecs)
            mags = np.abs(np.flip(evals))  # Store magnitudes for later
        elif V == 2:
            evals, evecs = np.linalg.eigh(cNb)
            basis = np.fliplr(evecs)
            mags = np.abs(np.flip(evals))  # Store magnitudes for later
        elif V == 3:
            trial_avg = np.mean(data, axis=2)  # shape (nunits, nconds)
            cov_matrix = np.cov(trial_avg)     # shape (nunits, nunits)
            evals, evecs = np.linalg.eigh(cov_matrix)
            basis = np.fliplr(evecs)
            mags = np.abs(np.flip(evals))  # Store magnitudes for later
        else:  # V == 4 => random orthonormal
            # Generate a random basis with same dimensions as eigenvector basis
            rand_mat = np.random.randn(nunits, nunits)  # Start with square matrix
            basis, _ = np.linalg.qr(rand_mat)
            # Only keep first nunits columns to match eigenvector basis dimensions
            basis = basis[:, :nunits]
            mags = np.ones(nunits)  # No meaningful magnitudes for random basis
    else:
        # If V not int => must be a numpy array
        if not isinstance(V, np.ndarray):
            raise ValueError("If V is not int, it must be a numpy array.")
        
        # Check orthonormality of user-supplied basis
        if V.shape[0] != nunits:
            raise ValueError(f"Basis must have {nunits} rows, got {V.shape[0]}")
        if V.shape[1] < 1:
            raise ValueError("Basis must have at least 1 column")
            
        # Check unit-length columns
        norms = np.linalg.norm(V, axis=0)
        if not np.allclose(norms, 1):
            raise ValueError("Basis columns must be unit length")
            
        # Check orthogonality
        gram = V.T @ V
        if not np.allclose(gram, np.eye(V.shape[1])):
            raise ValueError("Basis columns must be orthogonal")
            
        basis = V
        # For user-supplied basis, compute magnitudes based on variance in basis
        trial_avg = np.mean(data, axis=2)  # shape (nunits, nconds)
        trial_avg_reshaped = trial_avg.T  # shape (ncond, nvox)
        proj_data = trial_avg_reshaped @ basis  # shape (ncond, basis_dim)
        mags = np.var(proj_data, axis=0)  # variance along conditions for each basis dimension

    # Store the full basis and magnitudes for return
    fullbasis = basis.copy()
    stored_mags = mags.copy()  # Store magnitudes for later use

    # 6) Default cross-validation thresholds if not provided
    if 'cv_thresholds' not in opt:
        opt['cv_thresholds'] = np.arange(1, basis.shape[1] + 1)
    else:
        # Validate cv_thresholds
        thresholds = np.array(opt['cv_thresholds'])
        if not np.all(thresholds > 0):
            raise ValueError("cv_thresholds must be positive integers")
        if not np.all(thresholds == thresholds.astype(int)):
            raise ValueError("cv_thresholds must be integers")
        if not np.all(np.diff(thresholds) > 0):
            raise ValueError("cv_thresholds must be in sorted order with unique values")

    # Initialize return dictionary with None values
    results = {
        'denoiser': None,
        'cv_scores': None,
        'best_threshold': None,
        'denoiseddata': None,
        'fullbasis': fullbasis,
        'signalsubspace': None,
        'dimreduce': None,
        'mags': None,
        'dimsretained': None
    }

    # 7) Decide cross-validation or magnitude-threshold
    # We'll treat negative or zero cv_mode as "do magnitude thresholding."
    if opt['cv_mode'] >= 0:
        denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, signalsubspace, dimreduce = perform_cross_validation(data, basis, opt)
        
        # Update results dictionary
        results.update({
            'denoiser': denoiser,
            'cv_scores': cv_scores,
            'best_threshold': best_threshold,
            'denoiseddata': denoiseddata,
            'fullbasis': fullbasis
        })
        
        # Add population-specific returns if applicable
        if opt['cv_threshold_per'] == 'population':
            results.update({
                'signalsubspace': signalsubspace,
                'dimreduce': dimreduce
            })
    else:
        denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, signalsubspace, dimreduce, mags, dimsretained = perform_magnitude_thresholding(data, basis, gsn_results, opt, V)
        
        # Update results dictionary with all magnitude thresholding returns
        results.update({
            'denoiser': denoiser,
            'cv_scores': cv_scores,
            'best_threshold': best_threshold,
            'denoiseddata': denoiseddata,
            'fullbasis': fullbasis,
            'mags': stored_mags,
            'dimsretained': dimsretained,
            'signalsubspace': signalsubspace,
            'dimreduce': dimreduce
        })

    return results

def perform_cross_validation(data, basis, opt):
    """Perform cross-validation to find optimal threshold.
    
    cv_mode options:
    0: Denoise the average of n-1 trials, test against held out trial
    1: Denoise single trial, test against average of n-1 trials
    -1: No cross-validation, use magnitude thresholding instead
    """
    nunits, nconds, ntrials = data.shape
    cv_mode = opt['cv_mode']
    thresholds = opt['cv_thresholds']
    opt.setdefault('cv_scoring_fn', negative_mse_columns)
    threshold_per = opt.get('cv_threshold_per')
    scoring_fn = opt['cv_scoring_fn']
    denoisingtype = opt.get('denoisingtype', 0)
    
    # Initialize cv_scores
    cv_scores = np.zeros((len(thresholds), ntrials, nunits))

    if cv_mode == -1:
        # No cross-validation - use magnitude thresholding
        data_ctv = data.copy().transpose(1, 2, 0)  # (nconds, ntrials, nunits)
        ncsnrs = np.zeros(nunits)
        for i in range(data_ctv.shape[2]):
            this_eigv = basis[:, i]
            proj_data = np.dot(data_ctv, this_eigv)
            _, ncsnr, _, _ = compute_noise_ceiling(proj_data[np.newaxis, ...])
            ncsnrs[i] = ncsnr

        # Find first index where SNR hits 0
        if np.sum(ncsnrs == 0) == 0:
            raise ValueError('Basis SNR never hits 0. Adjust cross-validation settings.')
        best_threshold = np.argwhere(ncsnrs == 0)[0, 0]
        denoising_fn = np.concatenate([np.ones(best_threshold), np.zeros(basis.shape[1] - best_threshold)])
        denoiser = basis @ np.diag(denoising_fn) @ basis.T
        return denoiser, cv_scores, best_threshold, None, basis, None, None

    for tr in range(ntrials):
        # Define cross-validation splits based on cv_mode
        if cv_mode == 0:
            # Denoise average of n-1 trials, test against held out trial
            train_trials = np.setdiff1d(np.arange(ntrials), tr)
            train_avg = np.mean(data[:, :, train_trials], axis=2)  # Average n-1 trials
            test_data = data[:, :, tr]  # Single held-out trial
            
            for tt, threshold in enumerate(thresholds):
                safe_thr = min(threshold, basis.shape[1])
                denoising_fn = np.concatenate([np.ones(safe_thr), np.zeros(basis.shape[1] - safe_thr)])
                denoiser = basis @ np.diag(denoising_fn) @ basis.T
                
                # Denoise the training average
                train_denoised = (train_avg.T @ denoiser).T
                cv_scores[tt, tr] = scoring_fn(test_data.T, train_denoised.T)
                  
        elif cv_mode == 1:
            # Denoise single trial, test against average of n-1 trials
            dataA = data[:, :, tr].T  # Single trial (nconds x nunits)
            dataB = np.mean(data[:, :, np.setdiff1d(np.arange(ntrials), tr)], axis=2).T  # Mean of other trials
            
            for tt, threshold in enumerate(thresholds):
                safe_thr = min(threshold, basis.shape[1])
                denoising_fn = np.concatenate([np.ones(safe_thr), np.zeros(basis.shape[1] - safe_thr)])
                denoiser = basis @ np.diag(denoising_fn) @ basis.T
                
                # Denoise the single trial
                dataA_denoised = dataA @ denoiser
                cv_scores[tt, tr] = scoring_fn(dataB, dataA_denoised)
                
    # Decide best threshold
    if threshold_per == 'population':
        # Average over trials and units for population threshold
        avg_scores = np.mean(cv_scores, axis=(1, 2))  # (len(thresholds),)
        best_ix = np.argmax(avg_scores)
        best_threshold = thresholds[best_ix]
        safe_thr = min(best_threshold, basis.shape[1])
        denoiser = basis[:, :safe_thr] @ basis[:, :safe_thr].T
    else:
        # unit-wise: average over trials only
        avg_scores = np.mean(cv_scores, axis=1)  # (len(thresholds), nunits)
        best_thresh_unitwise = []
        for unit_i in range(nunits):
            best_idx = np.argmax(avg_scores[:, unit_i])
            best_thresh_unitwise.append(thresholds[best_idx])
        best_thresh_unitwise = np.array(best_thresh_unitwise)
        best_threshold = best_thresh_unitwise
        
        # Construct unit-wise denoiser
        denoiser = np.zeros((nunits, nunits))
        for unit_i in range(nunits):
            # For each unit, create its own denoising vector using its threshold
            safe_thr = min(int(best_threshold[unit_i]), basis.shape[1])
            unit_denoiser = basis[:, :safe_thr] @ basis[:, :safe_thr].T
            # Use the column corresponding to this unit
            denoiser[:, unit_i] = unit_denoiser[:, unit_i]

    # Calculate denoiseddata based on denoisingtype
    if denoisingtype == 0:
        # Trial-averaged denoising
        trial_avg = np.mean(data, axis=2)
        denoiseddata = (trial_avg.T @ denoiser).T
    else:
        # Single-trial denoising
        denoiseddata = np.zeros_like(data)
        for t in range(ntrials):
            denoiseddata[:, :, t] = (data[:, :, t].T @ denoiser).T

    # Calculate additional return values
    fullbasis = basis.copy()
    signalsubspace = basis[:, :safe_thr]
    
    # Project data onto signal subspace
    if denoisingtype == 0:
        trial_avg = np.mean(data, axis=2)
        dimreduce = signalsubspace.T @ trial_avg  # (safe_thr, nconds)
    else:
        dimreduce = np.zeros((safe_thr, nconds, ntrials))
        for t in range(ntrials):
            dimreduce[:, :, t] = signalsubspace.T @ data[:, :, t]

    # Return values based on threshold_per
    if threshold_per == 'population':
        return denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, signalsubspace, dimreduce
    else:  # 'unit'
        return denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, None, None

def perform_magnitude_thresholding(data, basis, gsn_results, opt, V):
    """Perform magnitude thresholding to determine the optimal number of dimensions to retain.

    Args:
        data: (nunits x nconds x ntrials) data array
        basis: (nunits x dims) basis matrix
        gsn_results: dictionary containing cSb and cNb matrices
        opt: dictionary of options
        V: basis selection mode (0..4) or custom matrix

    Returns:
        denoiser: (nunits x nunits) denoising matrix
        cv_scores: empty array (not used in magnitude thresholding)
        best_threshold: scalar integer indicating number of dimensions retained
        denoiseddata: (nunits x nconds) or (nunits x nconds x ntrials) denoised data
        fullbasis: (nunits x dims) full basis matrix
        signalsubspace: (nunits x dims) final set of basis functions
        dimreduce: (dims x nconds) or (dims x nconds x ntrials) projected data
        mags: array of component magnitudes
        dimsretained: scalar integer indicating number of dimensions retained
    """
    
    nunits, nconds, ntrials = data.shape
    mag_type = opt.get('mag_type', 0)
    mag_frac = opt.get('mag_frac', 0.01)
    mag_mode = opt.get('mag_mode', 0)
    threshold_per = opt.get('cv_threshold_per', 'unit')
    denoisingtype = opt.get('denoisingtype', 0)

    cv_scores = np.array([])  # Not used in magnitude thresholding

    # Get magnitudes based on mag_type
    if mag_type == 0:
        # Eigen-based threshold
        if isinstance(V, (int, np.integer)):
            if V == 0:
                evals = np.linalg.eigvalsh(gsn_results['cSb'])
                magnitudes = evals  # Keep original order for eigenvalues
            elif V == 1:
                cNb_inv = np.linalg.pinv(gsn_results['cNb'])
                matM = cNb_inv @ gsn_results['cSb']
                evals = np.linalg.eigvalsh(matM)
                magnitudes = evals  # Keep original order for eigenvalues
            elif V == 2:
                evals = np.linalg.eigvalsh(gsn_results['cNb'])
                magnitudes = evals  # Keep original order for eigenvalues
            elif V == 3:
                trial_avg = np.mean(data, axis=2)
                cov_mat = np.cov(trial_avg.T)
                evals = np.linalg.eigvalsh(cov_mat)
                magnitudes = evals  # Keep original order for eigenvalues
            else:  # V == 4
                magnitudes = np.ones(basis.shape[1])
        else:
            # For user-supplied basis, compute projection variances
            trial_avg = np.mean(data, axis=2)
            proj = trial_avg.T @ basis
            magnitudes = np.var(proj, axis=0)
    else:
        # Variance-based threshold in user basis
        trial_avg = np.mean(data, axis=2)
        proj = trial_avg.T @ basis
        magnitudes = np.var(proj, axis=0)

    # Determine threshold as fraction of maximum magnitude
    threshold_val = mag_frac * np.max(np.abs(magnitudes))
    surviving = np.abs(magnitudes) >= threshold_val

    # Find dimensions to retain based on mag_mode
    surv_idx = np.where(surviving)[0]

    if len(surv_idx) == 0:
        # If no dimensions survive, return zero matrices
        denoiser = np.zeros((nunits, nunits))
        denoiseddata = np.zeros((nunits, nconds)) if denoisingtype == 0 else np.zeros_like(data)
        signalsubspace = basis[:, :0]  # Empty but valid shape
        dimreduce = np.zeros((0, nconds)) if denoisingtype == 0 else np.zeros((0, nconds, ntrials))
        dimsretained = 0
        best_threshold = np.array([])
        return denoiser, cv_scores, best_threshold, denoiseddata, basis, signalsubspace, dimreduce, magnitudes, dimsretained

    if mag_mode == 0:  # Contiguous from left
        # For contiguous from left, we want the leftmost block
        # Find the first gap after the start
        if len(surv_idx) == 1:
            dimsretained = 1
            best_threshold = surv_idx
        else:
            # Take all dimensions up to the first gap
            gaps = np.where(np.diff(surv_idx) > 1)[0]
            if len(gaps) > 0:
                dimsretained = gaps[0] + 1
                best_threshold = surv_idx[:dimsretained]
            else:
                # No gaps, take all surviving dimensions
                dimsretained = len(surv_idx)
                best_threshold = surv_idx
    else:  # Keep all dimensions above threshold
        dimsretained = len(surv_idx)
        best_threshold = surv_idx

    # Create denoising matrix using retained dimensions
    denoising_fn = np.zeros(basis.shape[1])
    denoising_fn[best_threshold] = 1
    denoiser = basis @ np.diag(denoising_fn) @ basis.T

    # Calculate denoised data
    if denoisingtype == 0:
        # Trial-averaged denoising
        trial_avg = np.mean(data, axis=2)
        denoiseddata = (trial_avg.T @ denoiser).T
    else:
        # Single-trial denoising
        denoiseddata = np.zeros_like(data)
        for t in range(ntrials):
            denoiseddata[:, :, t] = (data[:, :, t].T @ denoiser).T

    # Calculate signal subspace and reduced dimensions
    signalsubspace = basis[:, best_threshold]
    if denoisingtype == 0:
        trial_avg = np.mean(data, axis=2)
        dimreduce = signalsubspace.T @ trial_avg
    else:
        dimreduce = np.zeros((len(best_threshold), nconds, ntrials))
        for t in range(ntrials):
            dimreduce[:, :, t] = signalsubspace.T @ data[:, :, t]

    return denoiser, cv_scores, best_threshold, denoiseddata, basis, signalsubspace, dimreduce, magnitudes, dimsretained

def negative_mse_columns(x, y):
    """Calculate the negative mean squared error per column.
    
    Args:
        x: array of shape (nconds, nunits)
        y: array of shape (nconds, nunits)
    Returns:
        array of shape (nunits,) containing negative MSE for each column
    """
    if x.shape[0] == 0 or y.shape[0] == 0:
        return np.zeros(x.shape[1])  # Return zeros for empty arrays
    return -np.mean((x - y) ** 2, axis=0)
