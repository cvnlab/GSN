import numpy as np
from gsn.perform_gsn import perform_gsn

def gsn_denoise(data, V=None, opt=None):
    """
    Denoise neural data using Generative Modeling of Signal and Noise (GSN).

    Algorithm Details:
    -----------------
    The GSN denoising algorithm works by identifying dimensions in the neural data that contain
    primarily signal rather than noise. It does this in several steps:

    1. Signal and Noise Estimation:
        - For each condition, computes mean response across trials (signal estimate)
        - For each condition, computes variance across trials (noise estimate)
        - Builds signal (cSb) and noise (cNb) covariance matrices across conditions

    2. Basis Selection (<V> parameter):
        - V=0: Uses eigenvectors of signal covariance (cSb)
        - V=1: Uses eigenvectors of signal covariance transformed by inverse noise covariance
        - V=2: Uses eigenvectors of noise covariance (cNb)
        - V=3: Uses PCA on trial-averaged data
        - V=4: Uses random orthonormal basis
        - V=matrix: Uses user-supplied orthonormal basis

    3. Dimension Selection:
        The algorithm must decide how many dimensions to keep. This can be done in two ways:

        a) Cross-validation (<cv_mode> >= 0):
            - Splits trials into training and testing sets
            - For training set:
                * Projects data onto different numbers of basis dimensions
                * Creates denoising matrix for each dimensionality
            - For test set:
                * Measures how well denoised training data predicts test data
                * Uses mean squared error (MSE) as prediction metric
            - Selects number of dimensions that gives best prediction
            - Can be done per-unit or for whole population

        b) Magnitude Thresholding (<cv_mode> = -1):
            - Computes "magnitude" for each dimension:
                * Either eigenvalues (signal strength)
                * Or variance explained in the data
            - Sets threshold as fraction of maximum magnitude
            - Keeps dimensions above threshold either:
                * Contiguously from strongest dimension
                * Or any dimension above threshold

    4. Denoising:
        - Creates denoising matrix using selected dimensions
        - For trial-averaged denoising:
            * Averages data across trials
            * Projects through denoising matrix
        - For single-trial denoising:
            * Projects each trial through denoising matrix
        - Returns denoised data and diagnostic information

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <data> - shape (nunits, nconds, ntrials). This indicates the measured
        responses to different conditions on distinct trials.
        The number of trials (ntrials) must be at least 2.
    <V> - shape (nunits, nunits) or scalar. Indicates the set of basis functions to use.
        0 means perform GSN and use the eigenvectors of the
          signal covariance estimate (cSb)
        1 means perform GSN and use the eigenvectors of the
          signal covariance estimate, transformed by the inverse of 
          the noise covariance estimate (inv(cNb)*cSb)
        2 means perform GSN and use the eigenvectors of the 
          noise covariance estimate (cNb)
        3 means naive PCA (i.e. eigenvectors of the covariance
          of the trial-averaged data)
        4 means use a randomly generated orthonormal basis (nunits, nunits)
        B means use user-supplied basis B. The dimensionality of B
          should be (nunits, D) where D >= 1. The columns of B should
          unit-length and pairwise orthogonal.
        Default: 0.
    <opt> - dict with the following optional fields:
        <cv_mode> - scalar. Indicates how to determine the optimal threshold:
          0 means cross-validation using n-1 (train) / 1 (test) splits of trials.
          1 means cross-validation using 1 (train) / n-1 (test) splits of trials.
         -1 means do not perform cross-validation and instead set the threshold
            based on when the magnitudes of components drop below
            a certain fraction (see <mag_frac>).
          Default: 0.
        <cv_threshold_per> - string. 'population' or 'unit', specifying 
          whether to use unit-wise thresholding (possibly different thresholds
          for different units) or population thresholding (one threshold for
          all units). Matters only when <cv_mode> is 0 or 1. Default: 'unit'.
        <cv_thresholds> - shape (1, n_thresholds). Vector of thresholds to evaluate in
          cross-validation. Matters only when <cv_mode> is 0 or 1.
          Each threshold is a positive integer indicating a potential 
          number of dimensions to retain. Should be in sorted order and 
          elements should be unique. Default: 1:D where D is the 
          maximum number of dimensions.
        <cv_scoring_fn> - function handle. For <cv_mode> 0 or 1 only.
          It is a function handle to compute denoiser performance.
          Default: negative_mse_columns. 
        <mag_type> - scalar. Indicates how to obtain component magnitudes.
          Matters only when <cv_mode> is -1.
          0 means use eigenvalues (<V> must be 0, 1, 2, or 3)
          1 means use signal variance computed from the data
          Default: 0.
        <mag_frac> - scalar. Indicates a fraction of the maximum magnitude
          component. Matters only when <cv_mode> is -1.
          Default: 0.01.
        <mag_mode> - scalar. Indicates how to select dimensions. Matters only 
          when <cv_mode> is -1.
          0 means use the smallest number of dimensions that all survive threshold.
            In this case, the dimensions returned are all contiguous from the left.
          1 means use all dimensions that survive the threshold.
            In this case, the dimensions returned are not necessarily contiguous.
          Default: 0.
        <denoisingtype> - scalar. Indicates denoising type:
          0 means denoising in the trial-averaged sense
          1 means single-trial-oriented denoising
          Note that if <cv_mode> is 0, you probably want <denoisingtype> to be 0,
          and if <cv_mode> is 1, you probably want <denoisingtype> to be 1, but
          the code is deliberately flexible for users to specify what they want.
          Default: 0.

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    Returns a dictionary with the following fields:

    Return in all cases:
        <denoiser> - shape (nunits, nunits). This is the denoising matrix.
        <fullbasis> - shape (nunits, dims). This is the full set of basis functions.

    In the case that <denoisingtype> is 0, we return:
        <denoiseddata> - shape (nunits, nconds). This is the trial-averaged data
          after applying the denoiser.

    In the case that <denoisingtype> is 1, we return:
        <denoiseddata> - shape (nunits, nconds, ntrials). This is the 
          single-trial data after applying the denoiser.

    In the case that <cv_mode> is 0 or 1 (cross-validation):
        If <cv_threshold_per> is 'population', we return:
          <best_threshold> - shape (1, 1). The optimal threshold (a single integer),
            indicating how many dimensions are retained.
          <signalsubspace> - shape (nunits, best_threshold). This is the final set of basis
            functions selected for denoising (i.e. the subspace into which
            we project). The number of basis functions is equal to <best_threshold>.
          <dimreduce> - shape (best_threshold, nconds) or (best_threshold, nconds, ntrials). This
            is the trial-averaged data (or single-trial data) after denoising.
            Importantly, we do not reconstruct the original units but leave
            the data projected into the set of reduced dimensions.
        If <cv_threshold_per> is 'unit', we return:
          <best_threshold> - shape (1, nunits). The optimal threshold for each unit.
        In both cases ('population' or 'unit'), we return:
          <denoised_cv_scores> - shape (n_thresholds, ntrials, nunits).
            Cross-validation performance scores for each threshold.

    In the case that <cv_mode> is -1 (magnitude-based):
        <mags> - shape (1, dims). Component magnitudes used for thresholding.
        <dimsretained> - shape (1, n_retained). The indices of the dimensions retained.
        <signalsubspace> - shape (nunits, n_retained). This is the final set of basis
          functions selected for denoising (i.e. the subspace into which
          we project).
        <dimreduce> - shape (n_retained, nconds) or (n_retained, nconds, ntrials). This
          is the trial-averaged data (or single-trial data) after denoising.
          Importantly, we do not reconstruct the original units but leave
          the data projected into the set of reduced dimensions.

    -------------------------------------------------------------------------
    Examples:
    -------------------------------------------------------------------------

        # Basic usage with default options
        data = np.random.randn(100, 200, 3)  # 100 voxels, 200 conditions, 3 trials
        opt = {
            'cv_mode': 0,  # n-1 train / 1 test split
            'cv_threshold_per': 'unit',  # Same threshold for all units
            'cv_thresholds': np.arange(100),  # Test all possible dimensions
            'cv_scoring_fn': negative_mse_columns,  # Use negative MSE as scoring function
            'denoisingtype': 1  # Single-trial denoising
        }
        results = gsn_denoise(data, None, opt)

        # Using magnitude thresholding
        opt = {
            'cv_mode': -1,  # Use magnitude thresholding
            'mag_frac': 0.1,  # Keep components > 10% of max
            'mag_mode': 0  # Use contiguous dimensions
        }
        results = gsn_denoise(data, 0, opt)

        # Unit-wise cross-validation
        opt = {
            'cv_mode': 0,  # Leave-one-out CV
            'cv_threshold_per': 'unit',  # Unit-specific thresholds
            'cv_thresholds': [1, 2, 3]  # Test these dimensions
        }
        results = gsn_denoise(data, 0, opt)

        # Single-trial denoising with population threshold
        opt = {
            'denoisingtype': 1,  # Single-trial mode
            'cv_threshold_per': 'population'  # Same dims for all units
        }
        results = gsn_denoise(data, 0, opt)
        denoised_trials = results['denoiseddata']  # [nunits x nconds x ntrials]

        # Custom basis
        nunits = data.shape[0]
        custom_basis = np.linalg.qr(np.random.randn(nunits, nunits))[0]
        results = gsn_denoise(data, custom_basis)

    -------------------------------------------------------------------------
    History:
    -------------------------------------------------------------------------

        - 2025/01/06 - Initial version.
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

    # Check if basis vectors are unit length and normalize if not
    if isinstance(V, np.ndarray):
        # First check and fix unit length
        vector_norms = np.sqrt(np.sum(V**2, axis=0))
        if not np.allclose(vector_norms, 1, rtol=0, atol=1e-10):
            print('Normalizing basis vectors to unit length...')
            V = V / vector_norms

        # Then check orthogonality
        gram = V.T @ V
        if not np.allclose(gram, np.eye(gram.shape[0]), rtol=0, atol=1e-10):
            print('Adjusting basis vectors to ensure orthogonality...')
            V = make_orthonormal(V)

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
        gsn_results = perform_gsn(data, {'wantverbose': False})
        cSb = gsn_results['cSb']
        cNb = gsn_results['cNb']

        # Helper for pseudo-inversion (in case cNb is singular)
        def inv_or_pinv(mat):
            return np.linalg.pinv(mat)

        if V == 0:
            # Just eigen-decompose cSb
            evals, evecs = np.linalg.eigh(cSb)
            # Sort by absolute value of eigenvalues
            idx = np.argsort(np.abs(evals))[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]
            basis = evecs
            magnitudes = np.abs(evals)  # No need to flip, already sorted
        elif V == 1:
            cNb_inv = inv_or_pinv(cNb)
            transformed_cov = cNb_inv @ cSb
            evals, evecs = np.linalg.eigh(transformed_cov)
            # Sort by absolute value of eigenvalues
            idx = np.argsort(np.abs(evals))[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]
            basis = evecs
            magnitudes = np.abs(evals)  # No need to flip, already sorted
        elif V == 2:
            evals, evecs = np.linalg.eigh(cNb)
            # Sort by absolute value of eigenvalues
            idx = np.argsort(np.abs(evals))[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]
            basis = evecs
            magnitudes = np.abs(evals)  # No need to flip, already sorted
        elif V == 3:
            trial_avg = np.mean(data, axis=2)
            cov_mat = np.cov(trial_avg)
            evals, evecs = np.linalg.eigh(cov_mat)
            # Sort by absolute value of eigenvalues
            idx = np.argsort(np.abs(evals))[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]
            basis = evecs
            magnitudes = np.abs(evals)  # No need to flip, already sorted
        else:  # V == 4
            # Generate a random basis with same dimensions as eigenvector basis
            rand_mat = np.random.randn(nunits, nunits)  # Start with square matrix
            basis, _ = np.linalg.qr(rand_mat)
            # Only keep first nunits columns to match eigenvector basis dimensions
            basis = basis[:, :nunits]
            magnitudes = np.ones(nunits)  # No meaningful magnitudes for random basis
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
            
        basis = V.copy()
        # For user-supplied basis, compute magnitudes based on variance in basis
        trial_avg = np.mean(data, axis=2)  # shape (nunits, nconds)
        trial_avg_reshaped = trial_avg.T  # shape (ncond, nvox)
        proj_data = trial_avg_reshaped @ basis  # shape (ncond, basis_dim)
        magnitudes = np.var(proj_data, axis=0, ddof=1)  # variance along conditions for each basis dimension
        
    # Store the full basis and magnitudes for return
    fullbasis = basis.copy()
    stored_mags = magnitudes.copy()  # Store magnitudes for later use

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
    """
    Perform cross-validation to determine optimal denoising dimensions.

    Uses cross-validation to determine how many dimensions to retain for denoising:
    1. Split trials into training and testing sets
    2. Project training data into basis
    3. Create denoising matrix for each dimensionality
    4. Measure prediction quality on test set
    5. Select threshold that gives best predictions

    The splitting can be done in two ways:
    - Leave-one-out: Use n-1 trials for training, 1 for testing
    - Keep-one-in: Use 1 trial for training, n-1 for testing

    Inputs:
    -----------
    <data> - shape (nunits, nconds, ntrials). Neural response data to denoise.
    <basis> - shape (nunits, dims). Orthonormal basis for denoising.
    <opt> - dict with fields:
        <cv_mode> - scalar. 
            0: n-1 train / 1 test split
            1: 1 train / n-1 test split
        <cv_threshold_per> - string.
            'unit': different thresholds per unit
            'population': same threshold for all units
        <cv_thresholds> - shape (1, n_thresholds).
            Dimensions to test
        <cv_scoring_fn> - function handle.
            Function to compute prediction error
        <denoisingtype> - scalar.
            0: trial-averaged denoising
            1: single-trial denoising

    Returns:
    --------
    <denoiser> - shape (nunits, nunits). Matrix that projects data onto denoised space.
    <cv_scores> - shape (n_thresholds, ntrials, nunits). Cross-validation scores for each threshold.
    <best_threshold> - shape (1, nunits) or scalar. Selected threshold(s).
    <denoiseddata> - shape (nunits, nconds) or (nunits, nconds, ntrials). Denoised neural responses.
    <fullbasis> - shape (nunits, dims). Complete basis used for denoising.
    <signalsubspace> - shape (nunits, best_threshold) or []. Final basis functions used for denoising.
    <dimreduce> - shape (best_threshold, nconds) or (best_threshold, nconds, ntrials) or []. 
        Data projected onto signal subspace.
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
    if threshold_per == 'population':
        signalsubspace = basis[:, :safe_thr]
        
        # Project data onto signal subspace
        if denoisingtype == 0:
            trial_avg = np.mean(data, axis=2)
            dimreduce = signalsubspace.T @ trial_avg  # (safe_thr, nconds)
        else:
            dimreduce = np.zeros((safe_thr, nconds, ntrials))
            for t in range(ntrials):
                dimreduce[:, :, t] = signalsubspace.T @ data[:, :, t]
    else:
        signalsubspace = None
        dimreduce = None

    return denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, signalsubspace, dimreduce

def perform_magnitude_thresholding(data, basis, gsn_results, opt, V):
    """
    Select dimensions using magnitude thresholding.

    Implements the magnitude thresholding procedure for GSN denoising.
    Selects dimensions based on their magnitudes (eigenvalues or variances)
    rather than using cross-validation.

    Supports two modes:
    - Contiguous selection of the left-most group of dimensions above threshold
    - Selection of any dimension above threshold

    Algorithm Details:
    1. Compute magnitudes for each dimension:
       - Either eigenvalues from decomposition
       - Or variance explained in the data
    2. Set threshold as fraction of maximum magnitude
    3. Select dimensions either:
       - Contiguously from strongest dimension
       - Or any dimension above threshold
    4. Create denoising matrix using selected dimensions

    Parameters:
    -----------
    <data> - shape (nunits, nconds, ntrials). Neural response data to denoise.
    <basis> - shape (nunits, dims). Orthonormal basis for denoising.
    <gsn_results> - dict. Results from GSN computation containing:
        <cSb> - shape (nunits, nunits). Signal covariance matrix.
        <cNb> - shape (nunits, nunits). Noise covariance matrix.
    <opt> - dict with fields:
        <mag_type> - scalar. How to obtain component magnitudes:
            0: use eigenvalues (<V> must be 0, 1, 2, or 3)
            1: use signal variance computed from data
        <mag_frac> - scalar. Fraction of maximum magnitude to use as threshold.
        <mag_mode> - scalar. How to select dimensions:
            0: contiguous from strongest dimension
            1: any dimension above threshold
        <denoisingtype> - scalar. Type of denoising:
            0: trial-averaged
            1: single-trial
    <V> - scalar or matrix. Basis selection mode or custom basis.

    Returns:
    --------
    <denoiser> - shape (nunits, nunits). Matrix that projects data onto denoised space.
    <cv_scores> - shape (0, 0). Empty array (not used in magnitude thresholding).
    <best_threshold> - shape (1, n_retained). Selected dimensions.
    <denoiseddata> - shape (nunits, nconds) or (nunits, nconds, ntrials). Denoised neural responses.
    <basis> - shape (nunits, dims). Complete basis used for denoising.
    <signalsubspace> - shape (nunits, n_retained). Final basis functions used for denoising.
    <dimreduce> - shape (n_retained, nconds) or (n_retained, nconds, ntrials). 
        Data projected onto signal subspace.
    <magnitudes> - shape (1, dims). Component magnitudes used for thresholding.
    <dimsretained> - scalar. Number of dimensions retained.
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
                # Get both eigenvalues and eigenvectors
                evals, evecs = np.linalg.eigh(gsn_results['cSb'])
                # Sort by magnitude in descending order
                sort_idx = np.argsort(-np.abs(evals))  # Descending order
                evals = evals[sort_idx]
                evecs = evecs[:, sort_idx]
                magnitudes = np.abs(evals)
                basis = evecs  # Use sorted eigenvectors as basis
            elif V == 1:
                cNb_inv = np.linalg.pinv(gsn_results['cNb'])
                matM = cNb_inv @ gsn_results['cSb']
                evals, evecs = np.linalg.eigh(matM)
                sort_idx = np.argsort(-np.abs(evals))  # Descending order
                evals = evals[sort_idx]
                evecs = evecs[:, sort_idx]
                magnitudes = np.abs(evals)
                basis = evecs
            elif V == 2:
                evals, evecs = np.linalg.eigh(gsn_results['cNb'])
                sort_idx = np.argsort(-np.abs(evals))  # Descending order
                evals = evals[sort_idx]
                evecs = evecs[:, sort_idx]
                magnitudes = np.abs(evals)
                basis = evecs
            elif V == 3:
                trial_avg = np.mean(data, axis=2)
                cov_mat = np.cov(trial_avg)
                evals, evecs = np.linalg.eigh(cov_mat)
                sort_idx = np.argsort(-np.abs(evals))  # Descending order
                evals = evals[sort_idx]
                evecs = evecs[:, sort_idx]
                magnitudes = np.abs(evals)
                basis = evecs
            else:  # V == 4
                magnitudes = np.ones(basis.shape[1])
        else:
            # For user-supplied basis, compute projection variances
            trial_avg = np.mean(data, axis=2)
            proj = trial_avg.T @ basis
            magnitudes = np.var(proj, axis=0, ddof=1)
    else:
        # Variance-based threshold in user basis
        trial_avg = np.mean(data, axis=2)
        proj = trial_avg.T @ basis
        magnitudes = np.var(proj, axis=0, ddof=1)

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
    """
    Calculate negative mean squared error between columns.

    Parameters:
    -----------
    <x> - shape (nconds, nunits). First matrix (usually test data).
    <y> - shape (nconds, nunits). Second matrix (usually predictions).
        Must have same shape as <x>.

    Returns:
    --------
    <scores> - shape (1, nunits). Negative MSE for each column/unit.
            0 indicates perfect prediction
            More negative values indicate worse predictions
            Each unit gets its own score

    Example:
    --------
        x = np.array([[1, 2], [3, 4]])  # 2 conditions, 2 units
        y = np.array([[1.1, 2.1], [2.9, 3.9]])  # Predictions
        scores = negative_mse_columns(x, y)  # Close to 0

    Notes:
    ------
        The function handles empty inputs gracefully by returning zeros, which is useful
        when no data survives thresholding.
    """
    if x.shape[0] == 0 or y.shape[0] == 0:
        return np.zeros(x.shape[1])  # Return zeros for empty arrays
    return -np.mean((x - y) ** 2, axis=0)

def make_orthonormal(V):
    """MAKE_ORTHONORMAL Find the nearest matrix with orthonormal columns.

    Uses Singular Value Decomposition (SVD) to find the nearest orthonormal matrix:
    1. Decompose <V> = <U>*<S>*<Vh> where <U> and <Vh> are orthogonal
    2. The nearest orthonormal matrix is <U>*<Vh>
    3. Take only the first n columns if m > n
    4. Verify orthonormality within numerical precision

    Inputs:
        <V> - m x n matrix where m >= n. Input matrix to be made orthonormal.
            The number of rows (m) must be at least as large as the number of
            columns (n).

    Returns:
        <V_orthonormal> - m x n matrix with orthonormal columns.
            The resulting matrix will have:
            1. All columns unit length
            2. All columns pairwise orthogonal

    Example:
        V = np.random.randn(5,3)  # Random 5x3 matrix
        V_ortho = make_orthonormal(V)
        # Check orthonormality
        gram = V_ortho.T @ V_ortho  # Should be very close to identity
        print(np.max(np.abs(gram - np.eye(gram.shape[0]))))  # Should be ~1e-15

    Notes:
        The SVD method guarantees orthonormality within numerical precision.
        A warning is issued if the result is not perfectly orthonormal.
    """
    # Check input dimensions
    m, n = V.shape
    if m < n:
        raise ValueError('Input matrix must have at least as many rows as columns')
    
    # Use SVD to find the nearest orthonormal matrix
    # SVD gives us V = U*S*Vh where U and Vh are orthogonal
    # The nearest orthonormal matrix is U*Vh
    U, _, Vh = np.linalg.svd(V, full_matrices=False)
    
    # Take only the first n columns of U if m > n
    V_orthonormal = U[:,:n] @ Vh
    
    # Double check that the result is orthonormal within numerical precision
    # This is mainly for debugging - the SVD method should guarantee this
    gram = V_orthonormal.T @ V_orthonormal
    if not np.allclose(gram, np.eye(n), rtol=0, atol=1e-10):
        print('Warning: Result may not be perfectly orthonormal due to numerical precision')
    
    return V_orthonormal
