import numpy as np
from gsn.perform_gsn import perform_gsn

def gsn_denoise(data, V=None, opt=None):
    """Denoise neural data using Generative Modeling of Signal and Noise (GSN) denoising.
    
    This function implements GSN denoising, which uses cross-validation or magnitude thresholding
    to identify and remove noise dimensions while preserving signal dimensions. The algorithm:
    
    1. Computes signal and noise covariance matrices from the data
    2. Selects a basis for denoising (several options available)
    3. Uses either cross-validation or magnitude thresholding to determine which dimensions to retain
    4. Constructs a denoising matrix that projects data onto the retained dimensions
    
    The denoising can be performed either:
    - On trial-averaged data (default)
    - On single trials
    - Using population-level thresholding (same dimensions for all units)
    - Using unit-wise thresholding (different dimensions for each unit)
    
    Algorithm Details:
    -----------------
    The GSN denoising algorithm works by identifying dimensions in the neural data that contain
    primarily signal rather than noise. It does this in several steps:

    1. Signal and Noise Estimation:
        - For each condition, computes mean response across trials (signal estimate)
        - For each condition, computes variance across trials (noise estimate)
        - Builds signal (cSb) and noise (cNb) covariance matrices across conditions
    
    2. Basis Selection (V parameter):
        - V=0: Uses eigenvectors of signal covariance (cSb)
        - V=1: Uses eigenvectors of signal covariance transformed by inverse noise covariance
        - V=2: Uses eigenvectors of noise covariance (cNb)
        - V=3: Uses PCA on trial-averaged data
        - V=4: Uses random orthonormal basis
        - V=matrix: Uses user-supplied orthonormal basis
    
    3. Dimension Selection:
        The algorithm must decide how many dimensions to keep. This can be done in two ways:

        a) Cross-validation (cv_mode >= 0):
            - Splits trials into training and testing sets
            - For training set:
                * Projects data onto different numbers of basis dimensions
                * Creates denoising matrix for each dimensionality
            - For test set:
                * Measures how well denoised training data predicts test data
                * Uses mean squared error (MSE) as prediction metric
            - Selects number of dimensions that gives best prediction
            - Can be done per-unit or for whole population
        
        b) Magnitude Thresholding (cv_mode < 0):
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
    
    The algorithm is particularly effective because:
    - It adapts to the structure of both signal and noise in the data
    - It can handle different types of neural response patterns
    - It allows for different denoising strategies (population vs unit-wise)
    - It provides cross-validation to prevent overfitting
    - It can denoise both trial-averaged and single-trial data
    
    Args:
        data: ndarray, shape (nunits, nconds, ntrials)
            Neural responses with dimensions:
            - nunits: number of units/neurons
            - nconds: number of conditions/stimuli
            - ntrials: number of repeated measurements
            Must have at least 2 trials and 2 conditions.
        
        V: int or ndarray, optional
            Basis selection mode (0,1,2,3,4) or custom basis matrix.
            If matrix: shape (nunits, D) with D >= 1, orthonormal columns.
            Default: 0 (signal covariance eigenvectors)
        
        opt: dict, optional
            Dictionary of options with fields:
            
            Cross-validation options:
            - cv_mode: int
                0: n-1 train / 1 test split (default)
                1: 1 train / n-1 test split
                -1: use magnitude thresholding instead
            - cv_threshold_per: str
                'unit': different thresholds per unit (default)
                'population': same threshold for all units
            - cv_thresholds: array
                Dimensions to test in cross-validation
                Default: 1 to nunits
            - cv_scoring_fn: callable
                Function to compute prediction error
                Default: negative mean squared error per unit
            
            Magnitude thresholding options:
            - mag_type: int
                0: use eigenvalues (default)
                1: use variances
            - mag_frac: float
                Fraction of maximum magnitude for threshold
                Default: 0.01
            - mag_mode: int
                0: use contiguous dimensions from left (default)
                1: use all dimensions above threshold
            
            General options:
            - denoisingtype: int
                0: trial-averaged denoising (default)
                1: single-trial denoising
    
    Returns:
        dict with fields:
            denoiser: ndarray, shape (nunits, nunits)
                Matrix that projects data onto denoised space
            
            cv_scores: ndarray
                Cross-validation scores for each threshold
                Shape depends on cv_mode and cv_threshold_per
            
            best_threshold: int or ndarray
                Selected threshold(s)
                Scalar for population mode
                Array of length nunits for unit mode
            
            denoiseddata: ndarray
                Denoised neural responses
                Shape (nunits, nconds) for trial-averaged
                Shape (nunits, nconds, ntrials) for single-trial
            
            fullbasis: ndarray, shape (nunits, dims)
                Complete basis used for denoising
            
            signalsubspace: ndarray or None
                Final basis functions used for denoising
                None for unit-wise mode
            
            dimreduce: ndarray or None
                Data projected onto signal subspace
                None for unit-wise mode
            
            mags: ndarray or None
                Component magnitudes (for magnitude thresholding)
            
            dimsretained: int or None
                Number of dimensions retained (for magnitude thresholding)
    
    Examples:
    --------
    Basic usage with default options:
    >>> data = np.random.randn(10, 20, 5)  # 10 units, 20 conditions, 5 trials
    >>> results = gsn_denoise(data)
    >>> denoised = results['denoiseddata']  # Get denoised data
    
    Using magnitude thresholding:
    >>> opt = {
    ...     'cv_mode': -1,  # Use magnitude thresholding
    ...     'mag_frac': 0.1,  # Keep components > 10% of max
    ...     'mag_mode': 0  # Use contiguous dimensions
    ... }
    >>> results = gsn_denoise(data, V=0, opt=opt)
    
    Unit-wise cross-validation:
    >>> opt = {
    ...     'cv_mode': 0,  # Leave-one-out CV
    ...     'cv_threshold_per': 'unit',  # Unit-specific thresholds
    ...     'cv_thresholds': [1, 2, 3]  # Test these dimensions
    ... }
    >>> results = gsn_denoise(data, V=0, opt=opt)
    
    Single-trial denoising:
    >>> opt = {
    ...     'denoisingtype': 1,  # Single-trial mode
    ...     'cv_threshold_per': 'population'  # Same dims for all units
    ... }
    >>> results = gsn_denoise(data, V=0, opt=opt)
    >>> denoised_trials = results['denoiseddata']  # (nunits, nconds, ntrials)
    
    Custom basis:
    >>> nunits = data.shape[0]
    >>> custom_basis = np.linalg.qr(np.random.randn(nunits, nunits))[0]
    >>> results = gsn_denoise(data, V=custom_basis)
    
    See Also:
    --------
    perform_gsn : Computes signal and noise covariance matrices
    perform_cross_validation : Implements cross-validation for dimension selection
    perform_magnitude_thresholding : Implements magnitude-based dimension selection
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
            
        basis = V.copy()
        # For user-supplied basis, compute magnitudes based on variance in basis
        trial_avg = np.mean(data, axis=2)  # shape (nunits, nconds)
        trial_avg_reshaped = trial_avg.T  # shape (ncond, nvox)
        proj_data = trial_avg_reshaped @ basis  # shape (ncond, basis_dim)
        mags = np.var(proj_data, axis=0, ddof=1)  # variance along conditions for each basis dimension
        
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
    """Perform cross-validation to determine optimal number of dimensions for denoising.
    
    This function implements leave-one-out or leave-n-out cross-validation to find
    the optimal number of dimensions to retain for denoising. It works by:
    1. Splitting data into training and test sets
    2. Creating denoising matrices with different numbers of dimensions
    3. Evaluating how well each denoiser predicts held-out data
    4. Selecting the dimensionality that gives best predictions
    
    Algorithm Details:
    -----------------
    1. Data Splitting:
        - cv_mode=0 (leave-one-out):
            * Training: Average of n-1 trials
            * Testing: Single held-out trial
        - cv_mode=1 (leave-n-out):
            * Training: Single trial
            * Testing: Average of n-1 trials
    
    2. Denoiser Construction:
        - For each candidate threshold:
            * Takes first k dimensions from basis
            * Creates projection matrix onto those dimensions
            * Applies to training data
            * Compares with test data
    
    3. Score Computation:
        - Population mode:
            * Averages scores across all units
            * Selects single best threshold
            * Creates one denoiser for all units
        - Unit mode:
            * Keeps separate scores for each unit
            * Selects best threshold per unit
            * Creates unit-specific denoisers
    
    Args:
        data: ndarray, shape (nunits, nconds, ntrials)
            Neural response data to denoise
        
        basis: ndarray, shape (nunits, dims)
            Orthonormal basis for denoising
            Usually eigenvectors of some covariance matrix
        
        opt: dict
            Dictionary of options with fields:
            - cv_mode: int
                0: n-1 train / 1 test split
                1: 1 train / n-1 test split
            - cv_threshold_per: str
                'unit' or 'population'
            - cv_thresholds: array
                Dimensions to test
            - cv_scoring_fn: callable
                Function to compute prediction error
            - denoisingtype: int
                0: trial-averaged, 1: single-trial
    
    Returns:
        denoiser: ndarray, shape (nunits, nunits)
            Matrix that projects data onto denoised space
        
        cv_scores: ndarray, shape (len(thresholds), ntrials, nunits)
            Cross-validation scores for each threshold/trial/unit
        
        best_threshold: int or ndarray
            Selected threshold(s)
            Scalar for population mode
            Array of length nunits for unit mode
        
        denoiseddata: ndarray
            Denoised neural responses
            Shape (nunits, nconds) for trial-averaged
            Shape (nunits, nconds, ntrials) for single-trial
        
        fullbasis: ndarray, shape (nunits, dims)
            Complete basis used for denoising
        
        signalsubspace: ndarray or None
            Final basis functions used for denoising
            None for unit-wise mode
        
        dimreduce: ndarray or None
            Data projected onto signal subspace
            None for unit-wise mode
    
    Examples:
    --------
    Population-level cross-validation:
    >>> data = np.random.randn(10, 20, 5)  # 10 units, 20 conditions, 5 trials
    >>> basis = np.eye(10)  # Identity basis for example
    >>> opt = {
    ...     'cv_mode': 0,
    ...     'cv_threshold_per': 'population',
    ...     'cv_thresholds': [1, 2, 3],
    ...     'denoisingtype': 0
    ... }
    >>> denoiser, scores, thresh, denoised, _, _, _ = perform_cross_validation(data, basis, opt)
    
    Unit-wise cross-validation:
    >>> opt['cv_threshold_per'] = 'unit'
    >>> denoiser, scores, thresholds, denoised, _, _, _ = perform_cross_validation(data, basis, opt)
    >>> print(f"Each unit got its own threshold: {thresholds}")
    
    Single-trial denoising:
    >>> opt.update({'denoisingtype': 1, 'cv_mode': 1})
    >>> denoiser, scores, thresh, denoised, _, _, _ = perform_cross_validation(data, basis, opt)
    >>> print(f"Denoised data shape: {denoised.shape}")  # (10, 20, 5)
    
    See Also:
    --------
    gsn_denoise : Main denoising function
    perform_magnitude_thresholding : Alternative to cross-validation
    negative_mse_columns : Default scoring function
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
    """Determine dimensions to retain based on their magnitudes (eigenvalues or variances).
    
    This function implements magnitude-based dimension selection as an alternative to
    cross-validation. It works by:
    1. Computing magnitudes for each dimension (eigenvalues or variances)
    2. Setting a threshold as a fraction of the maximum magnitude
    3. Retaining dimensions above the threshold
    4. Creating a denoiser using the retained dimensions
    
    Algorithm Details:
    -----------------
    1. Magnitude Computation (mag_type):
        - mag_type=0 (eigenvalue-based):
            * For V=0: Signal covariance eigenvalues
            * For V=1: Signal-to-noise eigenvalues
            * For V=2: Noise covariance eigenvalues
            * For V=3: Trial-averaged covariance eigenvalues
            * For V=4: All ones (random basis)
            * For custom V: Projection variances
        - mag_type=1 (variance-based):
            * Projects data onto basis
            * Computes variance in each dimension
    
    2. Threshold Selection:
        - Sets threshold as mag_frac * max(magnitudes)
        - Higher mag_frac means fewer dimensions retained
        - mag_frac=1.0 retains at most one dimension
        - mag_frac=0.0 retains all dimensions
    
    3. Dimension Selection (mag_mode):
        - mag_mode=0 (contiguous):
            * Keeps contiguous block from strongest dimension
            * Stops at first gap in surviving dimensions
            * Best for ordered bases (e.g., eigenvalues)
        - mag_mode=1 (any):
            * Keeps any dimension above threshold
            * Order doesn't matter
            * Best for custom bases
    
    Args:
        data: ndarray, shape (nunits, nconds, ntrials)
            Neural response data to denoise
        
        basis: ndarray, shape (nunits, dims)
            Orthonormal basis for denoising
        
        gsn_results: dict
            Results from perform_gsn containing:
            - cSb: Signal covariance matrix
            - cNb: Noise covariance matrix
        
        opt: dict
            Dictionary of options with fields:
            - mag_type: int
                0: eigenvalue-based
                1: variance-based
            - mag_frac: float
                Fraction of maximum magnitude for threshold
            - mag_mode: int
                0: contiguous from strongest
                1: any above threshold
            - denoisingtype: int
                0: trial-averaged
                1: single-trial
        
        V: int or ndarray
            Basis selection mode (0-4) or custom basis
    
    Returns:
        denoiser: ndarray, shape (nunits, nunits)
            Matrix that projects data onto denoised space
        
        cv_scores: empty ndarray
            Placeholder for compatibility with cross-validation
        
        best_threshold: ndarray
            Indices of retained dimensions
        
        denoiseddata: ndarray
            Denoised neural responses
            Shape (nunits, nconds) for trial-averaged
            Shape (nunits, nconds, ntrials) for single-trial
        
        basis: ndarray, shape (nunits, dims)
            Complete basis used for denoising
        
        signalsubspace: ndarray
            Final basis functions used for denoising
        
        dimreduce: ndarray
            Data projected onto signal subspace
        
        magnitudes: ndarray
            Component magnitudes used for thresholding
        
        dimsretained: int
            Number of dimensions retained
    
    Examples:
    --------
    Basic usage with eigenvalue-based thresholding:
    >>> data = np.random.randn(10, 20, 5)  # 10 units, 20 conditions, 5 trials
    >>> basis = np.eye(10)  # Identity basis for example
    >>> gsn_results = {'cSb': np.eye(10), 'cNb': np.eye(10)}
    >>> opt = {
    ...     'mag_type': 0,
    ...     'mag_frac': 0.1,  # Keep components > 10% of max
    ...     'mag_mode': 0,    # Contiguous from strongest
    ...     'denoisingtype': 0
    ... }
    >>> results = perform_magnitude_thresholding(data, basis, gsn_results, opt, V=0)
    >>> print(f"Retained {results[8]} dimensions")
    
    Variance-based thresholding with any dimensions:
    >>> opt.update({'mag_type': 1, 'mag_mode': 1})
    >>> results = perform_magnitude_thresholding(data, basis, gsn_results, opt, V=0)
    >>> print(f"Selected dimensions: {results[2]}")
    
    Single-trial denoising:
    >>> opt['denoisingtype'] = 1
    >>> results = perform_magnitude_thresholding(data, basis, gsn_results, opt, V=0)
    >>> print(f"Denoised data shape: {results[3].shape}")  # (10, 20, 5)
    
    See Also:
    --------
    gsn_denoise : Main denoising function
    perform_cross_validation : Alternative to magnitude thresholding
    perform_gsn : Computes signal and noise covariance matrices
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
    """Calculate the negative mean squared error between corresponding columns.
    
    This function computes the negative mean squared error (MSE) between each column
    of two matrices. It is primarily used as a scoring function for cross-validation
    in GSN denoising, where:
    - Each column represents a unit/neuron
    - Each row represents a condition/stimulus
    - The negative sign makes it compatible with maximization
        (higher scores = better predictions)
    
    The function handles empty inputs gracefully by returning zeros, which is useful
    when no data survives thresholding.
    
    Algorithm Details:
    -----------------
    For each column i:
    1. Computes squared differences: (x[:,i] - y[:,i])²
    2. Takes mean across rows (conditions)
    3. Multiplies by -1 to convert from error to score
    
    This results in a score where:
    - 0 indicates perfect prediction
    - More negative values indicate worse predictions
    - Each unit gets its own score
    
    Args:
        x: ndarray, shape (nconds, nunits)
            First matrix (usually test data)
            - nconds: number of conditions/stimuli
            - nunits: number of units/neurons
        
        y: ndarray, shape (nconds, nunits)
            Second matrix (usually predictions)
            Must have same shape as x
    
    Returns:
        ndarray, shape (nunits,)
            Negative MSE for each column/unit
            - Length matches number of columns in input
            - More negative = worse prediction
            - Zero = perfect prediction
    
    Examples:
    --------
    Basic usage:
    >>> x = np.array([[1, 2], [3, 4]])  # 2 conditions, 2 units
    >>> y = np.array([[1.1, 2.1], [2.9, 3.9]])  # Predictions
    >>> scores = negative_mse_columns(x, y)
    >>> print(f"Prediction scores: {scores}")  # Close to 0

    """
    if x.shape[0] == 0 or y.shape[0] == 0:
        return np.zeros(x.shape[1])  # Return zeros for empty arrays
    return -np.mean((x - y) ** 2, axis=0)

def make_orthonormal(V):
    """Find the nearest matrix with orthonormal columns.
    
    This function takes a matrix and finds the nearest matrix with orthonormal
    columns using polar decomposition. The resulting matrix will have:
    1. All columns unit length
    2. All columns pairwise orthogonal
    
    Args:
        V: ndarray of shape [m, n] where m >= n
    
    Returns:
        V_orthonormal: ndarray of shape [m, n] with orthonormal columns
    
    Example:
        >>> V = np.random.randn(5,3)  # Random 5x3 matrix
        >>> V_ortho = make_orthonormal(V)
        >>> # Check orthonormality
        >>> gram = V_ortho.T @ V_ortho  # Should be very close to identity
        >>> print(np.max(np.abs(gram - np.eye(gram.shape[0]))))  # Should be ~1e-15
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
