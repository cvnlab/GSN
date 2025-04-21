import numpy as np
from gsn.perform_gsn import perform_gsn
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

def gsn_denoise(data, V=None, opt=None, wantfig=True):
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
          0 means use signal variance computed from the data
          1 means use eigenvalues (<V> must be 0, 1, 2, or 3)
          Default: 0.
        <mag_frac> - scalar. Indicates a fraction of the maximum magnitude
          component. Matters only when <cv_mode> is -1.
          Default: 0.01.
        <mag_selection_mode> - scalar. Indicates how to select dimensions. Matters only 
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
    <wantfig> - bool. Whether to generate diagnostic figures showing the denoising results.
        Default: True.

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
        data = np.random.randn(100, 200, 3)  # 100 units, 200 conditions, 3 trials
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
            'mag_selection_mode': 0  # Use contiguous dimensions
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

    # Initialize return dictionary with None values
    results = {
        'denoiser': None,
        'cv_scores': None,
        'best_threshold': None,
        'denoiseddata': None,
        'fullbasis': None,
        'signalsubspace': None,
        'dimreduce': None,
        'mags': None,
        'dimsretained': None,
        'opt': opt, 
        'V': V 
    }

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
    opt.setdefault('mag_selection_mode', 0)
    opt.setdefault('denoisingtype', 0)  # Default to trial-averaged denoising
    
    # compute the unit means since they are removed during denoising and will be added back
    trial_avg = np.mean(data, axis=2)
    results['unit_means'] = np.mean(trial_avg, axis=1)

    # 5) If V is an integer => glean basis from GSN results
    if isinstance(V, int):
        if V not in [0, 1, 2, 3, 4]:
            raise ValueError("V must be in [0..4] (int) or a 2D numpy array.")
            
        gsn_results = perform_gsn(data, {'wantverbose': False, 'random_seed': 42})

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
            results['basis_source'] = cSb
        elif V == 1:
            cNb_inv = inv_or_pinv(cNb)
            transformed_cov = cNb_inv @ cSb
            # enforce symmetry of transformed_cov
            transformed_cov = (transformed_cov + transformed_cov.T) / 2
            evals, evecs = np.linalg.eigh(transformed_cov)
            # Sort by absolute value of eigenvalues
            idx = np.argsort(np.abs(evals))[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]
            basis = evecs
            magnitudes = np.abs(evals)  # No need to flip, already sorted
            results['basis_source'] = transformed_cov
        elif V == 2:
            evals, evecs = np.linalg.eigh(cNb)
            # Sort by absolute value of eigenvalues
            idx = np.argsort(np.abs(evals))[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]
            basis = evecs
            magnitudes = np.abs(evals)  # No need to flip, already sorted
            results['basis_source'] = cNb
        elif V == 3:
            # de-mean each row of trial_avg
            trial_avg = (trial_avg.T - results['unit_means']).T
            cov_mat = np.cov(trial_avg, ddof=1)
            evals, evecs = np.linalg.eigh(cov_mat)
            idx = np.argsort(np.abs(evals))[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]
            basis = evecs
            magnitudes = np.abs(evals)
            results['basis_source'] = cov_mat
        else:  # V == 4
            # Generate a random basis with same dimensions as eigenvector basis
            rand_mat = np.random.randn(nunits, nunits)  # Start with square matrix
            basis, _ = np.linalg.qr(rand_mat)
            # Only keep first nunits columns to match eigenvector basis dimensions
            basis = basis[:, :nunits]
            magnitudes = np.ones(nunits)  # No meaningful magnitudes for random basis
            results['basis_source'] = None  # No meaningful source matrix for random basis
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
        results['basis_source'] = None
        
    # Store the full basis and magnitudes for return
    fullbasis = basis.copy()
    stored_mags = magnitudes.copy()  # Store magnitudes for later use

    # Update results with computed values
    results['fullbasis'] = fullbasis
    results['mags'] = stored_mags

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

    # 7) Decide cross-validation or magnitude-threshold
    # We'll treat negative cv_mode as "do magnitude thresholding."
    if opt['cv_mode'] >= 0:
        denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, signalsubspace, dimreduce = perform_cross_validation(data, basis, opt, results=None)
        
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
        denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, signalsubspace, dimreduce, mags, dimsretained = perform_magnitude_thresholding(data, basis, opt, results)
        
        # Update results dictionary with all magnitude thresholding returns
        results.update({
            'denoiser': denoiser,
            'cv_scores': cv_scores,
            'best_threshold': best_threshold,
            'denoiseddata': denoiseddata,
            'fullbasis': fullbasis,
            'mags': mags,
            'dimsretained': dimsretained,
            'signalsubspace': signalsubspace,
            'dimreduce': dimreduce
        })

    # Store the input data and parameters in results for later visualization
    results['input_data'] = data.copy()
    results['V'] = V
    
    # Add a function handle to regenerate the visualization
    def regenerate_visualization(test_data=None):
        """
        Regenerate the diagnostic visualization.
        
        Parameters:
        -----------
        test_data : ndarray, optional
            Data to use for testing in the bottom row plots, shape (nunits, nconds, ntrials).
            If None, will use leave-one-out cross-validation on the training data.
        """
        plot_diagnostic_figures(results['input_data'], results, test_data)
    
    results['plot'] = regenerate_visualization

    if wantfig:
        plot_diagnostic_figures(data, results)

    return results

def perform_cross_validation(data, basis, opt, results=None):
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
    <dimreduce> - shape (best_threshold, nconds) or (best_threshold, nconds, ntrials). 
        Data projected onto signal subspace.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    nunits, nconds, ntrials = data.shape
    cv_mode = opt['cv_mode']
    thresholds = opt['cv_thresholds']
    opt.setdefault('cv_scoring_fn', negative_mse_columns)
    threshold_per = opt['cv_threshold_per']
    scoring_fn = opt['cv_scoring_fn']
    denoisingtype = opt['denoisingtype']
    
    # Initialize cv_scores
    cv_scores = np.zeros((len(thresholds), ntrials, nunits))

    for tr in tqdm(range(ntrials)):
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
            
    if results is not None and 'unit_means' in results:
        denoiseddata = (denoiseddata.T + results['unit_means']).T

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

def perform_magnitude_thresholding(data, basis, opt, results=None):
    """
    Select dimensions using magnitude thresholding.

    Implements the magnitude thresholding procedure for GSN denoising.
    Selects dimensions based on their magnitudes (eigenvalues or variances)
    rather than using cross-validation.

    Supports two modes:
    - Contiguous selection of the left-most group of dimensions above threshold
    - Selection of any dimension above threshold

    Algorithm Details:
    1. Get magnitudes either:
       - From signal variance of the data projected into the basis (mag_type=0)
       - Or precomputed basis eigenvalues (mag_type=1)
    2. Set threshold as fraction of maximum magnitude
    3. Select dimensions either:
       - Contiguously from strongest dimension
       - Or any dimension above threshold
    4. Create denoising matrix using selected dimensions

    Parameters:
    -----------
    <data> - shape (nunits, nconds, ntrials). Neural response data to denoise.
    <basis> - shape (nunits, dims). Orthonormal basis for denoising.
    <opt> - dict with fields:
        <mag_type> - scalar. How to obtain component magnitudes:
            0: use signal variance computed from data
            1: use pre-computed eigenvalues from results
        <mag_frac> - scalar. Fraction of maximum magnitude to use as threshold.
        <mag_selection_mode> - scalar. How to select dimensions:
            0: contiguous from strongest dimension
            1: any dimension above threshold
        <denoisingtype> - scalar. Type of denoising:
            0: trial-averaged
            1: single-trial
    <results> - dict containing pre-computed magnitudes in results['mags'] if mag_type=1

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
    mag_type = opt['mag_type']
    mag_frac = opt['mag_frac']
    mag_selection_mode = opt['mag_selection_mode']
    denoisingtype = opt['denoisingtype']

    cv_scores = np.array([])  # Not used in magnitude thresholding
    
    # Get magnitudes based on mag_type
    if mag_type == 1:
        # Use pre-computed magnitudes from results
        magnitudes = results['mags']
    else:
        # Variance-based threshold in user basis
        # Initialize list to store signal variances
        sigvars = []

        data_reshaped = data.transpose(1, 2, 0)
        # Compute signal variance for each basis dimension
        for i in range(basis.shape[1]):
            this_eigv = basis[:, i]  # Select the i-th eigenvector
            proj_data = np.dot(data_reshaped, this_eigv)  # Project data into this eigenvector's subspace

            # Compute signal variance (using same computation as in noise ceiling)
            noisevar = np.mean(np.std(proj_data, axis=1, ddof=1) ** 2)
            datavar = np.std(np.mean(proj_data, axis=1), ddof=1) ** 2
            signalvar = np.maximum(datavar - noisevar / proj_data.shape[1], 0)  # Ensure non-negative variance
            sigvars.append(float(signalvar))

        magnitudes = np.array(sigvars)
    
    # Determine threshold as fraction of maximum magnitude
    threshold_val = mag_frac * np.max(np.abs(magnitudes))
    surviving = np.abs(magnitudes) >= threshold_val
    
    # Find dimensions to retain based on mag_selection_mode
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

    if mag_selection_mode == 0:  # Contiguous from left
        # For contiguous from left, we want the leftmost block
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
            
    # add back the means
    if results is not None and 'unit_means' in results:
        denoiseddata = (denoiseddata.T + results['unit_means']).T

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

def compute_noise_ceiling(data_in):
    """
    Compute the noise ceiling signal-to-noise ratio (SNR) and percentage noise ceiling for each unit.
    
    Parameters:
    ----------
    data_in : np.ndarray
        A 3D array of shape (units/voxels, conditions, trials), representing the data for which to compute 
        the noise ceiling. Each unit requires more than 1 trial for each condition.

    Returns:
    -------
    noiseceiling : np.ndarray
        The noise ceiling for each unit, expressed as a percentage.
    ncsnr : np.ndarray
        The noise ceiling signal-to-noise ratio (SNR) for each unit.
    signalvar : np.ndarray
        The signal variance for each unit.
    noisevar : np.ndarray
        The noise variance for each unit.
    """
    # noisevar: mean variance across trials for each unit
    noisevar = np.mean(np.std(data_in, axis=2, ddof=1) ** 2, axis=1)

    # datavar: variance of the trial means across conditions for each unit
    datavar = np.std(np.mean(data_in, axis=2), axis=1, ddof=1) ** 2

    # signalvar: signal variance, obtained by subtracting noise variance from data variance
    signalvar = np.maximum(datavar - noisevar / data_in.shape[2], 0)  # Ensure non-negative variance

    # ncsnr: signal-to-noise ratio (SNR) for each unit
    ncsnr = np.sqrt(signalvar) / np.sqrt(noisevar)

    # noiseceiling: percentage noise ceiling based on SNR
    noiseceiling = 100 * (ncsnr ** 2 / (ncsnr ** 2 + 1 / data_in.shape[2]))

    return noiseceiling, ncsnr, signalvar, noisevar

def compute_r2(y_true, y_pred):
    """Compute R2 score between true and predicted values."""
    residual_ss = np.sum((y_true - y_pred) ** 2)
    total_ss = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (residual_ss / total_ss)
    return r2

def plot_diagnostic_figures(data, results, test_data=None):
    """
    Generate diagnostic figures for GSN denoising results.
    
    Parameters:
    -----------
    data : ndarray
        Training data used for denoising, shape (nunits, nconds, ntrials)
    results : dict
        Results dictionary from gsn_denoise
    test_data : ndarray, optional
        Data to use for testing in the bottom row plots, shape (nunits, nconds, ntrials).
        If None, will use leave-one-out cross-validation on the training data.
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create a single large figure with proper spacing
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)
    
    # Extract data dimensions
    nunits, nconds, ntrials = data.shape
    
    # Add text at the top of the figure
    V_type = results.get('V')  # Get V directly from results
    if V_type is None:
        V_type = results.get('opt', {}).get('V', 0)  # Fallback to opt dict if not found
    if isinstance(V_type, np.ndarray):
        V_desc = f"user-supplied {V_type.shape}"
    else:
        V_desc = str(V_type)
        
    # Create title text with data shape and GSN application info
    title_text = f"Data shape: {nunits} units × {nconds} conditions × {ntrials} trials    |    V = {V_desc}\n"
    
    # Add cv_mode and magnitude thresholding info to title
    cv_mode = results.get('opt', {}).get('cv_mode', 0)
    if cv_mode == -1:
        mag_type = results.get('opt', {}).get('mag_type', 0)
        mag_selection_mode = results.get('opt', {}).get('mag_selection_mode', 0)
        mag_frac = results.get('opt', {}).get('mag_frac', 0.01)
        mag_frac_str = f"{mag_frac:.3f}".rstrip('0').rstrip('.')
        title_text = (f"Data shape: {nunits} units × {nconds} conditions × {ntrials} trials    |    "
                     f"V = {V_desc}    |    cv_mode = {cv_mode}    |    "
                     f"mag_type = {mag_type}, mag_mode = {mag_selection_mode}, mag_frac = {mag_frac_str}\n")
    else:
        threshold_per = results.get('opt', {}).get('cv_threshold_per', 'unit')
        title_text = (f"Data shape: {nunits} units × {nconds} conditions × {ntrials} trials    |    "
                     f"V = {V_desc}    |    cv_mode = {cv_mode}    |    thresh = {threshold_per}\n")
    
    if test_data is None:
        title_text += f"gsn_denoise.py applied to all {ntrials} trials"
    else:
        title_text += f"gsn_denoise.py applied to {ntrials} trials, tested on 1 heldout trial"
    
    plt.figtext(0.5, 0.97, title_text,
                ha='center', va='top', fontsize=14)

    # Get raw and denoised data
    if results.get('opt', {}).get('denoisingtype', 0) == 0:
        raw_data = np.mean(data, axis=2)  # Average across trials for trial-averaged denoising
        denoised_data = results['denoiseddata']
    else:
        # For single-trial denoising, we'll plot the first trial
        raw_data = data[:, :, 0] if data.ndim == 3 else data
        denoised_data = results['denoiseddata'][:, :, 0] if results['denoiseddata'].ndim == 3 else results['denoiseddata']

    # Compute noise as difference
    noise = raw_data - denoised_data

    # Initialize lists for basis dimension analysis
    ncsnrs, sigvars, noisevars = [], [], []

    if 'fullbasis' in results and 'mags' in results:
        # Project data into basis
        data_reshaped = data.transpose(1, 2, 0)
        eigvecs = results['fullbasis']
        for i in range(eigvecs.shape[1]):
            this_eigv = eigvecs[:, i]
            proj_data = np.dot(data_reshaped, this_eigv)
            
            _, ncsnr, sigvar, noisevar = compute_noise_ceiling(proj_data[np.newaxis, ...])
            ncsnrs.append(float(ncsnr[0]))
            sigvars.append(float(sigvar[0]))
            noisevars.append(float(noisevar[0]))

        # Convert to numpy arrays
        sigvars = np.array(sigvars)
        ncsnrs = np.array(ncsnrs)
        noisevars = np.array(noisevars)
        S = results['mags']
        opt = results.get('opt', {})
        best_threshold = results.get('best_threshold', None)
        if best_threshold is None:
            best_threshold = results.get('dimsretained', [])

        # Plot 1: basis source matrix (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        V = results.get('V')
        
        if isinstance(V, (int, np.integer)):
            if V in [0, 1, 2, 3]:
                # Show the basis source matrix
                if 'basis_source' in results and results['basis_source'] is not None:
                    matrix_to_show = results['basis_source']
                    print("In plotting: matrix_to_show shape =", matrix_to_show.shape)  # Debug print
                    if V == 0:
                        title = 'GSN Signal Covariance (cSb)'
                    elif V == 1:
                        title = 'GSN Transformed Signal Cov\n(inv(cNb)*cSb)'
                    elif V == 2:
                        title = 'GSN Noise Covariance (cNb)'
                    else:  # V == 3
                        title = 'Naive Trial-avg Data\nCovariance'
                    
                    matrix_max = np.percentile(np.abs(matrix_to_show), 95)  # Use 95th percentile like example2
                    print(f"Matrix stats: shape={matrix_to_show.shape}, min={np.min(matrix_to_show):.3f}, max={np.max(matrix_to_show):.3f}, mean={np.mean(matrix_to_show):.3f}, has_nan={np.any(np.isnan(matrix_to_show))}, has_inf={np.any(np.isinf(matrix_to_show))}, max_95={matrix_max:.3f}")
                    
                    im1 = ax1.imshow(matrix_to_show, vmin=-matrix_max, vmax=matrix_max,
                                   aspect='equal', interpolation='nearest', cmap='RdBu_r')
                    plt.colorbar(im1, ax=ax1, label='Covariance')
                    ax1.set_title(title, pad=10)
                    ax1.set_xlabel('Units')
                    ax1.set_ylabel('Units')
                else:
                    ax1.text(0.5, 0.5, f'Covariance Matrix\nNot Available for V={V}',
                            ha='center', va='center', transform=ax1.transAxes)
                    ax1.set_title('')
            else:  # V == 4
                ax1.text(0.5, 0.5, 'Random Basis\n(No Matrix to Show)',
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('')
        elif isinstance(V, np.ndarray):
            # Handle case where V is a matrix
            ax1.text(0.5, 0.5, f'User-Supplied Basis\nShape: {V.shape}',
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('User-Supplied Basis')
        else:
            # Handle any other case
            ax1.text(0.5, 0.5, 'No Basis Information Available',
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('')

        # Plot 2: Full basis matrix (top middle-left)
        ax2 = fig.add_subplot(gs[0, 1])
        basis_max = np.percentile(np.abs(results['fullbasis']), 99)
        im2 = ax2.imshow(results['fullbasis'], aspect='auto', interpolation='none', 
                        clim=(-basis_max, basis_max), cmap='RdBu_r')
        plt.colorbar(im2, ax=ax2)
        ax2.set_title('Full Basis Matrix')
        ax2.set_xlabel('Dimension')
        ax2.set_ylabel('Units')
        
        # Plot 3: Eigenspectrum (top middle)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(S, linewidth=1, color='blue', label='Eigenvalues')  # Made line thinner
        
        # Calculate and plot threshold indicators based on mode
        cv_mode = results.get('opt', {}).get('cv_mode', 0)
        cv_threshold_per = results.get('opt', {}).get('cv_threshold_per', 'unit')
        mag_selection_mode = results.get('opt', {}).get('mag_selection_mode', 0)
        mag_type = results.get('opt', {}).get('mag_type', 0)
        
        if cv_mode >= 0:  # Cross-validation mode
            if cv_threshold_per == 'population':
                # Single line for population threshold
                if isinstance(best_threshold, (np.ndarray, list)):
                    best_threshold = int(best_threshold[0])  # Take first value if array
                ax3.axvline(x=float(best_threshold), color='r', linestyle='--', linewidth=1,
                          label=f'Population threshold: {best_threshold} dims')
            else:  # Unit mode
                # Mean line and asterisks for unit-specific thresholds
                if isinstance(best_threshold, (np.ndarray, list)):
                    mean_threshold = np.mean(best_threshold)
                    ax3.axvline(x=float(mean_threshold), color='r', linestyle='--', linewidth=1,
                              label=f'Mean threshold: {mean_threshold:.1f} dims')
                    # Add asterisks at the top for each unit's threshold
                    unique_thresholds = np.unique(best_threshold)
                    ylim = ax3.get_ylim()
                    for thresh in unique_thresholds:
                        ax3.plot(thresh, ylim[1], 'r*', markersize=5,
                               label="")
        else:  # Magnitude thresholding mode
            if mag_selection_mode == 0:  # Contiguous
                if isinstance(best_threshold, (np.ndarray, list)):
                    threshold_len = len(best_threshold)
                else:
                    threshold_len = best_threshold
                ax3.axvline(x=float(threshold_len), color='r', linestyle='--', linewidth=1,
                          label=f'Mag threshold: {threshold_len} dims')
            # Add circles for included dimensions only if mag_type=1
            if isinstance(best_threshold, (np.ndarray, list)) and mag_type == 1 and len(best_threshold) > 0:
                best_threshold_array = np.asarray(best_threshold, dtype=int)
                ax3.plot(best_threshold_array, S[best_threshold_array], 'ro', markersize=4,
                        label='Included dimensions' if mag_selection_mode == 1 else "")
        
        ax3.set_xlabel('Dimension')
        ax3.set_ylabel('Eigenvalue')
        ax3.set_title('Denoising Basis\nEigenspectrum')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Plot 4: Signal and noise variances with NCSNR (top right)
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.plot(sigvars, linewidth=1, label='Sig. var')
        ax4.plot(noisevars, linewidth=1, label='Noise var')
        
        # Handle thresholds based on mode
        cv_mode = results.get('opt', {}).get('cv_mode', 0)
        if cv_mode >= 0:  # Cross-validation mode
            if isinstance(best_threshold, (np.ndarray, list)):
                if len(best_threshold) > 0:
                    threshold_val = np.mean(best_threshold)
                    ax4.axvline(x=float(threshold_val), color='r', linestyle='--', linewidth=1,
                              label=f'Mean thresh: {threshold_val:.1f} dims')
            else:
                # Ensure scalar value for axvline
                ax4.axvline(x=float(best_threshold), color='r', linestyle='--', linewidth=1,
                           label=f'Thresh: {best_threshold} dims')
        else:  # Magnitude thresholding mode
            if isinstance(best_threshold, (np.ndarray, list)):
                threshold_len = len(best_threshold)
                ax4.axvline(x=float(threshold_len), color='r', linestyle='--', linewidth=1,
                           label=f'Dims retained: {threshold_len}')
                # Add circles for included dimensions if mag_type=0
                if mag_type == 0 and len(best_threshold) > 0:
                    best_threshold_array = np.asarray(best_threshold, dtype=int)
                    ax4.plot(best_threshold_array, sigvars[best_threshold_array], 'ro', markersize=4,
                            label='Included dimensions' if mag_selection_mode == 1 else "")
            else:
                # Ensure scalar value for axvline
                ax4.axvline(x=float(best_threshold), color='r', linestyle='--', linewidth=1,
                           label=f'Dims retained: {best_threshold}')
        
        # Add NCSNR on secondary y-axis
        ax4_twin = ax4.twinx()
        ax4_twin.plot(ncsnrs, linewidth=1, color='magenta', label='NCSNR')
        ax4_twin.set_ylabel('NCSNR', color='magenta')
        ax4_twin.tick_params(axis='y', labelcolor='magenta')
        
        # Combine legends from both axes
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='best')

        ax4.set_xlabel('Dimension')
        ax4.set_ylabel('Variance')
        ax4.set_title('Signal and Noise Variance for \nData Projected into Basis')
        ax4.grid(True, alpha=0.3, which='both', axis='both')  # Enable grid for both axes
        ax4_twin.grid(False)  # Disable grid for twin axis to avoid double grid lines

        # Plot 5: Cross-validation results (first subplot in middle row)
        ax5 = fig.add_subplot(gs[1, 0])
        if 'cv_scores' in results and results.get('opt', {}).get('cv_mode', 0) > -1:
            cv_data = stats.zscore(results['cv_scores'].mean(1),axis=0,ddof=1)
            vmin, vmax = np.percentile(cv_data, [1, 99])
            plt.imshow(cv_data.T, aspect='auto', interpolation='none', clim=(vmin, vmax))
            plt.colorbar()
            plt.xlabel('PC exclusion threshold')
            plt.ylabel('Units')
            plt.title('Cross-validation scores (z)')
            
            # Show fewer ticks by increasing step size
            # Get thresholds, handling both list and array types
            cv_thresholds = opt.get('cv_thresholds', np.arange(results['cv_scores'].shape[0]))
            if isinstance(cv_thresholds, list):
                thresholds = np.array(cv_thresholds)
            else:
                thresholds = cv_thresholds
            
            step = max(len(thresholds) // 10, 1)  # Show ~10 ticks or less
            plt.xticks(np.arange(len(thresholds))[::step], thresholds[::step])
            
            if results.get('opt', {}).get('cv_threshold_per') == 'unit':
                if isinstance(best_threshold, np.ndarray):
                    plt.plot(best_threshold-1, np.arange(nunits), 'r.', markersize=4,
                            label='Unit-specific thresholds')
            else:
                plt.text(0.5, 0.5, 'No Cross-validation\nScores Available',
                        ha='center', va='center', transform=ax5.transAxes)
                plt.title('Cross-validation scores')

        # Plot 6-8: Raw data, denoised data, noise (rest of middle row)
        all_data = np.concatenate([raw_data.flatten(), denoised_data.flatten(), noise.flatten()])
        max_abs_val = np.percentile(np.abs(all_data), 99)
        data_clim = (-max_abs_val, max_abs_val)
        
        # Raw data
        ax6 = fig.add_subplot(gs[1, 1])
        im6 = plt.imshow(raw_data, aspect='auto', interpolation='none', clim=data_clim, cmap='RdBu_r')
        plt.colorbar(im6)
        plt.title('Input Data (trial-averaged)')
        plt.xlabel('Conditions')
        plt.ylabel('Units')

        # Denoised data
        ax7 = fig.add_subplot(gs[1, 2])
        im7 = plt.imshow(denoised_data, aspect='auto', interpolation='none', clim=data_clim, cmap='RdBu_r')
        plt.colorbar(im7)
        plt.title('Data projected into basis')
        plt.xlabel('Conditions')
        plt.ylabel('Units')

        # Noise
        ax8 = fig.add_subplot(gs[1, 3])
        im8 = plt.imshow(noise, aspect='auto', interpolation='none', clim=data_clim, cmap='RdBu_r')
        plt.colorbar(im8)
        plt.title('Residual')
        plt.xlabel('Conditions')
        plt.ylabel('Units')

        # Plot denoising matrix (first subplot in bottom row)
        ax9 = fig.add_subplot(gs[2, 0])
        denoiser_max = np.percentile(np.abs(results['denoiser']), 99)
        denoiser_clim = (-denoiser_max, denoiser_max)
        im9 = plt.imshow(results['denoiser'], aspect='auto', interpolation='none', clim=denoiser_clim, cmap='RdBu_r')
        plt.colorbar(im9)
        plt.title('Optimal Basis Matrix')
        plt.xlabel('units')
        plt.ylabel('units')

        # Compute R2 and correlations for bottom row
        if test_data is None:
            # Use leave-one-out cross-validation on training data
            raw_r2_per_unit = np.zeros((ntrials, nunits))
            denoised_r2_per_unit = np.zeros((ntrials, nunits))
            raw_corr_per_unit = np.zeros((ntrials, nunits))
            denoised_corr_per_unit = np.zeros((ntrials, nunits))

            for tr in range(ntrials):
                train_trials = np.setdiff1d(np.arange(ntrials), tr)
                train_avg = np.mean(data[:, :, train_trials], axis=2)
                test_trial = data[:, :, tr]
                
                for v in range(nunits):
                    raw_r2_per_unit[tr, v] = compute_r2(test_trial[v], train_avg[v])
                    raw_corr_per_unit[tr, v] = np.corrcoef(test_trial[v], train_avg[v])[0, 1]
                
                train_avg_denoised = (train_avg.T @ results['denoiser']).T
                test_trial_denoised = (test_trial.T @ results['denoiser']).T
                    
                for v in range(nunits):
                    denoised_r2_per_unit[tr, v] = compute_r2(test_trial[v], train_avg_denoised[v])
                    denoised_corr_per_unit[tr, v] = np.corrcoef(test_trial[v], train_avg_denoised[v])[0, 1]
        else:
            # Use provided test data
            if np.ndim(test_data) > 2:
                test_avg = np.mean(test_data, axis=2)
            else:
                test_avg = test_data
            train_avg = np.mean(data, axis=2)
            
            raw_r2_per_unit = np.zeros((1, nunits))
            denoised_r2_per_unit = np.zeros((1, nunits))
            raw_corr_per_unit = np.zeros((1, nunits))
            denoised_corr_per_unit = np.zeros((1, nunits))
            
            for v in range(nunits):
                raw_r2_per_unit[0, v] = compute_r2(test_avg[v], train_avg[v])
                raw_corr_per_unit[0, v] = np.corrcoef(test_avg[v], train_avg[v])[0, 1]
            
            train_avg_denoised = (train_avg.T @ results['denoiser']).T
            test_avg_denoised = (test_avg.T @ results['denoiser']).T
                
            for v in range(nunits):
                denoised_r2_per_unit[0, v] = compute_r2(test_avg[v], train_avg_denoised[v])
                denoised_corr_per_unit[0, v] = np.corrcoef(test_avg[v], train_avg_denoised[v])[0, 1]

        # Compute mean and SEM
        raw_r2_mean = np.mean(raw_r2_per_unit, axis=0)
        raw_r2_sem = stats.sem(raw_r2_per_unit, axis=0)
        denoised_r2_mean = np.mean(denoised_r2_per_unit, axis=0)
        denoised_r2_sem = stats.sem(denoised_r2_per_unit, axis=0)

        raw_corr_mean = np.mean(raw_corr_per_unit, axis=0)
        raw_corr_sem = stats.sem(raw_corr_per_unit, axis=0)
        denoised_corr_mean = np.mean(denoised_corr_per_unit, axis=0)
        denoised_corr_sem = stats.sem(denoised_corr_per_unit, axis=0)

        # Function to plot bottom row with rotated histograms
        def plot_bottom_histogram(ax, r2_mean, corr_mean, r2_color, corr_color, title):
            plt.sca(ax)
            plt.axvline(x=0, color='k', linewidth=2, zorder=1)
            
            # Calculate histogram bins
            bins = np.linspace(-1, 1, 50)
            bin_width = bins[1] - bins[0]
            
            # Plot R2 histogram
            r2_hist, _ = np.histogram(r2_mean, bins=bins)  # Remove density=True
            plt.bar(bins[:-1] + bin_width/2, r2_hist, width=bin_width, 
                    color=r2_color, alpha=0.6, label=f'Mean R² = {np.mean(r2_mean):.3f}')
            
            # Plot correlation histogram
            corr_hist, _ = np.histogram(corr_mean, bins=bins)  # Remove density=True
            plt.bar(bins[:-1] + bin_width/2, corr_hist, width=bin_width, 
                    color=corr_color, alpha=0.6, label=f'Mean r = {np.mean(corr_mean):.3f}')
            
            plt.ylabel('# Units')  # Updated label
            plt.xlabel('R² / Pearson r')
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim(-1, 1)

        # Plot bottom row histograms and R² progression
        train_trials = ntrials-1 if test_data is None else data.shape[2]
        test_trials = 1 if test_data is None else (test_data.shape[2] if len(test_data.shape) > 2 else 1)
            
        plot_bottom_histogram(fig.add_subplot(gs[2, 1]), 
                            raw_r2_mean, raw_corr_mean,
                            'blue', 'lightblue',
                            f'Baseline Generalization\nTrial-avg Train ({train_trials} trials) vs\nTrial-avg Test ({test_trials} trials)')

        plot_bottom_histogram(fig.add_subplot(gs[2, 2]),
                            denoised_r2_mean, denoised_corr_mean,
                            'green', 'lightgreen',
                            f'Denoised Generalization\nTrial-avg Train + denoised ({train_trials} trials) vs\nTrial-avg Test ({test_trials} trials)')

        # Add R² progression plot
        ax_prog = fig.add_subplot(gs[2, 3])
        x_positions = [1, 2]  # Two positions for the two conditions
        
        # Plot lines for each unit
        for v in range(nunits):
            values = [raw_r2_mean[v], denoised_r2_mean[v]]
            plt.plot(x_positions, values, color='gray', alpha=0.2, linewidth=0.5)
            plt.scatter(x_positions[0], values[0], alpha=0.5, s=20, color='blue')
            plt.scatter(x_positions[1], values[1], alpha=0.5, s=20, color='green')
        
        # Plot mean performance
        mean_values = [np.mean(raw_r2_mean), np.mean(denoised_r2_mean)]
        plt.plot(x_positions, mean_values, color='pink', linewidth=2, label='Mean')
        plt.scatter(x_positions[0], mean_values[0], color='blue', s=100, edgecolor='pink', linewidth=2)
        plt.scatter(x_positions[1], mean_values[1], color='green', s=100, edgecolor='pink', linewidth=2)
        
        plt.xticks(x_positions, ['Trial Averaged', 'With Denoising'])
        plt.ylabel('R²')
        plt.title(f'Impact of denoising on R² ({nunits} units)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(0.5, 2.5)
        plt.ylim(-1, 1)
        plt.axhline(y=0, color='k', linewidth=2, zorder=1)

    