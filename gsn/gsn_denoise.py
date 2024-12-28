import numpy as np
from gsn.perform_gsn import perform_gsn
from gsn.calc_cod import calc_cod
from gsn.utilities import zerodiv
from gsn.convert_covariance_to_correlation import convert_covariance_to_correlation

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

def gsn_denoise(data, V=None, opt=None):
    """
    Main entry point for denoising:
    data: shape (nunits, nconds, ntrials)
    V: basis selection mode (0..4) or a numpy array. If None, defaults to 0.
    opt: dict with keys controlling cross-validation, magnitude thresholding, etc.
         Must include at least:
           'cv_mode': int (0, 1, or negative => 'magnitude thresholding')
           'cv_scoring_fn': a function or lambda that takes (nconds, nunits) for estimate and ground_truth
                            and returns either shape(nunits,) or scalar, used for cross-validation.
           'cv_threshold_per': 'unit' or 'population'
           'cv_thresholds': array of threshold values to test
           'mag_type': 0 => eigen-based thresholding, 1 => variance in basis
           'mag_frac': float from 0..1
           'mag_mode': 0 => contiguous, 1 => all that survive
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
    opt.setdefault('cv_scoring_fn', negative_mse_columns)
    opt.setdefault('cv_mode', 0)
    opt.setdefault('cv_threshold_per', 'unit')
    opt.setdefault('mag_type', 0)
    opt.setdefault('mag_frac', 0.01)
    opt.setdefault('mag_mode', 0)

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
        elif V == 1:
            cNb_inv = inv_or_pinv(cNb)
            transformed_cov = cNb_inv @ cSb
            evals, evecs = np.linalg.eigh(transformed_cov)
            basis = np.fliplr(evecs)
        elif V == 2:
            evals, evecs = np.linalg.eigh(cNb)
            basis = np.fliplr(evecs)
        elif V == 3:
            trial_avg = np.mean(data, axis=2)  # shape (nunits, nconds)
            cov_matrix = np.cov(trial_avg)     # shape (nunits, nunits)
            evals, evecs = np.linalg.eigh(cov_matrix)
            basis = np.fliplr(evecs)
        else:  # V == 4 => random orthonormal
            rand_mat = np.random.randn(nunits, nunits)
            basis, _ = np.linalg.qr(rand_mat)
    else:
        # If V not int => must be a numpy array
        if not isinstance(V, np.ndarray):
            raise ValueError("If V is not int, it must be a numpy array.")
        basis = V

    # 6) Default cross-validation thresholds if not provided
    if 'cv_thresholds' not in opt:
        opt['cv_thresholds'] = np.arange(1, basis.shape[1] + 1)

    # 7) Decide cross-validation or magnitude-threshold
    # We'll treat negative or zero cv_mode as "do magnitude thresholding."
    if opt['cv_mode'] >= 0:
        denoiser, cv_scores, best_threshold = perform_cross_validation(data, basis, opt)
    else:
        denoiser, cv_scores, best_threshold = perform_magnitude_thresholding(data, basis, gsn_results, opt, V)

    return denoiser, cv_scores, best_threshold


def perform_cross_validation(data, basis, opt):
    """
    Cross-validation to pick the best threshold dimension.
    We'll clamp thr to basis.shape[1] to avoid shape mismatches with rank-deficient basis.
    """
    nunits, nconds, ntrials = data.shape
    cv_mode = opt['cv_mode']
    thresholds = opt['cv_thresholds']
    opt.setdefault('cv_scoring_fn', negative_mse_columns)
    threshold_per = opt.get('cv_threshold_per')
    if threshold_per not in ['unit', 'population']:
        raise KeyError("cv_threshold_per must be either 'unit' or 'population'")
    scoring_fn = opt['cv_scoring_fn']

    if threshold_per == 'unit':
        cv_scores = np.zeros((len(thresholds), nunits))
    else:
        cv_scores = np.zeros(len(thresholds))

    if cv_mode in [0, 1]:
        valid_splits = 0  # Count valid splits for averaging
        for split_idx in range(ntrials):
            # Train/test split
            if cv_mode == 0:
                train_mask = [i for i in range(ntrials) if i != split_idx]
                test_mask = [split_idx]
            else:
                train_mask = [split_idx]
                test_mask = [i for i in range(ntrials) if i != split_idx]

            train_data = data[:, :, train_mask]
            test_data = data[:, :, test_mask]

            # Skip empty folds
            if train_data.shape[2] == 0 or test_data.shape[2] == 0:
                continue

            valid_splits += 1
            # Safe mean calculation
            train_avg = np.mean(train_data, axis=2) if train_data.shape[2] > 0 else np.zeros((nunits, nconds))
            test_avg = np.mean(test_data, axis=2) if test_data.shape[2] > 0 else np.zeros((nunits, nconds))

            for i, thr in enumerate(thresholds):
                # Clamp thr so we don't exceed basis columns
                safe_thr = min(thr, basis.shape[1])
                denoiser_i = basis[:, :safe_thr] @ basis[:, :safe_thr].T  # (nunits, nunits)
                reconstructed = (test_avg.T @ denoiser_i).T  # (nunits, nconds)

                # scoring_fn expects (nconds, nunits), so transpose
                score = scoring_fn(reconstructed.T, test_avg.T)

                if threshold_per == 'unit':
                    # Expect shape (nunits,) => add up
                    cv_scores[i, :] += score
                else:
                    # Expect scalar or shape(nunits,) => average
                    cv_scores[i] += np.mean(score) if np.size(score) > 0 else 0

        # Average scores over valid splits
        if valid_splits > 0:
            cv_scores /= valid_splits
    else:
        raise NotImplementedError(f"cv_mode={cv_mode} not implemented.")

    # Decide best threshold
    if threshold_per == 'population':
        best_ix = np.argmax(cv_scores)
        best_threshold = thresholds[best_ix]
        safe_thr = min(best_threshold, basis.shape[1])
        denoiser = basis[:, :safe_thr] @ basis[:, :safe_thr].T
    else:
        # unit-wise
        best_thresh_unitwise = []
        for unit_i in range(nunits):
            best_idx = np.argmax(cv_scores[:, unit_i])
            best_thresh_unitwise.append(thresholds[best_idx])
        best_thresh_unitwise = np.array(best_thresh_unitwise)
        best_threshold = best_thresh_unitwise
        max_thr = int(np.max(best_threshold))
        safe_thr = min(max_thr, basis.shape[1])
        denoiser = basis[:, :safe_thr] @ basis[:, :safe_thr].T

    return denoiser, cv_scores, best_threshold


def perform_magnitude_thresholding(data, basis, gsn_results, opt, V):
    """
    Use eigenvalue or variance-based thresholding.
    Handle mag_frac=0 => keep all. Use pinv if needed.
    """
    nunits, nconds, ntrials = data.shape
    mag_type = opt['mag_type']
    mag_frac = opt['mag_frac']
    mag_mode = opt['mag_mode']
    threshold_per = opt.get('cv_threshold_per', 'unit')

    # If no GSN results but mag_type=0 => we can't do eigenvalue-based
    if gsn_results is None and mag_type == 0:
        return np.zeros((nunits, nunits)), None, np.zeros(nunits) if threshold_per == 'unit' else 0

    # Gather magnitudes
    if mag_type == 0:
        # must be V in [0,1,2,3]
        cSb = gsn_results['cSb']
        cNb = gsn_results['cNb']

        def inv_or_pinv(x):
            return np.linalg.pinv(x)

        if V == 0:
            magnitudes = np.abs(np.linalg.eigvals(cSb))
        elif V == 1:
            cNb_inv = inv_or_pinv(cNb)
            M = cNb_inv @ cSb
            magnitudes = np.abs(np.linalg.eigvals(M))
        elif V == 2:
            magnitudes = np.abs(np.linalg.eigvals(cNb))
        else:  # V == 3
            if ntrials > 0:
                trial_avg = np.mean(data, axis=2)
                cov_mat = np.cov(trial_avg)
                magnitudes = np.abs(np.linalg.eigvals(cov_mat))
            else:
                magnitudes = np.array([])
    else:
        # signal variance in user basis
        if basis.shape[1] == 0:
            magnitudes = np.array([])
        else:
            trial_avg = np.mean(data, axis=2)  # shape (nvox, ncond)
            trial_avg_reshaped = trial_avg.T  # shape (ncond, nvox)
            proj_data = trial_avg_reshaped @ basis  # shape (ncond, basis_dim)
            magnitudes = np.var(proj_data, axis=0)  # variance along conditions for each basis dimension

    if magnitudes.size == 0:
        return np.zeros((nunits, nunits)), None, np.zeros(nunits) if threshold_per == 'unit' else 0

    # threshold_val
    threshold_val = mag_frac * np.max(magnitudes)
    # If mag_frac=0 => threshold_val=0 => keep all >= 0
    surviving = magnitudes >= threshold_val

    if mag_mode == 0:
        # contiguous => find the last surviving index, keep up to it
        surv_indices = np.where(surviving)[0]
        if len(surv_indices) > 0:
            last_idx = surv_indices[-1]
            best_threshold = np.arange(last_idx + 1)
        else:
            best_threshold = np.array([], dtype=int)
    else:
        # keep all that survive
        best_threshold = np.where(surviving)[0]

    denoiser = basis[:, best_threshold] @ basis[:, best_threshold].T

    # Return appropriate threshold format based on threshold_per
    if threshold_per == 'unit':
        # For unit-wise, return array of length nunits
        return denoiser, None, np.full(nunits, len(best_threshold))
    else:
        # For population, return scalar
        return denoiser, None, len(best_threshold)
    
    