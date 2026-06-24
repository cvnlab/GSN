from gsn.fast_perform_gsn import fast_perform_gsn

def perform_gsn(data, opt=None):
    """
    Perform GSN (generative modeling of signal and noise).

    Parameters:
    data (numpy.ndarray): An array of shape (voxels, conditions, trials). This indicates the measured responses to different conditions on distinct trials. The number of trials must be at least 2. 
        Not all conditions need to have the full set of trials (see more information below).

    opt (dict, optional): A dictionary with the following optional fields:
        wantverbose (bool, optional): Whether to print status statements. Default is True.
        wantshrinkage (bool, optional): Whether to use shrinkage in the estimation of covariance. Default is True.
        backend (str, optional): Which compute path to use. 'auto' (default)
            uses torch if it is installed and numpy otherwise; 'numpy' forces
            the reference numpy/scipy path; 'torch' forces the torch path and
            errors if torch is missing. The numpy path is the reference
            implementation; the torch path (CPU or GPU) is faster and is
            validated against numpy in the test suite.
        device (str, optional): Torch device for the batched shrinkage-NLL fast path.
            One of 'cpu' (default), 'cuda', 'mps', or 'auto' (picks cuda > mps > cpu by
            availability). 'cpu' is the right choice up to N ≈ 1000 voxels because
            GPU host↔device transfer dominates below that; 'cuda' / 'mps' open up
            the GPU path for larger N. Requires torch (`pip install gsn[fast]`).
        returns (iterable of str, optional): Which optional items to
            include in the result dict. Default ``('cN', 'cS', 'cNb',
            'cSb')`` — the four matrices the legacy perform_gsn always
            returned. Pass an iterable like ``['cSb', 'cNb']`` if you
            don't need cN / cS and want to save host memory at large N.
            Valid names:
              ``'cN', 'cS', 'cNb', 'cSb'``  — covariance matrices.
              ``'eigvecs_signal', 'eigvals_signal'``      — eigh of cSb.
              ``'eigvecs_difference', 'eigvals_difference'`` — eigh of
                  ``cSb - cNb / ntrial`` (symmetric, generally indefinite).
            Eigenvectors are columns, sorted by descending eigenvalue.
            Pre-computing these here saves the dominant cost of PSN
            at large N (eigh is O(N^3)) so downstream PSN calls can
            consume ``opt['basis'] = <matrix>`` and skip basis
            construction entirely.
        eigh_device ({'host', 'device'}, optional): Where to run the
            eigh that produces the eigvecs_* / eigvals_* returns.
            Default ``'host'`` uses numpy.linalg.eigh, which matches
            PSN's own eigh_descending_sym bit-for-bit (same LAPACK
            call + same deterministic sign convention) — so the
            cached eigvecs are a true drop-in for PSN's internal
            'signal' / 'difference' branches. ``'device'`` runs the
            eigh on the active torch device (typically much faster
            on GPU at large N) but picks a different orthonormal
            basis on cSb's zero-eigenvalue subspace; PSN's threshold
            selection is sensitive to that, so downstream results
            will diverge by a few percent from a PSN-internal eigh.
            Only relevant
            when an eigvecs_* / eigvals_* item is in ``returns``.

    Regarding uneven number of trials across conditions:
    - It is acceptable that different conditions may have different numbers
      of trials. To indicate the lack of data for certain trials, you can
      include NaNs --- specifically, it is okay if data[:, i, j]
      consists of NaNs for some combination(s) of i and j.
    - However, it must be the case that each condition has at least one 
      trial with valid data (i.e. data[:, i, :] must contain at least one
      valid trial).
    - Also, there must be a sufficient number of conditions with at least
      two trials of valid data (for estimation and cross-validation purposes).
      If this is ever not the case, we will issue an error and crash.
    - If the user provides data with uneven number of trials across
      conditions, our estimation strategy changes:
      - We estimate noise covariance for each condition (ignoring missing data)
        and average the estimated noise covariance across conditions. 
        Note that this approach does not perform any special compensation 
        for the differing numbers of trials available across conditions.
      - We determine the minimum number of trials and randomly select that
        many trials from each condition for the purposes of estimation of 
        data covariance. By doing so, we get a nice fully balanced data subset.
        Note that this approach introduces some stochasticity and
        ignores some portion of the data.
      - The biconvex optimization procedure proceeds as usual. (For the
        weighting step, we calculate the equivalent "average number of trials"
        that were actually used for noise covariance estimation.)

    The handling of uneven/missing data is selected by opt['uneven']:
      - 'fast' (default): NaN-aware whole-trial estimation (a trial counts only
        if every unit is present).
      - 'missing': per-unit missing-data estimation (a trial may have some units
        present and others missing); no valid data is discarded.
      - 'reference': the original rsa_noise_ceiling estimation path described
        above (stochastic min-trial subsetting), kept as a parity oracle.

    Returns:
    results: A dictionary with the results.

    Always present (cheap; no opt-in needed):
        mnN - the estimated mean of the noise (1 x voxels)
        mnS - the estimated mean of the signal (1 x voxels)
        ncsnr - the 'noise ceiling SNR' estimate for each voxel (1 x voxels).
                This is, for each voxel, the std dev of the estimated signal
                distribution divided by the std dev of the estimated noise
                distribution. Note that this is computed on the raw
                estimated covariances. Also, note that we apply positive
                rectification (to prevent non-sensical negative ncsnr values).
        shrinklevelN - shrinkage level chosen for cN
        shrinklevelD - shrinkage level chosen for the estimated data covariance
        numiters - the number of iterations used in the biconvex optimization.
                0 means the first estimate was already positive semi-definite.

    Present iff named in ``opt['returns']`` (see above for the default set
    and how to override):
        cN  - raw estimated covariance of the noise (voxels x voxels)
        cS  - raw estimated covariance of the signal (voxels x voxels)
        cNb - noise covariance after biconvex optimization (voxels x voxels)
        cSb - signal covariance after biconvex optimization (voxels x voxels)

    History:
    - 2025/07/15 - add support for uneven number of trials
    - 2024/08/24 - add results['numiters']
    - 2024/01/05 - (1) major change to use the biconvex optimization procedure --
                       we now have cSb and cNb as the final estimates;
                   (2) cSb no longer has the scaling baked in and instead we
                       create a separate temporary variable cSb_rsa;
                   (3) remove the rapprox output

    Example:
    data = np.random.randn(100, 40, 4) * 2 + np.random.randn(100, 40, 4)
    results = perform_gsn(data)
    """

    # Backwards-compat defaults. fast_perform_gsn's DEFAULT_RETURNS is
    # the legacy ('cN', 'cS', 'cNb', 'cSb') set so we don't need to
    # inject 'returns' here — passing opt straight through gives the
    # same result whether the caller omitted it or set it explicitly.
    # Copy so we never mutate the caller's opt dict.
    opt = {} if opt is None else dict(opt)
    if 'wantverbose' not in opt or opt['wantverbose'] is None:
        opt['wantverbose'] = 1
    if 'wantshrinkage' not in opt or opt['wantshrinkage'] is None:
        opt['wantshrinkage'] = 1

    return fast_perform_gsn(data, opt)
