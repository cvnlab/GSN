function results = gsndenoise(data, V, opt)
    % GSN_DENOISE Denoise neural data using Generative Modeling of Signal and Noise (GSN) denoising.
    %
    % This function implements GSN denoising, which uses cross-validation or magnitude thresholding
    % to identify and remove noise dimensions while preserving signal dimensions. The algorithm:
    %
    % 1. Computes signal and noise covariance matrices from the data
    % 2. Selects a basis for denoising (several options available)
    % 3. Uses either cross-validation or magnitude thresholding to determine which dimensions to retain
    % 4. Constructs a denoising matrix that projects data onto the retained dimensions
    %
    % The denoising can be performed either:
    % - On trial-averaged data (default)
    % - On single trials
    % - Using population-level thresholding (same dimensions for all units)
    % - Using unit-wise thresholding (different dimensions for each unit)
    %
    % Algorithm Details:
    % -----------------
    % The GSN denoising algorithm works by identifying dimensions in the neural data that contain
    % primarily signal rather than noise. It does this in several steps:
    %
    % 1. Signal and Noise Estimation:
    %     - For each condition, computes mean response across trials (signal estimate)
    %     - For each condition, computes variance across trials (noise estimate)
    %     - Builds signal (cSb) and noise (cNb) covariance matrices across conditions
    %
    % 2. Basis Selection (V parameter):
    %     - V=0: Uses eigenvectors of signal covariance (cSb)
    %     - V=1: Uses eigenvectors of signal covariance transformed by inverse noise covariance
    %     - V=2: Uses eigenvectors of noise covariance (cNb)
    %     - V=3: Uses PCA on trial-averaged data
    %     - V=4: Uses random orthonormal basis
    %     - V=matrix: Uses user-supplied orthonormal basis
    %
    % 3. Dimension Selection:
    %     The algorithm must decide how many dimensions to keep. This can be done in two ways:
    %
    %     a) Cross-validation (cv_mode >= 0):
    %         - Splits trials into training and testing sets
    %         - For training set:
    %             * Projects data onto different numbers of basis dimensions
    %             * Creates denoising matrix for each dimensionality
    %         - For test set:
    %             * Measures how well denoised training data predicts test data
    %             * Uses mean squared error (MSE) as prediction metric
    %         - Selects number of dimensions that gives best prediction
    %         - Can be done per-unit or for whole population
    %
    %     b) Magnitude Thresholding (cv_mode < 0):
    %         - Computes "magnitude" for each dimension:
    %             * Either eigenvalues (signal strength)
    %             * Or variance explained in the data
    %         - Sets threshold as fraction of maximum magnitude
    %         - Keeps dimensions above threshold either:
    %             * Contiguously from strongest dimension
    %             * Or any dimension above threshold
    %
    % 4. Denoising:
    %     - Creates denoising matrix using selected dimensions
    %     - For trial-averaged denoising:
    %         * Averages data across trials
    %         * Projects through denoising matrix
    %     - For single-trial denoising:
    %         * Projects each trial through denoising matrix
    %     - Returns denoised data and diagnostic information
    %
    % The algorithm is particularly effective because:
    % - It adapts to the structure of both signal and noise in the data
    % - It can handle different types of neural response patterns
    % - It allows for different denoising strategies (population vs unit-wise)
    % - It provides cross-validation to prevent overfitting
    % - It can denoise both trial-averaged and single-trial data
    %
    % Inputs:
    %   data - Neural responses array of size [nunits x nconds x ntrials]
    %          - nunits: number of units/neurons
    %          - nconds: number of conditions/stimuli
    %          - ntrials: number of repeated measurements
    %          Must have at least 2 trials and 2 conditions.
    %
    %   V    - (Optional) Basis selection mode (0,1,2,3,4) or custom basis matrix.
    %          If matrix: size [nunits x D] with D >= 1, orthonormal columns.
    %          Default: 0 (signal covariance eigenvectors)
    %
    %   opt  - (Optional) Options struct with fields:
    %
    %          Cross-validation options:
    %          .cv_mode - Integer
    %              0: n-1 train / 1 test split (default)
    %              1: 1 train / n-1 test split
    %              -1: use magnitude thresholding instead
    %          .cv_threshold_per - String
    %              'unit': different thresholds per unit (default)
    %              'population': same threshold for all units
    %          .cv_thresholds - Array
    %              Dimensions to test in cross-validation
    %              Default: 1 to nunits
    %          .cv_scoring_fn - Function handle
    %              Function to compute prediction error
    %              Default: negative mean squared error per unit
    %
    %          Magnitude thresholding options:
    %          .mag_type - Integer
    %              0: use eigenvalues (default)
    %              1: use variances
    %          .mag_frac - Double
    %              Fraction of maximum magnitude for threshold
    %              Default: 0.01
    %          .mag_mode - Integer
    %              0: use contiguous dimensions from left (default)
    %              1: use all dimensions above threshold
    %
    %          General options:
    %          .denoisingtype - Integer
    %              0: trial-averaged denoising (default)
    %              1: single-trial denoising
    %
    % Outputs:
    %   results - Struct with fields:
    %     .denoiser - Matrix [nunits x nunits]
    %         Matrix that projects data onto denoised space
    %
    %     .cv_scores - Array
    %         Cross-validation scores for each threshold
    %         Shape depends on cv_mode and cv_threshold_per
    %
    %     .best_threshold - Integer or array
    %         Selected threshold(s)
    %         Scalar for population mode
    %         Array of length nunits for unit mode
    %
    %     .denoiseddata - Array
    %         Denoised neural responses
    %         Size [nunits x nconds] for trial-averaged
    %         Size [nunits x nconds x ntrials] for single-trial
    %
    %     .fullbasis - Matrix [nunits x dims]
    %         Complete basis used for denoising
    %
    %     .signalsubspace - Matrix or []
    %         Final basis functions used for denoising
    %         Empty for unit-wise mode
    %
    %     .dimreduce - Matrix or []
    %         Data projected onto signal subspace
    %         Empty for unit-wise mode
    %
    %     .mags - Array or []
    %         Component magnitudes (for magnitude thresholding)
    %
    %     .dimsretained - Integer or []
    %         Number of dimensions retained (for magnitude thresholding)
    %
    % Examples:
    %   % Basic usage with default options:
    %   data = randn(10, 20, 5);  % 10 units, 20 conditions, 5 trials
    %   results = gsn_denoise(data);
    %   denoised = results.denoiseddata;  % Get denoised data
    %
    %   % Using magnitude thresholding:
    %   opt = struct();
    %   opt.cv_mode = -1;  % Use magnitude thresholding
    %   opt.mag_frac = 0.1;  % Keep components > 10% of max
    %   opt.mag_mode = 0;  % Use contiguous dimensions
    %   results = gsn_denoise(data, 0, opt);
    %
    %   % Unit-wise cross-validation:
    %   opt = struct();
    %   opt.cv_mode = 0;  % Leave-one-out CV
    %   opt.cv_threshold_per = 'unit';  % Unit-specific thresholds
    %   opt.cv_thresholds = [1, 2, 3];  % Test these dimensions
    %   results = gsn_denoise(data, 0, opt);
    %
    %   % Single-trial denoising:
    %   opt = struct();
    %   opt.denoisingtype = 1;  % Single-trial mode
    %   opt.cv_threshold_per = 'population';  % Same dims for all units
    %   results = gsn_denoise(data, 0, opt);
    %   denoised_trials = results.denoiseddata;  % [nunits x nconds x ntrials]
    %
    %   % Custom basis:
    %   nunits = size(data, 1);
    %   [custom_basis, ~] = qr(randn(nunits));
    %   results = gsn_denoise(data, custom_basis);
    %
    % See also:
    %   PERFORM_GSN, PERFORM_CROSS_VALIDATION, PERFORM_MAGNITUDE_THRESHOLDING

    % 1) Check for infinite or NaN data => error if found
    if any(~isfinite(data(:)))
        error('Data contains infinite or NaN values.');
    end

    [nunits, nconds, ntrials] = size(data);

    % 2) If we have fewer than 2 trials, raise an error
    if ntrials < 2
        error('Data must have at least 2 trials.');
    end

    % 2b) Check for minimum number of conditions
    if nconds < 2
        error('Data must have at least 2 conditions to estimate covariance.');
    end

    % 3) If V is not provided => treat it as 0
    if ~exist('V','var') || isempty(V)
        V = 0;
    end

    % 4) Prepare default opts
    if ~exist('opt','var') || isempty(opt)
        opt = struct();
    end

    if isfield(opt, 'cv_threshold_per')
        if ~any(strcmp(opt.cv_threshold_per, {'unit','population'}))
            error('cv_threshold_per must be ''unit'' or ''population''');
        end
    end

    if ~isfield(opt, 'cv_scoring_fn')
        opt.cv_scoring_fn = @negative_mse_columns;
    end
    if ~isfield(opt, 'cv_mode')
        opt.cv_mode = 0;
    end
    if ~isfield(opt, 'cv_threshold_per')
        opt.cv_threshold_per = 'unit';
    end
    if ~isfield(opt, 'mag_type')
        opt.mag_type = 0;
    end
    if ~isfield(opt, 'mag_frac')
        opt.mag_frac = 0.01;
    end
    if ~isfield(opt, 'mag_mode')
        opt.mag_mode = 0;
    end
    if ~isfield(opt, 'denoisingtype')
        opt.denoisingtype = 0;
    end

    gsn_results = [];

    % 5) If V is an integer => glean basis from GSN results
    if isnumeric(V) && isscalar(V)
        if ~ismember(V, [0, 1, 2, 3, 4])
            error('V must be in [0..4] (int) or a 2D numeric array.');
        end

        % We rely on a function "perform_gsn" here (not shown), which returns:
        % gsn_results.cSb and gsn_results.cNb
        gsn_opt = struct();
        gsn_opt.wantverbose = 0;
        gsn_opt.wantshrinkage = 1;
        gsn_results = performgsn(data, gsn_opt);
        cSb = gsn_results.cSb;
        cNb = gsn_results.cNb;

        % Helper for pseudo-inversion
        inv_or_pinv = @(mat) pinv(mat);

        if V == 0
            % Just eigen-decompose cSb
            [evecs, evals_diag] = eig(cSb);
            evals = diag(evals_diag);
            % Sort in descending order
            [evals, idx] = sort(evals, 'descend');
            evecs = evecs(:, idx);
            basis = fliplr(fliplr(evecs)); %#ok<FLR> (No actual effect if we just sorted desc)
            basis = evecs;  % We already sorted in desc order
            mags = abs(evals);
        elseif V == 1
            cNb_inv = inv_or_pinv(cNb);
            transformed_cov = cNb_inv * cSb;
            [evecs, evals_diag] = eig(transformed_cov);
            evals = diag(evals_diag);
            [evals, idx] = sort(evals, 'descend');
            evecs = evecs(:, idx);
            basis = evecs;
            mags = abs(evals);
        elseif V == 2
            [evecs, evals_diag] = eig(cNb);
            evals = diag(evals_diag);
            [evals, idx] = sort(evals, 'descend');
            evecs = evecs(:, idx);
            basis = evecs;
            mags = abs(evals);
        elseif V == 3
            trial_avg = mean(data, 3);  % shape [nunits x nconds]
            cov_matrix = cov(trial_avg.'); % shape [nunits x nunits]
            [evecs, evals_diag] = eig(cov_matrix);
            evals = diag(evals_diag);
            [evals, idx] = sort(evals, 'descend');
            evecs = evecs(:, idx);
            basis = evecs;
            mags = abs(evals);
        else
            rand_mat = randn(nunits);
            [Q, ~] = qr(rand_mat);
            basis = Q(:, 1:nunits);
            mags = ones(nunits, 1);
        end

    else
        % If V not int => must be a numeric array
        if ~ismatrix(V)
            error('If V is not int, it must be a numeric matrix.');
        end
        if size(V, 1) ~= nunits
            error('Basis must have %d rows, got %d.', nunits, size(V, 1));
        end
        if size(V, 2) < 1
            error('Basis must have at least 1 column.');
        end

        % Check unit-length columns
        norms = sqrt(sum(V.^2, 1));
        if ~all(abs(norms - 1) < 1e-10)
            error('Basis columns must be unit length.');
        end
        % Check orthogonality
        gram = V' * V;
        if ~all(all(abs(gram - eye(size(V,2))) < 1e-10))
            error('Basis columns must be orthogonal.');
        end

        basis = V;
        % For user-supplied basis, compute magnitudes based on variance in basis
        trial_avg = mean(data, 3);      % shape [nunits x nconds]
        trial_avg_reshaped = trial_avg.';  % shape [nconds x nunits]
        proj_data = trial_avg_reshaped * basis; % shape [nconds x basis_dim]
        mags = var(proj_data, 0, 1).';  % variance along conditions
    end

    % Store the full basis and magnitudes
    fullbasis = basis;
    stored_mags = mags;

    % 6) Default cross-validation thresholds if not provided
    if ~isfield(opt, 'cv_thresholds')
        opt.cv_thresholds = 1:size(basis, 2);
    else
        thresholds = opt.cv_thresholds;
        % Validate
        if any(thresholds <= 0)
            error('cv_thresholds must be positive integers.');
        end
        if any(thresholds ~= round(thresholds))
            error('cv_thresholds must be integers.');
        end
        if any(diff(thresholds) <= 0)
            error('cv_thresholds must be in sorted order with unique values.');
        end
    end

    % Initialize return structure
    results.denoiser = [];
    results.cv_scores = [];
    results.best_threshold = [];
    results.denoiseddata = [];
    results.fullbasis = fullbasis;
    results.signalsubspace = [];
    results.dimreduce = [];
    results.mags = [];
    results.dimsretained = [];

    % 7) Decide cross-validation or magnitude-threshold
    if opt.cv_mode >= 0
        [denoiser, cv_scores, best_threshold, denoiseddata, fullbasis_out, signalsubspace, dimreduce] = ...
            perform_cross_validation(data, basis, opt);

        results.denoiser = denoiser;
        results.cv_scores = cv_scores;
        results.best_threshold = best_threshold;
        results.denoiseddata = denoiseddata;
        results.fullbasis = fullbasis_out;

        if strcmp(opt.cv_threshold_per, 'population')
            results.signalsubspace = signalsubspace;
            results.dimreduce = dimreduce;
        end

    else
        [denoiser, cv_scores, best_threshold, denoiseddata, fullbasis_out, signalsubspace, dimreduce, mags_out, dimsretained] = ...
            perform_magnitude_thresholding(data, basis, gsn_results, opt, V);

        results.denoiser = denoiser;
        results.cv_scores = cv_scores;
        results.best_threshold = best_threshold;
        results.denoiseddata = denoiseddata;
        results.fullbasis = fullbasis_out;
        results.mags = stored_mags;
        results.dimsretained = dimsretained;
        results.signalsubspace = signalsubspace;
        results.dimreduce = dimreduce;
    end
end


function [denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, signalsubspace, dimreduce] = perform_cross_validation(data, basis, opt)
    % PERFORM_CROSS_VALIDATION Perform cross-validation to select optimal dimensions.
    %
    % This function implements the cross-validation procedure for GSN denoising.
    % It splits the data into training and test sets, then evaluates different
    % numbers of dimensions to find the optimal denoising threshold.
    %
    % The function supports both:
    % - Population-level thresholding (same dimensions for all units)
    % - Unit-wise thresholding (different dimensions for each unit)
    %
    % Algorithm Details:
    % -----------------
    % For each threshold value:
    % 1. Split data into training and test sets
    % 2. Project training data onto different numbers of dimensions
    % 3. Create denoising matrix for each dimensionality
    % 4. Measure prediction quality on test set
    % 5. Select threshold that gives best predictions
    %
    % The splitting can be done in two ways:
    % - Leave-one-out: Use n-1 trials for training, 1 for testing
    % - Hold-one-out: Use 1 trial for training, n-1 for testing
    %
    % Args:
    %   data: ndarray, shape (nunits, nconds, ntrials)
    %       Neural response data to denoise
    %   basis: ndarray, shape (nunits, dims)
    %       Orthonormal basis for denoising
    %   opt: struct with fields:
    %       - cv_mode: int
    %           0: n-1 train / 1 test split
    %           1: 1 train / n-1 test split
    %       - cv_threshold_per: str
    %           'unit': different thresholds per unit
    %           'population': same threshold for all units
    %       - cv_thresholds: array
    %           Dimensions to test
    %       - cv_scoring_fn: function handle
    %           Function to compute prediction error
    %       - denoisingtype: int
    %           0: trial-averaged denoising
    %           1: single-trial denoising
    %
    % Returns:
    %   denoiser: ndarray, shape (nunits, nunits)
    %       Matrix that projects data onto denoised space
    %   cv_scores: ndarray
    %       Cross-validation scores for each threshold
    %   best_threshold: int or array
    %       Selected threshold(s)
    %   denoiseddata: ndarray
    %       Denoised neural responses
    %   fullbasis: ndarray, shape (nunits, dims)
    %       Complete basis used for denoising
    %   signalsubspace: ndarray or []
    %       Final basis functions used for denoising
    %   dimreduce: ndarray or []
    %       Data projected onto signal subspace

    [nunits, nconds, ntrials] = size(data);
    cv_mode = opt.cv_mode;
    thresholds = opt.cv_thresholds;
    if ~isfield(opt,'cv_scoring_fn')
        opt.cv_scoring_fn = @negative_mse_columns;
    end
    threshold_per = opt.cv_threshold_per;
    scoring_fn = opt.cv_scoring_fn;
    denoisingtype = opt.denoisingtype;

    % Initialize cv_scores
    cv_scores = zeros(length(thresholds), ntrials, nunits);

    for tr = 1:ntrials
        if cv_mode == 0
            % Denoise average of n-1 trials, test against held out trial
            train_trials = setdiff(1:ntrials, tr);
            train_avg = mean(data(:, :, train_trials), 3);  % [nunits x nconds]
            test_data = data(:, :, tr);                     % [nunits x nconds]

            for tt = 1:length(thresholds)
                threshold = thresholds(tt);
                safe_thr = min(threshold, size(basis, 2));
                denoising_fn = [ones(1, safe_thr), zeros(1, size(basis,2) - safe_thr)];
                D = diag(denoising_fn);
                denoiser_tmp = basis * D * basis';

                % Denoise the training average
                train_denoised = (train_avg' * denoiser_tmp)';
                cv_scores(tt, tr, :) = scoring_fn(test_data', train_denoised');
            end

        elseif cv_mode == 1
            % Denoise single trial, test against average of n-1 trials
            dataA = data(:, :, tr)';  % [nconds x nunits]
            dataB = mean(data(:, :, setdiff(1:ntrials, tr)), 3)';  % [nconds x nunits]

            for tt = 1:length(thresholds)
                threshold = thresholds(tt);
                safe_thr = min(threshold, size(basis,2));
                denoising_fn = [ones(1, safe_thr), zeros(1, size(basis,2) - safe_thr)];
                D = diag(denoising_fn);
                denoiser_tmp = basis * D * basis';

                dataA_denoised = dataA * denoiser_tmp;
                cv_scores(tt, tr, :) = scoring_fn(dataB, dataA_denoised);
            end
        end
    end

    % Decide best threshold
    if strcmp(threshold_per, 'population')
        % Average over trials and units
        avg_scores = mean(mean(cv_scores, 3), 2);  % shape: [length(thresholds), 1]
        [~, best_ix] = max(avg_scores);
        best_threshold = thresholds(best_ix);
        safe_thr = min(best_threshold, size(basis,2));
        denoiser = basis(:, 1:safe_thr) * basis(:, 1:safe_thr)';
    else
        % unit-wise: average over trials only
        avg_scores = squeeze(mean(cv_scores, 2));  % shape: [length(thresholds), nunits]
        if size(avg_scores, 2) == 1
            avg_scores = avg_scores(:);  % Convert to column vector if only one unit
        end
        best_thresh_unitwise = zeros(1, nunits);
        for unit_i = 1:nunits
            if size(avg_scores, 2) >= unit_i
                [~, best_idx] = max(avg_scores(:, unit_i));
                if ~isempty(best_idx) && best_idx <= length(thresholds)
                    best_thresh_unitwise(unit_i) = thresholds(best_idx);
                else
                    best_thresh_unitwise(unit_i) = thresholds(1);
                end
            else
                % If we don't have enough scores, use the first threshold
                best_thresh_unitwise(unit_i) = thresholds(1);
            end
        end
        best_threshold = best_thresh_unitwise;

        % Construct unit-wise denoiser
        denoiser = zeros(nunits, nunits);
        for unit_i = 1:nunits
            safe_thr = min(best_threshold(unit_i), size(basis,2));
            D = diag([ones(1, safe_thr), zeros(1, size(basis,2) - safe_thr)]);
            unit_denoiser = basis * D * basis';
            denoiser(:, unit_i) = unit_denoiser(:, unit_i);
        end
    end

    % Calculate denoiseddata based on denoisingtype
    if denoisingtype == 0
        % Trial-averaged denoising
        trial_avg = mean(data, 3);  % [nunits x nconds]
        denoiseddata = (trial_avg' * denoiser)';
    else
        % Single-trial denoising
        denoiseddata = zeros(size(data));
        for t = 1:ntrials
            denoiseddata(:, :, t) = (data(:, :, t)' * denoiser)';
        end
    end

    fullbasis = basis;
    if strcmp(threshold_per, 'population')
        signalsubspace = basis(:, 1:safe_thr);
        % Project data onto signal subspace
        if denoisingtype == 0
            trial_avg = mean(data, 3);
            dimreduce = signalsubspace' * trial_avg;  % [safe_thr x nconds]
        else
            dimreduce = zeros(safe_thr, nconds, ntrials);
            for t = 1:ntrials
                dimreduce(:, :, t) = signalsubspace' * data(:, :, t);
            end
        end
    else
        signalsubspace = [];
        dimreduce = [];
    end
end


function [denoiser, cv_scores, best_threshold, denoiseddata, basis, signalsubspace, dimreduce, magnitudes, dimsretained] = ...
    perform_magnitude_thresholding(data, basis, gsn_results, opt, V)
    % PERFORM_MAGNITUDE_THRESHOLDING Select dimensions using magnitude thresholding.
    %
    % This function implements the magnitude thresholding procedure for GSN denoising.
    % It selects dimensions based on their magnitudes (eigenvalues or variances)
    % rather than using cross-validation.
    %
    % The function supports two modes:
    % - Contiguous selection from strongest dimension
    % - Selection of any dimension above threshold
    %
    % Algorithm Details:
    % -----------------
    % 1. Compute magnitudes for each dimension:
    %    - Either eigenvalues from decomposition
    %    - Or variance explained in the data
    % 2. Set threshold as fraction of maximum magnitude
    % 3. Select dimensions either:
    %    - Contiguously from strongest dimension
    %    - Or any dimension above threshold
    % 4. Create denoising matrix using selected dimensions
    %
    % Args:
    %   data: ndarray, shape (nunits, nconds, ntrials)
    %       Neural response data to denoise
    %   basis: ndarray, shape (nunits, dims)
    %       Orthonormal basis for denoising
    %   gsn_results: struct
    %       Results from GSN computation
    %   opt: struct
    %       Options for magnitude thresholding
    %   V: int or ndarray
    %       Basis selection mode or custom basis
    %
    % Returns:
    %   denoiser: ndarray, shape (nunits, nunits)
    %       Matrix that projects data onto denoised space
    %   cv_scores: ndarray
    %       Empty array (not used in magnitude thresholding)
    %   best_threshold: array
    %       Selected dimensions
    %   denoiseddata: ndarray
    %       Denoised neural responses
    %   basis: ndarray, shape (nunits, dims)
    %       Complete basis used for denoising
    %   signalsubspace: ndarray
    %       Final basis functions used for denoising
    %   dimreduce: ndarray
    %       Data projected onto signal subspace
    %   magnitudes: ndarray
    %       Component magnitudes used for thresholding
    %   dimsretained: int
    %       Number of dimensions retained

    [nunits, nconds, ntrials] = size(data);
    mag_type = opt.mag_type;
    mag_frac = opt.mag_frac;
    mag_mode = opt.mag_mode;
    denoisingtype = opt.denoisingtype;

    cv_scores = [];  % Not used in magnitude thresholding

    % Compute magnitudes
    if mag_type == 0
        % Eigenvalue-based
        if isnumeric(V) && isscalar(V)
            if V == 0
                evals = eig(gsn_results.cSb);
                magnitudes = evals;  % Keep original order
            elseif V == 1
                cNb_inv = pinv(gsn_results.cNb);
                matM = cNb_inv * gsn_results.cSb;
                evals = eig(matM);
                magnitudes = evals;
            elseif V == 2
                evals = eig(gsn_results.cNb);
                magnitudes = evals;
            elseif V == 3
                trial_avg = mean(data, 3);
                cov_mat = cov(trial_avg.');
                evals = eig(cov_mat);
                magnitudes = evals;
            else
                magnitudes = ones(size(basis,2),1);
            end
        else
            trial_avg = mean(data, 3);
            proj = (trial_avg.') * basis;
            magnitudes = var(proj, 0, 1).';
        end
    else
        % Variance-based
        trial_avg = mean(data, 3);
        proj = (trial_avg.') * basis;
        magnitudes = var(proj, 0, 1).';
    end

    threshold_val = mag_frac * max(abs(magnitudes));
    surviving = abs(magnitudes) >= threshold_val;
    surv_idx = find(surviving);

    if isempty(surv_idx)
        % No dimensions survive
        denoiser = zeros(nunits, nunits);
        if denoisingtype == 0
            denoiseddata = zeros(nunits, nconds);
        else
            denoiseddata = zeros(nunits, nconds, ntrials);
        end
        signalsubspace = basis(:, 1:0);  % Empty
        if denoisingtype == 0
            dimreduce = zeros(0, nconds);
        else
            dimreduce = zeros(0, nconds, ntrials);
        end
        dimsretained = 0;
        best_threshold = [];
        return
    end

    if mag_mode == 0
        % Contiguous from left
        if numel(surv_idx) == 1
            dimsretained = 1;
            best_threshold = surv_idx;
        else
            gaps = find(diff(surv_idx) > 1);
            if ~isempty(gaps)
                dimsretained = gaps(1);
                best_threshold = surv_idx(1:dimsretained);
            else
                dimsretained = length(surv_idx);
                best_threshold = surv_idx;
            end
        end
    else
        % Keep all dimensions above threshold
        dimsretained = length(surv_idx);
        best_threshold = surv_idx;
    end

    denoising_fn = zeros(1, size(basis,2));
    denoising_fn(best_threshold) = 1;
    D = diag(denoising_fn);
    denoiser = basis * D * basis';

    % Calculate denoised data
    if denoisingtype == 0
        % Trial-averaged denoising
        trial_avg = mean(data, 3);
        denoiseddata = (trial_avg' * denoiser)';
    else
        % Single-trial denoising
        denoiseddata = zeros(size(data));
        for t = 1:ntrials
            denoiseddata(:, :, t) = (data(:, :, t)' * denoiser)';
        end
    end

    signalsubspace = basis(:, best_threshold);
    if denoisingtype == 0
        trial_avg = mean(data, 3);
        dimreduce = signalsubspace' * trial_avg;
    else
        dimreduce = zeros(length(best_threshold), nconds, ntrials);
        for t = 1:ntrials
            dimreduce(:, :, t) = signalsubspace' * data(:, :, t);
        end
    end
end


function scores = negative_mse_columns(x, y)
    % NEGATIVE_MSE_COLUMNS Calculate the negative mean squared error between corresponding columns.
    %
    % This function computes the negative mean squared error (MSE) between each column
    % of two matrices. It is primarily used as a scoring function for cross-validation
    % in GSN denoising, where:
    % - Each column represents a unit/neuron
    % - Each row represents a condition/stimulus
    % - The negative sign makes it compatible with maximization
    %     (higher scores = better predictions)
    %
    % The function handles empty inputs gracefully by returning zeros, which is useful
    % when no data survives thresholding.
    %
    % Algorithm Details:
    % -----------------
    % For each column i:
    % 1. Computes squared differences: (x(:,i) - y(:,i)).^2
    % 2. Takes mean across rows (conditions)
    % 3. Multiplies by -1 to convert from error to score
    %
    % This results in a score where:
    % - 0 indicates perfect prediction
    % - More negative values indicate worse predictions
    % - Each unit gets its own score
    %
    % Args:
    %   x: ndarray, shape (nconds, nunits)
    %       First matrix (usually test data)
    %   y: ndarray, shape (nconds, nunits)
    %       Second matrix (usually predictions)
    %
    % Returns:
    %   scores: ndarray, shape (1, nunits)
    %       Negative MSE for each column/unit
    %
    % Examples:
    %   % Basic usage:
    %   x = [1 2; 3 4];
    %   y = [1.1 2.1; 2.9 3.9];
    %   scores = negative_mse_columns(x, y);
    %   % close to 0

    if isempty(x) || isempty(y)
        scores = zeros(1, size(x, 2));
        return
    end
    diff_sq = (x - y).^2;
    mse_cols = mean(diff_sq, 1);
    scores = -mse_cols;  % negative MSE
end
