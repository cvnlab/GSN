function [results] = gsndenoise(data, V, opt)
% GSNDENOISE Denoise neural data using Generative Modeling of Signal and Noise (GSN).
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
% 2. Basis Selection (<V> parameter):
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
%     a) Cross-validation (<cv_mode> >= 0):
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
%     b) Magnitude Thresholding (<cv_mode> = -1):
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
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <data> - shape [nunits x nconds x ntrials]. This indicates the measured
%   responses to different conditions on distinct trials.
%   The number of trials (<ntrials>) must be at least 2.
% <V> - shape [nunits x nunits] or scalar. Indicates the set of basis functions to use.
%   0 means perform GSN and use the eigenvectors of the
%     signal covariance estimate (cSb)
%   1 means perform GSN and use the eigenvectors of the
%     signal covariance estimate, transformed by the inverse of 
%     the noise covariance estimate (inv(cNb)*cSb)
%   2 means perform GSN and use the eigenvectors of the 
%     noise covariance estimate (cNb)
%   3 means naive PCA (i.e. eigenvectors of the covariance
%     of the trial-averaged data)
%   4 means use a randomly generated orthonormal basis [nunits x nunits]
%   B means use user-supplied basis B. The dimensionality of B
%     should be [nunits x D] where D >= 1. The columns of B should
%     unit-length and pairwise orthogonal.
%   Default: 0.
% <opt> - struct with the following optional fields:
%   <cv_mode> - scalar. Indicates how to determine the optimal threshold:
%     0 means cross-validation using n-1 (train) / 1 (test) splits of trials.
%     1 means cross-validation using 1 (train) / n-1 (test) splits of trials.
%    -1 means do not perform cross-validation and instead set the threshold
%       based on when the magnitudes of components drop below
%       a certain fraction (see <mag_frac>).
%     Default: 0.
%   <cv_threshold_per> - string. 'population' or 'unit', specifying 
%     whether to use unit-wise thresholding (possibly different thresholds
%     for different units) or population thresholding (one threshold for
%     all units). Matters only when <cv_mode> is 0 or 1. Default: 'unit'.
%   <cv_thresholds> - shape [1 x n_thresholds]. Vector of thresholds to evaluate in
%     cross-validation. Matters only when <cv_mode> is 0 or 1.
%     Each threshold is a positive integer indicating a potential 
%     number of dimensions to retain. Should be in sorted order and 
%     elements should be unique. Default: 1:D where D is the 
%     maximum number of dimensions.
%   <cv_scoring_fn> - function handle. For <cv_mode> 0 or 1 only.
%     It is a function handle to compute denoiser performance.
%     Default: @negative_mse_columns. 
%   <mag_type> - scalar. Indicates how to obtain component magnitudes.
%     Matters only when <cv_mode> is -1.
%     0 means use eigenvalues (<V> must be 0, 1, 2, or 3)
%     1 means use signal variance computed from the data
%     Default: 0.
%   <mag_frac> - scalar. Indicates a fraction of the maximum magnitude
%     component. Matters only when <cv_mode> is -1.
%     Default: 0.01.
%   <mag_mode> - scalar. Indicates how to select dimensions. Matters only 
%     when <cv_mode> is -1.
%     0 means use the smallest number of dimensions that all survive threshold.
%       In this case, the dimensions returned are all contiguous from the left.
%     1 means use all dimensions that survive the threshold.
%       In this case, the dimensions returned are not necessarily contiguous.
%     Default: 0.
%   <denoisingtype> - scalar. Indicates denoising type:
%     0 means denoising in the trial-averaged sense
%     1 means single-trial-oriented denoising
%     Note that if <cv_mode> is 0, you probably want <denoisingtype> to be 0,
%     and if <cv_mode> is 1, you probably want <denoisingtype> to be 1, but
%     the code is deliberately flexible for users to specify what they want.
%     Default: 0.
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% Return in all cases:
%   <denoiser> - shape [nunits x nunits]. This is the denoising matrix.
%   <fullbasis> - shape [nunits x dims]. This is the full set of basis functions.
%
% In the case that <denoisingtype> is 0, we return:
%   <denoiseddata> - shape [nunits x nconds]. This is the trial-averaged data
%     after applying the denoiser.
%
% In the case that <denoisingtype> is 1, we return:
%   <denoiseddata> - shape [nunits x nconds x ntrials]. This is the 
%     single-trial data after applying the denoiser.
%
% In the case that <cv_mode> is 0 or 1 (cross-validation):
%   If <cv_threshold_per> is 'population', we return:
%     <best_threshold> - shape [1 x 1]. The optimal threshold (a single integer),
%       indicating how many dimensions are retained.
%     <signalsubspace> - shape [nunits x best_threshold]. This is the final set of basis
%       functions selected for denoising (i.e. the subspace into which
%       we project). The number of basis functions is equal to <best_threshold>.
%     <dimreduce> - shape [best_threshold x nconds] or [best_threshold x nconds x ntrials]. This
%       is the trial-averaged data (or single-trial data) after denoising.
%       Importantly, we do not reconstruct the original units but leave
%       the data projected into the set of reduced dimensions.
%   If <cv_threshold_per> is 'unit', we return:
%     <best_threshold> - shape [1 x nunits]. The optimal threshold for each unit.
%   In both cases ('population' or 'unit'), we return:
%     <denoised_cv_scores> - shape [n_thresholds x ntrials x nunits].
%       Cross-validation performance scores for each threshold.
%
% In the case that <cv_mode> is -1 (magnitude-based):
%   <mags> - shape [1 x dims]. Component magnitudes used for thresholding.
%   <dimsretained> - shape [1 x n_retained]. The indices of the dimensions retained.
%   <signalsubspace> - shape [nunits x n_retained]. This is the final set of basis
%     functions selected for denoising (i.e. the subspace into which
%     we project).
%   <dimreduce> - shape [n_retained x nconds] or [n_retained x nconds x ntrials]. This
%     is the trial-averaged data (or single-trial data) after denoising.
%     Importantly, we do not reconstruct the original units but leave
%     the data projected into the set of reduced dimensions.
%
% -------------------------------------------------------------------------
% Examples:
% -------------------------------------------------------------------------
%
%   % Basic usage with default options
%   data = randn(100, 200, 3);  % 100 voxels, 200 conditions, 3 trials
%   opt.cv_mode = 0;  % n-1 train / 1 test split
%   opt.cv_threshold_per = 'unit';  % Same threshold for all units
%   opt.cv_thresholds = 1:100;  % Test all possible dimensions
%   opt.cv_scoring_fn = @negative_mse_columns;  % Use negative MSE as scoring function
%   opt.denoisingtype = 1;  % Single-trial denoising
%   results = gsndenoise(data, [], opt);
%
%   % Using magnitude thresholding
%   opt = struct();
%   opt.cv_mode = -1;  % Use magnitude thresholding
%   opt.mag_frac = 0.1;  % Keep components > 10% of max
%   opt.mag_mode = 0;  % Use contiguous dimensions
%   results = gsndenoise(data, 0, opt);
%
%   % Unit-wise cross-validation
%   opt = struct();
%   opt.cv_mode = 0;  % Leave-one-out CV
%   opt.cv_threshold_per = 'unit';  % Unit-specific thresholds
%   opt.cv_thresholds = [1, 2, 3];  % Test these dimensions
%   results = gsndenoise(data, 0, opt);
%
%   % Single-trial denoising with population threshold
%   opt = struct();
%   opt.denoisingtype = 1;  % Single-trial mode
%   opt.cv_threshold_per = 'population';  % Same dims for all units
%   results = gsndenoise(data, 0, opt);
%   denoised_trials = results.denoiseddata;  % [nunits x nconds x ntrials]
%
%   % Custom basis
%   nunits = size(data, 1);
%   [custom_basis, ~] = qr(randn(nunits));
%   results = gsndenoise(data, custom_basis);
%
% -------------------------------------------------------------------------
% History:
% -------------------------------------------------------------------------
%
%   - 2025/01/06 - Initial version.

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

    % Check if basis vectors are unit length and normalize if not
    if isnumeric(V) && ~isscalar(V)
        % First check and fix unit length
        vector_norms = sqrt(sum(V.^2, 1));
        if any(abs(vector_norms - 1) > 1e-10)
            fprintf('Normalizing basis vectors to unit length...\n');
            V = V ./ vector_norms;
        end

        % Then check orthogonality
        gram = V' * V;
        if ~all(abs(gram - eye(size(gram))) < 1e-10, 'all')
            fprintf('Adjusting basis vectors to ensure orthogonality...\n');
            V = make_orthonormal(V);
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
            [evecs, evals] = eig(cSb, 'vector');  % Use vector output for eigenvalues
            [~, idx] = sort(abs(evals), 'descend');  % Sort by magnitude
            basis = evecs(:, idx);
            mags = abs(evals(idx));
        elseif V == 1
            cNb_inv = inv_or_pinv(cNb);
            transformed_cov = cNb_inv * cSb;
            [evecs, evals] = eig(transformed_cov, 'vector');  % Use vector output
            [~, idx] = sort(abs(evals), 'descend');  % Sort by magnitude
            basis = evecs(:, idx);
            mags = abs(evals(idx));
        elseif V == 2
            [evecs, evals] = eig(cNb, 'vector');  % Use vector output
            [~, idx] = sort(abs(evals), 'descend');  % Sort by magnitude
            basis = evecs(:, idx);
            mags = abs(evals(idx));
        elseif V == 3
            trial_avg = mean(data, 3);  % shape [nunits x nconds]
            cov_matrix = cov(trial_avg.');  % shape [nunits x nunits]
            [evecs, evals] = eig(cov_matrix, 'vector');  % Use vector output
            [~, idx] = sort(abs(evals), 'descend');  % Sort by magnitude
            basis = evecs(:, idx);
            mags = abs(evals(idx));
        else
            % V == 4 => random orthonormal
            rng('default');  % Reset to default generator
            rng(42, 'twister');  % Set seed to match Python
            rand_mat = randn(nunits);
            [basis, ~] = qr(rand_mat, 0);  % Use economy QR
            basis = basis(:, 1:nunits);
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
% PERFORM_CROSS_VALIDATION Perform cross-validation to determine optimal denoising dimensions.
%
% Uses cross-validation to determine how many dimensions to retain for denoising:
% 1. Split trials into training and testing sets
% 2. Project training data into basis
% 3. Create denoising matrix for each dimensionality
% 4. Measure prediction quality on test set
% 5. Select threshold that gives best predictions
%
% The splitting can be done in two ways:
% - Leave-one-out: Use n-1 trials for training, 1 for testing
% - Keep-one-in: Use 1 trial for training, n-1 for testing
%
% Inputs:
%   <data> - shape [nunits x nconds x ntrials]. Neural response data to denoise.
%   <basis> - shape [nunits x dims]. Orthonormal basis for denoising.
%   <opt> - struct with fields:
%     <cv_mode> - scalar. 
%         0: n-1 train / 1 test split
%         1: 1 train / n-1 test split
%     <cv_threshold_per> - string.
%         'unit': different thresholds per unit
%         'population': same threshold for all units
%     <cv_thresholds> - shape [1 x n_thresholds].
%         Dimensions to test
%     <cv_scoring_fn> - function handle.
%         Function to compute prediction error
%     <denoisingtype> - scalar.
%         0: trial-averaged denoising
%         1: single-trial denoising
%
% Returns:
%   <denoiser> - shape [nunits x nunits]. Matrix that projects data onto denoised space.
%   <cv_scores> - shape [n_thresholds x ntrials x nunits]. Cross-validation scores for each threshold.
%   <best_threshold> - shape [1 x nunits] or scalar. Selected threshold(s).
%   <denoiseddata> - shape [nunits x nconds] or [nunits x nconds x ntrials]. Denoised neural responses.
%   <fullbasis> - shape [nunits x dims]. Complete basis used for denoising.
%   <signalsubspace> - shape [nunits x best_threshold] or []. Final basis functions used for denoising.
%   <dimreduce> - shape [best_threshold x nconds] or [best_threshold x nconds x ntrials] or []. 
%       Data projected onto signal subspace.

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
% Implements the magnitude thresholding procedure for GSN denoising.
% Selects dimensions based on their magnitudes (eigenvalues or variances)
% rather than using cross-validation.
%
% Supports two modes:
% - Contiguous selection of the left-most group of dimensions above threshold
% - Selection of any dimension above threshold
%
% Algorithm Details:
% 1. Compute magnitudes for each dimension:
%    - Either eigenvalues from decomposition
%    - Or variance explained in the data
% 2. Set threshold as fraction of maximum magnitude
% 3. Select dimensions either:
%    - Contiguously from strongest dimension
%    - Or any dimension above threshold
% 4. Create denoising matrix using selected dimensions
%
% Inputs:
%   <data> - shape [nunits x nconds x ntrials]. Neural response data to denoise.
%   <basis> - shape [nunits x dims]. Orthonormal basis for denoising.
%   <gsn_results> - struct. Results from GSN computation containing:
%       <cSb> - shape [nunits x nunits]. Signal covariance matrix.
%       <cNb> - shape [nunits x nunits]. Noise covariance matrix.
%   <opt> - struct with fields:
%       <mag_type> - scalar. How to obtain component magnitudes:
%           0: use eigenvalues (<V> must be 0, 1, 2, or 3)
%           1: use signal variance computed from data
%       <mag_frac> - scalar. Fraction of maximum magnitude to use as threshold.
%       <mag_mode> - scalar. How to select dimensions:
%           0: contiguous from strongest dimension
%           1: any dimension above threshold
%       <denoisingtype> - scalar. Type of denoising:
%           0: trial-averaged
%           1: single-trial
%   <V> - scalar or matrix. Basis selection mode or custom basis.
%
% Returns:
%   <denoiser> - shape [nunits x nunits]. Matrix that projects data onto denoised space.
%   <cv_scores> - shape [0 x 0]. Empty array (not used in magnitude thresholding).
%   <best_threshold> - shape [1 x n_retained]. Selected dimensions.
%   <denoiseddata> - shape [nunits x nconds] or [nunits x nconds x ntrials]. Denoised neural responses.
%   <basis> - shape [nunits x dims]. Complete basis used for denoising.
%   <signalsubspace> - shape [nunits x n_retained]. Final basis functions used for denoising.
%   <dimreduce> - shape [n_retained x nconds] or [n_retained x nconds x ntrials]. 
%       Data projected onto signal subspace.
%   <magnitudes> - shape [1 x dims]. Component magnitudes used for thresholding.
%   <dimsretained> - scalar. Number of dimensions retained.

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
    % NEGATIVE_MSE_COLUMNS Calculate negative mean squared error between columns.
    %
    % Inputs:
    %   <x> - nconds x nunits. First matrix (usually test data).
    %   <y> - nconds x nunits. Second matrix (usually predictions).
    %       Must have same shape as <x>.
    %
    % Returns:
    %   <scores> - 1 x nunits. Negative MSE for each column/unit.
    %           0 indicates perfect prediction
    %           More negative values indicate worse predictions
    %           Each unit gets its own score
    %
    % Example:
    %   x = [1 2; 3 4];  % 2 conditions, 2 units
    %   y = [1.1 2.1; 2.9 3.9];  % Predictions
    %   scores = negative_mse_columns(x, y);  % Close to 0
    %
    % Notes:
    %   The function handles empty inputs gracefully by returning zeros, which is useful
    %   when no data survives thresholding.

    % Calculate negative mean squared error for each column
    scores = -mean((x - y).^2, 1);
end

function V_orthonormal = make_orthonormal(V)
    % MAKE_ORTHONORMAL Find the nearest matrix with orthonormal columns.
    %
    % Uses Singular Value Decomposition (SVD) to find the nearest orthonormal matrix:
    % 1. Decompose <V> = <U>*<S>*<Vh> where <U> and <Vh> are orthogonal
    % 2. The nearest orthonormal matrix is <U>*<Vh>
    % 3. Take only the first n columns if m > n
    % 4. Verify orthonormality within numerical precision
    %
    % Inputs:
    %   <V> - m x n matrix where m >= n. Input matrix to be made orthonormal.
    %       The number of rows (m) must be at least as large as the number of
    %       columns (n).
    %
    % Returns:
    %   <V_orthonormal> - m x n matrix with orthonormal columns.
    %                   The resulting matrix will have:
    %                   1. All columns unit length
    %                   2. All columns pairwise orthogonal
    %
    % Example:
    %   V = randn(5,3);  % Random 5x3 matrix
    %   V_ortho = make_orthonormal(V);
    %   % Check orthonormality
    %   gram = V_ortho' * V_ortho;  % Should be very close to identity
    %   disp(max(abs(gram - eye(size(gram))), [], 'all'));  % Should be ~1e-15
    %
    % Notes:
    %   The SVD method guarantees orthonormality within numerical precision.
    %   A warning is issued if the result is not perfectly orthonormal.
    
    % Check input dimensions
    [m, n] = size(V);
    if m < n
        error('Input matrix must have at least as many rows as columns');
    end
    
    % Use SVD to find the nearest orthonormal matrix
    % SVD gives us V = U*S*Vh where U and Vh are orthogonal
    % The nearest orthonormal matrix is U*Vh
    [U, ~, Vh] = svd(V, 'econ');
    
    % Take only the first n columns of U if m > n
    V_orthonormal = U(:,1:n) * Vh';
    
    % Double check that the result is orthonormal within numerical precision
    % This is mainly for debugging - the SVD method should guarantee this
    gram = V_orthonormal' * V_orthonormal;
    if ~all(abs(gram - eye(n)) < 1e-10, 'all')
        warning('Result may not be perfectly orthonormal due to numerical precision');
    end
end
