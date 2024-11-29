
function [denoiser, denoised_cv_scores, best_threshold] = computedenoiser(data, V, opt)
% COMPUTE_DENOISER Compute an optimal denoising matrix for data based on
% cross-validation performance.
%
% <data> is nunits x nconds x ntrials. This indicates the measured
%   responses to different conditions on distinct trials. The number of
%   trials must be at least 2.
% <V> is nunits x nunits basis matrix used for projection and denoising.
% <opt> (optional) is a struct with the following optional fields:
%   <thresholds> (optional) list or array of thresholds for denoising.
%   Default: 1:nunits. <scoring_fn> (optional) function handle to compute
%   denoised performance. Default: @negative_mse_columns. <threshold_per>
%   (optional) 'population' or 'unit', specifying thresholding method.
%   Default: 'population'. <cv_mode> (optional) int, cross-validation mode:
%       0 - Denoise using single trial against the mean of other trials
%       (default). 1 - Denoise using the mean of trials against a single
%       trial.
%      -1 - Do not perform cross-validation, instead set the threshold to
%      where signal-to-noise ratio (SNR) of the input basis reaches zero.
%
% Perform denoising by projecting the data into a lower-dimensional
% subspace defined by the basis matrix `V` and applying a threshold to the
% eigenvalues of the projection. Cross-validation ensures that denoising
% does not overfit to the data.
%
% Return:
%   <denoiser> as a nunits x nunits optimal denoising matrix.
%   <denoised_cv_scores> as a len(thresholds) x ntrials x nunits
%   cross-validation performance scores for each threshold.
%   <best_threshold> as an int or nunits array of optimal thresholds. For
%   'population', this is a single integer. For 'unit', this is an array of
%   thresholds per unit.
%
% Raises:
%   Error if 'cv_mode' is -1 and no threshold achieves zero noise ceiling
%   SNR. Error if 'denoised_cv_scores' contains NaN or infinite values.
%
% Example:
%   data = randn(100, 10, 20); V = eye(100); opt.thresholds = 1:50;
%   opt.cv_mode = 0; [denoiser, scores, best_thresh] =
%   compute_denoiser(data, V, opt);
%
% History:
%   - 2024/11/26 - Initial port from Python to MATLAB with corrections.

    % Extract dimensions
    [nunits, nconds, ntrials] = size(data);
    
    % Set default options if not provided
    if nargin < 3 || isempty(opt)
        opt = struct();
    end
    if ~isfield(opt, 'thresholds')
        opt.thresholds = 0:nunits;
    end
    if ~isfield(opt, 'scoring_fn')
        opt.scoring_fn = @negative_mse_columns;
    end
    if ~isfield(opt, 'threshold_per')
        opt.threshold_per = 'population';
    end
    if ~isfield(opt, 'cv_mode')
        opt.cv_mode = 0;
    end
    
    thresholds = opt.thresholds;
    scoring_fn = opt.scoring_fn;
    threshold_per = opt.threshold_per;
    cv_mode = opt.cv_mode;
    
    num_thresholds = length(thresholds);
    
    % Initialize array to hold denoised tuning correlations
    denoised_cv_scores = zeros(num_thresholds, ntrials, nunits);
    
    % Prepare for unit-specific thresholds if needed using a preallocated
    % 4D matrix
    if strcmp(threshold_per, 'unit')
        % Initialize a 4D matrix: thresholds x trials x nunits x nunits
        all_denoisers = zeros(num_thresholds, ntrials, nunits, nunits);
    end

    % Iterate through thresholds and trials
    for tt = 1:num_thresholds
        threshold = thresholds(tt);
        
        for tr = 1:ntrials
            % Define cross-validation splits based on cv_mode
            switch cv_mode
                case 0
                    % Single trial as test, others as training
                    dataA = squeeze(data(:, :, tr))'; % nconds x nunits
                    other_trials = setdiff(1:ntrials, tr);
                    dataB = mean(squeeze(data(:, :, other_trials)), 3)'; % nconds x nunits
                case 1
                    % Average trials as test, single trial as training
                    other_trials = setdiff(1:ntrials, tr);
                    dataA = mean(squeeze(data(:, :, other_trials)), 3)'; % nconds x nunits
                    dataB = squeeze(data(:, :, tr))'; % nconds x nunits
                case -1
                    % Special case: No cross-validation
                    data_ctv = permute(data, [2, 3, 1]); % nconds x ntrials x nunits
                    ncsnrs = zeros(nunits, 1);
                    for i = 1:nunits
                        this_eigv = V(:, i);
                        proj_data = data_ctv(:, :, i) * this_eigv; % nconds x 1
                        [~, ncsnr, ~, ~] = compute_noise_ceiling(proj_data');
                        ncsnrs(i) = ncsnr;
                    end
                    
                    % Find the first index where SNR is zero
                    zero_snr_indices = find(ncsnrs == 0, 1, 'first');
                    if isempty(zero_snr_indices)
                        error('Basis SNR never hits 0. Adjust cross-validation settings.');
                    else
                        best_threshold = zero_snr_indices;
                        denoising_fn = [ones(best_threshold, 1); zeros(nunits - best_threshold, 1)];
                        denoiser = V * diag(denoising_fn) * V';
                        return;
                    end
                otherwise
                    error('Invalid cv_mode. Must be 0, 1, or -1.');
            end
            
            % Define denoising function for current threshold
            denoising_fn = [ones(threshold, 1); zeros(nunits - threshold, 1)];
            denoiser_matrix = V * diag(denoising_fn) * V'; % nunits x nunits
            dataA_denoised = dataA * denoiser_matrix; % nconds x nunits
            
            if strcmp(threshold_per, 'unit')
                % Store the denoiser matrix for this trial and threshold
                all_denoisers(tt, tr, :, :) = denoiser_matrix;
            end
            
            % Calculate cross-validation score
            denoised_cv_scores(tt, tr, :) = scoring_fn(dataB, dataA_denoised)';
        end
    end
    
    % Check for invalid values in scores
    if any(~isfinite(denoised_cv_scores(:)))
        error('denoised_cv_scores contains NaN or inf values.');
    end
    
    % Average scores across trials
    mean_denoised_cv_scores = mean(denoised_cv_scores, 2); % num_thresholds x 1 x nunits
    mean_denoised_cv_scores = squeeze(mean_denoised_cv_scores); % num_thresholds x nunits
    
    % Select the best threshold based on population or unit-specific
    % scoring
    if strcmp(threshold_per, 'population')
        % Mean across units for each threshold
        mean_scores_per_threshold = mean(mean_denoised_cv_scores, 2); % num_thresholds x 1
        [~, best_idx] = max(mean_scores_per_threshold);
        best_threshold = thresholds(best_idx);
        denoising_fn = [ones(best_threshold, 1); zeros(nunits - best_threshold, 1)];
        denoiser = V * diag(denoising_fn) * V';
    elseif strcmp(threshold_per, 'unit')
        % Find best threshold per unit
        [~, best_threshold_indices] = max(mean_denoised_cv_scores, [], 1); % 1 x nunits
        best_threshold = thresholds(best_threshold_indices); % 1 x nunits
        
        denoiser = zeros(nunits, nunits);
        
        for u = 1:nunits
            current_threshold = best_threshold(u);
            % Find the index of the current threshold
            tt = find(thresholds == current_threshold, 1);
            if isempty(tt)
                error('Threshold %d not found in thresholds array.', current_threshold);
            end
            % Retrieve the denoiser matrices for the current threshold
            denoiser_trial = squeeze(all_denoisers(tt, :, :, :)); % ntrials x nunits x nunits
            % Average across trials
            this_denoiser = squeeze(mean(denoiser_trial, 1)); % nunits x nunits
            denoiser(:, u) = this_denoiser(:, u);
        end
    else
        error('Invalid threshold_per option. Must be "population" or "unit".');
    end
end


