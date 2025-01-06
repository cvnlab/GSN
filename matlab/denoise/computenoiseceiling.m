function [noiseceiling, ncsnr, signalvar, noisevar] = computenoiseceiling(data_in)
% COMPUTE_NOISE_CEILING Compute the noise ceiling signal-to-noise ratio (SNR) and percentage noise ceiling.
%
% Assumes input data is in the shape (vox, cond, trial). The noise ceiling is computed
% based on the variance explained by the signal and noise components for each voxel.
%
% Parameters:
%   <data_in>: 3D array
%     A (vox, cond, trial) array where:
%     - vox: Number of voxels
%     - cond: Number of conditions
%     - trial: Number of trials
%
% Returns:
%   <noiseceiling>: (vox x 1) array, noise ceiling in percentage for each voxel.
%   <ncsnr>: (vox x 1) array, noise ceiling SNR for each voxel.
%   <signalvar>: (vox x 1) array, estimated signal variance for each voxel.
%   <noisevar>: (vox x 1) array, noise variance for each voxel.

    % Check input dimensions
    if ndims(data_in) ~= 3
        error('Input data must be a 3D array of shape (vox, cond, trial).');
    end

    % Compute noise standard deviation (noisesd)
    % Take std across trials (dim 3), square, average across conditions (dim 2), then take the sqrt.
    noisesd = sqrt(mean(std(data_in, [], 3).^2, 2)); % (vox x 1)
    noisevar = noisesd.^2;

    % Compute total variance of the single-trial betas
    % Reshape into (vox x all_single_trials), take std along the 2nd dim, and square.
    totalvar = std(reshape(data_in, size(data_in, 1), []), 0, 2).^2; % (vox x 1)

    % Compute signal variance
    % Subtract noise variance from total variance, positively rectify (no negatives).
    signalvar = totalvar - noisevar;
    signalvar(signalvar < 0) = 0; % Ensure non-negative variance

    % Compute noise ceiling SNR (ncsnr)
    % Ratio of signal standard deviation to noise standard deviation
    ncsnr = sqrt(signalvar) ./ noisesd; % (vox x 1)

    % Compute noise ceiling as percentage of explainable variance
    % For the case of 3 trials
    ntrials = size(data_in, 3); % Number of trials
    noiseceiling = 100 * (ncsnr.^2 ./ (ncsnr.^2 + 1 / ntrials)); % (vox x 1)
end