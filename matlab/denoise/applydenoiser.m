
function [denoised_data, denoiser_out, noise, ncsnrs, ncs] = applydenoiser(data, denoiser, rescue)
% APPLY_DENOISER Apply a denoising matrix to trial-wise data, optionally
% rescuing any degraded voxels.
%
% <data> is nunits x nconds x ntrials. This indicates the measured
%   responses to different conditions on distinct trials. The number of
%   trials must be at least 2.
% <denoiser> is nunits x nunits denoising matrix used to project data into
% a denoised subspace. <rescue> (optional) is a boolean indicating whether
% to "rescue" voxels whose noise ceiling
%   signal-to-noise ratio (ncsnr) decreases after denoising. Default:
%   false.
%
% Perform denoising by applying the denoising matrix to each trial's data.
% Optionally, "rescue" any voxels that exhibit degraded performance after
% denoising by retaining their original values and adjusting the denoising
% matrix accordingly.
%
% Return:
%   <denoised_data> as a nunits x nconds x ntrials array of denoised data.
%   <denoiser_out> as a nunits x nunits adjusted denoising matrix. If
%   `rescue=true`,
%     voxels that are rescued act as an identity matrix; otherwise, it is
%     the same as the input denoiser.
%   <noise> as a nunits x nconds x ntrials array of residual noise data,
%   defined as the difference
%     between the original and denoised data.
%   <ncsnrs> as a 1 x (2 + rescue) array of signal-to-noise ratios:
%     - Before denoising - After denoising - After rescuing (if applicable)
%   <ncs> as a 1 x (2 + rescue) array of noise ceilings:
%     - Before denoising - After denoising - After rescuing (if applicable)
%
% Raises:
%   Error if input data dimensions are inconsistent.
%
% Example:
%   data = randn(100, 10, 20); denoiser = eye(100); [denoised_data,
%   denoiser_out, noise, ncsnrs, ncs] = apply_denoiser(data, denoiser,
%   true);
%
% History:
%   - 2024/11/26 - Ported from Python to MATLAB with detailed
%   documentation.

    % Check input dimensions
    [nunits, nconds, ntrials] = size(data);
    [d_nunits1, d_nunits2] = size(denoiser);
    if d_nunits1 ~= nunits || d_nunits2 ~= nunits
        error('Denoiser matrix dimensions must be nunits x nunits.');
    end
    
    % Set default value for rescue if not provided
    if nargin < 3 || isempty(rescue)
        rescue = false;
    end
    
    % Initialize array for denoised data
    denoised_data = zeros(nunits, nconds, ntrials);
    
    % Apply denoising to each trial
    for tr = 1:ntrials
        % Extract data for this trial (nunits x nconds)
        this_data = squeeze(data(:, :, tr))'; % nunits x nconds
        
        % Apply denoising matrix (nunits x nunits) * (nunits x nconds) =
        % nunits x nconds
        denoised_trial = (this_data * denoiser)';
        
        % Store denoised data
        denoised_data(:, :, tr) = denoised_trial;
    end
    
    % Compute initial and final noise ceiling and signal-to-noise ratio
    [nc_init, ncsnr_init, ~, ~] = computenoiseceiling(data);
    [nc_final, ncsnr_final, ~, ~] = computenoiseceiling(denoised_data);
    
    % Store signal-to-noise ratios and noise ceilings
    ncsnrs = [ncsnr_init, ncsnr_final];
    ncs = [nc_init, nc_final];
    
    % Initialize denoiser_out
    denoiser_out = denoiser;
    
    % Handle voxel "rescue" if performance degrades after denoising
    if rescue
        % Identify voxels with degraded performance
        restore_vox = ncsnr_init > ncsnr_final; % Logical array: 1 x nunits
        
        % Restore original data for these voxels
        denoised_data(restore_vox, :, :) = data(restore_vox, :, :);
        
        % Adjust denoising matrix to act as identity for rescued voxels
        identity_matrix = eye(nunits);
        denoiser_out(:, restore_vox) = identity_matrix(:, restore_vox);
        
        % Recompute noise ceiling and SNR after rescue
        [nc_rescued, ncsnr_rescued, ~, ~] = computenoiseceiling(denoised_data);
        ncsnrs = [ncsnrs, ncsnr_rescued];
        ncs = [ncs, nc_rescued];
    end
    
    % Compute residual noise as the difference between original and
    % denoised data
    noise = data - denoised_data; % nunits x nconds x ntrials

end


