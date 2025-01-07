function [train_data, test_data, ground_truth] = simulatedata(varargin)
% SIMULATEDATA Generate synthetic neural data with controlled signal and noise properties.
%
% This function generates synthetic neural data with specific covariance structures
% for both signal and noise components. The data generation process allows for:
% - Control over signal and noise decay rates
% - Alignment between signal and noise principal components
% - Separate train and test datasets with matched properties
%
% Usage:
%   [train_data, test_data, ground_truth] = simulatedata('param1', value1, ...)
%
% Parameters:
%   'nvox'  - Number of voxels/units (default: 50)
%   'ncond' - Number of conditions (default: 100)
%   'ntrial' - Number of trials per condition (default: 3)
%   'signal_decay' - Rate of eigenvalue decay for signal covariance (default: 1.0)
%   'noise_decay' - Rate of eigenvalue decay for noise covariance (default: 1.0)
%   'noise_multiplier' - Scaling factor for noise variance (default: 1.0)
%   'align_alpha' - Alignment between signal & noise PCs (0=aligned, 1=orthogonal) (default: 0.0)
%   'align_k' - Number of top PCs to align (default: 0)
%   'random_seed' - Random seed for reproducibility (default: [])
%
% Returns:
%   train_data - [nvox x ncond x ntrial] Training data
%   test_data  - [nvox x ncond x ntrial] Test data
%   ground_truth - struct with fields:
%       'signal'      - [ncond x nvox] True signal
%       'signal_cov'  - [nvox x nvox] Signal covariance
%       'noise_cov'   - [nvox x nvox] Noise covariance
%       'U_signal'    - Original eigenvectors for signal
%       'U_noise'     - Original eigenvectors for noise
%       'signal_eigs' - Signal eigenvalues
%       'noise_eigs'  - Noise eigenvalues

% Parse inputs
p = inputParser;
addParameter(p, 'nvox', 50, @isnumeric);
addParameter(p, 'ncond', 100, @isnumeric);
addParameter(p, 'ntrial', 3, @isnumeric);
addParameter(p, 'signal_decay', 1.0, @isnumeric);
addParameter(p, 'noise_decay', 1.0, @isnumeric);
addParameter(p, 'noise_multiplier', 1.0, @isnumeric);
addParameter(p, 'align_alpha', 0.0, @isnumeric);
addParameter(p, 'align_k', 0, @isnumeric);
addParameter(p, 'random_seed', [], @(x) isempty(x) || isnumeric(x));
parse(p, varargin{:});

% Extract parameters
nvox = p.Results.nvox;
ncond = p.Results.ncond;
ntrial = p.Results.ntrial;
signal_decay = p.Results.signal_decay;
noise_decay = p.Results.noise_decay;
noise_multiplier = p.Results.noise_multiplier;
align_alpha = p.Results.align_alpha;
align_k = p.Results.align_k;
random_seed = p.Results.random_seed;

% Set random seed if provided
if ~isempty(random_seed)
    rng(random_seed, 'twister');
end

% Generate random orthonormal matrices for signal & noise
[U_signal, ~] = qr(randn(nvox));
[U_noise, ~] = qr(randn(nvox));

% Possibly adjust noise eigenvectors alignment
if align_k > 0
    U_noise = adjust_alignment(U_signal, U_noise, align_alpha, align_k);
end

% Create diagonal eigenvalues
signal_eigs = 1.0 ./ ((1:nvox) .^ signal_decay)';
noise_eigs = noise_multiplier ./ ((1:nvox) .^ noise_decay)';

% Build covariance matrices
signal_cov = U_signal * diag(signal_eigs) * U_signal';
noise_cov = U_noise * diag(noise_eigs) * U_noise';

% Generate the ground truth signal
true_signal = mvnrnd(zeros(1,nvox), signal_cov, ncond);  % shape (ncond, nvox)

% Preallocate train/test data
train_data = zeros(nvox, ncond, ntrial);
test_data = zeros(nvox, ncond, ntrial);

% Generate data
for t = 1:ntrial
    % Independent noise for each trial
    train_noise = mvnrnd(zeros(1,nvox), noise_cov, ncond);  % shape (ncond, nvox)
    test_noise = mvnrnd(zeros(1,nvox), noise_cov, ncond);   % shape (ncond, nvox)
    
    % Add noise to signal
    train_data(:,:,t) = (true_signal + train_noise)';
    test_data(:,:,t) = (true_signal + test_noise)';
end

% Package ground truth information
ground_truth = struct(...
    'signal', true_signal, ...
    'signal_cov', signal_cov, ...
    'noise_cov', noise_cov, ...
    'U_signal', U_signal, ...
    'U_noise', U_noise, ...
    'signal_eigs', signal_eigs, ...
    'noise_eigs', noise_eigs);

end

function U_noise_adjusted = adjust_alignment(U_signal, U_noise, alpha, k, tolerance)
% ADJUST_ALIGNMENT Adjust alignment between the top-k columns of U_signal and U_noise.
%
% Parameters:
%   U_signal - [nvox x nvox] orthonormal (columns are principal dirs)
%   U_noise  - [nvox x nvox] orthonormal
%   alpha    - float in [0,1], where 1 => perfect alignment, 0 => orthogonal
%   k        - int, number of top PCs to align
%   tolerance- numeric tolerance for final orthonormal checks (default: 1e-9)

if nargin < 5
    tolerance = 1e-9;
end

% Clamp alpha to [0,1]
if alpha < 0 || alpha > 1
    warning('alpha must be in [0,1]; will be clamped.');
    alpha = max(0, min(alpha, 1));
end

nvox = size(U_signal, 1);
if k > nvox
    error('k cannot exceed the number of columns in U_signal.');
end

% If k=0, return original noise basis
if k == 0
    U_noise_adjusted = U_noise;
    return;
end

% Start with a copy of U_noise
U_noise_adjusted = U_noise;

% For each of the first k components
for i = 1:k
    v_sig = U_signal(:,i);
    v_noise = U_noise(:,i);
    
    % Create a vector that's orthogonal to v_sig
    v_orth = v_noise - (dot(v_noise, v_sig) * v_sig);
    v_orth_norm = norm(v_orth);
    
    if v_orth_norm < 1e-10
        % If v_noise is too close to v_sig, find another orthogonal vector
        for j = 1:nvox
            e_j = zeros(nvox,1);
            e_j(j) = 1.0;
            v_candidate = e_j - (dot(e_j, v_sig) * v_sig);
            v_candidate_norm = norm(v_candidate);
            if v_candidate_norm > 1e-10
                v_orth = v_candidate / v_candidate_norm;
                break;
            end
        end
    else
        v_orth = v_orth / v_orth_norm;
    end
    
    % Create the aligned vector as a weighted combination
    v_aligned = alpha * v_sig + sqrt(1 - alpha^2) * v_orth;
    v_aligned = v_aligned / norm(v_aligned);
    
    % Update the i-th column
    U_noise_adjusted(:,i) = v_aligned;
    
    % Orthogonalize all remaining columns with respect to this one
    for j = (i+1):nvox
        v_j = U_noise_adjusted(:,j);
        v_j = v_j - (dot(v_j, v_aligned) * v_aligned);
        v_j_norm = norm(v_j);
        if v_j_norm > 1e-10
            U_noise_adjusted(:,j) = v_j / v_j_norm;
        else
            % If the vector becomes degenerate, find a replacement
            for idx = 1:nvox
                e_idx = zeros(nvox,1);
                e_idx(idx) = 1.0;
                % Make orthogonal to all previous vectors
                for m = 1:j-1
                    e_idx = e_idx - (dot(e_idx, U_noise_adjusted(:,m)) * U_noise_adjusted(:,m));
                end
                e_norm = norm(e_idx);
                if e_norm > 1e-10
                    U_noise_adjusted(:,j) = e_idx / e_norm;
                    break;
                end
            end
        end
    end
end

% Final orthogonalization pass to ensure numerical stability
for i = 1:nvox
    v_i = U_noise_adjusted(:,i);
    % Orthogonalize with respect to all previous vectors
    for j = 1:i-1
        v_i = v_i - (dot(v_i, U_noise_adjusted(:,j)) * U_noise_adjusted(:,j));
    end
    v_i_norm = norm(v_i);
    if v_i_norm > 1e-10
        U_noise_adjusted(:,i) = v_i / v_i_norm;
    end
end

% Verify orthonormality
gram = U_noise_adjusted' * U_noise_adjusted;
if ~all(abs(gram - eye(nvox)) < tolerance, 'all')
    warning('Result may not be perfectly orthonormal due to numerical precision');
end

end 