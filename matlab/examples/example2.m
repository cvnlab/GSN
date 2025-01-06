% GSN Denoising Example in MATLAB
%==================================================
% This MATLAB script demonstrates the application of GSNdenoise on fMRI data.
% It includes steps for data preparation, denoising using GSN, and visualization of results.
%
% Users encountering bugs, unexpected outputs, or other issues regarding GSN should
% raise an issue on GitHub: https://github.com/cvnlab/GSN/issues
%
% The example data has dimensionality 100 voxels x 200 conditions x 3 trials.
% The data are from an fMRI experiment measuring responses to auditory sentences.
% The values reflect fMRI response amplitudes in percent BOLD signal change units.
% The voxels are taken from a brain region in the left hemisphere.
%==================================================

% Clear workspace and command window
clear; clc; close all;

%% Define Paths
% Define paths for home, data, and output directories

homedir = fileparts(pwd);  % Get the parent directory of the current directory
datadir = fullfile(homedir, 'examples', 'data');
outputdir = fullfile(homedir, 'examples', 'example2outputs');

% Create directories if they do not exist
if ~exist(datadir, 'dir')
    mkdir(datadir);
end

if ~exist(outputdir, 'dir')
    mkdir(outputdir);
end

% Display the configured directories
fprintf('Directory to save example dataset:\n\t%s\n\n', datadir);
fprintf('Directory to save example outputs:\n\t%s\n\n', outputdir);

%% Download Dataset
% Define the file name for the dataset
datafn = fullfile(datadir, 'exampledata.mat');

% Check if the dataset already exists; if not, download it
if ~isfile(datafn)
    fprintf('Downloading example dataset and saving to:\n%s\n', datafn);
    dataurl = 'https://osf.io/download/utfpq/';
    websave(datafn, dataurl);
end

% Load the dataset from the .mat file
X = load(datafn);
data = X.data;

%% Prepare Data
% The dataset contains 100 voxels x 200 conditions x 3 trials
% Split data into training (even indices) and testing (odd indices)

train_data = data(:, 1:2:end, :);
test_data = data(:, 2:2:end, :);

% Extract dataset dimensions
[nvox, ncond, ntrial] = size(train_data);
fprintf('Number of voxels: %d, conditions: %d, trials: %d\n', nvox, ncond, ntrial);

%% Perform GSN Denoising
% First get covariance estimates using performgsn
gsn_opt = struct();
gsn_opt.wantshrinkage = true;
gsn_results = performgsn(train_data, gsn_opt);

% Get eigenvectors from GSN's signal covariance matrix
[V, S] = eig(gsn_results.cSb);
S = diag(S);
[S, idx] = sort(real(S), 'descend');
V = real(V(:, idx));

% Set up options for GSN denoising
opt = struct();
opt.cv_mode = 0;  % n-1 train / 1 test split
opt.cv_threshold_per = 'unit';  % Same threshold for all units
opt.cv_thresholds = 1:nvox;  % Test all possible dimensions
opt.cv_scoring_fn = @negative_mse_columns;  % Use negative MSE as scoring function
opt.denoisingtype = 1;  % Single-trial denoising

% Perform GSN denoising on training data using GSN basis
results = gsndenoise(train_data, V, opt);

%% Visualize Covariance Estimates from GSN
% Define the range for color limits in visualizations
rng_vals = [-0.5, 0.5];

% Create a figure for visualizing covariance estimates
figure('Position', [100, 100, 1000, 800]);

% Noise covariance estimate
subplot(2, 2, 1);
imagesc(gsn_results.cN, rng_vals);
colorbar;
title('Raw Noise Covariance (cN)');
axis tight;
axis equal;

% Final noise covariance estimate
subplot(2, 2, 2);
imagesc(gsn_results.cNb, rng_vals);
colorbar;
title('Final Noise Covariance (cNb)');
axis tight;
axis equal;

% Signal covariance estimate
subplot(2, 2, 3);
imagesc(gsn_results.cS, rng_vals);
colorbar;
title('Raw Signal Covariance (cS)');
axis tight;
axis equal;

% Final signal covariance estimate
subplot(2, 2, 4);
imagesc(gsn_results.cSb, rng_vals);
colorbar;
title('Final Signal Covariance (cSb)');
axis tight;
axis equal;

%% Compute and Visualize Eigenspectrum
% Plot Eigenspectrum and NCSNR
figure('Position', [100, 100, 1600, 400]);

% Plot Eigenspectrum
subplot(1, 3, 1);
plot(S, 'k', 'LineWidth', 3);
title({'Signal Covariance', 'Eigenspectrum'});
xlabel('Dimension');
ylabel('Eigenvalue');
set(gca, 'YScale', 'log');
grid on;

% Initialize arrays for signal/noise analysis
ncsnrs = zeros(nvox, 1);
sigvars = zeros(nvox, 1);
noisevars = zeros(nvox, 1);

% Compute metrics for each basis dimension
for i = 1:nvox
    this_eigv = V(:,i);  % Get i-th eigenvector
    % Project data onto this dimension
    proj_data = zeros(1, ncond, ntrial);  % Initialize with correct dimensions
    for t = 1:ntrial
        trial_data = train_data(:,:,t);  % Get data for this trial [nvox x ncond]
        proj_data(1,:,t) = this_eigv' * trial_data;  % Project this trial's data
    end
    [~, ncsnr_i, sigvar_i, noisevar_i] = compute_noise_ceiling(proj_data);
    ncsnrs(i) = ncsnr_i;
    sigvars(i) = sigvar_i;
    noisevars(i) = noisevar_i;
end

% Plot Signal and Noise SD
subplot(1, 3, 2);
plot(sqrt(sigvars), 'LineWidth', 3, 'DisplayName', 'Signal SD');
hold on;
plot(sqrt(noisevars), 'LineWidth', 3, 'DisplayName', 'Noise SD');
plot([0, nvox], [0, 0], 'k--', 'LineWidth', 0.4, 'HandleVisibility', 'off');
if strcmp(opt.cv_threshold_per, 'population')
    xline(results.best_threshold, 'r--', ['Cutoff: ' num2str(results.best_threshold) ' dims'], 'HandleVisibility', 'off');
else
    % Plot each threshold with transparency
    for idx = 1:length(results.best_threshold)
        if idx == length(results.best_threshold)
            xline(results.best_threshold(idx), 'r--', ['Retained dims: ' num2str(mean(results.best_threshold))], 'HandleVisibility', 'off');
        else
            xline(results.best_threshold(idx), 'r--', 'Alpha', 0.3, 'HandleVisibility', 'off');
        end
    end
end
xlabel('Dimension');
ylabel('Standard Deviation');
title({'Signal and Noise SD', 'in Signal Covariance Basis'});
ylim([-0.2, 5.1]);
grid on;
legend('Location', 'best');
hold off;

% Plot NCSNR
subplot(1, 3, 3);
plot(ncsnrs, 'LineWidth', 3, 'Color', 'm', 'DisplayName', 'NCSNR');
hold on;
plot([0, nvox], [0, 0], 'k--', 'LineWidth', 0.4, 'HandleVisibility', 'off');
if strcmp(opt.cv_threshold_per, 'population')
    xline(results.best_threshold, 'r--', ['Cutoff: ' num2str(results.best_threshold) ' dims'], 'HandleVisibility', 'off');
else
    % Plot each threshold with transparency
    for idx = 1:length(results.best_threshold)
        if idx == length(results.best_threshold)
            xline(results.best_threshold(idx), 'r--', ['Retained dims: ' num2str(mean(results.best_threshold))], 'HandleVisibility', 'off');
        else
            xline(results.best_threshold(idx), 'r--', 'Alpha', 0.3, 'HandleVisibility', 'off');
        end
    end
end
xlabel('Dimension');
ylabel('NCSNR');
title({'Noise Ceiling SNR', 'in Signal Covariance Basis'});
ylim([-0.05, 1.3]);
grid on;
legend('Location', 'best');
hold off;

% Print summary statistics
fprintf('\nNCNSR Analysis Summary:\n');
fprintf('├── Signal SD range: [%.3f, %.3f]\n', min(sqrt(sigvars)), max(sqrt(sigvars)));
fprintf('├── Noise SD range: [%.3f, %.3f]\n', min(sqrt(noisevars)), max(sqrt(noisevars)));
fprintf('├── NCSNR range: [%.3f, %.3f]\n', min(ncsnrs), max(ncsnrs));

%% Cross-Validation Scores
figure('Position', [100, 100, 1200, 400]);

% Plot the cross-validation scores
% For single-trial denoising, we need to average across trials
cv_scores = squeeze(mean(results.cv_scores, 2));  % Average across trials if needed
imagesc(cv_scores);
colorbar;
clim([-1, -0.2]);

xlabel('Voxels');
ylabel('PC exclusion threshold');
title('Cross-validation scores per threshold');

hold on;
if strcmp(opt.cv_threshold_per,'unit')
% Add a red dashed line to indicate the optimal threshold
    plot(results.best_threshold, 'ro-', 'LineWidth', 2);
else
    % Add a red dashed line to indicate the optimal threshold
    yline(results.best_threshold, 'r--', 'LineWidth', 2);
end

hold off;

%% Compare Initial Data, Denoised Data, and Noise
% Apply denoising to test data
test_results = gsndenoise(test_data, results.fullbasis, opt);
denoised_data = test_results.denoiseddata;
noise = test_data - denoised_data;  % For single-trial denoising

figure('Position', [100, 100, 2200, 400]);
subplot(1, 4, 1);
imagesc(mean(test_data, 3)', [-4, 4]);
colormap(redblue(256));
colorbar;
title('Initial Data (Trial-Averaged)');
xlabel('Voxels');
ylabel('Conditions');

subplot(1, 4, 2);
imagesc(mean(denoised_data, 3)', [-4, 4]);
colormap(redblue(256));
colorbar;
title('Denoised Data (Trial-Averaged)');
xlabel('Voxels');
ylabel('Conditions');

subplot(1, 4, 3);
imagesc(mean(noise, 3)', [-4, 4]);
colormap(redblue(256));
colorbar;
xlabel('Voxels');
ylabel('Conditions');
title('Noise (Trial-Averaged)');

subplot(1, 4, 4);
imagesc(results.denoiser, [-0.3, 0.3]);
colormap(redblue(256));
colorbar;
xlabel('Voxels');
ylabel('Voxels');
title('Denoising Matrix');

%% Changes in NCSNR After Denoising
figure('Position', [100, 100, 1400, 600]);

% Get NCSNR for test data before and after denoising
[initial_nc, initial_ncsnr, ~, ~] = compute_noise_ceiling(test_data);
[final_nc, final_ncsnr, ~, ~] = compute_noise_ceiling(denoised_data);

% Scatter plot of initial vs. denoised NCSNR
subplot(1, 2, 1);
scatter(initial_ncsnr, final_ncsnr, 10, 'filled');
hold on;
plot([0, 1.5], [0, 1.5], 'r--');
hold off;
xlabel(['Initial Mean NCSNR:', newline, num2str(round(mean(initial_ncsnr), 3))]);
ylabel(['Denoised Mean NCSNR:', newline, num2str(round(mean(final_ncsnr), 3))]);
title(['Change in NCSNR', newline, 'Optimal PC Threshold = ', num2str(mean(results.best_threshold))]);

% Scatter plot of initial vs. denoised NC
subplot(1, 2, 2);
scatter(initial_nc, final_nc, 10, 'filled');
hold on;
plot([0, 100], [0, 100], 'r--');
hold off;
xlabel(['Initial Mean NC:', newline, num2str(round(mean(initial_nc), 3))]);
ylabel(['Denoised Mean NC:', newline, num2str(round(mean(final_nc), 3))]);
title(['Change in Noise Ceiling Pct', newline, 'Optimal PC Threshold = ', num2str(mean(results.best_threshold))]);

%% Helper Functions
function cmap = redblue(m)
    % Red-Blue colormap, diverging.
    % Creates a colormap transitioning from blue to white to red.
    % Input:
    %   m - Number of levels in the colormap (default = 64).
    % Output:
    %   cmap - m x 3 colormap array.

    if nargin < 1
        m = 64; % Default number of colors
    end

    % Generate blue to white
    blue_to_white = [linspace(0, 1, m/2)', linspace(0, 1, m/2)', ones(m/2, 1)];

    % Generate white to red
    white_to_red = [ones(m/2, 1), linspace(1, 0, m/2)', linspace(1, 0, m/2)'];

    % Combine the two
    cmap = [blue_to_white; white_to_red];
end

function scores = negative_mse_columns(pred, actual)
    % Compute negative mean squared error for each column
    scores = -mean((pred - actual).^2, 1);
end

function [noise_ceiling, ncsnr, signal_var, noise_var] = compute_noise_ceiling(data_in)
    % Compute the noise ceiling signal-to-noise ratio (SNR) and percentage noise ceiling for each unit.
    %
    % Parameters:
    %   data_in (array): Data with shape (voxels, conditions, trials), where each voxel has more than 1 trial per condition.
    %
    % Returns:
    %   noise_ceiling (array): Noise ceiling as a percentage for each voxel.
    %   ncsnr (array): Noise ceiling SNR for each voxel.
    %   signal_var (array): Signal variance for each voxel.
    %   noise_var (array): Noise variance for each voxel.

    % Calculate noise variance as mean variance across trials for each voxel
    noise_var = mean(var(data_in, 0, 3), 2);

    % Calculate data variance as variance of the trial means across conditions for each voxel
    data_var = var(mean(data_in, 3), 0, 2);

    % Calculate signal variance by subtracting noise variance from data variance
    signal_var = max(data_var - noise_var / size(data_in, 3), 0);  % Ensure non-negative variance

    % Compute noise ceiling SNR
    ncsnr = sqrt(signal_var) ./ sqrt(noise_var);

    % Calculate noise ceiling as percentage based on SNR
    noise_ceiling = 100 * (ncsnr .^ 2 ./ (ncsnr .^ 2 + 1 / size(data_in, 3)));
end
