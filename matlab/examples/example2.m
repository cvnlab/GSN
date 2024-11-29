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

train_data = data(:, 1:end, :);
test_data = data(:, 1:end, :);

% Extract dataset dimensions
[nvox, ncond, ntrial] = size(train_data);
fprintf('Number of voxels: %d, conditions: %d, trials: %d\n', nvox, ncond, ntrial);

%% Perform GSN Denoising
% Perform GSN denoising on training data with shrinkage enabled

% Ensure that GSN functions are in the MATLAB path
% addpath(fullfile(homedir, 'path_to_gsn_functions')); % Adjust the path as needed

% Perform GSN denoising
options.wantshrinkage = true;
results = performgsn(train_data, options);


%% Visualize Covariance Estimates
% Define the range for color limits in visualizations
rng_vals = [-0.5, 0.5];

% Create a figure for visualizing covariance estimates
figure('Position', [100, 100, 1000, 800]);

% Noise covariance estimate
subplot(2, 2, 1);
imagesc(results.cN, rng_vals);
colorbar;
title('Raw Noise Covariance (cN)');
axis tight;
axis equal;

% Final noise covariance estimate
subplot(2, 2, 2);
imagesc(results.cNb, rng_vals);
colorbar;
title('Final Noise Covariance (cNb)');
axis tight;
axis equal;

% Signal covariance estimate
subplot(2, 2, 3);
imagesc(results.cS, rng_vals);
colorbar;
title('Raw Signal Covariance (cS)');
axis tight;
axis equal;

% Final signal covariance estimate
subplot(2, 2, 4);
imagesc(results.cSb, rng_vals);
colorbar;
title('Final Signal Covariance (cSb)');
axis tight;
axis equal;

% Save or display the figure as needed
% saveas(gcf, fullfile(outputdir, 'covariance_estimates.png'));

%% Overview of Results
% We have several outputs in results:
% mnN is the mean of the noise distribution
% cN is the raw estimate of the covariance of the noise distribution 
% cNb is the final estimate of the covariance of the noise distribution
% mnS is the mean of the signal distribution
% cS is the raw estimate of the covariance of the signal distribution 
% cSb is the final estimate of the covariance of the signal distribution
% shrinklevelN is the shrinkage fraction used when estimating the noise distribution
% shrinklevelD is the shrinkage fraction used when estimating the data distribution
% ncsnr is the noise ceiling snr for each voxel (signal sd divided by noise sd)

%% Compute Basis for Denoising
% Perform eigendecomposition for the chosen (signal) basis

basis = 'cSb';
[V, S] = eig(results.(basis));
S = diag(S);
[S, idx] = sort(real(S), 'descend');
V = real(V(:, idx));

%% Compute the Denoising Matrix
% Configure options for the denoiser

opt.threshold_per = 'unit';
opt.scoring_fn = @negativemsecolumns;  % Define or implement this function
opt.thresholds = 1:nvox;
opt.cv_mode = 0;

% Compute the optimal denoiser based on cross-validation
[denoiser, cv_scores, best_threshold] = computedenoiser(train_data, V, opt);

%% Apply the Denoising Matrix to Held-Out Data
% Apply the computed denoiser to the test data

[denoised_data, denoiser_out, noise, ncsnrs, ncs] = applydenoiser(test_data, denoiser, false);

% Verify that the original data is reconstructable from denoised data and noise
assert(all(abs(mean(test_data, 3) - (mean(denoised_data, 3) + mean(noise, 3))) < 1e-10, 'all'), ...
    'Reconstruction from denoised data and noise failed.');

%% Diagnostic Plots

% Plot Eigenspectrum and NCSNR
figure('Position', [100, 100, 1600, 400]);

% Plot Eigenspectrum
subplot(1, 3, 1);
plot(S, 'k', 'LineWidth', 3);
title(['Eigenspectrum of ', basis, ' Basis']);
xlabel('Dimension');
ylabel('Variance');
grid on;

% Plot Signal/Noise SD and NCSNR
plotbasisdimncsnr(train_data, V, basis, best_threshold);

% Add Overall Title
sgtitle('Eigenspectrum and NCSNR Analysis');

% Save or display the figure as needed
% saveas(gcf, fullfile(outputdir, 'eigenspectrum_ncsnr.png'));

%% Cross-Validation Scores per Voxel/Threshold
figure('Position', [100, 100, 1200, 400]);

% Plot the mean cross-validation scores for each threshold
imagesc(squeeze(mean(cv_scores, 2)), 'XData', [1, nvox], 'YData', [opt.thresholds(1), opt.thresholds(end)]);
axis tight;
colorbar;
clim([-1, -0.2]);

% Set y-axis ticks to display thresholds every 5 steps
yticks(1:5:length(opt.thresholds));
yticklabels(string(opt.thresholds(1:5:end)));

hold on;

% Add a red dashed line to indicate the optimal threshold
if strcmp(opt.threshold_per, 'population')
    yline(mean(best_threshold), 'r--', 'LineWidth', 2);
else
    plot(1:nvox, best_threshold, 'ro-', 'LineWidth', 1);
end
hold off;

xlabel('Voxels');
ylabel('PC exclusion threshold');
title([basis, ': Cross-validation scores per voxel/threshold']);

% Save or display the figure as needed
% saveas(gcf, fullfile(outputdir, 'eigenspectrum_ncsnr.png'));

% Save or display the figure as needed
% saveas(gcf, fullfile(outputdir, 'cv_scores.png'));

%% Compare Initial Data, Denoised Data, Noise, and Denoising Matrix
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
title(['Basis: ', basis, newline, 'Denoised Data (Trial-Averaged)']);
xlabel('Voxels');
ylabel('Conditions');

subplot(1, 4, 3);
imagesc(mean(noise, 3)', [-4, 4]);
colormap(redblue(256));
colorbar;
xlabel('Conditions');
ylabel('Voxels');
title('Noise (Trial-Averaged)');

subplot(1, 4, 4);
imagesc(denoiser, [-0.3, 0.3]);
colormap(redblue(256));
colorbar;
xlabel('Voxels');
ylabel('Voxels');
title('Optimal Denoising Matrix');
% Save or display the figure as needed
% saveas(gcf, fullfile(outputdir, 'denoising_comparison.png'));

%% Changes in NCSNR and NC After Denoising
figure('Position', [100, 100, 1400, 600]);

% Scatter plot of initial vs. denoised NCSNR
subplot(1, 2, 1);
scatter(ncsnrs(:, 1), ncsnrs(:, end), 10, 'filled');
hold on;
plot([0, 1.5], [0, 1.5], 'r--');
hold off;
xlabel(['Initial Mean NCSNR:', newline, num2str(round(mean(ncsnrs(:,1)), 3))]);
ylabel(['Denoised Mean NCSNR:', newline, num2str(round(mean(ncsnrs(:,end)), 3))]);
if strcmp(opt.threshold_per, 'population')
    title([basis, ' Change in NCSNR', newline, 'Optimal PC Threshold = ', num2str(best_threshold)]);
else
    title([basis, ' Change in NCSNR', newline, 'Optimal PC Threshold = ', num2str(mean(best_threshold))]);
end

% Scatter plot of initial vs. denoised NC
subplot(1, 2, 2);
scatter(ncs(:, 1), ncs(:, end), 10, 'filled');
hold on;
plot([0, 100], [0, 100], 'r--');
hold off;
xlabel(['Initial Mean NC:',newline,num2str(round(mean(ncs(:, 1)), 3))]);
ylabel(['Denoised Mean NC:',newline, num2str(round(mean(ncs(:, end)), 3))]);
if strcmp(opt.threshold_per, 'population')
    title([basis, ' Change in NC', newline, 'Optimal PC Threshold = ', num2str(best_threshold)]);
else
    title([basis, ' Change in NC', newline, 'Optimal PC Threshold = ', num2str(mean(best_threshold))]);
end

% Save or display the figure as needed
% saveas(gcf, fullfile(outputdir, 'ncsnr_nc_changes.png'));

%% Compute Pearson Correlation Between Pairwise Distances
% Calculate pairwise correlation distances
data_pdist = pdist(mean(test_data, 3)', 'correlation');
noise_pdist = pdist(mean(noise, 3)', 'correlation');

% Compute Pearson correlation
[r, p] = corrcoef(data_pdist, noise_pdist);
fprintf('Pearson correlation: %.4f, p-value: %.4f\n', r(1,2), p(1,2));

%%

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
    blue_to_white = [linspace(0, 1, m)', linspace(0, 1, m)', ones(m, 1)];

    % Generate white to red
    white_to_red = [ones(m, 1), linspace(1, 0, m)', linspace(1, 0, m)'];

    % Combine the two
    cmap = [blue_to_white; white_to_red];
end
