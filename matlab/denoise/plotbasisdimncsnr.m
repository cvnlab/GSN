function plotbasisdimncsnr(data, eigvecs, basisname, threshold)
    % Function to plot signal and noise standard deviations (SD) and noise ceiling SNR (ncsnr)
    % for data projected into a given basis.

    % Parameters:
    % ----------
    % data: 3D array
    %       Input data with dimensions (nunits, nconds, ntrials).
    % eigvecs: 2D array
    %       Eigenvector matrix (nunits x nunits) representing the basis.
    % basisname: string
    %       Name of the basis (used for plot titles).
    % threshold: float (optional)
    %       Threshold indicating the optimal principal component (PC) dimension.

    % Initialize lists to store results for each basis dimension
    ncsnrs = zeros(1, size(data, 1));    % Noise ceiling signal-to-noise ratios
    sigvars = zeros(1, size(data, 1));  % Signal variances
    noisevars = zeros(1, size(data, 1)); % Noise variances

    % Compute ncsnr, signal variance, and noise variance for each basis dimension
    nunits = size(data, 1);
    for i = 1:nunits
        thiseigvec = eigvecs(:, i); % Select the i-th eigenvector
        projdata = zeros(size(data, 2), size(data, 3)); % Initialize projected data

        % Compute dot product for each condition and trial (manual broadcasting)
        for cond = 1:size(data, 2)
            for trial = 1:size(data, 3)
                projdata(cond, trial) = dot(data(:, cond, trial), thiseigvec);
            end
        end

        % Validate projdata dimensions
        if size(projdata, 1) ~= size(data, 2) || size(projdata, 2) ~= size(data, 3)
            error('projdata dimensions are incorrect. Check dot product logic.');
        end

        % Compute noise ceiling metrics using the provided function
        [noiseceiling, ncsnr, sigvar, noisevar] = computenoiseceiling(reshape(projdata, [1, size(projdata)]));

        % Validate outputs of computenoiseceiling
        if ~isscalar(ncsnr) || ~isscalar(sigvar) || ~isscalar(noisevar)
            error('computenoiseceiling outputs must be scalars. Check its implementation.');
        end

        % Store results
        ncsnrs(i) = ncsnr;
        sigvars(i) = sigvar;
        noisevars(i) = noisevar;
    end

    % Create a figure for the plots

    % Plot signal and noise standard deviations
    subplot(1, 3, 2);
    plot(sqrt(sigvars), 'LineWidth', 3, 'DisplayName', 'Signal SD'); hold on;
    plot(sqrt(noisevars), 'LineWidth', 3, 'DisplayName', 'Noise SD');
    xlabel('Dimension');
    ylabel('Standard Deviation');
    title(['Signal and Noise SD of Data Projected into ', basisname, ' Basis']);
    yline(0, 'k--', 'LineWidth', 0.4); % Zero line

    % Add threshold line if specified
    if ~isempty(threshold)
        xline(mean(threshold), 'g--', 'LineWidth', 2, ...
              'DisplayName', ['Optimal PC Threshold: ', num2str(mean(threshold),3)]);
    end
    ylim([-0.2, max(sqrt(noisevars)) + 1]);
    %legend();
    hold off;

    % Plot noise ceiling signal-to-noise ratio (ncsnr)
    subplot(1, 3, 3);
    plot(ncsnrs, 'LineWidth', 3, 'Color', 'm', 'DisplayName', 'ncsnr'); hold on;
    xlabel('Dimension');
    ylabel('NCSNR');
    title(['NCSNR of Data Projected into ', basisname, ' Basis']);
    yline(0, 'k--', 'LineWidth', 0.4); % Zero line

    % Add threshold line if specified
    if ~isempty(threshold)
        xline(mean(threshold), 'g--', 'LineWidth', 2, ...
              'DisplayName', ['Optimal PC Threshold: ', num2str(mean(threshold),3)]);
    end
    ylim([-0.05, max(ncsnrs) + 0.1]);
    %legend();
    hold off;
end
