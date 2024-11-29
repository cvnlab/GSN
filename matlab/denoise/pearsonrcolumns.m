

function pearson_r = pearsonrcolumns(mat1, mat2)
% PEARSON_R_COLUMNS Computes the Pearson correlation coefficient between
% corresponding columns of two matrices.
%
% <mat1> and <mat2> are n_rows x n_cols matrices, where each column
% represents a variable.
%
% Parameters:
%   <mat1> : array
%     An n_rows x n_cols matrix representing the first set of variables.
%   <mat2> : array
%     An n_rows x n_cols matrix representing the second set of variables.
%
% Return:
%   <pearson_r> as a 1 x n_cols array of Pearson correlation coefficients
%   between corresponding columns of mat1 and mat2.
%
% Notes:
%   - The function computes the Pearson correlation coefficient for each
%   pair of corresponding columns in mat1 and mat2. - A small constant
%   (1e-8) is added to the denominator to prevent division by zero.
%
% Example:
%   mat1 = randn(100, 50); mat2 = randn(100, 50); pearson_r =
%   pearson_r_columns(mat1, mat2);


    % Check if mat1 and mat2 have the same size
    if ~isequal(size(mat1), size(mat2))
        error('Input matrices must have the same shape.');
    end
    
    % Compute column-wise means
    mean1 = mean(mat1, 1); % 1 x n_cols
    mean2 = mean(mat2, 1); % 1 x n_cols
    
    % Subtract the means from each element (center the data)
    mat1_centered = mat1 - mean1; % n_rows x n_cols
    mat2_centered = mat2 - mean2; % n_rows x n_cols
    
    % Compute the numerator (covariance between corresponding columns)
    covariance = sum(mat1_centered .* mat2_centered, 1); % 1 x n_cols
    
    % Compute the denominator (standard deviations of corresponding
    % columns)
    std1 = sqrt(sum(mat1_centered .^ 2, 1)); % 1 x n_cols
    std2 = sqrt(sum(mat2_centered .^ 2, 1)); % 1 x n_cols
    
    % Compute Pearson correlation coefficient
    pearson_r = covariance ./ (std1 .* std2 + 1e-8); % 1 x n_cols

end


