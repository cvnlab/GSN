function score = negativemsecolumns(dataA, dataB)
% NEGATIVE_MSE_COLUMNS Compute the negative mean squared error between
% corresponding columns of dataA and dataB.
%
% <dataA> and <dataB> are nconds x nunits matrices.
%
% Parameters:
%   <dataA> : array
%     A nconds x nunits matrix representing the first set of data.
%   <dataB> : array
%     A nconds x nunits matrix representing the second set of data.
%
% Return:
%   <score> as a 1 x nunits array of negative mean squared errors.
%
% Notes:
%   - The function computes the mean squared error (MSE) between each pair
%   of corresponding columns in dataA and dataB. - The MSE is then negated
%   to produce the score.
%
% Example:
%   dataA = randn(10, 100); dataB = randn(10, 100); score =
%   negative_mse_columns(dataA, dataB);
%
    % Check if dataA and dataB have the same size
    if ~isequal(size(dataA), size(dataB))
        error('dataA and dataB must have the same dimensions.');
    end
    
    % Compute mean squared error for each column
    mse = mean((dataA - dataB) .^ 2, 1); % 1 x nunits
    
    % Compute negative MSE
    score = -mse; % 1 x nunits

end