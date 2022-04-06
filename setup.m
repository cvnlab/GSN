% This script adds GSN to the MATLAB path.

% Add GSN to the MATLAB path (in case the user has not already done so).
GSN_dir = fileparts(mfilename('fullfile'));

addpath(fullfile(GSN_dir, 'matlab'));
addpath(fullfile(GSN_dir, 'matlab', 'utilities'));

clear GSN_dir;
