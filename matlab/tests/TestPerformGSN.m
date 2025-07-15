classdef TestPerformGSN < matlab.unittest.TestCase
    % TestPerformGSN - Unit tests for performgsn.m function
    %
    % This test class verifies that the performgsn function:
    % - Runs without errors on basic synthetic data
    % - Returns expected output structure with correct fields
    % - Handles edge cases and different input configurations
    % - Works with uneven trials across conditions
    % - Produces reasonable covariance estimates
    
    methods(Test)
        
        function testBasicFunctionality(testCase)
            % Test basic functionality with simple synthetic data
            fprintf('Testing basic functionality...\n');
            
            % Generate simple test data: voxels x conditions x trials
            nvox = 20;
            ncond = 10;
            ntrial = 4;
            
            % Create data with signal + noise structure
            rng(42, 'twister'); % For reproducibility
            signal = 2 * randn(nvox, ncond);
            data = repmat(signal, [1, 1, ntrial]) + 0.5 * randn(nvox, ncond, ntrial);
            
            % Test basic call
            results = performgsn(data);
            
            % Verify output structure
            testCase.verifyTrue(isstruct(results), 'Output should be a struct');
            
            % Check required fields
            required_fields = {'mnN', 'cN', 'cNb', 'shrinklevelN', 'shrinklevelD', ...
                              'mnS', 'cS', 'cSb', 'ncsnr', 'numiters'};
            for i = 1:length(required_fields)
                testCase.verifyTrue(isfield(results, required_fields{i}), ...
                    sprintf('Missing field: %s', required_fields{i}));
            end
            
            % Check dimensions
            testCase.verifyEqual(size(results.mnN), [1, nvox], 'mnN dimensions incorrect');
            testCase.verifyEqual(size(results.mnS), [1, nvox], 'mnS dimensions incorrect');
            testCase.verifyEqual(size(results.cN), [nvox, nvox], 'cN dimensions incorrect');
            testCase.verifyEqual(size(results.cS), [nvox, nvox], 'cS dimensions incorrect');
            testCase.verifyEqual(size(results.cNb), [nvox, nvox], 'cNb dimensions incorrect');
            testCase.verifyEqual(size(results.cSb), [nvox, nvox], 'cSb dimensions incorrect');
            testCase.verifyEqual(size(results.ncsnr), [1, nvox], 'ncsnr dimensions incorrect');
            
            % Check that covariances are symmetric
            testCase.verifyEqual(results.cN, results.cN', 'RelTol', 1e-10, 'cN should be symmetric');
            testCase.verifyEqual(results.cS, results.cS', 'RelTol', 1e-10, 'cS should be symmetric');
            testCase.verifyEqual(results.cNb, results.cNb', 'RelTol', 1e-10, 'cNb should be symmetric');
            testCase.verifyEqual(results.cSb, results.cSb', 'RelTol', 1e-10, 'cSb should be symmetric');
            
            % Check that final covariances are positive semi-definite
            testCase.verifyGreaterThanOrEqual(min(eig(results.cNb)), -1e-10, 'cNb should be PSD');
            testCase.verifyGreaterThanOrEqual(min(eig(results.cSb)), -1e-10, 'cSb should be PSD');
            
            % Check that ncsnr values are non-negative
            testCase.verifyGreaterThanOrEqual(results.ncsnr, 0, 'ncsnr should be non-negative');
            
            % Check that numiters is non-negative integer
            testCase.verifyGreaterThanOrEqual(results.numiters, 0, 'numiters should be non-negative');
            testCase.verifyEqual(results.numiters, round(results.numiters), 'numiters should be integer');
            
            fprintf('Basic functionality test passed!\n');
        end
        
        function testWithOptions(testCase)
            % Test function with different option configurations
            fprintf('Testing with different options...\n');
            
            % Generate test data
            rng(123, 'twister');
            nvox = 15;
            ncond = 8;
            ntrial = 3;
            data = 2 * randn(nvox, ncond, ntrial) + 0.5 * randn(nvox, ncond, ntrial);
            
            % Test with verbose off
            opt1.wantverbose = 0;
            results1 = performgsn(data, opt1);
            testCase.verifyTrue(isstruct(results1), 'Should work with verbose off');
            
            % Test with shrinkage off
            opt2.wantshrinkage = 0;
            results2 = performgsn(data, opt2);
            testCase.verifyTrue(isstruct(results2), 'Should work with shrinkage off');
            
            % Test with both options
            opt3.wantverbose = 0;
            opt3.wantshrinkage = 0;
            results3 = performgsn(data, opt3);
            testCase.verifyTrue(isstruct(results3), 'Should work with both options set');
            
            fprintf('Options test passed!\n');
        end
        
        function testMinimalTrials(testCase)
            % Test with minimum number of trials (2)
            fprintf('Testing with minimal trials...\n');
            
            rng(456, 'twister');
            nvox = 10;
            ncond = 5;
            ntrial = 2; % Minimum required
            
            data = randn(nvox, ncond, ntrial);
            
            results = performgsn(data);
            testCase.verifyTrue(isstruct(results), 'Should work with 2 trials');
            
            fprintf('Minimal trials test passed!\n');
        end
        
        function testLargerData(testCase)
            % Test with larger dataset
            fprintf('Testing with larger dataset...\n');
            
            rng(789, 'twister');
            nvox = 50;
            ncond = 20;
            ntrial = 6;
            
            % Create structured data
            signal = 3 * randn(nvox, ncond);
            noise = 1 * randn(nvox, ncond, ntrial);
            data = repmat(signal, [1, 1, ntrial]) + noise;
            
            results = performgsn(data);
            testCase.verifyTrue(isstruct(results), 'Should work with larger data');
            
            % Verify reasonable SNR values
            testCase.verifyTrue(all(results.ncsnr >= 0), 'SNR should be non-negative');
            testCase.verifyTrue(any(results.ncsnr > 0), 'Should have some positive SNR');
            
            fprintf('Larger dataset test passed!\n');
        end
        
        function testUnevenTrials(testCase)
            % Test with uneven number of trials across conditions
            fprintf('Testing with uneven trials...\n');
            
            rng(101112, 'twister');
            nvox = 12;
            ncond = 6;
            max_trials = 5;
            
            % Create data with varying numbers of trials per condition
            data = nan(nvox, ncond, max_trials);
            
            for c = 1:ncond
                % Each condition has 2 to max_trials valid trials
                ntrials_this_cond = 2 + mod(c-1, max_trials-1);
                data(:, c, 1:ntrials_this_cond) = randn(nvox, ntrials_this_cond);
            end
            
            results = performgsn(data);
            testCase.verifyTrue(isstruct(results), 'Should work with uneven trials');
            
            % All standard checks should still pass
            testCase.verifyEqual(size(results.mnN), [1, nvox], 'mnN dimensions incorrect');
            testCase.verifyEqual(size(results.cN), [nvox, nvox], 'cN dimensions incorrect');
            testCase.verifyGreaterThanOrEqual(min(eig(results.cNb)), -1e-10, 'cNb should be PSD');
            testCase.verifyGreaterThanOrEqual(min(eig(results.cSb)), -1e-10, 'cSb should be PSD');
            
            fprintf('Uneven trials test passed!\n');
        end
        
        function testEmptyOptions(testCase)
            % Test with empty options struct
            fprintf('Testing with empty options...\n');
            
            rng(131415, 'twister');
            data = randn(8, 4, 3);
            
            % Test with empty struct
            results1 = performgsn(data, struct());
            testCase.verifyTrue(isstruct(results1), 'Should work with empty struct');
            
            % Test with no options argument
            results2 = performgsn(data);
            testCase.verifyTrue(isstruct(results2), 'Should work with no options');
            
            fprintf('Empty options test passed!\n');
        end
        
        function testConsistentResults(testCase)
            % Test that results are consistent with same random seed
            fprintf('Testing result consistency...\n');
            
            nvox = 10;
            ncond = 6;
            ntrial = 3;
            
            % Run twice with same random seed
            rng(161718, 'twister');
            data1 = randn(nvox, ncond, ntrial);
            results1 = performgsn(data1);
            
            rng(161718, 'twister');
            data2 = randn(nvox, ncond, ntrial);
            results2 = performgsn(data2);
            
            % Results should be identical
            testCase.verifyEqual(results1.mnN, results2.mnN, 'RelTol', 1e-12, 'mnN should be consistent');
            testCase.verifyEqual(results1.cN, results2.cN, 'RelTol', 1e-12, 'cN should be consistent');
            testCase.verifyEqual(results1.numiters, results2.numiters, 'numiters should be consistent');
            
            fprintf('Consistency test passed!\n');
        end
        
        function testErrorConditions(testCase)
            % Test various error conditions
            fprintf('Testing error conditions...\n');
            
            % Test with insufficient trials (should error)
            data_bad = randn(5, 3, 1); % Only 1 trial
            testCase.verifyError(@() performgsn(data_bad), 'MATLAB:assertion:failed', ...
                'Should error with only 1 trial');
            
            % Test with all NaN condition (should error)
            data_bad2 = randn(5, 3, 3);
            data_bad2(:, 1, :) = NaN; % First condition all NaN
            testCase.verifyError(@() performgsn(data_bad2), '', ...
                'Should error when condition has all NaN trials');
            
            fprintf('Error conditions test passed!\n');
        end
        
        function testCalcShrunkenCovarianceValidation(testCase)
            % Test validation assertions in calcshrunkencovariance.m
            fprintf('Testing calcshrunkencovariance validation assertions...\n');
            
            % Test 1: NaNs in 2D data should fail
            data_2d_nan = randn(20, 10);
            data_2d_nan(1, 1) = NaN;
            
            testCase.verifyError(@() calcshrunkencovariance(data_2d_nan), '', ...
                'Should error with NaNs in 2D data');
            
            % Test 2: All trials NaN for one condition should fail
            data_all_nan = randn(5, 10, 8);
            data_all_nan(:, :, 1) = NaN; % All trials for first condition
            
            testCase.verifyError(@() calcshrunkencovariance(data_all_nan), '', ...
                'Should error when all conditions must have at least 1 valid trial');
            
            % Test 3: Insufficient conditions with multiple trials should fail
            % Create a case that will pass basic validation but fail cross-validation
            data_insufficient = randn(4, 10, 5);  % 5 conditions with 4 observations each
            % Make most conditions have only 1 observation
            data_insufficient(2:end, :, 1) = NaN;  % Condition 1: only 1 observation
            data_insufficient(2:end, :, 2) = NaN;  % Condition 2: only 1 observation
            data_insufficient(2:end, :, 3) = NaN;  % Condition 3: only 1 observation
            data_insufficient(3:end, :, 4) = NaN;  % Condition 4: only 2 observations
            % Condition 5 keeps all 4 observations - so only 1 condition has 2+ observations
            
            testCase.verifyError(@() calcshrunkencovariance(data_insufficient), '', ...
                'Should error when insufficient conditions for cross-validation');
            
            % Test 4: Valid uneven data should work
            data_valid = randn(4, 10, 5);  % 5 conditions with up to 4 observations each
            data_valid(4, :, 1) = NaN;     % Condition 1: 3 observations
            data_valid(3:4, :, 2) = NaN;   % Condition 2: 2 observations
            % Conditions 3,4,5 have all 4 observations
            
            [mn, c, shrinklevel, nll] = calcshrunkencovariance(data_valid);
            testCase.verifyTrue(isstruct(struct('mn', mn, 'c', c, 'shrinklevel', shrinklevel, 'nll', nll)), ...
                'Valid uneven data should process successfully');
            testCase.verifyEqual(size(mn), [1, 10], 'Mean should have correct dimensions');
            testCase.verifyEqual(size(c), [10, 10], 'Covariance should have correct dimensions');
            testCase.verifyTrue(shrinklevel >= 0 && shrinklevel <= 1, 'Shrinkage level should be in [0,1]');
            testCase.verifyEqual(length(nll), 51, 'Should have 51 log-likelihood values by default');
            
            fprintf('CalcShrunkenCovariance validation test passed!\n');
        end
        
        function testNumericalStability(testCase)
            % Test numerical stability with extreme values
            fprintf('Testing numerical stability...\n');
            
            % Test with very small values
            rng(192021, 'twister');
            data_small = 1e-10 * randn(8, 5, 3);
            results_small = performgsn(data_small);
            testCase.verifyTrue(isstruct(results_small), 'Should handle small values');
            testCase.verifyTrue(all(isfinite(results_small.ncsnr)), 'SNR should be finite');
            
            % Test with larger values
            data_large = 1e3 * randn(8, 5, 3);
            results_large = performgsn(data_large);
            testCase.verifyTrue(isstruct(results_large), 'Should handle large values');
            testCase.verifyTrue(all(isfinite(results_large.ncsnr)), 'SNR should be finite');
            
            fprintf('Numerical stability test passed!\n');
        end
        
        function testUnevenTrialsComprehensive(testCase)
            % Comprehensive test of uneven trials functionality
            fprintf('Testing comprehensive uneven trials functionality...\n');
            
            rng(300301, 'twister');
            nvox = 15;
            ncond = 8;
            max_trials = 6;
            
            % Create data where each condition has a different number of trials
            data = nan(nvox, ncond, max_trials);
            trial_counts = [6, 5, 4, 3, 2, 6, 4, 3]; % Different for each condition
            
            for c = 1:ncond
                ntrials = trial_counts(c);
                % Generate signal + noise structure
                signal = 2 * randn(nvox, 1);
                noise = 0.5 * randn(nvox, ntrials);
                data(:, c, 1:ntrials) = repmat(signal, 1, ntrials) + noise;
            end
            
            results = performgsn(data);
            
            % Verify basic structure
            testCase.verifyTrue(isstruct(results), 'Should work with comprehensive uneven trials');
            testCase.verifyEqual(size(results.mnN), [1, nvox], 'mnN dimensions incorrect');
            testCase.verifyEqual(size(results.cN), [nvox, nvox], 'cN dimensions incorrect');
            testCase.verifyEqual(size(results.cSb), [nvox, nvox], 'cSb dimensions incorrect');
            
            % Check mathematical properties
            testCase.verifyEqual(results.cN, results.cN', 'RelTol', 1e-10, 'cN should be symmetric');
            testCase.verifyEqual(results.cSb, results.cSb', 'RelTol', 1e-10, 'cSb should be symmetric');
            testCase.verifyGreaterThanOrEqual(min(eig(results.cNb)), -1e-10, 'cNb should be PSD');
            testCase.verifyGreaterThanOrEqual(min(eig(results.cSb)), -1e-10, 'cSb should be PSD');
            
            % Verify reasonable SNR values
            testCase.verifyTrue(all(results.ncsnr >= 0), 'SNR should be non-negative');
            testCase.verifyTrue(any(results.ncsnr > 0), 'Should have some positive SNR');
            
            fprintf('Comprehensive uneven trials test passed!\n');
        end
        
        function testTrailingNaNEquivalence(testCase)
            % Test that appending NaN trials produces identical results to not having them
            fprintf('Testing trailing NaN equivalence...\n');
            
            rng(400401, 'twister');
            nvox = 12;
            ncond = 6;
            base_trials = 4;
            extra_nan_trials = 3;
            
            % Create base data without extra NaN trials
            data_base = nan(nvox, ncond, base_trials);
            for c = 1:ncond
                % Generate consistent signal + noise
                signal = 1.5 * randn(nvox, 1);
                noise = 0.8 * randn(nvox, base_trials);
                data_base(:, c, :) = repmat(signal, 1, base_trials) + noise;
            end
            
            % Create extended data with NaN trials appended to ALL conditions
            data_extended = nan(nvox, ncond, base_trials + extra_nan_trials);
            data_extended(:, :, 1:base_trials) = data_base;
            % The remaining trials are already NaN from initialization
            
            % Run GSN on both datasets
            results_base = performgsn(data_base);
            results_extended = performgsn(data_extended);
            
            % Results should be numerically identical
            tolerance = 1e-12;
            
            testCase.verifyEqual(results_base.mnN, results_extended.mnN, 'RelTol', tolerance, ...
                'mnN should be identical when trailing NaNs added');
            testCase.verifyEqual(results_base.mnS, results_extended.mnS, 'RelTol', tolerance, ...
                'mnS should be identical when trailing NaNs added');
            testCase.verifyEqual(results_base.cN, results_extended.cN, 'RelTol', tolerance, ...
                'cN should be identical when trailing NaNs added');
            testCase.verifyEqual(results_base.cS, results_extended.cS, 'RelTol', tolerance, ...
                'cS should be identical when trailing NaNs added');
            testCase.verifyEqual(results_base.cNb, results_extended.cNb, 'RelTol', tolerance, ...
                'cNb should be identical when trailing NaNs added');
            testCase.verifyEqual(results_base.cSb, results_extended.cSb, 'RelTol', tolerance, ...
                'cSb should be identical when trailing NaNs added');
            testCase.verifyEqual(results_base.ncsnr, results_extended.ncsnr, 'RelTol', tolerance, ...
                'ncsnr should be identical when trailing NaNs added');
            testCase.verifyEqual(results_base.shrinklevelN, results_extended.shrinklevelN, 'RelTol', tolerance, ...
                'shrinklevelN should be identical when trailing NaNs added');
            testCase.verifyEqual(results_base.shrinklevelD, results_extended.shrinklevelD, 'RelTol', tolerance, ...
                'shrinklevelD should be identical when trailing NaNs added');
            testCase.verifyEqual(results_base.numiters, results_extended.numiters, ...
                'numiters should be identical when trailing NaNs added');
            
            fprintf('Trailing NaN equivalence test passed!\n');
        end
        
        function testMixedNaNPatterns(testCase)
            % Test various patterns of NaN placement within trials
            fprintf('Testing mixed NaN patterns...\n');
            
            rng(500501, 'twister');
            nvox = 10;
            ncond = 5;
            max_trials = 5;
            
            % Create data with different NaN patterns
            data = nan(nvox, ncond, max_trials);
            
            % Condition 1: All trials valid
            data(:, 1, :) = randn(nvox, max_trials);
            
            % Condition 2: Only first 3 trials valid
            data(:, 2, 1:3) = randn(nvox, 3);
            
            % Condition 3: Only first 2 trials valid
            data(:, 3, 1:2) = randn(nvox, 2);
            
            % Condition 4: First 4 trials valid
            data(:, 4, 1:4) = randn(nvox, 4);
            
            % Condition 5: Only first 2 trials valid (minimum)
            data(:, 5, 1:2) = randn(nvox, 2);
            
            results = performgsn(data);
            
            % Verify the function handles mixed patterns correctly
            testCase.verifyTrue(isstruct(results), 'Should handle mixed NaN patterns');
            testCase.verifyEqual(size(results.mnN), [1, nvox], 'mnN dimensions correct with mixed patterns');
            testCase.verifyEqual(size(results.cN), [nvox, nvox], 'cN dimensions correct with mixed patterns');
            
            % Check mathematical properties are preserved
            testCase.verifyEqual(results.cN, results.cN', 'RelTol', 1e-10, 'cN should be symmetric with mixed patterns');
            testCase.verifyGreaterThanOrEqual(min(eig(results.cNb)), -1e-10, 'cNb should be PSD with mixed patterns');
            testCase.verifyGreaterThanOrEqual(min(eig(results.cSb)), -1e-10, 'cSb should be PSD with mixed patterns');
            
            fprintf('Mixed NaN patterns test passed!\n');
        end
        
        function testUnevenTrialsVsEqualSubset(testCase)
            % Test that uneven trials gives same result as equal subset when applicable
            fprintf('Testing uneven trials vs equal subset equivalence...\n');
            
            rng(600601, 'twister');
            nvox = 8;
            ncond = 4;
            full_trials = 6;
            subset_trials = 3;
            
            % Create full dataset
            data_full = randn(nvox, ncond, full_trials);
            
            % Create subset with only first subset_trials from each condition
            data_subset = data_full(:, :, 1:subset_trials);
            
            % Create uneven dataset that matches the subset (first subset_trials valid, rest NaN)
            data_uneven = nan(nvox, ncond, full_trials);
            data_uneven(:, :, 1:subset_trials) = data_full(:, :, 1:subset_trials);
            
            % Run GSN on both
            results_subset = performgsn(data_subset);
            results_uneven = performgsn(data_uneven);
            
            % Results should be identical
            tolerance = 1e-12;
            testCase.verifyEqual(results_subset.mnN, results_uneven.mnN, 'RelTol', tolerance, ...
                'mnN should match between subset and uneven versions');
            testCase.verifyEqual(results_subset.cN, results_uneven.cN, 'RelTol', tolerance, ...
                'cN should match between subset and uneven versions');
            testCase.verifyEqual(results_subset.cSb, results_uneven.cSb, 'RelTol', tolerance, ...
                'cSb should match between subset and uneven versions');
            testCase.verifyEqual(results_subset.ncsnr, results_uneven.ncsnr, 'RelTol', tolerance, ...
                'ncsnr should match between subset and uneven versions');
            
            fprintf('Uneven vs equal subset equivalence test passed!\n');
        end
        
        function testMinimalUnevenTrials(testCase)
            % Test edge case with minimal trials in uneven scenario
            fprintf('Testing minimal uneven trials...\n');
            
            rng(700701, 'twister');
            nvox = 6;
            ncond = 4;
            max_trials = 4;
            
            % Create scenario where some conditions have only 2 trials (minimum)
            data = nan(nvox, ncond, max_trials);
            trial_pattern = [2, 3, 2, 4]; % Some conditions have minimum trials
            
            for c = 1:ncond
                ntrials = trial_pattern(c);
                data(:, c, 1:ntrials) = randn(nvox, ntrials);
            end
            
            results = performgsn(data);
            
            % Should still work with minimal trials in some conditions
            testCase.verifyTrue(isstruct(results), 'Should work with minimal trials in uneven scenario');
            testCase.verifyEqual(size(results.ncsnr), [1, nvox], 'ncsnr should have correct dimensions');
            testCase.verifyTrue(all(results.ncsnr >= 0), 'SNR should be non-negative with minimal uneven trials');
            
            % Mathematical properties should hold
            testCase.verifyEqual(results.cN, results.cN', 'RelTol', 1e-10, 'cN should be symmetric with minimal uneven');
            testCase.verifyGreaterThanOrEqual(min(eig(results.cNb)), -1e-10, 'cNb should be PSD with minimal uneven');
            
            fprintf('Minimal uneven trials test passed!\n');
        end
        
        function testUnevenTrialsErrorConditions(testCase)
            % Test error conditions specific to uneven trials
            fprintf('Testing uneven trials error conditions...\n');
            
            rng(800801, 'twister');
            nvox = 6;
            ncond = 3;
            max_trials = 3;
            
            % Test case where one condition has insufficient trials (only 1)
            data_bad1 = nan(nvox, ncond, max_trials);
            data_bad1(:, 1, 1:2) = randn(nvox, 2); % Condition 1: 2 trials (OK)
            data_bad1(:, 2, 1) = randn(nvox, 1);   % Condition 2: 1 trial (BAD)
            data_bad1(:, 3, 1:3) = randn(nvox, 3); % Condition 3: 3 trials (OK)
            
            testCase.verifyError(@() performgsn(data_bad1), '', ...
                'Should error when any condition has insufficient trials');
            
            % Test case where one condition has all NaN trials
            data_bad2 = nan(nvox, ncond, max_trials);
            data_bad2(:, 1, 1:2) = randn(nvox, 2); % Condition 1: 2 trials (OK)
            % Condition 2: all NaN (BAD)
            data_bad2(:, 3, 1:3) = randn(nvox, 3); % Condition 3: 3 trials (OK)
            
            testCase.verifyError(@() performgsn(data_bad2), '', ...
                'Should error when any condition has all NaN trials');
            
            fprintf('Uneven trials error conditions test passed!\n');
        end
        
    end
    
    methods(TestMethodSetup)
        function setupTest(testCase)
            % Add the matlab directory to path if needed
            current_dir = fileparts(mfilename('fullpath'));
            matlab_dir = fileparts(current_dir);
            addpath(matlab_dir);
            addpath(fullfile(matlab_dir, 'utilities'));
        end
    end
    
end
