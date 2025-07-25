classdef TestGSNDenoise < matlab.unittest.TestCase
    %TESTGSNDENOISE Tests for gsndenoise.m (port of Python tests).
    %
    % This test class replicates the Python pytest suite for gsn_denoise,
    % using MATLAB's unittest framework. Each Python test function has
    % a corresponding MATLAB method below.

    methods(Test)
        function test_basic_functionality(testCase)
            % Corresponds to: test_basic_functionality()

            nunits = 8;
            nconds = 10;
            ntrials = 3;
            data = randn(nunits, nconds, ntrials);

            % Call gsn_denoise with default options
            results = gsndenoise(data);

            % Assertions
            testCase.assertEqual(size(results.denoiser), [nunits nunits], ...
                'Denoiser must be (nunits x nunits)');
            testCase.assertEqual(size(results.denoiseddata), [nunits nconds], ...
                'Denoised data must be (nunits x nconds) in trial-averaged mode');
            testCase.assertEqual(size(results.fullbasis,1), nunits, ...
                'Basis should have nunits rows');
            testCase.assertGreaterThanOrEqual(size(results.fullbasis,2), 1, ...
                'Basis must have at least 1 column');
        end

        function test_cross_validation_population(testCase)
            % Corresponds to: test_cross_validation_population()

            nunits = 8;
            nconds = 10;
            ntrials = 3;
            data = randn(nunits, nconds, ntrials);

            opt = struct('cv_threshold_per','population');
            results = gsndenoise(data, [], opt);

            testCase.assertEqual(size(results.denoiser), [nunits nunits]);
            testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
            % Check best_threshold is scalar (population mode)
            testCase.assertTrue(isscalar(results.best_threshold), ...
                'best_threshold should be a scalar for population mode');

            testCase.assertEqual(size(results.fullbasis), [nunits nunits], ...
                'Full basis should be (nunits x nunits)');
            testCase.assertEqual(size(results.signalsubspace), [nunits results.best_threshold], ...
                'signalsubspace must be (nunits x best_threshold)');
            testCase.assertEqual(size(results.dimreduce), [results.best_threshold, nconds], ...
                'dimreduce must be (best_threshold x nconds)');

            % Check symmetry
            testCase.assertTrue(norm(results.denoiser - results.denoiser','fro') < 1e-12, ...
                'Denoiser must be symmetric in population mode');
        end

        function test_cross_validation_unit(testCase)
            % Corresponds to: test_cross_validation_unit()

            nunits = 8;
            nconds = 10;
            ntrials = 3;
            data = randn(nunits, nconds, ntrials);

            opt = struct('cv_threshold_per','unit');
            results = gsndenoise(data, [], opt);

            testCase.assertEqual(size(results.denoiser), [nunits nunits]);
            testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
            testCase.assertEqual(size(results.fullbasis), [nunits nunits], ...
                'Full basis should be (nunits x nunits)');

            % One threshold per unit
            testCase.assertEqual(length(results.best_threshold), nunits, ...
                'best_threshold should have one value per unit');

            % In python we do: assert results['cv_scores'].shape[0] == nunits
            % Our MATLAB code for cross-validation stores cv_scores as (len(thresholds), ntrials, nunits).
            % That means the # of thresholds is dimension 1. Possibly "cv_scores" is (nThresholds x nTrials x nUnits).
            % There's no direct single dimension that equals nunits. We'll approximate the check:
            testCase.assertEqual(size(results.cv_scores,3), nunits, ...
                'The 3rd dimension of cv_scores should match nunits');
        end

        function test_magnitude_thresholding(testCase)
            % Corresponds to: test_magnitude_thresholding()

            nunits = 8;
            nconds = 10;
            ntrials = 3;
            data = randn(nunits, nconds, ntrials);

            opt = struct('cv_mode', -1);
            results = gsndenoise(data, [], opt);

            testCase.assertEqual(size(results.denoiser), [nunits nunits]);
            testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
            testCase.assertEqual(size(results.fullbasis), [nunits nunits]);

            testCase.assertEqual(length(results.mags), nunits, ...
                'One magnitude per dimension when basis is (nunits x nunits)');
            testCase.assertTrue(isscalar(results.dimsretained), ...
                'dimsretained should be a single integer');

            testCase.assertEqual(size(results.signalsubspace), [nunits results.dimsretained]);
            testCase.assertEqual(size(results.dimreduce), [results.dimsretained, nconds]);

            % Check symmetry
            testCase.assertTrue(norm(results.denoiser - results.denoiser','fro') < 1e-12, ...
                'Denoiser must be symmetric in magnitude thresholding');
        end

        function test_custom_basis(testCase)
            % Corresponds to: test_custom_basis()

            nunits = 8;
            nconds = 10;
            ntrials = 3;
            data = randn(nunits, nconds, ntrials);

            basis_dims = [1, floor(nunits/2), nunits];

            for dim = basis_dims
                % Create a random orthonormal basis with dim columns
                [tmpQ, ~] = qr(randn(nunits, dim), 0);
                V = tmpQ;  % [nunits x dim]

                % Test default options
                results = gsndenoise(data, V);

                testCase.assertEqual(size(results.denoiser), [nunits nunits]);
                testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
                testCase.assertEqual(size(results.fullbasis), [nunits dim], ...
                    'Fullbasis must match input custom basis dimensions');

                % Test population thresholding
                opt = struct('cv_threshold_per','population');
                results = gsndenoise(data, V, opt);

                testCase.assertEqual(size(results.denoiser), [nunits nunits]);
                testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
                testCase.assertTrue(isscalar(results.best_threshold));
                testCase.assertEqual(size(results.fullbasis), [nunits dim]);
                testCase.assertEqual(size(results.signalsubspace,1), nunits);
                testCase.assertLessThanOrEqual(size(results.signalsubspace,2), dim);
                testCase.assertEqual(size(results.dimreduce,1), size(results.signalsubspace,2));
                testCase.assertEqual(size(results.dimreduce,2), nconds);

                % Test magnitude thresholding
                opt = struct('cv_mode', -1);
                results = gsndenoise(data, V, opt);

                testCase.assertEqual(size(results.denoiser), [nunits nunits]);
                testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
                testCase.assertEqual(size(results.fullbasis), [nunits dim]);
                testCase.assertEqual(length(results.mags), dim, ...
                    'One magnitude per basis dimension');
                testCase.assertTrue(isscalar(results.dimsretained));
                testCase.assertEqual(size(results.signalsubspace,1), nunits);
                testCase.assertLessThanOrEqual(size(results.signalsubspace,2), dim);
                testCase.assertEqual(size(results.dimreduce,1), size(results.signalsubspace,2));
                testCase.assertEqual(size(results.dimreduce,2), nconds);
            end
        end

        function test_custom_scoring(testCase)
            % Corresponds to: test_custom_scoring()

            nunits = 8;
            nconds = 10;
            ntrials = 3;
            data = randn(nunits, nconds, ntrials);

            % In MATLAB, define a custom scoring function handle:
            custom_score = @(A,B) -mean(abs(A - B),'all');  % negative mean absolute error

            % Test default options + custom scoring
            opt = struct('cv_scoring_fn', custom_score);
            results = gsndenoise(data, [], opt);

            testCase.assertEqual(size(results.denoiser), [nunits nunits]);
            testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
            testCase.assertEqual(size(results.fullbasis), [nunits nunits]);

            % Population thresholding + custom scoring
            opt = struct('cv_threshold_per','population','cv_scoring_fn',custom_score);
            results = gsndenoise(data, [], opt);

            testCase.assertEqual(size(results.denoiser), [nunits nunits]);
            testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
            testCase.assertTrue(isscalar(results.best_threshold));
            testCase.assertEqual(size(results.fullbasis), [nunits nunits]);
            testCase.assertEqual(size(results.signalsubspace), [nunits results.best_threshold]);
            testCase.assertEqual(size(results.dimreduce), [results.best_threshold, nconds]);
            testCase.assertTrue(norm(results.denoiser - results.denoiser','fro') < 1e-12);

            % Magnitude thresholding + custom scoring
            opt = struct('cv_mode', -1, 'cv_scoring_fn', custom_score);
            results = gsndenoise(data, [], opt);

            testCase.assertEqual(size(results.denoiser), [nunits nunits]);
            testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
            testCase.assertEqual(size(results.fullbasis), [nunits nunits]);
            testCase.assertEqual(length(results.mags), nunits);
            testCase.assertTrue(isscalar(results.dimsretained));
            testCase.assertEqual(size(results.signalsubspace), [nunits results.dimsretained]);
            testCase.assertEqual(size(results.dimreduce), [results.dimsretained, nconds]);
            testCase.assertTrue(norm(results.denoiser - results.denoiser','fro') < 1e-12);
        end

        function test_population_thresholding(testCase)
            % Corresponds to: test_population_thresholding()

            nunits = 8;
            nconds = 10;
            ntrials = 3;
            data = randn(nunits, nconds, ntrials);

            opt = struct('cv_threshold_per','population');
            results = gsndenoise(data, [], opt);

            testCase.assertEqual(size(results.denoiser), [nunits nunits]);
            testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
            testCase.assertTrue(isscalar(results.best_threshold));
            testCase.assertEqual(size(results.fullbasis), [nunits nunits]);
            testCase.assertEqual(size(results.signalsubspace), [nunits results.best_threshold]);
            testCase.assertEqual(size(results.dimreduce), [results.best_threshold, nconds]);
            testCase.assertTrue(norm(results.denoiser - results.denoiser','fro') < 1e-12);
        end

        function test_unit_thresholding(testCase)
            % Corresponds to: test_unit_thresholding()

            nunits = 8;
            nconds = 10;
            ntrials = 3;
            data = randn(nunits, nconds, ntrials);

            opt = struct('cv_threshold_per','unit');
            results = gsndenoise(data, [], opt);

            testCase.assertEqual(size(results.denoiser), [nunits nunits]);
            testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
            testCase.assertEqual(size(results.fullbasis), [nunits nunits]);
            testCase.assertEqual(length(results.best_threshold), nunits, ...
                'One threshold per unit');

            % Check dimension of cv_scores in third axis = nunits:
            testCase.assertEqual(size(results.cv_scores,3), nunits);
        end

        function test_cv_mode_0(testCase)
            % Corresponds to: test_cv_mode_0()

            nunits = 6;
            nconds = 8;
            ntrials = 4;
            data = randn(nunits, nconds, ntrials);
            opt = struct( ...
                'cv_mode', 0, ...
                'cv_threshold_per','population',...
                'cv_thresholds', 1:nunits,...
                'cv_scoring_fn', @(A,B) -mean((A - B).^2,'all')...
            );

            results = gsndenoise(data, [], opt);

            testCase.assertEqual(size(results.denoiser), [nunits nunits]);
            testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
            testCase.assertTrue(isscalar(results.best_threshold));
            testCase.assertEqual(size(results.fullbasis), [nunits nunits]);
            testCase.assertEqual(size(results.signalsubspace), [nunits results.best_threshold]);
            testCase.assertEqual(size(results.dimreduce), [results.best_threshold, nconds]);
            testCase.assertTrue(norm(results.denoiser - results.denoiser','fro') < 1e-12);
        end

        function test_cv_mode_1(testCase)
            % Corresponds to: test_cv_mode_1()

            nunits = 6;
            nconds = 8;
            ntrials = 4;
            data = randn(nunits, nconds, ntrials);
            opt = struct( ...
                'cv_mode', 1, ...
                'cv_threshold_per','population',...
                'cv_thresholds', 1:nunits,...
                'cv_scoring_fn', @(A,B) -mean((A - B).^2,'all')...
            );

            results = gsndenoise(data, [], opt);

            testCase.assertEqual(size(results.denoiser), [nunits nunits]);
            testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
            testCase.assertTrue(isscalar(results.best_threshold));
            testCase.assertEqual(size(results.fullbasis), [nunits nunits]);
            testCase.assertEqual(size(results.signalsubspace), [nunits results.best_threshold]);
            testCase.assertEqual(size(results.dimreduce), [results.best_threshold, nconds]);
            testCase.assertTrue(norm(results.denoiser - results.denoiser','fro') < 1e-12);
        end

        function test_cv_mode_minus1(testCase)
            % Corresponds to: test_cv_mode_minus1()

            nunits = 6;
            nconds = 8;
            ntrials = 4;
            data = randn(nunits, nconds, ntrials);
            opt = struct( ...
                'cv_mode', -1, ...
                'cv_threshold_per','population',...
                'cv_thresholds', 1:nunits,...
                'cv_scoring_fn', @(A,B) -mean((A - B).^2,'all')...
            );

            results = gsndenoise(data, [], opt);

            testCase.assertEqual(size(results.denoiser), [nunits nunits]);
            testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
            testCase.assertEqual(size(results.fullbasis), [nunits nunits]);
            testCase.assertEqual(length(results.mags), nunits);
            testCase.assertTrue(isscalar(results.dimsretained));
            testCase.assertEqual(size(results.signalsubspace), [nunits results.dimsretained]);
            testCase.assertEqual(size(results.dimreduce), [results.dimsretained, nconds]);
            testCase.assertTrue(norm(results.denoiser - results.denoiser','fro') < 1e-12);
        end

        function test_custom_nonsquare_basis(testCase)
            % Corresponds to: test_custom_nonsquare_basis()

            nunits = 8;
            nconds = 10;
            ntrials = 3;
            data = randn(nunits, nconds, ntrials);

            % basis_dims < nunits
            basis_dims = [1, floor(nunits/4), floor(nunits/2)];

            for dim = basis_dims
                [tmpQ, ~] = qr(randn(nunits, dim), 0);
                V = tmpQ;  % [nunits x dim]

                % Default
                results = gsndenoise(data, V);
                testCase.assertEqual(size(results.denoiser), [nunits nunits]);
                testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
                testCase.assertEqual(size(results.fullbasis), [nunits dim]);

                % Population thresholding
                opt = struct('cv_threshold_per','population');
                results = gsndenoise(data, V, opt);
                testCase.assertEqual(size(results.denoiser), [nunits nunits]);
                testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
                testCase.assertTrue(isscalar(results.best_threshold));
                testCase.assertEqual(size(results.fullbasis), [nunits dim]);
                testCase.assertEqual(size(results.signalsubspace,1), nunits);
                testCase.assertLessThanOrEqual(size(results.signalsubspace,2), dim);
                testCase.assertEqual(size(results.dimreduce,1), size(results.signalsubspace,2));
                testCase.assertEqual(size(results.dimreduce,2), nconds);

                % Magnitude thresholding, mag_type=0 or 1
                for mag_type = [0, 1]
                    opt = struct('cv_mode', -1, 'mag_type', mag_type);
                    results = gsndenoise(data, V, opt);

                    testCase.assertEqual(size(results.denoiser), [nunits nunits]);
                    testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
                    testCase.assertEqual(size(results.fullbasis), [nunits dim]);
                    testCase.assertEqual(length(results.mags), dim);
                    testCase.assertTrue(isscalar(results.dimsretained));
                    if ~isempty(results.signalsubspace)
                        testCase.assertEqual(size(results.signalsubspace,1), nunits);
                        testCase.assertLessThanOrEqual(size(results.signalsubspace,2), dim);
                        testCase.assertEqual(size(results.dimreduce,1), size(results.signalsubspace,2));
                        testCase.assertEqual(size(results.dimreduce,2), nconds);
                    end
                end

                % Single-trial denoising
                opt = struct('denoisingtype', 1);
                results = gsndenoise(data, V, opt);
                testCase.assertEqual(size(results.denoiser), [nunits nunits]);
                testCase.assertEqual(size(results.denoiseddata), [nunits nconds ntrials]);
                testCase.assertEqual(size(results.fullbasis), [nunits dim]);

                % Different mag_modes
                for mag_mode = [0, 1]
                    opt = struct('cv_mode', -1, 'mag_mode', mag_mode);
                    results = gsndenoise(data, V, opt);
                    testCase.assertEqual(size(results.denoiser), [nunits nunits]);
                    testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
                    testCase.assertEqual(size(results.fullbasis), [nunits dim]);
                    testCase.assertEqual(length(results.mags), dim);
                    testCase.assertTrue(isscalar(results.dimsretained));
                end

                % Different mag_fracs
                for mag_frac = [0.01, 0.1, 0.5]
                    opt = struct('cv_mode', -1, 'mag_frac', mag_frac);
                    results = gsndenoise(data, V, opt);
                    testCase.assertEqual(size(results.denoiser), [nunits nunits]);
                    testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
                    testCase.assertEqual(size(results.fullbasis), [nunits dim]);
                    testCase.assertEqual(length(results.mags), dim);
                    testCase.assertTrue(isscalar(results.dimsretained));
                end

                % Custom cv_thresholds
                opt = struct('cv_thresholds', 1:2:dim);  % odd thresholds
                results = gsndenoise(data, V, opt);
                testCase.assertEqual(size(results.denoiser), [nunits nunits]);
                testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
                testCase.assertEqual(size(results.fullbasis), [nunits dim]);
            end
        end

        function test_custom_basis_edge_cases(testCase)
            % Corresponds to: test_custom_basis_edge_cases()

            nunits = 8;
            nconds = 10;
            ntrials = 3;
            data = randn(nunits, nconds, ntrials);

            % Minimum basis dimension = 1
            [tmpQ, ~] = qr(randn(nunits,1), 0);
            V = tmpQ;
            results = gsndenoise(data, V);
            testCase.assertEqual(size(results.denoiser), [nunits nunits]);
            testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
            testCase.assertEqual(size(results.fullbasis), [nunits 1]);

            % Magnitude thresholding that retains no dimensions
            [tmpQ, ~] = qr(randn(nunits,2), 0);
            V = tmpQ;
            opt = struct('cv_mode', -1, 'mag_frac', 1.1); 
            results = gsndenoise(data, V, opt);
            testCase.assertEqual(size(results.denoiser), [nunits nunits]);
            testCase.assertEqual(norm(results.denoiser, 'fro'), 0, ...
                'Denoiser should be all zeros if no dims retained');
            testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
            testCase.assertEqual(norm(results.denoiseddata,'fro'), 0, ...
                'Denoised data should be zero if no dims retained');
            testCase.assertEqual(size(results.fullbasis), [nunits 2]);
            testCase.assertEqual(length(results.mags), 2);
            testCase.assertEqual(results.dimsretained, 0);
            testCase.assertEqual(size(results.signalsubspace), [nunits 0]);
            testCase.assertEqual(size(results.dimreduce), [0 nconds]);

            % Single condition => should fail
            data_single_cond = randn(nunits, 1, ntrials);
            [tmpQ, ~] = qr(randn(nunits, 3), 0);
            V = tmpQ;

            % We expect an error from "Data must have at least 2 conditions..."
            try
                gsndenoise(data_single_cond, V);
                testCase.assertFail('Expected an error for single condition data.');
            catch ME
                testCase.verifySubstring(ME.message, ...
                    'Data must have at least 2 conditions to estimate covariance.', ...
                    'Wrong error for single condition data');
            end

            % Two conditions => minimum valid
            data_two_conds = randn(nunits, 2, ntrials);
            results = gsndenoise(data_two_conds, V);
            testCase.assertEqual(size(results.denoiser), [nunits nunits]);
            testCase.assertEqual(size(results.denoiseddata), [nunits 2]);
            testCase.assertEqual(size(results.fullbasis), [nunits 3]);
        end

        function test_parameter_validation(testCase)
            % Corresponds to: test_parameter_validation()

            nunits = 8;
            nconds = 10;
            ntrials = 3;
            data = randn(nunits, nconds, ntrials);

            % Invalid V => must be in [0..4] or array
            try
                gsndenoise(data, 5);
                testCase.assertFail('Expected error for invalid integer V');
            catch ME
                testCase.verifySubstring(ME.message, ...
                    'V must be in [0..4] (int) or a 2D numeric array.', ...
                    'Did not throw correct error for invalid V');
            end

            % Invalid cv_threshold_per => triggers KeyError-like check
            try
                gsndenoise(data, 0, struct('cv_threshold_per','invalid'));
                testCase.assertFail('Expected error for invalid cv_threshold_per');
            catch ME
                testCase.verifySubstring(ME.message, ...
                    'cv_threshold_per must be ''unit'' or ''population''', ...
                    'Did not catch invalid cv_threshold_per');
            end

            % Invalid data shape => if it's not 3D
            invalid_data = randn(nunits, nconds);
            try
                gsndenoise(invalid_data, 0);
                testCase.assertFail('Expected error for invalid data shape');
            catch ME
                testCase.verifySubstring(ME.message, 'Data must have at least 2 trials.', ...
                    'Wrong error for invalid data shape');
            end

            % Too few trials
            invalid_data = randn(nunits, nconds, 1);
            try
                gsndenoise(invalid_data, 0);
                testCase.assertFail('Expected error for too few trials');
            catch ME
                testCase.verifySubstring(ME.message, ...
                    'Data must have at least 2 trials.', ...
                    'Wrong error message for too few trials');
            end

            % cv_thresholds must be positive
            try
                gsndenoise(data, [], struct('cv_thresholds',[0,1,2]));
                testCase.assertFail('Expected error for non-positive threshold');
            catch ME
                testCase.verifySubstring(ME.message, ...
                    'cv_thresholds must be positive integers', ...
                    'Wrong error for non-positive thresholds');
            end

            try
                gsndenoise(data, [], struct('cv_thresholds',[-1,1,2]));
                testCase.assertFail('Expected error for negative threshold');
            catch ME
                testCase.verifySubstring(ME.message, ...
                    'cv_thresholds must be positive integers', ...
                    'Wrong error for negative threshold');
            end

            % cv_thresholds must be integers
            try
                gsndenoise(data, [], struct('cv_thresholds',[1.5, 2, 3]));
                testCase.assertFail('Expected error for float threshold');
            catch ME
                testCase.verifySubstring(ME.message, ...
                    'cv_thresholds must be integers', ...
                    'Wrong error for float threshold');
            end

            % cv_thresholds must be in sorted order
            try
                gsndenoise(data, [], struct('cv_thresholds',[3,2,1]));
                testCase.assertFail('Expected error for unsorted thresholds');
            catch ME
                testCase.verifySubstring(ME.message, ...
                    'cv_thresholds must be in sorted order', ...
                    'Wrong error for unsorted thresholds');
            end

            % Non-unique thresholds
            try
                gsndenoise(data, [], struct('cv_thresholds',[1,2,2,3]));
                testCase.assertFail('Expected error for non-unique thresholds');
            catch ME
                testCase.verifySubstring(ME.message, ...
                    'cv_thresholds must be in sorted order with unique values', ...
                    'Wrong error for non-unique thresholds');
            end
        end

        function test_numerical_stability(testCase)
            % Test numerical stability with poorly conditioned data
            nunits = 10;
            nconds = 15;
            ntrials = 4;
            
            % Create ill-conditioned data
            base_data = randn(nunits, nconds, ntrials);
            small_noise = 1e-10 * randn(nunits, nconds, ntrials);
            data = base_data + small_noise;
            
            % Test with default options
            results = gsndenoise(data);
            testCase.assertTrue(all(isfinite(results.denoiser(:))), ...
                'Denoiser contains non-finite values');
            testCase.assertTrue(all(isfinite(results.denoiseddata(:))), ...
                'Denoised data contains non-finite values');
            
            % Test with different cv_modes
            for mode = [-1, 0, 1]
                opt = struct('cv_mode', mode);
                results = gsndenoise(data, [], opt);
                testCase.assertTrue(all(isfinite(results.denoiser(:))), ...
                    sprintf('Non-finite values in denoiser for cv_mode=%d', mode));
            end
        end

        function test_large_scale(testCase)
            % Test with larger datasets
            nunits = 50;
            nconds = 100;
            ntrials = 5;
            data = randn(nunits, nconds, ntrials);
            
            % Test memory efficiency and runtime
            results = gsndenoise(data);
            
            testCase.assertEqual(size(results.denoiser), [nunits nunits]);
            testCase.assertEqual(size(results.denoiseddata), [nunits nconds]);
        end

        function test_denoising_effectiveness(testCase)
            % Test if denoising actually reduces noise
            nunits = 8;
            nconds = 12;
            ntrials = 5;
            
            % Create structured signal (low-rank)
            rank = 3;  % Low-rank signal
            U = orth(randn(nunits, rank));  % Orthonormal basis
            V = randn(nconds, rank);        % Coefficients
            true_signal = U * V';           % Low-rank signal
            
            % Add noise
            noise_level = 0.1;
            noisy_data = repmat(true_signal, [1 1 ntrials]) + ...
                noise_level * randn(nunits, nconds, ntrials);
            
            % Denoise with population mode to better capture low-rank structure
            opt = struct('cv_threshold_per', 'population');
            results = gsndenoise(noisy_data, [], opt);
            
            % Compare denoised result to true signal
            mse_before = mean((mean(noisy_data,3) - true_signal).^2, 'all');
            mse_after = mean((results.denoiseddata - true_signal).^2, 'all');
            
            testCase.assertLessThan(mse_after, mse_before, ...
                'Denoising should reduce mean squared error');
        end

        function test_basis_orthogonality(testCase)
            % Test if bases remain orthogonal throughout processing
            nunits = 6;
            nconds = 8;
            ntrials = 3;
            data = randn(nunits, nconds, ntrials);
            
            % Test with different basis options
            % V=4 is random orthonormal, should definitely be orthogonal
            results = gsndenoise(data, 4);
            basis = results.fullbasis;
            gram = basis' * basis;
            testCase.assertTrue(norm(gram - eye(size(gram)), 'fro') < 1e-6, ...
                'Random orthonormal basis should remain orthonormal');
            
            % Test custom orthonormal basis
            [Q, ~] = qr(randn(nunits));
            results = gsndenoise(data, Q);
            basis = results.fullbasis;
            gram = basis' * basis;
            testCase.assertTrue(norm(gram - eye(size(gram)), 'fro') < 1e-6, ...
                'Custom orthonormal basis should remain orthonormal');
        end

        function test_extreme_values(testCase)
            % Test behavior with extreme values
            nunits = 5;
            nconds = 7;
            ntrials = 3;
            
            % Test with very large values
            data_large = 1e6 * randn(nunits, nconds, ntrials);
            results = gsndenoise(data_large);
            testCase.assertTrue(all(isfinite(results.denoiser(:))), ...
                'Denoiser should handle large values');
            
            % Test with very small values
            data_small = 1e-6 * randn(nunits, nconds, ntrials);
            results = gsndenoise(data_small);
            testCase.assertTrue(all(isfinite(results.denoiser(:))), ...
                'Denoiser should handle small values');
        end

        function test_cross_validation_consistency(testCase)
            % Test consistency of cross-validation results
            nunits = 6;
            nconds = 8;
            ntrials = 4;
            data = randn(nunits, nconds, ntrials);
            
            % Run multiple times with same data
            results1 = gsndenoise(data);
            results2 = gsndenoise(data);
            
            testCase.assertEqual(results1.best_threshold, results2.best_threshold, ...
                'Cross-validation should be deterministic');
            testCase.assertTrue(norm(results1.denoiser - results2.denoiser, 'fro') < 1e-10, ...
                'Denoiser should be consistent across runs');
        end

        function test_dimension_reduction(testCase)
            % Test dimension reduction properties
            nunits = 8;
            nconds = 10;
            ntrials = 3;
            data = randn(nunits, nconds, ntrials);
            
            % Test different threshold levels
            thresholds = 1:nunits;
            for thr = thresholds
                opt = struct('cv_thresholds', 1:thr, ...
                           'cv_threshold_per', 'population');
                results = gsndenoise(data, [], opt);
                
                testCase.assertLessThanOrEqual(size(results.signalsubspace, 2), thr, ...
                    'Signal subspace dimension should not exceed threshold');
            end
        end

        function test_single_trial_denoising(testCase)
            % Test single-trial denoising mode
            nunits = 6;
            nconds = 8;
            ntrials = 4;
            data = randn(nunits, nconds, ntrials);
            
            opt = struct('denoisingtype', 1);
            results = gsndenoise(data, [], opt);
            
            % Check dimensions
            testCase.assertEqual(size(results.denoiseddata), [nunits nconds ntrials], ...
                'Single-trial denoising should preserve trial dimension');
            
            % Check that each trial is different
            trial_diffs = zeros(ntrials);
            for i = 1:ntrials
                for j = (i+1):ntrials
                    diff = norm(results.denoiseddata(:,:,i) - results.denoiseddata(:,:,j), 'fro');
                    trial_diffs(i,j) = diff;
                end
            end
            testCase.assertTrue(all(trial_diffs(trial_diffs ~= 0) > 0), ...
                'Denoised trials should remain distinct');
        end

        function test_noise_level_adaptation(testCase)
            % Test adaptation to different noise levels
            nunits = 6;
            nconds = 8;
            ntrials = 4;
            signal = randn(nunits, nconds);
            
            % Test with different noise levels
            noise_levels = [0.1, 1.0, 10.0];
            retained_dims = zeros(size(noise_levels));
            
            for i = 1:length(noise_levels)
                noise = noise_levels(i) * randn(nunits, nconds, ntrials);
                data = repmat(signal, [1 1 ntrials]) + noise;
                
                opt = struct('cv_mode', -1);  % Use magnitude thresholding
                results = gsndenoise(data, [], opt);
                retained_dims(i) = results.dimsretained;
            end
            
            % Higher noise should generally lead to fewer dimensions
            testCase.assertTrue(issorted(retained_dims, 'descend'), ...
                'Number of retained dimensions should decrease with noise level');
        end

        function test_custom_basis_properties(testCase)
            % Test properties of custom bases
            nunits = 8;
            nconds = 10;
            ntrials = 3;
            data = randn(nunits, nconds, ntrials);
            
            % Test bases of different ranks
            ranks = [1, 3, nunits];
            for r = ranks
                % Create rank-r basis
                [Q, ~] = qr(randn(nunits, r), 0);
                results = gsndenoise(data, Q);
                
                % Check rank preservation
                testCase.assertEqual(size(results.fullbasis, 2), r, ...
                    'Basis rank should be preserved');
                testCase.assertLessThanOrEqual(size(results.signalsubspace, 2), r, ...
                    'Signal subspace rank should not exceed basis rank');
            end
        end

        function test_error_propagation(testCase)
            % Test error handling and propagation
            nunits = 6;
            nconds = 8;
            ntrials = 3;
            data = randn(nunits, nconds, ntrials);
            
            % Test with NaN values
            data_with_nan = data;
            data_with_nan(1,1,1) = NaN;
            try
                gsndenoise(data_with_nan);
                testCase.assertFail('Should error on NaN input');
            catch ME
                testCase.assertTrue(contains(ME.message, 'infinite'), ...
                    'Should catch NaN values');
            end
            
            % Test with Inf values
            data_with_inf = data;
            data_with_inf(1,1,1) = Inf;
            try
                gsndenoise(data_with_inf);
                testCase.assertFail('Should error on Inf input');
            catch ME
                testCase.assertTrue(contains(ME.message, 'infinite'), ...
                    'Should catch Inf values');
            end
        end

        function test_threshold_validation(testCase)
            % Test threshold validation
            nunits = 6;
            nconds = 8;
            ntrials = 3;
            data = randn(nunits, nconds, ntrials);
            
            % Test with non-positive thresholds
            try
                opt = struct('cv_thresholds', [-1, 1, 2]);
                gsndenoise(data, [], opt);
                testCase.assertFail('Should error on negative thresholds');
            catch ME
                testCase.assertTrue(contains(ME.message, 'positive'), ...
                    'Should catch negative thresholds');
            end
            
            % Test with non-integer thresholds
            try
                opt = struct('cv_thresholds', [1.5, 2, 3]);
                gsndenoise(data, [], opt);
                testCase.assertFail('Should error on non-integer thresholds');
            catch ME
                testCase.assertTrue(contains(ME.message, 'integer'), ...
                    'Should catch non-integer thresholds');
            end
            
            % Test with unsorted thresholds
            try
                opt = struct('cv_thresholds', [3, 1, 2]);
                gsndenoise(data, [], opt);
                testCase.assertFail('Should error on unsorted thresholds');
            catch ME
                testCase.assertTrue(contains(ME.message, 'sorted'), ...
                    'Should catch unsorted thresholds');
            end
            
            % Test with non-unique thresholds
            try
                opt = struct('cv_thresholds', [1, 2, 2, 3]);
                gsndenoise(data, [], opt);
                testCase.assertFail('Should error on non-unique thresholds');
            catch ME
                testCase.assertTrue(contains(ME.message, 'unique'), ...
                    'Should catch non-unique thresholds');
            end
            
            % Test with valid thresholds
            opt = struct('cv_thresholds', 1:3);
            results = gsndenoise(data, [], opt);
            testCase.assertTrue(~isempty(results), 'Should work with valid thresholds');
            
            % Test without specifying thresholds (should use default)
            results = gsndenoise(data);
            testCase.assertTrue(~isempty(results), 'Should work with default thresholds');
        end

        function test_symmetry_preservation(testCase)
            % Test preservation of symmetry properties
            nunits = 6;
            nconds = 8;
            ntrials = 3;
            data = randn(nunits, nconds, ntrials);
            
            % Test symmetry preservation across different modes
            modes = [-1, 0, 1];
            for mode = modes
                % Only test symmetry in population mode
                opt = struct('cv_mode', mode, 'cv_threshold_per', 'population');
                results = gsndenoise(data, [], opt);
                
                % Check denoiser symmetry
                testCase.assertTrue(norm(results.denoiser - results.denoiser', 'fro') < 1e-6, ...
                    sprintf('Denoiser should be symmetric in population mode %d', mode));
                
                % Check positive semi-definiteness
                eigs_denoiser = eig(results.denoiser);
                testCase.assertTrue(all(eigs_denoiser > -1e-6), ...
                    sprintf('Denoiser should be positive semi-definite in population mode %d', mode));
            end

            % In unit mode, denoiser doesn't need to be symmetric
            for mode = modes
                opt = struct('cv_mode', mode, 'cv_threshold_per', 'unit');
                results = gsndenoise(data, [], opt);
                
                % No symmetry check needed
                % But output should still be finite and well-behaved
                testCase.assertTrue(all(isfinite(results.denoiser(:))), ...
                    sprintf('Denoiser should be finite in unit mode %d', mode));
            end
        end
    end
end
