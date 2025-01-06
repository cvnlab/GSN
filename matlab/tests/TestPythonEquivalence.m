classdef TestPythonEquivalence < matlab.unittest.TestCase
    % Test equivalence between Python and MATLAB implementations of GSN denoising
    
    properties (TestParameter)
        test_case = {'default', 'population_threshold', 'magnitude_threshold', 'custom_basis'}
    end
    
    methods (Test)
        function test_denoiser_equivalence(testCase, test_case)
            % Load Python results
            matfile = fullfile('test_data', ['test_case_' test_case '.mat']);
            py_data = load(matfile);
            
            % Extract data and options
            data = py_data.data;
            opt = struct();
            
            % Get options from Python results
            fields = fieldnames(py_data);
            for i = 1:length(fields)
                field = fields{i};
                if startsWith(field, 'opt_')
                    opt_name = field(5:end);  % Remove 'opt_' prefix
                    opt.(opt_name) = py_data.(field);
                end
            end
            
            % Get V parameter if it exists
            V = [];
            if isfield(py_data, 'V')
                V = py_data.V;
            end
            
            % Run MATLAB implementation
            results = gsndenoise(data, V, opt);
            
            % Compare results
            testCase.verifyEqual(size(results.denoiser), size(py_data.python_denoiser), ...
                'Denoiser matrices have different sizes');
            testCase.verifyEqual(size(results.denoiseddata), size(py_data.python_denoiseddata), ...
                'Denoised data have different sizes');
            
            % Compare numerical values with tolerance
            tol = 1e-6;
            
            % Compare denoiser matrices (allowing for sign flips)
            diff_denoiser = min(norm(results.denoiser - py_data.python_denoiser, 'fro'), ...
                              norm(results.denoiser + py_data.python_denoiser, 'fro'));
            testCase.verifyLessThan(diff_denoiser, tol, ...
                sprintf('Denoiser matrices differ by %g', diff_denoiser));
            
            % Compare denoised data (allowing for sign flips)
            diff_denoised = min(norm(results.denoiseddata - py_data.python_denoiseddata, 'fro'), ...
                              norm(results.denoiseddata + py_data.python_denoiseddata, 'fro'));
            testCase.verifyLessThan(diff_denoised, tol, ...
                sprintf('Denoised data differ by %g', diff_denoised));
            
            % Compare other fields if they exist
            if ~isempty(py_data.python_cv_scores) && ~isempty(results.cv_scores)
                % CV scores should be invariant to sign flips
                diff_scores = norm(results.cv_scores - py_data.python_cv_scores, 'fro');
                testCase.verifyLessThan(diff_scores, tol, ...
                    sprintf('CV scores differ by %g', diff_scores));
            end
            
            if ~isempty(py_data.python_best_threshold) && ~isempty(results.best_threshold)
                % Best thresholds should be identical
                matlab_thresh = double(results.best_threshold);
                python_thresh = double(py_data.python_best_threshold);
                diff_thresh = norm(matlab_thresh - python_thresh);
                testCase.verifyLessThan(diff_thresh, tol, ...
                    sprintf('Best thresholds differ by %g', diff_thresh));
            end
            
            if ~isempty(py_data.python_fullbasis) && ~isempty(results.fullbasis)
                % Compare basis vectors allowing for sign flips
                diff_basis = min(norm(results.fullbasis - py_data.python_fullbasis, 'fro'), ...
                               norm(results.fullbasis + py_data.python_fullbasis, 'fro'));
                testCase.verifyLessThan(diff_basis, tol, ...
                    sprintf('Full bases differ by %g', diff_basis));
            end
            
            if ~isempty(py_data.python_signalsubspace) && ~isempty(results.signalsubspace)
                % Compare signal subspaces allowing for sign flips
                diff_signal = min(norm(results.signalsubspace - py_data.python_signalsubspace, 'fro'), ...
                                norm(results.signalsubspace + py_data.python_signalsubspace, 'fro'));
                testCase.verifyLessThan(diff_signal, tol, ...
                    sprintf('Signal subspaces differ by %g', diff_signal));
            end
            
            if ~isempty(py_data.python_dimreduce) && ~isempty(results.dimreduce)
                % Compare dimension reductions allowing for sign flips
                diff_dimred = min(norm(results.dimreduce - py_data.python_dimreduce, 'fro'), ...
                                norm(results.dimreduce + py_data.python_dimreduce, 'fro'));
                testCase.verifyLessThan(diff_dimred, tol, ...
                    sprintf('Dimension reductions differ by %g', diff_dimred));
            end
            
            if ~isempty(py_data.python_mags) && ~isempty(results.mags)
                % Magnitudes should be invariant to sign flips
                diff_mags = norm(abs(double(results.mags)) - abs(double(py_data.python_mags)));
                testCase.verifyLessThan(diff_mags, tol, ...
                    sprintf('Magnitudes differ by %g', diff_mags));
            end
            
            if ~isempty(py_data.python_dimsretained) && ~isempty(results.dimsretained)
                % Dimensions retained should be identical
                diff_dims = abs(double(results.dimsretained) - double(py_data.python_dimsretained));
                testCase.verifyLessThan(diff_dims, tol, ...
                    sprintf('Dimensions retained differ by %g', diff_dims));
            end
        end
        
        function test_ground_truth_preservation(testCase, test_case)
            % Load Python results
            matfile = fullfile('test_data', ['test_case_' test_case '.mat']);
            py_data = load(matfile);
            
            % Verify ground truth data is preserved correctly
            testCase.verifyEqual(size(py_data.ground_truth_signal), ...
                [size(py_data.data, 2), size(py_data.data, 1)], ...
                'Ground truth signal has wrong dimensions');
            
            testCase.verifyEqual(size(py_data.ground_truth_signal_cov), ...
                [size(py_data.data, 1), size(py_data.data, 1)], ...
                'Ground truth signal covariance has wrong dimensions');
            
            testCase.verifyEqual(size(py_data.ground_truth_noise_cov), ...
                [size(py_data.data, 1), size(py_data.data, 1)], ...
                'Ground truth noise covariance has wrong dimensions');
            
            % Verify covariance matrices are symmetric
            diff_sig = norm(py_data.ground_truth_signal_cov - py_data.ground_truth_signal_cov', 'fro');
            testCase.verifyLessThan(diff_sig, 1e-6, ...
                'Ground truth signal covariance is not symmetric');
            
            diff_noise = norm(py_data.ground_truth_noise_cov - py_data.ground_truth_noise_cov', 'fro');
            testCase.verifyLessThan(diff_noise, 1e-6, ...
                'Ground truth noise covariance is not symmetric');
            
            % Verify covariance matrices are positive semi-definite
            eig_sig = eig(py_data.ground_truth_signal_cov);
            testCase.verifyGreaterThanOrEqual(min(eig_sig), -1e-6, ...
                'Ground truth signal covariance is not positive semi-definite');
            
            eig_noise = eig(py_data.ground_truth_noise_cov);
            testCase.verifyGreaterThanOrEqual(min(eig_noise), -1e-6, ...
                'Ground truth noise covariance is not positive semi-definite');
        end
    end
end 