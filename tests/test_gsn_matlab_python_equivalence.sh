#!/bin/bash

# test_gsn_matlab_python_equivalence.sh
#
# This script tests numeric equivalence between Python and MATLAB implementations
# of GSN (perform_gsn / performgsn) using simulated data.
#
# Strategy:
#   1. Generate test data once in Python (with deterministic seed) and save as
#      both .npy and .mat so Python and MATLAB pipelines see identical inputs.
#   2. Run perform_gsn (Python) and performgsn (MATLAB) on the same data.
#   3. Compare the output covariances/means and ncsnr field-by-field.
#
# Both implementations rely on `deterministic_randperm` (in gsn/utilities.py and
# matlab/utilities/deterministic_randperm.m) so the trial-shuffling step inside
# calc_shrunken_covariance produces identical permutations across platforms.
#
# Usage:
#   ./test_gsn_matlab_python_equivalence.sh [test_number] [--overwrite]
#
# Arguments:
#   test_number  - Optional. Run only a specific test (1-N), or "all" (default).
#   --overwrite  - Optional. If set, regenerate test data and re-run python /
#                  matlab pipelines for the selected tests even if cached
#                  artifacts already exist. Without this flag, existing
#                  test data and result files are reused (the comparison
#                  step always runs).
#
# Examples:
#   ./test_gsn_matlab_python_equivalence.sh                     # run all, reuse caches
#   ./test_gsn_matlab_python_equivalence.sh 3                   # run only test 3, reuse cache
#   ./test_gsn_matlab_python_equivalence.sh 3 --overwrite       # rerun test 3 from scratch
#   ./test_gsn_matlab_python_equivalence.sh all --overwrite     # rerun everything from scratch
#
# IMPORTANT: MATLAB is considered the ground truth implementation.
# If discrepancies are found, the Python implementation should be modified
# to match MATLAB, not the other way around.

set -e  # Exit on any error

# ---- Parse command-line arguments ----
TEST_TO_RUN="all"
OVERWRITE=0

for arg in "$@"; do
    case "$arg" in
        --overwrite|-f)
            OVERWRITE=1
            ;;
        all)
            TEST_TO_RUN="all"
            ;;
        ''|*[!0-9]*)
            echo "Error: Unrecognized argument '$arg'."
            echo "Usage: $0 [test_number|all] [--overwrite]"
            exit 1
            ;;
        *)
            TEST_TO_RUN="$arg"
            ;;
    esac
done

if [[ "$TEST_TO_RUN" != "all" ]] && ! [[ "$TEST_TO_RUN" =~ ^[1-9]$|^[1-9][0-9]$ ]]; then
    echo "Error: Invalid test number '$TEST_TO_RUN'. Must be a positive integer or 'all'"
    echo "Usage: $0 [test_number|all] [--overwrite]"
    exit 1
fi

# ==================== CONFIGURATION ====================
# MATLAB configuration
MATLAB_PATH="/Applications/MATLAB_R2022b.app/bin/matlab"

# Python configuration
PYTHON_CMD="python3"

# Tolerance / correlation thresholds for equivalence checking
TOLERANCE="1e-8"          # Absolute tolerance for scalar / small-value comparisons
MIN_CORRELATION="0.999999"  # Minimum correlation for array comparisons
# ========================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSN_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_DIR="$GSN_ROOT/gsn"
MATLAB_DIR="$GSN_ROOT/matlab"
TEST_DATA_DIR="$SCRIPT_DIR/gsn_equivalence_test_data"

mkdir -p "$TEST_DATA_DIR"

echo "=========================================="
echo "GSN Python-MATLAB Equivalence Test"
echo "=========================================="
echo "Test data directory: $TEST_DATA_DIR"
echo "Tolerance: $TOLERANCE"
echo "Min correlation: $MIN_CORRELATION"
if [ "$OVERWRITE" -eq 1 ]; then
    echo "Overwrite: YES (cached artifacts will be regenerated)"
else
    echo "Overwrite: NO  (cached artifacts will be reused if present)"
fi
echo ""

# Check if MATLAB is available
if [ ! -f "$MATLAB_PATH" ]; then
    echo "MATLAB not found at $MATLAB_PATH"
    if ! command -v matlab &> /dev/null; then
        echo "ERROR: MATLAB not found. Please ensure MATLAB is installed."
        exit 1
    else
        MATLAB_PATH="matlab"
        echo "Using system MATLAB from PATH"
    fi
fi

# Check if Python and required packages are available
if ! $PYTHON_CMD -c "import numpy, scipy, sys; sys.path.insert(0, '$GSN_ROOT'); from gsn.perform_gsn import perform_gsn" &> /dev/null; then
    echo "ERROR: Python or required packages not available."
    echo "Tried to use: $PYTHON_CMD"
    echo "Required packages: numpy, scipy"
    echo "Required: 'from gsn.perform_gsn import perform_gsn' must work from $GSN_ROOT"
    exit 1
fi

echo "Environment checks passed."
echo ""

# Function to check if a test should run based on command-line argument
should_run_test() {
    local test_num="$1"
    if [[ "$TEST_TO_RUN" == "all" ]] || [[ "$TEST_TO_RUN" == "$test_num" ]]; then
        return 0
    else
        return 1
    fi
}

# Function to generate test data and run GSN equivalence test.
#
# Args:
#   test_name      - identifier used for filenames
#   nvox           - number of voxels
#   ncond          - number of conditions
#   ntrial         - number of trials
#   wantshrinkage  - 1 to use shrinkage, 0 to force full estimation
#   uneven_frac    - fraction of (cond,trial) entries to mark NaN (0 = balanced)
#   low_rank_spec  - optional "<rank_signal>:<rank_noise>:<noise_scale>" string. When
#                    set, data is generated from a truly low-rank signal+noise model
#                    so the empirical sample covariance is rank-deficient at the
#                    population level. Used to stress the 1e-6*I ridge bug in the
#                    2D path of calc_shrunken_covariance.py.
run_gsn_equivalence_test() {
    local test_name="$1"
    local nvox="$2"
    local ncond="$3"
    local ntrial="$4"
    local wantshrinkage="$5"
    local uneven_frac="${6:-0}"
    local low_rank_spec="${7:-}"

    echo "=========================================="
    echo "Testing: $test_name"
    echo "Parameters: $nvox voxels, $ncond conditions, $ntrial trials"
    echo "wantshrinkage=$wantshrinkage, uneven_frac=$uneven_frac, low_rank_spec=${low_rank_spec:-<none>}"
    echo "=========================================="

    # ---- Generate data with Python (shared input for both pipelines) ----
    if [ -z "$low_rank_spec" ]; then
        cat > "$TEST_DATA_DIR/generate_${test_name}_data.py" << 'EOFPYTHON'
import numpy as np
import scipy.io
import sys
import os

sys.path.insert(0, 'GSN_ROOT_PLACEHOLDER')
from gsn.simulate_data import generate_data

np.random.seed(42)

print("Generating simulated data...")
print(f"Shape: NVOX_PLACEHOLDER voxels x NCOND_PLACEHOLDER conditions x NTRIAL_PLACEHOLDER trials")

train_data, _, _ = generate_data(
    nvox=NVOX_PLACEHOLDER,
    ncond=NCOND_PLACEHOLDER,
    ntrial=NTRIAL_PLACEHOLDER,
    signal_decay=1.0,
    noise_decay=1.0,
    noise_multiplier=1.0,
    align_alpha=0.5,
    align_k=min(5, NVOX_PLACEHOLDER // 2),
    random_seed=42,
)

# Optionally drop a fraction of (condition, trial) cells to NaN for uneven-trials test.
uneven_frac = UNEVEN_FRAC_PLACEHOLDER
if uneven_frac > 0:
    rng = np.random.RandomState(123)
    nvox_, ncond_, ntrial_ = train_data.shape
    n_drop = int(round(uneven_frac * ncond_ * ntrial_))
    # Pick (cond, trial) cells but ensure each condition keeps at least 2 trials valid.
    candidates = [(c, t) for c in range(ncond_) for t in range(ntrial_)]
    rng.shuffle(candidates)
    dropped_per_cond = np.zeros(ncond_, dtype=int)
    max_drop_per_cond = ntrial_ - 2  # keep >= 2 valid trials per condition
    actually_dropped = 0
    for (c, t) in candidates:
        if actually_dropped >= n_drop:
            break
        if dropped_per_cond[c] < max_drop_per_cond:
            train_data[:, c, t] = np.nan
            dropped_per_cond[c] += 1
            actually_dropped += 1
    print(f"Marked {actually_dropped} (cond,trial) cells as NaN")

print(f"Generated data shape: {train_data.shape}")
finite_mask = np.isfinite(train_data)
print(f"Data range (finite): [{np.min(train_data[finite_mask]):.6f}, {np.max(train_data[finite_mask]):.6f}]")
print(f"Data mean (finite):  {np.mean(train_data[finite_mask]):.6f}")
print(f"Data std  (finite):  {np.std(train_data[finite_mask]):.6f}")

np.save('TEST_DATA_DIR_PLACEHOLDER/TEST_NAME_PLACEHOLDER_data.npy', train_data)
scipy.io.savemat('TEST_DATA_DIR_PLACEHOLDER/TEST_NAME_PLACEHOLDER_data.mat', {'data': train_data})

print("Test data generated and saved successfully")
EOFPYTHON
    else
        # Truly low-rank signal+noise generator. The population covariance has
        # rank (rank_signal + rank_noise) < nvox, so the sample cov is also rank-
        # deficient — and crucially, validation conditions live in the SAME
        # low-rank subspace as training, which makes the 1e-6*I ridge mask the
        # singularity at alpha=1 with a spuriously very-negative NLL (log-det
        # dominates because the quadratic form along ridge-padded null directions
        # is essentially zero).
        cat > "$TEST_DATA_DIR/generate_${test_name}_data.py" << 'EOFPYTHON'
import numpy as np
import scipy.io

rank_signal, rank_noise, noise_scale = LOW_RANK_SPEC_TUPLE_PLACEHOLDER

nvox     = NVOX_PLACEHOLDER
ncond    = NCOND_PLACEHOLDER
ntrial   = NTRIAL_PLACEHOLDER

assert rank_signal + rank_noise < nvox, \
    f"low-rank generator requires rank_signal+rank_noise < nvox " \
    f"(got {rank_signal}+{rank_noise} >= {nvox})"

rng = np.random.RandomState(42)
print(f"Generating low-rank data: nvox={nvox} ncond={ncond} ntrial={ntrial} "
      f"rank_signal={rank_signal} rank_noise={rank_noise} noise_scale={noise_scale}")

U_s, _ = np.linalg.qr(rng.randn(nvox, rank_signal))
U_n, _ = np.linalg.qr(rng.randn(nvox, rank_noise))
sig_s  = np.diag(np.linspace(1.0, 0.2, rank_signal))
sig_n  = np.diag(noise_scale * np.linspace(1.0, 0.2, rank_noise))

z_cond = rng.randn(rank_signal, ncond)
signal = U_s @ sig_s @ z_cond           # (nvox, ncond)

train_data = np.empty((nvox, ncond, ntrial), dtype=float)
for t in range(ntrial):
    z_noise = rng.randn(rank_noise, ncond)
    train_data[:, :, t] = signal + U_n @ sig_n @ z_noise

print(f"Generated data shape: {train_data.shape}")
print(f"Data range: [{train_data.min():.6f}, {train_data.max():.6f}]")
print(f"Data mean:  {train_data.mean():.6f}")
print(f"Data std:   {train_data.std():.6f}")

np.save('TEST_DATA_DIR_PLACEHOLDER/TEST_NAME_PLACEHOLDER_data.npy', train_data)
scipy.io.savemat('TEST_DATA_DIR_PLACEHOLDER/TEST_NAME_PLACEHOLDER_data.mat', {'data': train_data})

print("Test data generated and saved successfully")
EOFPYTHON
    fi

    # Convert low_rank_spec "r_s:r_n:noise_scale" to a Python tuple literal.
    local low_rank_tuple
    if [ -n "$low_rank_spec" ]; then
        IFS=':' read -r rs rn ns <<< "$low_rank_spec"
        low_rank_tuple="(${rs}, ${rn}, ${ns})"
    else
        low_rank_tuple="(0, 0, 0.0)"
    fi

    sed "s|GSN_ROOT_PLACEHOLDER|$GSN_ROOT|g; \
         s|TEST_DATA_DIR_PLACEHOLDER|$TEST_DATA_DIR|g; \
         s|TEST_NAME_PLACEHOLDER|$test_name|g; \
         s|NVOX_PLACEHOLDER|$nvox|g; \
         s|NCOND_PLACEHOLDER|$ncond|g; \
         s|NTRIAL_PLACEHOLDER|$ntrial|g; \
         s|UNEVEN_FRAC_PLACEHOLDER|$uneven_frac|g; \
         s|LOW_RANK_SPEC_TUPLE_PLACEHOLDER|$low_rank_tuple|g" \
        "$TEST_DATA_DIR/generate_${test_name}_data.py" > "$TEST_DATA_DIR/generate_${test_name}_data_tmp.py"
    mv "$TEST_DATA_DIR/generate_${test_name}_data_tmp.py" "$TEST_DATA_DIR/generate_${test_name}_data.py"

    local data_npy="$TEST_DATA_DIR/${test_name}_data.npy"
    local data_mat="$TEST_DATA_DIR/${test_name}_data.mat"
    if [ "$OVERWRITE" -eq 0 ] && [ -f "$data_npy" ] && [ -f "$data_mat" ]; then
        echo "[cache] Reusing existing test data: $data_npy / $data_mat"
    else
        $PYTHON_CMD "$TEST_DATA_DIR/generate_${test_name}_data.py"
    fi

    # ---- Run Python perform_gsn ----
    cat > "$TEST_DATA_DIR/run_python_${test_name}.py" << 'EOFPYTHON'
import numpy as np
import scipy.io
import sys
import os
import time

sys.path.insert(0, 'GSN_ROOT_PLACEHOLDER')
from gsn.perform_gsn import perform_gsn

np.random.seed(42)

data = np.load('TEST_DATA_DIR_PLACEHOLDER/TEST_NAME_PLACEHOLDER_data.npy')
print(f"Running Python perform_gsn on TEST_NAME_PLACEHOLDER data...")
print(f"Data shape: {data.shape}")

opt = {
    'wantshrinkage': bool(WANTSHRINKAGE_PLACEHOLDER),
    'wantverbose': False,
}

start_time = time.time()
results = perform_gsn(data, opt)
elapsed = time.time() - start_time
print(f"Python computation completed in {elapsed:.2f} seconds")

# Save results (only numerical fields)
matlab_results = {}
for key, value in results.items():
    if value is None:
        continue
    if isinstance(value, np.ndarray):
        matlab_results[key] = value
    elif isinstance(value, (int, float, np.integer, np.floating, bool, np.bool_)):
        matlab_results[key] = np.array([[float(value)]])
    elif isinstance(value, dict):
        print(f"Skipping dict field '{key}' for MATLAB-compatible save")
        continue
    else:
        try:
            matlab_results[key] = np.array(value)
        except Exception:
            print(f"Could not convert field '{key}' (type={type(value).__name__})")

scipy.io.savemat('TEST_DATA_DIR_PLACEHOLDER/TEST_NAME_PLACEHOLDER_python_results.mat', matlab_results)

print("Results summary:")
print(f"  Available fields: {sorted(matlab_results.keys())}")
for k in ['cN', 'cNb', 'cS', 'cSb', 'mnN', 'mnS', 'ncsnr', 'numiters',
          'shrinklevelN', 'shrinklevelD']:
    if k in matlab_results:
        v = matlab_results[k]
        if v.size <= 4:
            print(f"  {k}: shape={v.shape}, values={v.flatten().tolist()}")
        else:
            print(f"  {k}: shape={v.shape}, range=[{np.min(v):.6g}, {np.max(v):.6g}]")
print("Python results saved successfully")
EOFPYTHON

    sed "s|GSN_ROOT_PLACEHOLDER|$GSN_ROOT|g; \
         s|TEST_DATA_DIR_PLACEHOLDER|$TEST_DATA_DIR|g; \
         s|TEST_NAME_PLACEHOLDER|$test_name|g; \
         s|WANTSHRINKAGE_PLACEHOLDER|$wantshrinkage|g" \
        "$TEST_DATA_DIR/run_python_${test_name}.py" > "$TEST_DATA_DIR/run_python_${test_name}_tmp.py"
    mv "$TEST_DATA_DIR/run_python_${test_name}_tmp.py" "$TEST_DATA_DIR/run_python_${test_name}.py"

    local python_results="$TEST_DATA_DIR/${test_name}_python_results.mat"
    if [ "$OVERWRITE" -eq 0 ] && [ -f "$python_results" ]; then
        echo "[cache] Reusing existing Python results: $python_results"
    else
        $PYTHON_CMD "$TEST_DATA_DIR/run_python_${test_name}.py"
    fi

    # ---- Run MATLAB performgsn ----
    cat > "$TEST_DATA_DIR/run_matlab_${test_name}.m" << 'EOFMATLAB'
try
    addpath(genpath('MATLAB_DIR_PLACEHOLDER'));

    rng('default');
    rng(42, 'twister');

    loaded = load('TEST_DATA_DIR_PLACEHOLDER/TEST_NAME_PLACEHOLDER_data.mat');
    data = loaded.data;
    fprintf('Running MATLAB performgsn on TEST_NAME_PLACEHOLDER data...\n');
    fprintf('Data shape: [%s]\n', num2str(size(data)));

    opt = struct();
    opt.wantshrinkage = WANTSHRINKAGE_PLACEHOLDER;
    opt.wantverbose = 0;

    tic;
    results = performgsn(data, opt);
    elapsed = toc;
    fprintf('MATLAB computation completed in %.2f seconds\n', elapsed);

    save('TEST_DATA_DIR_PLACEHOLDER/TEST_NAME_PLACEHOLDER_matlab_results.mat', '-struct', 'results');

    fprintf('Results summary:\n');
    fns = fieldnames(results);
    for i = 1:length(fns)
        v = results.(fns{i});
        if isnumeric(v) && ~isempty(v)
            if numel(v) <= 4
                fprintf('  %s: shape=[%s], values=%s\n', fns{i}, num2str(size(v)), mat2str(v(:)'));
            else
                fprintf('  %s: shape=[%s], range=[%g, %g]\n', fns{i}, num2str(size(v)), min(v(:)), max(v(:)));
            end
        end
    end
    fprintf('MATLAB results saved successfully\n');
    exit(0);
catch ME
    fprintf('Error in MATLAB performgsn:\n');
    fprintf('Message: %s\n', ME.message);
    fprintf('Identifier: %s\n', ME.identifier);
    for i = 1:length(ME.stack)
        fprintf('  File: %s, Function: %s, Line: %d\n', ...
                ME.stack(i).file, ME.stack(i).name, ME.stack(i).line);
    end
    exit(1);
end
EOFMATLAB

    sed "s|MATLAB_DIR_PLACEHOLDER|$MATLAB_DIR|g; \
         s|TEST_DATA_DIR_PLACEHOLDER|$TEST_DATA_DIR|g; \
         s|TEST_NAME_PLACEHOLDER|$test_name|g; \
         s|WANTSHRINKAGE_PLACEHOLDER|$wantshrinkage|g" \
        "$TEST_DATA_DIR/run_matlab_${test_name}.m" > "$TEST_DATA_DIR/run_matlab_${test_name}_tmp.m"
    mv "$TEST_DATA_DIR/run_matlab_${test_name}_tmp.m" "$TEST_DATA_DIR/run_matlab_${test_name}.m"

    local matlab_results="$TEST_DATA_DIR/${test_name}_matlab_results.mat"
    if [ "$OVERWRITE" -eq 0 ] && [ -f "$matlab_results" ]; then
        echo "[cache] Reusing existing MATLAB results: $matlab_results"
    else
        "$MATLAB_PATH" -nosplash -nodesktop -r "try; run('$TEST_DATA_DIR/run_matlab_${test_name}.m'); catch ME; disp('Error running MATLAB:'); disp(ME.message); disp(ME.stack); exit(1); end"
    fi

    # ---- Compare results ----
    cat > "$TEST_DATA_DIR/compare_${test_name}.py" << 'EOFPYTHON'
import numpy as np
import scipy.io
import sys

py = scipy.io.loadmat('TEST_DATA_DIR_PLACEHOLDER/TEST_NAME_PLACEHOLDER_python_results.mat')
mat = scipy.io.loadmat('TEST_DATA_DIR_PLACEHOLDER/TEST_NAME_PLACEHOLDER_matlab_results.mat')

for k in ['__header__', '__version__', '__globals__']:
    py.pop(k, None)
    mat.pop(k, None)

print(f"Python result fields: {sorted(py.keys())}")
print(f"MATLAB result fields: {sorted(mat.keys())}")
print("")

tolerance = TOLERANCE_PLACEHOLDER
min_correlation = MIN_CORRELATION_PLACEHOLDER

# Core perform_gsn outputs we want to compare.
fields_to_compare = [
    'mnN', 'cN', 'cNb', 'shrinklevelN',
    'mnS', 'cS', 'cSb', 'shrinklevelD',
    'ncsnr', 'numiters',
]

print("=" * 60)
print(f"COMPARING PYTHON vs MATLAB RESULTS FOR TEST_NAME_PLACEHOLDER")
print("=" * 60)
print(f"Tolerance: {tolerance}")
print(f"Minimum correlation: {min_correlation}")
print("")

max_error = 0.0
failed_fields = []
correlation_failed_fields = []

for field in fields_to_compare:
    if field not in py or py[field] is None or field not in mat:
        missing_py = field not in py or py[field] is None
        print(f"SKIP: {field} - Missing in {'Python' if missing_py else 'MATLAB'}")
        continue

    py_val = py[field]
    mat_val = mat[field]

    if hasattr(py_val, 'squeeze'):
        py_val = py_val.squeeze()
    if hasattr(mat_val, 'squeeze'):
        mat_val = mat_val.squeeze()

    py_val = np.atleast_1d(np.asarray(py_val, dtype=float))
    mat_val = np.atleast_1d(np.asarray(mat_val, dtype=float))

    if py_val.shape != mat_val.shape:
        if py_val.ndim == mat_val.ndim and py_val.shape == mat_val.T.shape:
            mat_val = mat_val.T
            print(f"NOTE: {field} - Transposed MATLAB result to match Python shape")
        else:
            print(f"FAIL: {field} - Shape mismatch: Python {py_val.shape} vs MATLAB {mat_val.shape}")
            failed_fields.append(field)
            continue

    diff = py_val - mat_val
    if diff.size == 0:
        print(f"PASS: {field} - Both arrays empty with shape {py_val.shape}")
        continue

    abs_error = float(np.max(np.abs(diff)))
    rel_error = float(np.max(np.abs(diff) / (np.abs(py_val) + 1e-15)))
    max_error = max(max_error, max(abs_error, rel_error))

    py_flat = py_val.flatten()
    mat_flat = mat_val.flatten()
    valid = np.isfinite(py_flat) & np.isfinite(mat_flat)
    if np.sum(valid) > 1:
        if np.allclose(py_flat[valid], mat_flat[valid], atol=tolerance):
            correlation = 1.0
        else:
            try:
                correlation = float(np.corrcoef(py_flat[valid], mat_flat[valid])[0, 1])
                if np.isnan(correlation):
                    correlation = 1.0 if np.allclose(py_flat[valid], mat_flat[valid], atol=tolerance) else 0.0
            except Exception:
                correlation = 0.0
    else:
        correlation = 1.0 if np.allclose(py_flat[valid], mat_flat[valid], atol=tolerance) else 0.0

    if py_val.size == 1:
        if abs_error <= tolerance:
            print(f"PASS: {field} - Scalar value Python={py_val.item():.10f} MATLAB={mat_val.item():.10f} (abs diff: {abs_error:.2e})")
        else:
            print(f"FAIL: {field} - Scalar Python={py_val.item():.10f} MATLAB={mat_val.item():.10f} (abs diff: {abs_error:.2e})")
            failed_fields.append(field)
    else:
        if correlation >= min_correlation:
            print(f"PASS: {field} - Max abs err: {abs_error:.2e}, Max rel err: {rel_error:.2e}, Corr: {correlation:.10f}")
        else:
            print(f"FAIL: {field} - Max abs err: {abs_error:.2e}, Max rel err: {rel_error:.2e}, Corr: {correlation:.10f} (CORRELATION)")
            correlation_failed_fields.append(field)
            failed_fields.append(field)

print("")
print("=" * 60)
if len(failed_fields) == 0:
    print(f"ALL TESTS PASSED FOR TEST_NAME_PLACEHOLDER")
    print(f"  All fields passed correlation/tolerance test")
    print("=" * 60)
    sys.exit(0)
else:
    print(f"{len(failed_fields)} TESTS FAILED FOR TEST_NAME_PLACEHOLDER")
    print(f"  Failed fields: {failed_fields}")
    if correlation_failed_fields:
        print(f"  Fields failing correlation: {correlation_failed_fields}")
    print(f"  Maximum error encountered: {max_error:.6e}")
    print("")
    print("IMPORTANT: MATLAB is the ground truth implementation.")
    print("Failures indicate the Python implementation needs to be modified")
    print("to match MATLAB behavior, not vice versa.")
    print("=" * 60)
    sys.exit(1)
EOFPYTHON

    sed "s|TEST_DATA_DIR_PLACEHOLDER|$TEST_DATA_DIR|g; \
         s|TEST_NAME_PLACEHOLDER|$test_name|g; \
         s|TOLERANCE_PLACEHOLDER|$TOLERANCE|g; \
         s|MIN_CORRELATION_PLACEHOLDER|$MIN_CORRELATION|g" \
        "$TEST_DATA_DIR/compare_${test_name}.py" > "$TEST_DATA_DIR/compare_${test_name}_tmp.py"
    mv "$TEST_DATA_DIR/compare_${test_name}_tmp.py" "$TEST_DATA_DIR/compare_${test_name}.py"

    $PYTHON_CMD "$TEST_DATA_DIR/compare_${test_name}.py"
}

# ==================== TEST CONFIGURATIONS ====================
echo "Starting GSN equivalence tests..."
echo ""

# Each entry: "test_name|nvox|ncond|ntrial|wantshrinkage|uneven_frac|short_label"
#
# Test 9 (nvox=15, ncond=8, ntrial=3) is intentionally chosen to expose a
# suspected bug in the Python port. The 3D path of calc_shrunken_covariance
# does NOT add the +1e-6*I rank fix that the 2D path uses; with nvox > ntrial-1
# and few conditions, the averaged per-condition covariance is rank-deficient,
# so high-shrinkage levels produce a singular shrunken covariance and yield
# nll = NaN at those levels. The Python port then calls np.argmin(nll) (line
# 246 of gsn/calc_shrunken_covariance.py) which treats NaN as the minimum and
# returns its index, while the MATLAB port (line 223 of calcshrunkencovariance.m)
# uses min(nll) which silently ignores NaN. Result: Python picks shrinklevelN=1.0,
# MATLAB picks the true min (e.g. ~0.08). This test should FAIL on the unfixed
# Python code; the fix is np.argmin -> np.nanargmin.
TEST_DEFS=(
    "test1_small_shrinkage|10|30|4|1|0|Small/shrinkage"
    "test2_small_noshrinkage|10|30|4|0|0|Small/no-shrinkage"
    "test3_medium_shrinkage|25|60|5|1|0|Medium/shrinkage"
    "test4_medium_noshrinkage|25|60|5|0|0|Medium/no-shrinkage"
    "test5_wide_shrinkage|40|80|6|1|0|Wide/shrinkage"
    "test6_uneven_shrinkage|15|50|5|1|0.15|Uneven/shrinkage"
    "test7_uneven_noshrinkage|15|50|5|0|0.15|Uneven/no-shrinkage"
    "test8_tiny|5|12|3|1|0|Tiny edge case"
    "test9_argmin_nan_bug|15|8|3|1|0|argmin/NaN bug (rank-deficient cN)"
    # Test 10: stresses the 2D data-cov ridge bug in calc_shrunken_covariance.py.
    # Uses a truly low-rank generator (rank_signal=5, rank_noise=10 in a 50-dim
    # space) so the POPULATION covariance has rank 15 < nvox=50. The 2D path's
    # training cov is consequently rank-deficient and Python's lines 179/182 add
    # c + 1e-6*I, making alpha=1 non-singular. Because the validation conditions
    # live in the same rank-15 subspace as training, the quadratic form along
    # the ridge-padded null directions is essentially zero, and the spuriously
    # very-negative log-det dominates the NLL at alpha=1.
    # Result: Python picks shrinklevelD=1.0 with NLL≈-205, while MATLAB's
    # cholcov fails at alpha=1 -> NaN -> min() skips it and picks alpha=0.98.
    # Downstream cSb / cNb diverge accordingly. The fix is to remove the two
    # `c = c + np.eye(...) * 1e-6` lines in calc_shrunken_covariance.py.
    "test10_ridge_2d_data_cov|50|40|3|1|0|Ridge bug 2D data-cov path|5:10:0.3"
)

# Parallel arrays keyed by test number (1-based). Sparse — only filled for tests that actually run.
declare -a test_result_by_num=()
declare -a test_label_by_num=()

for i in "${!TEST_DEFS[@]}"; do
    num=$((i + 1))
    if ! should_run_test "$num"; then
        continue
    fi
    IFS='|' read -r tname nvox ncond ntrial wantshrinkage uneven_frac label low_rank_spec <<< "${TEST_DEFS[$i]}"
    echo "=== Test $num: $label ==="
    if run_gsn_equivalence_test "$tname" "$nvox" "$ncond" "$ntrial" "$wantshrinkage" "$uneven_frac" "$low_rank_spec"; then
        test_result_by_num[$num]="PASSED"
    else
        test_result_by_num[$num]="FAILED"
    fi
    test_label_by_num[$num]="$label"
    echo ""
done

# ==================== FINAL SUMMARY ====================
echo ""
echo "=========================================="
echo "FINAL SUMMARY"
echo "=========================================="

total_tests=0
passed_tests=0

for num in "${!test_result_by_num[@]}"; do
    result="${test_result_by_num[$num]}"
    label="${test_label_by_num[$num]}"
    total_tests=$((total_tests + 1))
    if [ "$result" = "PASSED" ]; then
        echo "PASS: Test $num: $label"
        passed_tests=$((passed_tests + 1))
    else
        echo "FAIL: Test $num: $label"
    fi
done

echo ""
echo "Overall Results: $passed_tests/$total_tests tests passed"

if [ $passed_tests -eq $total_tests ]; then
    echo ""
    echo "ALL GSN EQUIVALENCE TESTS PASSED!"
    echo "Python and MATLAB implementations are numerically equivalent"
    echo "across all tested data configurations."
    final_exit_code=0
else
    failed_tests=$((total_tests - passed_tests))
    echo ""
    echo "$failed_tests/$total_tests tests failed."
    echo ""
    echo "IMPORTANT: MATLAB is the ground truth implementation."
    echo "Any failures indicate the Python implementation needs to be"
    echo "modified to match MATLAB behavior, not vice versa."
    final_exit_code=1
fi

echo ""
echo "Test data saved in: $TEST_DATA_DIR"
echo "You can inspect individual results and comparison summaries there."
echo ""
echo "GSN equivalence testing complete!"

exit $final_exit_code
