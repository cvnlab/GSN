#!/usr/bin/env python3
"""Benchmark GSN runtimes across implementations available on this machine.

Auto-detects which backends are present and times perform_gsn / performgsn
across a sweep of nunits, with K repeats per cell. Saves a summary figure
and a JSON dump of the raw timings.

Methods (auto-detected):
    python-numpy       Pure numpy + scipy fallback (forces _HAS_TORCH=False).
    python-torch-cpu   Torch CPU backend (default when torch is installed).
    python-torch-cuda  Torch CUDA backend (if torch.cuda.is_available()).
    python-torch-mps   Torch MPS backend (Apple Silicon, if available).
    matlab             performgsn via a single batched MATLAB subprocess.

Usage
-----
    python tests/test_speedup_magnitude.py
    python tests/test_speedup_magnitude.py --nunits 50 100 200 350 500 --repeats 3
    python tests/test_speedup_magnitude.py --skip-matlab

Outputs:
    tests/speedup_bench_data/        (data .mat files + MATLAB results JSON)
    tests/speedup_magnitude.png      (figure)
    tests/speedup_magnitude.json     (raw timings)

Note: file is named ``test_*`` to live next to the unit tests, but it
defines no ``test_*`` functions, so pytest collection is a no-op. Run
the script directly.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from gsn.perform_gsn import perform_gsn  # noqa: E402
import gsn.batched_nll as bn             # noqa: E402

_HAS_TORCH = bn._HAS_TORCH
if _HAS_TORCH:
    import torch


# ---------------------------------------------------------------------------
# Setup detection
# ---------------------------------------------------------------------------

def _find_matlab():
    """Return path to a usable matlab binary or None.

    Probes common macOS install locations first (matching the equivalence
    .sh), then falls back to whatever's on PATH.
    """
    candidates = [
        '/Applications/MATLAB_R2024b.app/bin/matlab',
        '/Applications/MATLAB_R2024a.app/bin/matlab',
        '/Applications/MATLAB_R2023b.app/bin/matlab',
        '/Applications/MATLAB_R2023a.app/bin/matlab',
        '/Applications/MATLAB_R2022b.app/bin/matlab',
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return shutil.which('matlab')


def detect_methods(want_matlab=True):
    methods = ['python-numpy']
    if _HAS_TORCH:
        methods.append('python-torch-cpu')
        if torch.cuda.is_available():
            methods.append('python-torch-cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            methods.append('python-torch-mps')
    if want_matlab and _find_matlab():
        methods.append('matlab')
    return methods


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_synthetic(nvox, ncond, ntrial, rank=20, seed=0):
    """Low-rank signal + iid noise. Matches speedup3/benchmark.py."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((nvox, rank)) / np.sqrt(rank)
    Z = rng.standard_normal((rank, ncond))
    signal = U @ Z
    noise = rng.standard_normal((nvox, ncond, ntrial))
    return signal[:, :, None] + noise


# ---------------------------------------------------------------------------
# Python timing
# ---------------------------------------------------------------------------

def _sync_device(device):
    """Wait for outstanding GPU work before reading the wall clock."""
    if not _HAS_TORCH:
        return
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == 'mps' and hasattr(torch.backends, 'mps') \
            and torch.backends.mps.is_available():
        torch.mps.synchronize()


def time_python(data, method, repeats):
    """Time perform_gsn K times, return list of seconds.

    method is one of 'numpy', 'cpu', 'cuda', 'mps'.
    'numpy' monkey-patches _HAS_TORCH=False so the numpy + scipy fallback
    is exercised even when torch is installed.
    """
    force_numpy = (method == 'numpy')
    device = 'cpu' if force_numpy else method
    opt = {'wantverbose': 0, 'device': device}
    saved = bn._HAS_TORCH
    if force_numpy:
        bn._HAS_TORCH = False
    try:
        # Warmup — first call pays JIT / lazy-import / kernel-load costs.
        perform_gsn(data, opt)
        _sync_device(device)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            perform_gsn(data, opt)
            _sync_device(device)
            times.append(time.perf_counter() - t0)
        return times
    finally:
        bn._HAS_TORCH = saved


# ---------------------------------------------------------------------------
# MATLAB timing
# ---------------------------------------------------------------------------

MATLAB_SCRIPT_TEMPLATE = """\
try
    addpath(genpath('{matlab_dir}'));
    data_dir = '{data_dir}';
    nunits_list = [{nunits_csv}];
    repeats = {repeats};
    results = struct();
    for i = 1:length(nunits_list)
        N = nunits_list(i);
        loaded = load(fullfile(data_dir, sprintf('bench_N%d.mat', N)));
        data = loaded.data;
        opt = struct('wantverbose', 0, 'wantshrinkage', 1);
        performgsn(data, opt);   % warmup
        times = zeros(1, repeats);
        for r = 1:repeats
            t = tic;
            performgsn(data, opt);
            times(r) = toc(t);
        end
        results.(sprintf('N%d', N)) = times;
    end
    fid = fopen('{results_file}', 'w');
    fprintf(fid, '%s', jsonencode(results));
    fclose(fid);
    exit(0);
catch ME
    fprintf('matlab error: %s\\n', ME.message);
    exit(1);
end
"""


def time_matlab(nunits_list, ncond, ntrial, repeats, matlab_bin, data_dir):
    """Run MATLAB once over the full sweep. Returns {nunits: [times...]}."""
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    for N in nunits_list:
        f = data_dir / f'bench_N{N}.mat'
        if not f.exists():
            scipy.io.savemat(str(f), {'data': make_synthetic(N, ncond, ntrial)})

    matlab_dir = ROOT / 'matlab'
    results_file = data_dir / 'matlab_results.json'
    script_file = data_dir / 'run_matlab_bench.m'

    script = MATLAB_SCRIPT_TEMPLATE.format(
        matlab_dir=str(matlab_dir),
        data_dir=str(data_dir),
        nunits_csv=', '.join(str(n) for n in nunits_list),
        repeats=repeats,
        results_file=str(results_file),
    )
    script_file.write_text(script)

    # -batch expects a single MATLAB statement; the simplest portable
    # approach is to write the script to a .m file and `run()` it. We
    # also add data_dir to the MATLAB path so the script file resolves.
    cmd = f"addpath('{data_dir}'); run_matlab_bench"
    subprocess.run([matlab_bin, '-batch', cmd], check=True)

    raw = json.loads(results_file.read_text())
    return {N: list(np.atleast_1d(raw[f'N{N}'])) for N in nunits_list}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_COLORS = {
    'python-main-reference': '#8c564b',  # old main-branch reference (no speedups)
    'python-numpy':          '#1f77b4',
    'python-torch-cpu':      '#ff7f0e',
    'python-torch-cuda':     '#2ca02c',
    'python-torch-mps':      '#d62728',
    'matlab':                '#9467bd',
}


def save_figure(results, args, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax_abs, ax_extrap, ax_rel) = plt.subplots(
        1, 3, figsize=(18, 5.5), gridspec_kw={'wspace': 0.3})

    nunits = np.array(args.nunits, dtype=float)
    # Missing (method, N) cells get NaN so methods can opt out of late N
    # values without crashing the renderer (e.g. matlab skipped at N=5000).
    medians = {
        m: np.array([
            float(np.median(results[m][N])) if N in results[m] else np.nan
            for N in args.nunits
        ]) for m in results
    }

    # ---- Panel 1: absolute timing on measured range (linear axes)
    for method in results:
        m = np.isfinite(medians[method])
        ax_abs.plot(nunits[m], medians[method][m], 'o-', label=method,
                    color=_COLORS.get(method))
    ax_abs.set_xlabel('nunits (voxels)')
    ax_abs.set_ylabel('median seconds per perform_gsn call')
    ax_abs.set_title('Absolute runtime (measured)')
    ax_abs.grid(True, alpha=0.3)
    ax_abs.legend()

    # ---- Panel 2: power-law extrapolation to N=1e6 (log Y).
    # GSN's bottleneck is per-call O(N^3) Cholesky / eigh, so wall-clock
    # follows time = a * N^b with b near 3. Small-N points are
    # overhead-dominated (Python interpreter / MATLAB startup tax) and
    # pull the fitted exponent below the true asymptotic, so we restrict
    # both the displayed markers and the fit to N > 1000.
    FIT_MIN_N = 1000
    in_range = nunits > FIT_MIN_N
    N_extrap = np.logspace(np.log10(FIT_MIN_N), 6, 200)
    for method in results:
        ts = medians[method]
        ok = np.isfinite(ts) & (ts > 0) & in_range
        if ok.sum() < 2:
            continue
        slope, intercept = np.polyfit(np.log(nunits[ok]), np.log(ts[ok]), 1)
        a = np.exp(intercept)
        t_extrap = a * N_extrap ** slope
        color = _COLORS.get(method)
        ax_extrap.plot(nunits[ok], ts[ok], 'o', color=color, markersize=7)
        ax_extrap.plot(N_extrap, t_extrap, '--', color=color,
                       label=f'{method}  (~ N^{slope:.2f})')

    # horizontal reference lines for human-scale costs
    for sec, label in [(1, '1 sec'), (60, '1 min'), (3600, '1 hour'),
                       (86400, '1 day')]:
        ax_extrap.axhline(sec, color='gray', linestyle=':', alpha=0.5)
        ax_extrap.text(N_extrap[-1] * 0.95, sec * 1.15, label,
                       fontsize=8, color='gray', ha='right', va='bottom')

    ax_extrap.set_xscale('log')
    ax_extrap.set_yscale('log')
    ax_extrap.set_xlabel('nunits (voxels, log)')
    ax_extrap.set_ylabel('extrapolated seconds per perform_gsn call (log)')
    ax_extrap.set_title(f'Power-law extrapolation to N = 1,000,000  (fit on N > {FIT_MIN_N})')
    ax_extrap.set_xlim(FIT_MIN_N * 0.8, 1.2e6)
    ax_extrap.grid(True, which='both', alpha=0.3)
    ax_extrap.legend(loc='upper left')

    # ---- Panel 3: speedup relative to a chosen baseline (linear axes).
    # Prefer python-main-reference when present so the headline numbers
    # reflect the cumulative branch win; fall back to python-numpy (the
    # already-optimized fallback) or whichever method has the highest
    # mean wall clock.
    if 'python-main-reference' in medians and np.any(np.isfinite(medians['python-main-reference'])):
        baseline_name = 'python-main-reference'
    elif 'python-numpy' in medians:
        baseline_name = 'python-numpy'
    else:
        baseline_name = max(medians, key=lambda m: float(np.nanmean(medians[m])))
    baseline = medians[baseline_name]
    for method in results:
        if method == baseline_name:
            continue
        m = np.isfinite(medians[method]) & np.isfinite(baseline)
        ax_rel.plot(nunits[m], (baseline / medians[method])[m], 'o-',
                    label=method, color=_COLORS.get(method))
    ax_rel.axhline(1.0, color='black', linestyle='--', alpha=0.4,
                   label=f'{baseline_name} = 1.0')
    ax_rel.set_xlabel('nunits (voxels)')
    ax_rel.set_ylabel(f'speedup vs {baseline_name}')
    ax_rel.set_title('Relative speedup')
    ax_rel.grid(True, alpha=0.3)
    ax_rel.legend()

    fig.suptitle(
        f'GSN benchmark — ncond={args.ncond}, ntrial={args.ntrial}, '
        f'repeats={args.repeats}',
        fontsize=11)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    print(f"\nFigure saved to {path}")


def print_table(results, args):
    methods = list(results)
    width = max(14, max(len(m) for m in methods) + 2)
    header = f"{'nunits':>8}" + ''.join(f"{m:>{width}}" for m in methods)
    print()
    print(header)
    print('-' * len(header))
    for N in args.nunits:
        row = f"{N:>8}"
        for m in methods:
            t = np.median(results[m][N]) * 1000
            row += f"{t:>{width-2}.1f}ms"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--nunits', type=int, nargs='+',
                        default=[50, 100, 200, 500, 1000, 2000, 5000],
                        help='voxel counts to sweep (default: 50 100 200 500 1000 2000 5000)')
    parser.add_argument('--ncond', type=int, default=80)
    parser.add_argument('--ntrial', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3,
                        help='timed calls per (method, nunits) cell after warmup')
    parser.add_argument('--skip-matlab', action='store_true')
    parser.add_argument('--data-dir', type=str,
                        default=str(HERE / 'speedup_bench_data'))
    parser.add_argument('--figure', type=str,
                        default=str(HERE / 'speedup_magnitude.png'))
    parser.add_argument('--results-json', type=str,
                        default=str(HERE / 'speedup_magnitude.json'))
    args = parser.parse_args(argv)

    methods = detect_methods(want_matlab=not args.skip_matlab)

    print('GSN speedup magnitude benchmark')
    print('-' * 60)
    print(f'  methods detected:  {methods}')
    print(f'  nunits:            {args.nunits}')
    print(f'  ncond={args.ncond}  ntrial={args.ntrial}  repeats={args.repeats}')
    print()

    results = {m: {} for m in methods}

    # Python sweeps first (cheap)
    for N in args.nunits:
        print(f'=== nunits={N} (python) ===', flush=True)
        data = make_synthetic(N, args.ncond, args.ntrial)
        for method in methods:
            if method == 'matlab':
                continue
            # 'python-numpy' -> 'numpy'; 'python-torch-cpu' -> 'cpu', etc.
            kind = (method.replace('python-torch-', '')
                          .replace('python-', ''))
            times = time_python(data, kind, args.repeats)
            results[method][N] = times
            print(f'  {method:14s} median {np.median(times)*1000:7.1f} ms '
                  f'(min {np.min(times)*1000:.1f}, '
                  f'max {np.max(times)*1000:.1f})',
                  flush=True)

    # MATLAB in one process for the whole sweep
    if 'matlab' in methods:
        matlab_bin = _find_matlab()
        print(f'\n=== MATLAB sweep ({matlab_bin}) ===', flush=True)
        try:
            mat_times = time_matlab(args.nunits, args.ncond, args.ntrial,
                                    args.repeats, matlab_bin, args.data_dir)
            for N in args.nunits:
                results['matlab'][N] = mat_times[N]
                print(f'  matlab N={N:<4d}  median '
                      f'{np.median(mat_times[N])*1000:7.1f} ms',
                      flush=True)
        except subprocess.CalledProcessError as e:
            print(f'  MATLAB subprocess failed: {e}', flush=True)
            results.pop('matlab', None)
            methods.remove('matlab')

    print_table(results, args)

    Path(args.results_json).write_text(json.dumps(
        {m: {str(N): list(map(float, ts)) for N, ts in cells.items()}
         for m, cells in results.items()},
        indent=2,
    ))
    print(f'\nRaw timings saved to {args.results_json}')

    save_figure(results, args, args.figure)

    return 0


if __name__ == '__main__':
    sys.exit(main())
