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
    'numpy' forces the numpy + scipy reference path via opt['backend']='numpy'
    so it is exercised even when torch is installed.
    """
    force_numpy = (method == 'numpy')
    device = 'cpu' if force_numpy else method
    backend = 'numpy' if force_numpy else 'torch'
    # Match the legacy main reference so the cross-backend wall-clock is
    # apples-to-apples; the default 'returns' adds three eighs that the
    # reference doesn't do.
    opt = {'wantverbose': 0, 'backend': backend, 'device': device,
           'returns': ['cSb', 'cNb']}
    # Warmup: first call pays JIT / lazy-import / kernel-load costs.
    perform_gsn(data, opt)
    _sync_device(device)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        perform_gsn(data, opt)
        _sync_device(device)
        times.append(time.perf_counter() - t0)
    return times


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
    'python-main-reference': '#000000',  # reference: black
    'python-numpy':          '#2ca02c',  # fast_numpy: green
    'python-torch-cpu':      '#1f77b4',  # fast_torch_cpu: blue
    'python-torch-cuda':     '#d62728',  # fast_torch_cuda: red
    'python-torch-mps':      '#ff7f0e',
    'matlab':                '#9467bd',
}

_MARKERS = {
    'python-main-reference': 's',  # square
    'python-numpy':          'D',  # diamond
    'python-torch-cpu':      '^',  # triangle
    'python-torch-cuda':     'o',  # circle
    'python-torch-mps':      'v',
    'matlab':                'P',
}

# How each backend appears in the legend. Anything not listed falls back
# to the raw key. The naming distinguishes (a) the unoptimized reference
# from (b) the speedup branch, and within (b) which 51-level loop strategy
# is used: per-slot Python for-loop using numpy + scipy.linalg.solve_triangular
# vs. one batched torch.linalg.cholesky_ex over the entire stack on CPU or GPU.
_DISPLAY_NAMES = {
    'python-main-reference': 'gsn.perform_gsn (reference)',
    'python-numpy':          'numpy + scipy.linalg loop',
    'python-torch-cpu':      'torch CPU (batched)',
    'python-torch-cuda':     'torch CUDA (batched)',
    'python-torch-mps':      'torch MPS (batched)',
    'matlab':                'matlab',
}


def save_figure(results, args, path, *,
                title='perform_gsn runtime scaling',
                extrap_max=2e5, fit_min_n=1000):
    """Single-panel log-log figure with power-law extrapolation.

    Layout:
      - solid line + filled markers on the measured range
      - dashed extrapolation extending from the last measured point out
        to ``extrap_max`` using a power-law fit on the N >= fit_min_n tail
      - horizontal reference lines at 1 min / 10 min / 1 hr / 10 hr with
        labels on the right edge
      - legend lists each backend as "name (measured)" plus a separate
        "extrap: t ≈ a·N^b" entry per backend
      - gray shading over the extrapolation region
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(11, 7))

    nunits = np.array(args.nunits, dtype=float)
    medians = {
        m: np.array([
            float(np.median(results[m][N])) if N in results[m] else np.nan
            for N in args.nunits
        ]) for m in results
    }

    # Find the largest measured N across any method; the extrapolation
    # starts there and runs to extrap_max.
    finite_n_per_method = {
        m: nunits[np.isfinite(medians[m]) & (medians[m] > 0)]
        for m in results
    }
    all_max_measured = max((arr.max() for arr in finite_n_per_method.values()
                            if arr.size > 0), default=None)
    if all_max_measured is None:
        raise RuntimeError('no measured points to plot')

    # Shade the extrapolated x-range so it reads as a projection, not data.
    ax.axvspan(all_max_measured, extrap_max, color='black', alpha=0.04, zorder=0)

    # ---- Plot each backend: measured line+markers, then dashed extrapolation
    legend_handles = []
    legend_labels = []
    for method in results:
        ts = medians[method]
        ok = np.isfinite(ts) & (ts > 0)
        if ok.sum() < 1:
            continue
        color = _COLORS.get(method, 'gray')
        marker = _MARKERS.get(method, 'o')
        disp = _DISPLAY_NAMES.get(method, method)

        # Solid line + filled markers on measured range
        ax.plot(nunits[ok], ts[ok], marker=marker, linestyle='-', color=color,
                markersize=8, linewidth=1.8, zorder=3)
        legend_handles.append(Line2D([0], [0], marker=marker, color=color,
                                     linestyle='-', linewidth=1.8,
                                     markersize=8))
        legend_labels.append(f'{disp} (measured)')

        # Power-law fit on the tail (N >= fit_min_n) when available;
        # otherwise use all measured points.
        tail = ok & (nunits >= fit_min_n)
        fit_mask = tail if tail.sum() >= 2 else ok
        if fit_mask.sum() < 2:
            continue
        slope, intercept = np.polyfit(np.log(nunits[fit_mask]),
                                       np.log(ts[fit_mask]), 1)
        a = np.exp(intercept)
        # Extrapolation starts at the method's own largest measured N so
        # the solid + dashed line is continuous.
        n_start = nunits[ok].max()
        N_extrap = np.logspace(np.log10(n_start), np.log10(extrap_max), 200)
        t_extrap = a * N_extrap ** slope
        ax.plot(N_extrap, t_extrap, linestyle='--', color=color,
                linewidth=1.3, zorder=2, alpha=0.85)
        legend_handles.append(Line2D([0], [0], color=color, linestyle='--',
                                     linewidth=1.3))
        legend_labels.append(f'extrap: t ≈ {a:.2e} · N^{slope:.2f}')

    # ---- Horizontal time-reference lines on the right edge
    ref_lines = [
        (60,    '1 min',  'goldenrod'),
        (600,   '10 min', 'darkorange'),
        (3600,  '1 hr',   'firebrick'),
        (36000, '10 hr',  'darkred'),
    ]
    for sec, label, lcolor in ref_lines:
        ax.axhline(sec, color=lcolor, linestyle='-', linewidth=0.8,
                   alpha=0.6, zorder=1)
        ax.text(extrap_max * 1.02, sec, label, color=lcolor,
                fontsize=10, va='center', ha='left', fontweight='bold')

    # ---- Axes, ticks, labels
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(nunits.min() * 0.8, extrap_max)
    ax.set_xlabel('nunits (log scale)', fontsize=12)
    ax.set_ylabel('perform_gsn wall-clock time (s, log scale)', fontsize=12)
    ax.set_title(f'{title} — ncond={args.ncond}, ntrial={args.ntrial}, '
                 f'repeats={args.repeats}', fontsize=12)
    ax.grid(True, which='major', alpha=0.4)
    ax.grid(True, which='minor', alpha=0.15)

    ax.legend(legend_handles, legend_labels, loc='upper left', fontsize=9,
              framealpha=0.92)

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
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
