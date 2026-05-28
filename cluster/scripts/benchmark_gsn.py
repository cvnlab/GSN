#!/usr/bin/env python3
"""Cluster-side runtime benchmark driver for gsn.perform_gsn.

Times perform_gsn with K repeats per backend at a single nunits value,
writes a JSON shard. Designed to be called from
cluster/jobs/benchmark_gsn.sh as one SLURM array task per nunits — the
shards are merged by cluster/run.sh after the array completes.

Backends:
    python-numpy        gsn.batched_nll._HAS_TORCH monkey-patched to False
    python-torch-cpu    opt['device']='cpu'
    python-torch-cuda   opt['device']='cuda' (skipped if cuda unavailable)

Output JSON schema:
    {
      "nunits": int, "nconds": int, "ntrials": int, "repeats": int,
      "hostname": str, "gpu": str | null, "torch": str | null,
      "results": {
        "python-numpy":     {"times": [..K..], "median": float} | null,
        "python-torch-cpu": {...},
        "python-torch-cuda":{...}
      }
    }
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from gsn.perform_gsn import perform_gsn  # noqa: E402
import gsn.batched_nll as bn             # noqa: E402

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False


def make_synthetic(nvox: int, ncond: int, ntrial: int,
                   rank: int = 20, seed: int = 0) -> np.ndarray:
    """Same generator as tests/test_speedup_magnitude.py so results from
    the two scripts are directly comparable."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((nvox, rank)) / np.sqrt(rank)
    Z = rng.standard_normal((rank, ncond))
    signal = U @ Z
    noise = rng.standard_normal((nvox, ncond, ntrial))
    return signal[:, :, None] + noise


def _sync_cuda():
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


def time_backend(data: np.ndarray, *, backend: str,
                 repeats: int) -> Optional[dict]:
    """Time `perform_gsn` `repeats` times after one warmup; return None if
    the backend is unavailable on this node."""
    force_numpy = (backend == 'python-numpy')
    if backend == 'python-torch-cuda':
        if not (_HAS_TORCH and torch.cuda.is_available()):
            return None
        device = 'cuda'
    elif backend == 'python-torch-cpu':
        if not _HAS_TORCH:
            return None
        device = 'cpu'
    elif backend == 'python-numpy':
        device = 'cpu'
    else:
        raise ValueError(f"unknown backend {backend!r}")

    saved = bn._HAS_TORCH
    if force_numpy:
        bn._HAS_TORCH = False
    try:
        opt = {'wantverbose': 0, 'device': device}
        # Warmup — first call pays kernel-load / JIT / lazy-import tax.
        perform_gsn(data, opt)
        _sync_cuda()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            perform_gsn(data, opt)
            _sync_cuda()
            times.append(time.perf_counter() - t0)
        return {'times': times, 'median': float(np.median(times))}
    finally:
        bn._HAS_TORCH = saved


def _detect_gpu() -> Optional[str]:
    if _HAS_TORCH and torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return 'cuda (unnamed)'
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--nunits', type=int, required=True)
    parser.add_argument('--nconds', type=int, default=80)
    parser.add_argument('--ntrials', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--rank', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--backends', type=str, nargs='+',
                        default=['python-numpy', 'python-torch-cpu',
                                 'python-torch-cuda'])
    args = parser.parse_args(argv)

    print(f'[benchmark_gsn] N={args.nunits} ncond={args.nconds} '
          f'ntrial={args.ntrials} repeats={args.repeats}', flush=True)
    data = make_synthetic(args.nunits, args.nconds, args.ntrials,
                          rank=args.rank, seed=args.seed)

    results = {}
    for backend in args.backends:
        print(f'[benchmark_gsn] timing {backend}...', flush=True)
        t0 = time.perf_counter()
        info = time_backend(data, backend=backend, repeats=args.repeats)
        elapsed = time.perf_counter() - t0
        if info is None:
            print(f'[benchmark_gsn]   {backend}: SKIPPED (unavailable)',
                  flush=True)
            results[backend] = None
        else:
            print(f'[benchmark_gsn]   {backend}: '
                  f'median {info["median"]*1000:.1f} ms  '
                  f'(wall {elapsed:.1f}s)', flush=True)
            results[backend] = info

    payload = {
        'nunits':   args.nunits,
        'nconds':   args.nconds,
        'ntrials':  args.ntrials,
        'repeats':  args.repeats,
        'rank':     args.rank,
        'seed':     args.seed,
        'hostname': socket.gethostname(),
        'gpu':      _detect_gpu(),
        'torch':    torch.__version__ if _HAS_TORCH else None,
        'results':  results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically via a tmp file so a concurrent reader (rsync from
    # the local driver during polling) never sees a half-written file.
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, out)
    print(f'[benchmark_gsn] wrote {out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
