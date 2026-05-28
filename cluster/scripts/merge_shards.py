#!/usr/bin/env python3
"""Merge per-nunits benchmark shards into one consolidated JSON.

Each shard from cluster/scripts/benchmark_gsn.py looks like:
    { "nunits": 500, ..., "results": { backend: {"times": [...], "median": ...}, ...} }

The merged file is keyed by backend so it can be fed straight into the
local tests/test_speedup_magnitude.py save_figure() routine:

    { backend: { "<N>": [times...], ... }, ..., "meta": [shard metadata, ...] }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard-dir', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args(argv)

    shard_dir = Path(args.shard_dir)
    shards = sorted(shard_dir.glob('shard_N*.json'))
    if not shards:
        print(f'No shards in {shard_dir}', file=sys.stderr)
        return 1

    merged = {}
    meta = []
    for f in shards:
        payload = json.loads(f.read_text())
        N = int(payload['nunits'])
        for backend, info in payload['results'].items():
            if info is None:
                continue
            merged.setdefault(backend, {})[str(N)] = info['times']
        meta.append({k: payload.get(k) for k in (
            'nunits', 'nconds', 'ntrials', 'repeats', 'hostname',
            'gpu', 'torch')})

    merged['meta'] = meta
    Path(args.output).write_text(json.dumps(merged, indent=2))
    backends = [k for k in merged if k != 'meta']
    print(f'merged {len(shards)} shards across backends {backends}')
    print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
