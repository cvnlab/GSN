# GSN cluster workflow (Kempner H100)

Scripts for pushing the GSN code to the Kempner cluster, running
GPU-backed benchmarks of `gsn.perform_gsn` across backends, and pulling
results back. Modeled on the PSN-updated `cluster/` layout.

## One-time setup

```bash
bash cluster/setup.sh
```

Syncs the repo to `$CLUSTER_PROJECT` on `cannon`, creates output
directories, and builds a conda env at `$CONDA_PREFIX` with torch+CUDA
12.1 and GSN installed in editable mode.

Edit `cluster/config.sh` first if any defaults are wrong — especially
`CLUSTER_USER`, `CLUSTER_PROJECT`, `SLURM_ACCOUNT`, `CONDA_PREFIX`.

## Running a benchmark

```bash
bash cluster/run.sh benchmark_gsn                     # full pipeline
bash cluster/run.sh benchmark_gsn --no-poll           # fire-and-forget
bash cluster/run.sh benchmark_gsn --merge-only        # re-merge existing shards
```

Override the sweep with:
```bash
BENCH_NUNITS="500 1000 2000" bash cluster/run.sh benchmark_gsn
BENCH_NUNITS="1000 2000 5000" BENCH_NCONDS=80 BENCH_REPEATS=2 \
    bash cluster/run.sh benchmark_gsn
```

## What the benchmark measures

For each `nunits` (one SLURM array task per size — all run in parallel
on different H100 nodes), times `perform_gsn` with K repeats per
backend:

  - **python-numpy**       — `gsn.batched_nll._HAS_TORCH` monkey-patched
    to `False` (the fallback numpy + scipy loop).
  - **python-torch-cpu**   — `opt['device']='cpu'` (the default torch
    path using LAPACK batched-Cholesky on CPU).
  - **python-torch-cuda**  — `opt['device']='cuda'` (the H100 path).

Each shard is written to
`cluster/outputs/benchmark_gsn_shards/shard_N<N>_C<nconds>_T<ntrials>_rep<repeats>.json`
and the consolidated merge lands at
`cluster/outputs/benchmark_gsn_merged.json` in a schema compatible with
`tests/test_speedup_magnitude.py`'s `save_figure()` helper.

## Inspecting status

```bash
bash cluster/status.sh        # squeue + recent sacct
bash cluster/sync-results.sh  # pull outputs/logs without re-running
```

## File layout

```
cluster/
├── config.sh                   shared config (host, paths, SLURM, conda)
├── setup.sh                    one-time install
├── sync-code.sh                rsync gsn/ + tests/ + cluster/ → cluster
├── sync-results.sh             rsync outputs/ + logs/ ← cluster
├── run.sh                      submit + poll + merge pipeline
├── status.sh                   squeue + sacct
├── requirements.txt            cluster python deps
├── jobs/
│   └── benchmark_gsn.sh        SLURM array script (1 task per nunits)
└── scripts/
    ├── benchmark_gsn.py        Python driver — times all backends, writes shard
    └── merge_shards.py         consolidate shards into one JSON
```
