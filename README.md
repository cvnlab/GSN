# GSN (Generative Modeling of Signal and Noise)

![image](https://github.com/cvnlab/GSN/assets/35503086/eb433f19-a957-47e5-88c1-652779ef6d78)

-------------------------------------------------------------------------------------------

GSN is a toolbox for accurately modeling signal and noise distributions in neural datasets. We provide both MATLAB and Python implementations. 

GSN is detailed in the following paper:

**[Kay, K., Prince, J.S., Gebhart, T., Tuckute, G., Zhou, J., Naselaris, T., Schütt, H.H. Disentangling signal and noise in neural responses through generative modeling. *PLOS Computational Biology* (2025).](https://doi.org/10.1371/journal.pcbi.1012092)**

If you have questions or discussion points, please use the Discussions
feature of this GitHub repository. If you find a bug, 
please let us know by raising a GitHub Issue.

## MATLAB

To install: 

```bash
git clone https://github.com/cvnlab/GSN.git
```

To use the GSN toolbox, add it to your MATLAB path by running the `setup.m` script.

## Python

To install: 

```bash
pip install git+https://github.com/cvnlab/GSN.git
```

For the optional fast / GPU-accelerated backend (PyTorch), install the `fast` extra:

```bash
pip install "gsn[fast] @ git+https://github.com/cvnlab/GSN.git"
```

Running the example scripts requires:

- installing jupyter notebook or jupyter lab
- installing matplotlib
- cloning the GSN repository in order to get the example scripts located in `examples`:

```bash
pip install jupyterlab
pip install "matplotlib<3.9.0"
git clone https://github.com/cvnlab/GSN.git
```

Code dependencies: see [requirements.txt](./requirements.txt)

## Example scripts

We provide a number of example scripts that demonstrate usage of GSN. You can browse these example scripts here:

(Python Example 1 - running GSN on a small dataset of 100 voxels x 200 conditions x 3 trials) 
https://github.com/cvnlab/GSN/blob/main/examples/example1.ipynb

(MATLAB Example 1 - running GSN on a small dataset of 100 voxels x 200 conditions x 3 trials) 
https://github.com/cvnlab/GSN/blob/main/matlab/examples/example1.m

If you would like to run these example scripts, the Python versions are available in `/GSN/examples`, and the MATLAB versions are available in `/GSN/matlab/examples`.

These notebooks contain a full walkthrough of the process of loading an example dataset, estimating signal and noise distributions using GSN, examining voxel-level noise ceiling estimates, computing the eigenspectrum of both signal and noise covariance matrices, and estimating dimensionality of each. 

## Performance and GPU acceleration (Python)

`perform_gsn` has two interchangeable compute paths: a NumPy/SciPy path that is
the reference implementation and runs out of the box, and an optional PyTorch
path (CPU or GPU) that accelerates the covariance and cross-validated shrinkage
estimation, which is the runtime bottleneck at large numbers of units (thousands
to tens of thousands). Torch is an optional dependency (`pip install gsn[fast]`).
By default GSN uses the torch path automatically when torch is installed and the
NumPy path otherwise; the two are validated against each other in `tests/`.

These behaviors are controlled through optional fields of the `opt` dict passed
to `perform_gsn(data, opt)`:

- `opt['backend']`: `'auto'` (default), `'numpy'`, or `'torch'`. `'auto'` uses
  the torch path when torch is installed and the NumPy path otherwise; `'numpy'`
  forces the reference NumPy/SciPy path; `'torch'` forces the torch path (and
  errors if torch is not installed).
- `opt['device']`: `'cpu'` (default), `'cuda'`, `'mps'`, or `'auto'` (picks
  cuda > mps > cpu by availability). Used only by the torch path. `'cpu'` is the
  right choice up to ~1000 units, where GPU host-to-device transfer dominates;
  `'cuda'` / `'mps'` open up the GPU path for larger N. If torch is not installed
  a non-cpu request falls back to the NumPy CPU path, but if torch is installed
  and the requested GPU is unavailable the call raises rather than silently
  falling back.
- `opt['returns']`: which items to include in the result dict. The default is
  the four covariance matrices the legacy `perform_gsn` always returned
  (`'cN', 'cS', 'cNb', 'cSb'`). Request only what you need (e.g.
  `['cSb', 'cNb']`) to save memory at large N, and optionally include the
  eigendecomposition of `cSb` and of `cSb - cNb/ntrial`
  (`'eigvecs_signal'`, `'eigvals_signal'`, `'eigvecs_difference'`,
  `'eigvals_difference'`). Precomputing these lets [PSN (Partitioning Signal and
  Noise)](https://github.com/jacob-prince/PSN), a companion toolbox that denoises
  neural data using GSN's signal/noise covariance estimates, skip the O(N^3)
  eigendecomposition.
- `opt['eigh_device']`: `'host'` (default, NumPy `eigh`) or `'device'` (torch
  `eigh`, faster on GPU at large N). Only relevant when an
  `eigvecs_*` / `eigvals_*` item is requested.
- `opt['uneven']`: how missing data is handled. `'fast'` (default) is the
  NaN-aware whole-trial path (a trial counts only if every unit is present);
  `'missing'` handles per-unit missing data (a trial may have some units present
  and others missing) without discarding good data; `'reference'` delegates to
  the original estimation path as a parity oracle.

The NumPy path is the reference implementation; the torch paths are validated
against it in `tests/` (with relative tolerances and an argmin-agreement check
on the GPU eigendecomposition).

## Additional information

Terms of use: This content is licensed under a BSD 3-Clause License.

If you use GSN in your research, please cite the following paper:

* **[Kay, K., Prince, J.S., Gebhart, T., Tuckute, G., Zhou, J., Naselaris, T., Schütt, H.H. Disentangling signal and noise in neural responses through generative modeling. *PLOS Computational Biology* (2025).](https://doi.org/10.1371/journal.pcbi.1012092)**

## Change history

* 2025/08/31 - Version 1.1 of GSN released. Accompanies the PLOS Computational Biology paper.
* 2024/04/28 - Version 1.0 of GSN released. Accompanies the bioRxiv preprint.

[pre-release updates; early-stage testing]
* 2024/02/25 - Completed port of matlab algorithmic changes to python. 
* 2024/01/05 - Major overhaul of GSN matlab functionality by incorporating the biconvex optimization procedure, other minor tweaks.
* 2022/04/13 - Algorithmic changes to covariance estimation added.
* 2022/04/08 - Initial python code version is completed. 
* 2022/04/06 - Initial matlab code version is completed. 

