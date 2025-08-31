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

