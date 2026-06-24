from setuptools import setup, find_packages

requires = []
with open('requirements.txt') as reqfile:
    requires = reqfile.read().splitlines()

with open('README.md', encoding='utf-8') as readmefile:
    long_description = readmefile.read()


setup(
    name='GSN',
    version='0.0.1',
    description='Python GSN',
    url='https://github.com/cvnlab/GSN',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
      "Programming Language :: Python",
      "Development Status :: 1 - Planning",
      "License :: OSI Approved :: BSD License",
      "Topic :: Scientific/Engineering",
      "Intended Audience :: Science/Research",
      ],
    maintainer='Jacob Prince',
    maintainer_email='jacob.samuel.prince@gmail.com',
    keywords='neuroscience ',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=requires,
    extras_require={
        # `pip install gsn[fast]` enables the batched-Cholesky path in
        # calc_shrunken_covariance, which collapses the 51-level shrinkage
        # loop into a single batched torch.linalg.cholesky_ex + batched
        # solve_triangular. 10-100x on CPU, more on CUDA/MPS.
        'fast': ['torch>=2.0'],
    },
)