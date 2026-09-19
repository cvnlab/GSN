"""
calc_mmd.py

Python port of calcmmd.m from the GSN toolbox (JW 09/2026).
Outputs match matlab outputs up to floating point error/random draws and eigenvector signs.
Note that I implemented the same random sample draw for MMD and MED.

  results = calc_mmd(signalCovB, noiseCovB, numpairs=1000, noisepct=90)

<signalCovB> and <noiseCovB> are cSb and cNb from performgsn / perform_gsn
<numpairs> (optional) is the number of pairs for which to calculate
  distances. Default: 1000.
<noisepct> (optional) is a non-empty vector of percentages. We apply
  whitening for the number of noise dimensions that achieves each
  percentage of variance of the noise distribution. If you supply
  more than one percentage, we automatically sort them. Default: 90.

Return <results> as a dict with:
  <vSb> and <dSb> as eigenvectors and eigenvalues of <cSb>
  <vNb> and <dNb> as eigenvectors and eigenvalues of <cNb>
  <ncsnr> as median ratio of signal std to noise std
  <totvarsignal> as sum of all signal variances
  <totvarnoise> as sum of all noise variances
  <totvarsnr> as ratio of <totvarsignal> and <totvarnoise>
  <EDsignal> as the effective dimensionality of <cSb>
  <EDnoise> as the effective dimensionality of <cNb>
  <totvarsignalALT> as sum of all signal variances
    after normalization such that noise variances are 1
  <totvarnoiseALT> as sum of all noise variances (after normalization)
  <totvarsnrALT> as ratio of <totvarsignalALT> and <totvarnoiseALT>
  <med> as median Euclidean distance
  <mmd_uncorr> as median Mahalanobis distance (ignoring any noise correlations)
  <mmd> as a 1 x len(<noisepct>) vector with median Mahalanobis distances,
    whitening only the noise dimensions corresponding to <noisepct>

Note that for <vSb>, <dSb>, <vNb>, and <dNb>, eigenvalues are provided
in descending order with all eigenvalues forced to be real and non-negative.
"""

import numpy as np
from gsn.utilities import posrect


def calc_mmd(signalCovB, noiseCovB, numpairs=None, noisepct=None):



    #### setup/inputs ####
    # set empty variables
    if numpairs is None:
        numpairs = 1000
    if noisepct is None:
        noisepct = 90  # [50 75 90 95]
    # deal with inputs
    noisepct = np.sort(np.atleast_1d(np.asarray(noisepct, dtype=float)))
    # constants
    effectiveDim = lambda x: np.sum(x) ** 2 / np.sum(x ** 2)



    #### eigendecomposition of signal ####
    signalEigvalsB, signalEigvecsB = np.linalg.eigh(signalCovB)
    signalEigvalsB = posrect(np.real(signalEigvalsB))
    sortIdx = np.argsort(-signalEigvalsB, kind='stable')
    signalEigvalsB = signalEigvalsB[sortIdx]
    signalEigvecsB = signalEigvecsB[:, sortIdx]



    #### eigendecomposition of noise ####
    noiseEigvalsB, noiseEigvecsB = np.linalg.eigh(noiseCovB)
    noiseEigvalsB = posrect(np.real(noiseEigvalsB))
    sortIdx = np.argsort(-noiseEigvalsB, kind='stable')
    noiseEigvalsB = noiseEigvalsB[sortIdx]
    noiseEigvecsB = noiseEigvecsB[:, sortIdx]



    #### calc simple metrics ####
    # ncsnr
    ncsnr = np.median(np.sqrt(np.diag(signalCovB) / np.diag(noiseCovB)))
    # total variance
    totVarSignal = np.sum(np.diag(signalCovB))
    totVarNoise = np.sum(np.diag(noiseCovB))
    totVarSNR = totVarSignal / totVarNoise
    # ED
    signalED = effectiveDim(signalEigvalsB)
    noiseED = effectiveDim(noiseEigvalsB)



    #### proceed to MMD ####
    # construct normalization matrix
    noiseStd = np.sqrt(np.diag(noiseCovB))
    noiseStdOuter = np.outer(noiseStd, noiseStd)
    # divide by normalization matrix such that noise variances equal 1
    noiseCorr = noiseCovB / noiseStdOuter
    signalCovNorm = signalCovB / noiseStdOuter
    # total variance (alternative)
    totVarSignalNorm = np.sum(np.diag(signalCovNorm))
    totVarNoiseNorm = np.sum(np.diag(noiseCorr))
    totVarSNRNorm = totVarSignalNorm / totVarNoiseNorm
    # draw random samples from signal distribution
    samples = np.random.multivariate_normal(np.zeros(signalCovNorm.shape[0]), signalCovNorm, 2 * numpairs).T  # dim x 2*N
    # reshape
    samplePairs = np.reshape(samples, (samples.shape[0], -1, 2), order='F')  # dim x N x 2
    # calculate median Mahalanobis distance assuming the noise is uncorrelated
    pairDist = np.sqrt(np.sum(np.diff(samplePairs, axis=2) ** 2, axis=0))
    mmd_uncorr = np.median(pairDist)



    #### proceed to MED ####
    # DEVIATION FROM MATLAB: Using the same random draw here as we did in MMD.
    # scale the normalized samples back into the original units
    samplesRaw = noiseStd[:, np.newaxis] * samples  # dim x 2*N
    # reshape
    samplePairsRaw = np.reshape(samplesRaw, (samplesRaw.shape[0], -1, 2), order='F')  # dim x N x 2
    # calculate median Euclidean distance
    pairDistRaw = np.sqrt(np.sum(np.diff(samplePairsRaw, axis=2) ** 2, axis=0))
    med = np.median(pairDistRaw)



    #### do the full version of MMD ####
    # eigendecomposition of the normalized noise
    noiseCorrEigvals, noiseCorrEigvecs = np.linalg.eigh(noiseCorr)
    noiseCorrEigvals = posrect(np.real(noiseCorrEigvals))
    sortIdx = np.argsort(-noiseCorrEigvals, kind='stable')
    noiseCorrEigvals = noiseCorrEigvals[sortIdx]
    noiseCorrEigvecs = noiseCorrEigvecs[:, sortIdx]
    # compute cumulative sum of noise variance
    cumNoisePct = np.cumsum(noiseCorrEigvals) / np.sum(noiseCorrEigvals) * 100
    # loop over how much of the noise to take into account
    mmd = np.zeros(len(noisepct))
    for p in range(len(noisepct)):
        # find number of dimensions to retain
        aboveIdx = np.flatnonzero(cumNoisePct >= noisepct[p])
        assert len(aboveIdx) >= 1
        # construct vector of scalings
        whitenScale = np.sqrt(1 / noiseCorrEigvals)
        whitenScale[aboveIdx[0] + 1:] = 1
        # construct noise whitening matrix
        whitenMatrix = noiseCorrEigvecs @ np.diag(whitenScale) @ noiseCorrEigvecs.T
        # multiply and reshape
        samplePairs = np.reshape(whitenMatrix @ samples, (samples.shape[0], -1, 2), order='F')  # dim x N x 2
        # calculate median Mahalanobis distance
        pairDist = np.sqrt(np.sum(np.diff(samplePairs, axis=2) ** 2, axis=0))
        mmd[p] = np.median(pairDist)



    #### outputs ####
    results = {
        'vSb': signalEigvecsB,
        'dSb': signalEigvalsB,
        'vNb': noiseEigvecsB,
        'dNb': noiseEigvalsB,
        'ncsnr': ncsnr,
        'totvarsignal': totVarSignal,
        'totvarnoise': totVarNoise,
        'totvarsnr': totVarSNR,
        'EDsignal': signalED,
        'EDnoise': noiseED,
        'totvarsignalALT': totVarSignalNorm,
        'totvarnoiseALT': totVarNoiseNorm,
        'totvarsnrALT': totVarSNRNorm,
        'med': med,
        'mmd_uncorr': mmd_uncorr,
        'mmd': mmd,
    }
    return results
