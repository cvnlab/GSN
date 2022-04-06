function [nc,ncdist,results] = rsanoiseceiling(data,rdmfun,comparefun,numsim,nctrials,shrinklevels,mode)

% function [nc,ncdist,results] = rsanoiseceiling(data,rdmfun,comparefun,numsim,nctrials)
%
% <data> is voxels x conditions x trials
% <rdmfun> (optional) is a function that constructs an RDM. Specifically,
%   the function should accept as input a data matrix (e.g. voxels x conditions x trials)
%   and output a RDM with some dimensionality (can be a column vector, 2D matrix, etc.).
%   Default: @(data) pdist(mean(data,3)','correlation')'. This default simply 
%   computes the mean across trials, calculates dissimilarity as 1-r, and then
%   extracts the lower triangle (excluding the diagonal) as a column vector.
% <comparefun> (optional) is a function that quantifies the similarity of two RDMs.
%   Specifically, the function should accept as input two RDMs (in the format
%   that is returned by <rdmfun>) and output a scalar. Default: @corr.
% <numsim> (optional) is the number of Monte Carlo simulations to run.
%   The final answer is computed as the median across simulations. Default: 20.
% <nctrials> (optional) is the number of trials over which to average for
%   the purposes of the noise ceiling estimate. For example, setting
%   <nctrials> to 10 will result in the calculation of a noise ceiling 
%   estimate for the case in which responses are averaged across 10 trials
%   measured for each condition. Default: size(data,3).
%
% Use the GSN (generative modeling of signal and noise) method to estimate
% an RSA noise ceiling.
%
% Note: if <comparefun> ever returns NaN, we automatically replace these
% cases with 0. This is a convenient workaround for degenerate cases, 
% e.g., cases where the signal is generated as all zero.
%
% Return:
%   <nc> as a scalar with the noise ceiling estimate.
%   <ncdist> as 1 x <numsim> with the result of each simulation.
%     Note that <nc> is simply the median of <ncdist>.
%   <results> as a struct with additional details:
%     mnN - the estimated mean of the noise (1 x voxels)
%     cN  - the estimated covariance of the noise (voxels x voxels)
%     mnS - the estimated mean of the signal (1 x voxels)
%     cS  - the estimated covariance of the signal (voxels x voxels)
%     cSb - the regularized estimated covariance of the signal (voxels x voxels).
%           this estimate reflects both a nearest-approximation and 
%           a post-hoc scaling, and is used in the Monte Carlo simulations.
%     rapprox - the correlation between the nearest-approximation of the 
%               signal covariance and the original signal covariance
%
% Example:
% data = repmat(randn(100,40),[1 1 4]) + 2*randn(100,40,4);
% [nc,ncdist,results] = rsanoiseceiling(data);

% internal inputs:
%
% <shrinklevels> (optional) is like the input to calcshrunkencovariance.m.
%   Default: [].
% <mode> (optional) is
%   0 means do the normal thing
%   1 means to omit the gain adjustment

% inputs
if ~exist('rdmfun','var') || isempty(rdmfun)
  rdmfun = @(data) pdist(mean(data,3)','correlation')';
end
if ~exist('comparefun','var') || isempty(comparefun)
  comparefun = @corr;
end
if ~exist('numsim','var') || isempty(numsim)
  numsim = 20;
end
if ~exist('nctrials','var') || isempty(nctrials)
  nctrials = size(data,3);
end
if ~exist('shrinklevels','var') || isempty(shrinklevels)
  shrinklevels = [];
end
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end

% calc
nvox   = size(data,1);
ncond  = size(data,2);
ntrial = size(data,3);

% estimate noise covariance
[mnN,cN,shrinklevelN,nllN] = calcshrunkencovariance(permute(data,[3 1 2]),[],shrinklevels,1);

% estimate data covariance
[mnD,cD,shrinklevelD,nllD] = calcshrunkencovariance(mean(data,3)',        [],shrinklevels,1);

% estimate signal covariance
mnS = mnD - mnN;
cS  =  cD - cN/ntrial;

% calculate nearest approximation for the noise.
% this is expected to be PSD already. however, small numerical issues
% may occassionally arise. so, our strategy is to go ahead and
% run it through the approximation, and to do a quick assertion to 
% check sanity. note that we just overwrite cN.
[cN,rapprox0] = constructnearestpsdcovariance(cN);
assert(rapprox0 > 0.99);

% calculate nearest approximation for the signal.
% this is the more critical case!
[cSb,rapprox] = constructnearestpsdcovariance(cS);

% scale the nearest approximation to match the average variance 
% that is observed in the original estimate of the signal covariance.
switch mode
case 0
  sc = posrect(mean(diag(cS))) / mean(diag(cSb));  % notice the posrect to ensure non-negative scaling
  cSb = constructnearestpsdcovariance(cSb * sc);   % impose scaling and run it through constructnearestpsdcovariance.m for good measure
case 1
  % do nothing
end

% perform Monte Carlo simulations
ncdist = zeros(1,numsim);
for rr=1:numsim
  signal = mvnrnd(mnS,cSb,ncond);         % cond x voxels
  noise = mvnrnd(mnN,cN,ncond*nctrials);  % ncond*nctrials x voxels
  measurement = signal + squish(mean(reshape(noise,[ncond nctrials nvox]),2),2);  % cond x voxels
  ncdist(rr) = comparefun(rdmfun(signal'),rdmfun(measurement'));
end

% if comparefun ever outputs NaN, set these cases to 0.
% for example, you might be correlating an all-zero signal
% with some data, which may result in NaN.
ncdist(isnan(ncdist)) = 0;

% compute median across simulations
nc = median(ncdist);

% prepare additional outputs
clear results;
varstosave = {'mnN' 'cN' 'mnS' 'cS' 'cSb' 'rapprox'};
for p=1:length(varstosave)
  results.(varstosave{p}) = eval(varstosave{p});
end
