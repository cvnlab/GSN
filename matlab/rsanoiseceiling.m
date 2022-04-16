function [nc,ncdist,results] = rsanoiseceiling(data,opt) 

% function [nc,ncdist,results] = rsanoiseceiling(data,opt)
%
% <data> is voxels x conditions x trials. This indicates the measured
%   responses to different conditions on distinct trials. The number of 
%   trials must be at least 2.
% <opt> (optional) is a struct with the following optional fields:
%   <wantverbose> (optional) is whether to print status statements. Default: 1.
%   <rdmfun> (optional) is a function that constructs an RDM. Specifically,
%     the function should accept as input a data matrix (e.g. voxels x conditions)
%     and output a RDM with some dimensionality (can be a column vector, 2D matrix, etc.).
%     Default: @(data) pdist(data','correlation')'. This default simply 
%     calculates dissimilarity as 1-r, and then extracts the lower triangle
%     (excluding the diagonal) as a column vector.
%   <comparefun> (optional) is a function that quantifies the similarity of two RDMs.
%     Specifically, the function should accept as input two RDMs (in the format
%     that is returned by <rdmfun>) and output a scalar. Default: @corr.
%   <wantfig> (optional) is
%     0 means do not make a figure
%     1 means plot a figure in a new figure window
%     A means write a figure to filename prefix A (e.g., '/path/to/figure'
%       will result in /path/to/figure.png being written)
%     Default: 1.
%   <ncsims> (optional) is the number of Monte Carlo simulations to run for
%     the purposes of estimating the RSA noise ceiling. The final answer 
%     is computed as the median across simulation results. Default: 50.
%   <ncconds> (optional) is the number of conditions to simulate in the Monte
%     Carlo simulations. In theory, the RSA noise ceiling estimate should be 
%     invariant to number of conditions simulated, but the higher the number,
%     the more stable/accurate the results. Default: 50.
%   <nctrials> (optional) is the number of trials to target for the RSA noise 
%     ceiling estimate. For example, setting <nctrials> to 10 will result in 
%     the calculation of a noise ceiling estimate for the case in which 
%     responses are averaged across 10 trials per condition.
%     Default: size(data,3).
%   <splitmode> (optional) controls the way in which trials are divided
%     for the data reliability calculation. Specifically, <splitmode> is:
%     0 means use only the maximum split-half number of trials. In the case
%       of an odd number of trials T, we use floor(T/2).
%     1 means use numbers of trials increasing by a factor of 2 starting
%       at 1 and including the maximum split-half number of trials.
%       For example, if there are 10 trials total, we use [1 2 4 5].
%     2 means use the maximum split-half number of trials as well as half
%       of that number (rounding down if necessary). For example, if 
%       there are 11 trials total, we use [2 5].
%     The primary value of using multiple numbers of trials (e.g. options 
%     1 and 2) is to provide greater insight for the figure inspection that is
%     created. However, in terms of accuracy of RSA noise ceiling estimates,
%     option 0 should be fine (and will result in faster execution).
%     Default: 0.
%   <scs> (optional) controls the way in which the posthoc scaling factor
%     is determined. Specifically, <scs> is:
%     A where A is a vector of non-negative values. This specifies the
%       specific scale factors to evaluate. There is a trade-off between
%       speed of execution and the discretization/precision of the results.
%     Default: 0:.1:2.
%   <simchunk> (optional) is the chunk size for the data-splitting
%     simulations. Default: 50, which indicates to perform 50 simulations 
%     for each case, and then increment in steps of 50 if necessary to
%     achieve <simthresh>. Must be 2 or greater.
%   <simthresh> (optional) is the value for the robustness metric that must
%     be exceeded in order to halt the data-splitting simulations. 
%     The lower this number, the faster the execution time, but the less 
%     accurate the results. Default: 10.
%   <maxsimnum> (optional) is the maximum number of simulations to perform
%     for the data-splitting simulations. Default: 1000.
%
% Use the GSN (generative modeling of signal and noise) method to estimate
% an RSA noise ceiling.
%
% Note: if <comparefun> ever returns NaN, we automatically replace these
% cases with 0. This is a convenient workaround for degenerate cases that
% might arise, e.g., cases where the signal is generated as all zero.
%
% Return:
%   <nc> as a scalar with the noise ceiling estimate.
%   <ncdist> as 1 x <ncsims> with the result of each Monte Carlo simulation.
%     Note that <nc> is simply the median of <ncdist>.
%   <results> as a struct with additional details:
%     mnN - the estimated mean of the noise (1 x voxels)
%     cN  - the estimated covariance of the noise (voxels x voxels)
%     mnS - the estimated mean of the signal (1 x voxels)
%     cS  - the estimated covariance of the signal (voxels x voxels)
%     cSb - the regularized estimated covariance of the signal (voxels x voxels).
%           This estimate reflects both a nearest-approximation and 
%           a post-hoc scaling that is designed to match the data reliability
%           estimate. It is this regularized signal covariance that is
%           used in the Monte Carlo simulations.
%     rapprox - the correlation between the nearest-approximation of the 
%               signal covariance and the original signal covariance
%     sc - the post-hoc scaling factor that was selected
%     splitr - the data split-half reliability value that was obtained for the
%              largest trial number that was evaluated. (We return the median
%              result across the simulations that were conducted.)
%     ncsnr - the 'noise ceiling SNR' estimate for each voxel (1 x voxels).
%             This is, for each voxel, the std dev of the estimated signal
%             distribution divided by the std dev of the estimated noise
%             distribution. Note that this is computed on the originally
%             estimated signal covariance and not the regularized signal
%             covariance. Also, note that we apply positive rectification to 
%             the signal std dev (to prevent non-sensical negative ncsnr values).
%
% Example:
% data = repmat(2*randn(100,40),[1 1 4]) + 1*randn(100,40,4);
% [nc,ncdist,results] = rsanoiseceiling(data,struct('splitmode',1));

% internal options (not for general use):
%   <shrinklevels> (optional) is like the input to calcshrunkencovariance.m.
%     Default: [].
%   <mode> (optional) is
%     0 means use the data-reliability method
%     1 means use no scaling.
%     2 means scale to match the un-regularized average variance
%     Default: 0.

% inputs
if ~exist('opt','var') || isempty(opt)
  opt = struct;
end
if ~isfield(opt,'wantverbose') || isempty(opt.wantverbose)
  opt.wantverbose = 1;
end
if ~isfield(opt,'rdmfun') || isempty(opt.rdmfun)
  opt.rdmfun = @(data) pdist(data','correlation')';
end
if ~isfield(opt,'comparefun') || isempty(opt.comparefun)
  opt.comparefun = @corr;
end
if ~isfield(opt,'wantfig') || isempty(opt.wantfig)
  opt.wantfig = 1;
end
if ~isfield(opt,'ncsims') || isempty(opt.ncsims)
  opt.ncsims = 50;
end
if ~isfield(opt,'ncconds') || isempty(opt.ncconds)
  opt.ncconds = 50;
end
if ~isfield(opt,'nctrials') || isempty(opt.nctrials)
  opt.nctrials = size(data,3);
end
if ~isfield(opt,'splitmode') || isempty(opt.splitmode)
  opt.splitmode = 0;
end
if ~isfield(opt,'scs') || isempty(opt.scs)
  opt.scs = 0:.1:2;
end
if ~isfield(opt,'simchunk') || isempty(opt.simchunk)
  opt.simchunk = 50;
end
if ~isfield(opt,'simthresh') || isempty(opt.simthresh)
  opt.simthresh = 10;
end
if ~isfield(opt,'maxsimnum') || isempty(opt.maxsimnum)
  opt.maxsimnum = 1000;
end
if ~isfield(opt,'shrinklevels') || isempty(opt.shrinklevels)
  opt.shrinklevels = [];
end
if ~isfield(opt,'mode') || isempty(opt.mode)
  opt.mode = 0;
end

% calc
nvox   = size(data,1);
ncond  = size(data,2);
ntrial = size(data,3);

% deal with massaging inputs and sanity checks
assert(ntrial >= 2);
opt.scs = unique(opt.scs);
assert(all(opt.scs>=0));
assert(opt.simchunk >= 2);

%% %%%%% ESTIMATION OF COVARIANCES

% estimate noise covariance
if opt.wantverbose, fprintf('Estimating noise covariance...');, end
[mnN,cN,shrinklevelN,nllN] = calcshrunkencovariance(permute(data,[3 1 2]),[],opt.shrinklevels,1);
if opt.wantverbose, fprintf('done.\n');, end

% estimate data covariance
if opt.wantverbose, fprintf('Estimating data covariance...');, end
[mnD,cD,shrinklevelD,nllD] = calcshrunkencovariance(mean(data,3)',        [],opt.shrinklevels,1);
if opt.wantverbose, fprintf('done.\n');, end

% estimate signal covariance
if opt.wantverbose, fprintf('Estimating signal covariance...');, end
mnS = mnD - mnN;
cS  =  cD - cN/ntrial;
if opt.wantverbose, fprintf('done.\n');, end

% prepare some outputs
sd_noise = sqrt(diag(cN))';   % std of the noise (1 x voxels)
sd_signal = sqrt(posrect(diag(cS)))';  % std of the signal (1 x voxels)
ncsnr = sd_signal ./ sd_noise;   % noise ceiling SNR (1 x voxels)

%% %%%%% REGULARIZATION OF COVARIANCES

if opt.wantverbose, fprintf('Regularizing...');, end

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

% deal with scaling of the signal covariance matrix
switch opt.mode
case 1
  % do nothing
  sc = 1;
  splitr = [];
case 2
  % scale the nearest approximation to match the average variance 
  % that is observed in the original estimate of the signal covariance.
  sc = posrect(mean(diag(cS))) / mean(diag(cSb));  % notice the posrect to ensure non-negative scaling
  cSb = constructnearestpsdcovariance(cSb * sc);   % impose scaling and run it through constructnearestpsdcovariance.m for good measure
  splitr = [];
case 0

  % calculate the number of trials to put into the two splits
  switch opt.splitmode
  case 0
    splitnums = [floor(ntrial/2)];
  case 1
    splitnums = unique([2.^(0:floor(log2(ntrial/2))) floor(ntrial/2)]);
  case 2
    splitnums = [floor(floor(ntrial/2)/2) floor(ntrial/2)];
    splitnums = splitnums(splitnums > 0);
  end

  % calculate data split reliability
  if opt.wantverbose, fprintf('Calculating data split reliability...');, end
  iicur = 1;               % current sim number
  iimax = opt.simchunk;    % current targeted max
  datasplitr = zeros(length(splitnums),iimax);
  while 1
    for nn=1:length(splitnums)
      for si=iicur:iimax
        temp = randperm(ntrial);
        datasplitr(nn,si) = nanreplace(opt.comparefun(opt.rdmfun(mean(data(:,:,temp(1:splitnums(nn))),3)), ...
                                                  opt.rdmfun(mean(data(:,:,temp(splitnums(nn)+(1:splitnums(nn)))),3))));
      end
    end
    temp = datasplitr;
    robustness = mean(abs(median(temp,2)) ./ ((iqr(temp,2)/2)/sqrt(size(temp,2))));
    if robustness > opt.simthresh
      break;
    end
    iicur = iimax + 1;
    iimax = iimax + opt.simchunk;
    if iimax > opt.maxsimnum
      break;
    end
    datasplitr(1,iimax) = 0;  % pre-allocate
  end
  splitr = median(datasplitr(end,:),2);  % 1 x 1 with the median result for the "most trials" data split case
  if opt.wantverbose, fprintf('done.\n');, end

  % calculate model-based split reliability
  if opt.wantverbose, fprintf('Calculating model split reliability...');, end
  iicur = 1;               % current sim number
  iimax = opt.simchunk;    % current targeted max
  modelsplitr = zeros(length(opt.scs),length(splitnums),iimax);
    % precompute
  tempcS = zeros([size(cSb) length(opt.scs)]);
  for sci=1:length(opt.scs)
    tempcS(:,:,sci) = constructnearestpsdcovariance(cSb*opt.scs(sci));
  end
  while 1
    robustness = zeros(1,length(opt.scs));
    for sci=1:length(opt.scs)
      for nn=1:length(splitnums)
        for si=iicur:iimax
          signal = mvnrnd(mnS,tempcS(:,:,sci),opt.ncconds);         % cond x voxels
          noise  = mvnrnd(mnN,cN/splitnums(nn),opt.ncconds*2);      % 2*cond x voxels
          measurement1 = signal + noise(1:opt.ncconds,:);           % cond x voxels
          measurement2 = signal + noise(opt.ncconds+1:end,:);       % cond x voxels
          modelsplitr(sci,nn,si) = nanreplace(opt.comparefun(opt.rdmfun(measurement1'),opt.rdmfun(measurement2')));
        end
      end
      temp = squish(modelsplitr(sci,:,:),2);
      robustness(sci) = mean(abs(median(temp,2)) ./ ((iqr(temp,2)/2)/sqrt(size(temp,2))));
    end
    robustness = mean(robustness);
    if robustness > opt.simthresh
      break;
    end
    iicur = iimax + 1;
    iimax = iimax + opt.simchunk;
    if iimax > opt.maxsimnum
      break;
    end
    modelsplitr(1,1,iimax) = 0;  % pre-allocate
  end
  if opt.wantverbose, fprintf('done.\n');, end

  % calculate R^2 between model-based results and the data results and find the max
  if opt.wantverbose, fprintf('Finding best model...');, end
  R2s = calccod(median(modelsplitr,3),repmat(median(datasplitr,2)',[length(opt.scs) 1]),2,[],0);  % scales x 1
  [~,bestii] = max(R2s);
  sc = opt.scs(bestii);
  if opt.wantverbose, fprintf('done.\n');, end
  
  % impose scaling and run it through constructnearestpsdcovariance.m for good measure
  cSb = constructnearestpsdcovariance(cSb * sc);
  
end

if opt.wantverbose, fprintf('done.\n');, end

%% %%%%% MONTE CARLO SIMULATIONS FOR RSA NOISE CEILING

if opt.wantverbose, fprintf('Performing Monte Carlo simulations...');, end

% perform Monte Carlo simulations
ncdist = zeros(1,opt.ncsims);
for rr=1:opt.ncsims
  signal = mvnrnd(mnS,cSb,opt.ncconds);              % ncconds x voxels
  noise  = mvnrnd(mnN,cN/opt.nctrials,opt.ncconds);  % ncconds x voxels
  measurement = signal + noise;                      % ncconds x voxels
  ncdist(rr) = nanreplace(opt.comparefun(opt.rdmfun(signal'),opt.rdmfun(measurement')));
end

if opt.wantverbose, fprintf('done.\n');, end

%% %%%%% FINISH UP

% compute median across simulations
nc = median(ncdist);

% prepare additional outputs
clear results;
varstosave = {'mnN' 'cN' 'mnS' 'cS' 'cSb' 'rapprox' 'sc' 'splitr' 'ncsnr'};
for p=1:length(varstosave)
  results.(varstosave{p}) = eval(varstosave{p});
end

%% %%%%% MAKE A FIGURE

if opt.mode == 0 && ~isequal(opt.wantfig,0)

  if opt.wantverbose, fprintf('Creating figure...');, end

  if isequal(opt.wantfig,1)
    figure; setfigurepos([100 100 750 750]);
  else
    figureprep([100 100 750 750]);  % this makes an invisible figure window
  end

  subplot(4,6,[1 2]); hold on;
  hist(mnS);
  ylabel('Frequency');
  title('Mean of Signal');
  
  subplot(4,6,[3 4]); hold on;
  mx = max(abs(cS(:)));
  imagesc(cS,[-mx mx]); axis image tight; set(gca,'YDir','reverse'); colormap(parula); colorbar;
  title('Covariance of Signal');

  subplot(4,6,[5 6]); hold on;
  mx = max(abs(cSb(:)));
  imagesc(cSb,[-mx mx]); axis image tight; set(gca,'YDir','reverse'); colormap(parula); colorbar;
  title('Regularized and scaled');
  
  subplot(4,6,[7 8]); hold on;
  hist(mnN);
  ylabel('Frequency');
  title('Mean of Noise');

  subplot(4,6,[9 10]); hold on;
  mx = max(abs(cN(:)));
  imagesc(cN,[-mx mx]); axis image tight; set(gca,'YDir','reverse'); colormap(parula); colorbar;
  title('Covariance of Noise');

  subplot(4,6,[11 12]); hold on;
  hist(ncsnr);
  ylabel('Frequency');
  title('Noise ceiling SNR');

  subplot(4,6,[13 14 15 19 20 21]); hold on;
  cmap0 = parula(length(opt.scs));  % cmapturbo
  hs = [];
  for sci=1:length(opt.scs)
    md0 = median(modelsplitr(sci,:,:),3);  % 1 x n
    se0 = (iqr(modelsplitr(sci,:,:),3)/2) ./ sqrt(size(modelsplitr,3));  % 1 x n
    h0 = errorbar2(splitnums,md0,se0,'v','r-');
    set(h0,'Color',cmap0(sci,:));
    hs = [hs h0];
    if opt.scs(sci)==sc
      lw0 = 3;
      mark0 = 'o';
    else
      lw0 = 1;
      mark0 = 'x';
    end
    plot(splitnums,md0,['r' mark0 '-'],'Color',cmap0(sci,:),'LineWidth',lw0);
  end
  uistack(hs,'bottom');
  md0 = median(datasplitr,2)';
  sd0 = (iqr(datasplitr,2)/2)';
  se0 = sd0 ./ sqrt(size(datasplitr,2));
  set(errorbar2(splitnums,md0,sd0,'v','k-'),'LineWidth',1);
  set(errorbar2(splitnums,md0,se0,'v','k-'),'LineWidth',3);
  plot(splitnums,md0,'kd-','LineWidth',3);
  xlim([min(splitnums)-1 max(splitnums)+1]);
  set(gca,'XTick',unique(round(get(gca,'XTick'))));
  xlabel('Number of trials in each split');
  ylabel('Similarity (comparefun output)');
  title(sprintf('Data (%d sims); Model (%d sims); splitr=%.3f',size(datasplitr,2),size(modelsplitr,3),splitr));

  subplot(4,6,[16 17 18 22 23 24]); hold on;
  plot(opt.scs,R2s,'ro-');
  straightline(sc,'v','k-');
  xlabel('Scaling factor');
  ylabel('R^2 between model and data (%)');
  title(sprintf('rapprox=%.2f, sc=%.2f, nc=%.3f +/- %.3f',rapprox,sc,nc,iqr(ncdist)/2/sqrt(length(ncdist))));
  
  if isequal(opt.wantfig,1)
  else
    [dir0,file0] = stripfile(opt.wantfig);
    figurewrite(file0,[],[],dir0);
  end

  if opt.wantverbose, fprintf('done.\n');, end

end
