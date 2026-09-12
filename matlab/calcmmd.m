function results = calcmmd(cSb,cNb,numpairs,noisepct)

% function results = calcmmd(cSb,cNb,numpairs,noisepct)
%
% <cSb> and <cNb> are outputs from performgsn.m
% <numpairs> (optional) is the number of pairs for which to calculate
%   distances. Default: 1000.
% <noisepct> (optional) is a non-empty vector of percentages. We apply 
%   whitening for the number of noise dimensions that achieves each
%   percentage of variance of the noise distribution. If you supply
%   more than one percentage, we automatically sort them. Default: 90.
% 
% Return <results> as a struct with:
%   <vSb> and <dSb> as eigenvectors and eigenvalues of <cSb>
%   <vNb> and <dNb> as eigenvectors and eigenvalues of <cNb>
%   <ncsnr> as median ratio of signal std to noise std
%   <totvarsignal> as sum of all signal variances
%   <totvarnoise> as sum of all noise variances
%   <totvarsnr> as ratio of <totvarsignal> and <totvarnoise>
%   <EDsignal> as the effective dimensionality of <cSb>
%   <EDnoise> as the effective dimensionality of <cNb>
%   <totvarsignalALT> as sum of all signal variances
%     after normalization such that noise variances are 1
%   <totvarnoiseALT> as sum of all noise variances (after normalization)
%   <totvarsnrALT> as ratio of <totvarsignalALT> and <totvarnoiseALT>
%   <med> as median Euclidean distance
%   <mmd_uncorr> as median Mahalanobis distance (ignoring any noise correlations)
%   <mmd> as a 1 x length(<noisepct>) vector with median Mahalanobis distances,
%     whitening only the noise dimensions corresponding to <noisepct>
%
% Note that for <vSb>, <dSb>, <vNb>, and <dNb>, eigenvalues are provided 
% in descending order with all eigenvalues forced to be real and non-negative.
%
% Example:
% data = repmat(2*randn(100,300),[1 1 4]) + 1*randn(100,300,4);
% results = performgsn(data);
% results2 = calcmmd(results.cSb,results.cNb)

% inputs
if ~exist('numpairs','var') || isempty(numpairs)
  numpairs = 1000;
end
% if ~exist('wantfig','var') || isempty(wantfig)
%   wantfig = 1;
% end
if ~exist('noisepct','var') || isempty(noisepct)
  noisepct = 90;  %[50 75 90 95];
end

% deal with inputs
noisepct = sort(noisepct);

% constants
edfun = @(x) sum(x)^2/sum(x.^2);

% eigendecomposition of signal
[vSb,dSb] = eig(cSb,'vector');
dSb = posrect(real(dSb));
[~,ix] = sort(dSb,'descend');
dSb = dSb(ix);
vSb = vSb(:,ix);

% eigendecomposition of noise
[vNb,dNb] = eig(cNb,'vector');
dNb = posrect(real(dNb));
[~,ix] = sort(dNb,'descend');
dNb = dNb(ix);
vNb = vNb(:,ix);

%% simple metrics

% ncsnr
ncsnr = median(sqrt(diag(cSb)./diag(cNb)));

% total variance
totvarsignal = sum(diag(cSb));
totvarnoise = sum(diag(cNb));
totvarsnr = totvarsignal / totvarnoise;

% ED
EDsignal = edfun(dSb);
EDnoise = edfun(dNb);

%% proceed to MED

% draw random samples from signal distribution
pts = mvnrnd(zeros(1,size(cSb,1)),cSb,2*numpairs)';  % dim x 2*N

% reshape
pts2 = reshape(pts,size(pts,1),[],2);  % dim x N x 2

% calculate median Euclidean distance
dist = sqrt(sum(diff(pts2,[],3).^2,1));
med = median(dist);

%% proceed to MMD

% construct normalization matrix
T = sqrt(diag(cNb));
A = T*T';

% divide by normalization matrix such that noise variances equal 1
cNc = cNb ./ A;
cSc = cSb ./ A;

% total variance (alternative)
totvarsignalALT = sum(diag(cSc));
totvarnoiseALT = sum(diag(cNc));
totvarsnrALT = totvarsignalALT / totvarnoiseALT;

% draw random samples from signal distribution
pts = mvnrnd(zeros(1,size(cSc,1)),cSc,2*numpairs)';  % dim x 2*N

% reshape
pts2 = reshape(pts,size(pts,1),[],2);  % dim x N x 2

% calculate median Mahalanobis distance assuming the noise is uncorrelated
dist = sqrt(sum(diff(pts2,[],3).^2,1));
mmd_uncorr = median(dist);
  %se = std(bootstrp(1000,@median,dist'));

%% do the full version of MMD

% eigendecomposition of the normalized noise
[vNc,dNc] = eig(cNc,'vector');
dNc = posrect(real(dNc));
[~,ix] = sort(dNc,'descend');
dNc = dNc(ix);
vNc = vNc(:,ix);

% compute cum sum of noise variance
curv = cumsum(dNc)/sum(dNc)*100;

% % start figure
% if wantfig
%   figure;
%   subplot(2,1,1); hold on;
%   bar(curv,1);
%   ylim([0 100]);
%   ylabel('Percent noise variance');
% end

% loop over how much of the noise to take into account
mmd = [];
for p=1:length(noisepct)

  % find number of dimensions to retain
  iix = find(curv >= noisepct(p));
  assert(length(iix)>=1);
%   subplot(2,1,1); hold on;
%   straightline(iix(1),'v','r-');
  
  % construct vector of scalings
  elt = sqrt(1 ./ dNc);
  elt(iix(1)+1:end) = 1;
%   subplot(2,1,2); hold on;
%   plot(log2(elt));

  % construct noise whitening matrix
  wmatrix = vNc * diag(elt) * vNc';
  
  % multiply and reshape
  pts2 = reshape(wmatrix*pts,size(pts,1),[],2);  % dim x N x 2
  
  % calculate median Mahalanobis distance
  dist = sqrt(sum(diff(pts2,[],3).^2,1));
  mmd(p) = median(dist);

end

%% deal with outputs

clear results;
varstosave = ...
{'vSb' 'dSb' 'vNb' 'dNb' ...
 'ncsnr' 'totvarsignal' 'totvarnoise' 'totvarsnr' ...
 'EDsignal' 'EDnoise' ...
 'totvarsignalALT' 'totvarnoiseALT' 'totvarsnrALT' ...
 'med' ...
 'mmd_uncorr' ...
 'mmd'};
for p=1:length(varstosave)
  results.(varstosave{p}) = eval(varstosave{p});
end
