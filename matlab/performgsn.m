function results = performgsn(data,opt) 

% function results = performgsn(data,opt) 
%
% <data> is voxels x conditions x trials. This indicates the measured
%   responses to different conditions on distinct trials. The number of 
%   trials must be at least 2.
% <opt> (optional) is a struct with the following optional fields:
%   <wantverbose> (optional) is whether to print status statements. Default: 1.
%   <wantshrinkage> (optional) is whether to use shrinkage in the estimation
%     of covariance. Default: 0.
%
% Perform GSN (generative modeling of signal and noise).
%
% Return:
%   <results> as a struct with:
%     mnN - the estimated mean of the noise (1 x voxels)
%     cN  - the raw estimated covariance of the noise (voxels x voxels)
%     cNb - the final estimated covariance after biconvex optimization
%     shrinklevelN - shrinkage level chosen for cN
%     shrinklevelD - shrinkage level chosen for the estimated data covariance
%     mnS - the estimated mean of the signal (1 x voxels)
%     cS  - the raw estimated covariance of the signal (voxels x voxels)
%     cSb - the final estimated covariance after biconvex optimization
%     ncsnr - the 'noise ceiling SNR' estimate for each voxel (1 x voxels).
%             This is, for each voxel, the std dev of the estimated signal
%             distribution divided by the std dev of the estimated noise
%             distribution. Note that this is computed on the raw
%             estimated covariances. Also, note that we apply positive 
%             rectification (to prevent non-sensical negative ncsnr values).
%
% History:
% - 2024/01/05 - (1) major change to use the biconvex optimization procedure --
%                    we now have cSb and cNb as the final estimates;
%                (2) cSb no longer has the scaling baked in and instead we 
%                    create a separate temporary variable cSb_rsa;
%                (3) remove the rapprox output
%
% Example:
% data = repmat(2*randn(100,40),[1 1 4]) + 1*randn(100,40,4);
% results = performgsn(data);

% inputs
if ~exist('opt','var') || isempty(opt)
  opt = struct;
end
if ~isfield(opt,'wantverbose') || isempty(opt.wantverbose)
  opt.wantverbose = 1;
end
if ~isfield(opt,'wantshrinkage') || isempty(opt.wantshrinkage)
  opt.wantshrinkage = 0;
end

% prepare opt for rsanoiseceiling.m
opt.mode = 1;
opt.ncsims = 0;
opt.wantfig = 0;
if opt.wantshrinkage
  opt.shrinklevels = [];  % this will allow default shrinkage levels
else
  opt.shrinklevels = 1;   % this forces only full estimation
end

% do it
[~,~,results] = rsanoiseceiling(data,opt);
results = rmfield(results,{'sc' 'splitr'});
