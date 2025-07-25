function [mn,c,shrinklevel,nll] = calcshrunkencovariance(data,leaveout,shrinklevels,wantfull)

% function [mn,c,shrinklevel,nll] = calcshrunkencovariance(data,leaveout,shrinklevels,wantfull)
%
% <data> is a matrix with dimensions observations x variables.
%   There can be more variables than observations. In addition, the 
%   dimensions of <data> can be observations x variables x cases (where
%   the number of cases at least 2). In this special scenario where
%   multiple cases are specified, we perform handling of "mean-subtraction"
%   for each case (see details below) and it is acceptable that
%   some values may be NaN (see more information below).
% <leaveout> (optional) is N >= 2 which means leave out 1/N of the data for
%   cross-validation purposes. The selection of data points is random.
%   Default: 5.
% <shrinklevels> (optional) is a non-empty vector (1 x F) of shrinkage fractions
%   (between 0 and 1 inclusive) to evaluate. For example, 0.8 means to
%   shrink values to 80% of their original size. Default: 0:.02:1.
% <wantfull> (optional) is whether to use the identified optimal shrinkage
%   fraction to re-estimate the mean and covariance using the full dataset
%   (i.e. including the initially left-out data). Default: 0.
%
% Using (N-1)/N of the data (randomly selected), calculate a covariance 
% matrix and shrink the off-diagonal elements to 0 according to
% <shrinklevels>. (The diagonal elements are left untouched.) The shrinkage
% level that maximizes the likelihood (i.e., minimizes the negative log 
% likelihood) of the left-out 1/N of the data is chosen. If <wantfull>,
% we re-estimate the mean and covariance using all of the data and the
% identified shrinkage level. If not <wantfull>, we just return the 
% shrunken estimate from (N-1)/N of the data.
%
% Note that we try to detect pathological cases in the case of multiple
% <shrinklevels>, and if detected, we will issue warning messages.
%
% The case where <data> has multiple cases along the third dimension
% is useful for when you have multiple sets of measurements, each of
% which has an unknown mean. In this scenario, we calculate the covariance
% of each case separately (thereby ignoring the mean of each sample)
% and then average (pool) covariance estimates across cases.
% Note that in this special scenario, we perform cross-validation on 
% cases (not observations), the returned <mn> is necessarily all zero,
% and the left-out data are also mean-subtracted before evaluating the
% cross-validated likelihood of covariance estimates.
%
% Special case of uneven number of trials:
% - When multiple cases are provided in <data>, it is acceptable that
%   some of the rows in <data> are NaN. Conceptually, this is interpreted
%   as allowing for the possibility that different cases have different
%   numbers of observations. To be specific, it is okay if data(i,:,j)
%   consists of NaNs for some combination(s) of i and j. However, it must
%   be the case that each case must contain at least one set of observations
%   with valid data (i.e. data(:,:,j) must contain at least one row of
%   valid data). Also, bear in mind that training/testing splits must 
%   have at least one condition with at least two sets of observations with 
%   valid data --- don't worry, since if this fails to be true, we will 
%   issue an error.
% - The strategy for uneven number of trials is simply to estimate
%   noise covariance for each condition (ignoring missing trials)
%   and then average the results across conditions.
%
% Return:
%  <mn> as 1 x variables with the estimated mean
%  <c> as variables x variables with the estimated covariance matrix
%  <shrinklevel> as the shrinkage fraction that was chosen
%  <nll> as 1 x F with the mean negative log likelihood on the left-out data.
%    one or more values can be NaN (e.g. singular covariance matrices).
%
% Example:
% numvar = 100;     % number of variables
% for n=[90 1000]   % number of observations
%   sigma = 0.5*ones(numvar);
%   sigma(logical(eye(numvar))) = 1;
%   x = randnmulti(n,[],sigma,[]);
%   xcov = cov(x);
%   [mn,c,shrinklevel,nll] = calcshrunkencovariance(x);
%   figureprep([100 100 800 300],1);
%   subplot(1,3,1);
%   imagesc(xcov); colormap(jet); colorbar; axis image tight;
%   title(sprintf('original'));
%   subplot(1,3,2);
%   imagesc(c);    colormap(jet); colorbar; axis image tight;
%   title(sprintf('shrinkage = %.2f',shrinklevel));
%   subplot(1,3,3);
%   plot(0:.02:1,nll,'ro-');
%   title('mean negative log likelihood');
% end

% inputs
if ~exist('leaveout','var') || isempty(leaveout)
  leaveout = 5;
end
if ~exist('shrinklevels','var') || isempty(shrinklevels)
  shrinklevels = linspace(0,1,51);  % 1 means use full data (no shrinkage)
end
if ~exist('wantfull','var') || isempty(wantfull)
  wantfull = 0;
end

% check whether we are in the special case of uneven trials across conditions
isuneven = any(isnan(data(:)));
if isuneven  % if it seems like it is, let's do some stringent sanity checks
  assert(ndims(data) == 3,'NaNs are allowed only in the multi-case scenario (3D data required)');
  assert(size(data,3) > 1,'NaNs are allowed only in the multi-case scenario (number of cases at least 2)');
  validcnt = sum(~any(isnan(data),2),1);  % 1 x 1 x cases with number of rows that DO NOT have NaNs
  assert(all(validcnt(:) >= 1),'all conditions must have at least 1 valid trial (no NaNs)');
  
  % Additional check for edge cases: ensure we have enough conditions with >=2 trials for cross-validation
  repconds = sum(validcnt(:) >= 2);
  assert(repconds >= 2, ...
    sprintf('need at least 2 conditions with 2+ trials for cross-validation, but only have %d', repconds));
end

% handle special scenario
if size(data,3) > 1

  % divide into training and validation
  permuted_indices = deterministic_randperm(size(data,3));
  validation_size = round((1/leaveout)*size(data,3));
  ii = permuted_indices(1:validation_size);
  iinot = permuted_indices(validation_size+1:end);

  % handle the regular case of all valid trials
  if ~isuneven
  
    % calculate covariance from the training data (variables x variables).
    % notice that we separate compute covariance from each case (thereby
    % ignoring the mean of each variable within each case), and then
    % average across the results of each case.
    c = 0;
    for pp=1:length(iinot)
      c = c + cov(data(:,:,iinot(pp))) / length(iinot);
    end
  
  else

    % same idea, but tread with caution
    c = 0;
    validcnt = 0;  % how many conditions actually have at least 2 trials (so we can calculate covariance)
    for pp=1:length(iinot)
      validix = ~any(isnan(data(:,:,iinot(pp))),2);  % logical indicating rows that have valid data
      if sum(validix) > 1
        validcnt = validcnt + 1;
        c = c + cov(data(:,:,iinot(pp)),'omitrows');
      end
    end
    assert(validcnt >= 1,'training data did not have a condition with at least two valid observations');
    c = c / validcnt;

  end

  % we know the mean from the training data is just zero (1 x variables)
  mn = zeros(1,size(data,2),class(data));

% otherwise, do the regular thing
else

  % divide into training and validation
  permuted_indices = deterministic_randperm(size(data,1));
  validation_size = round((1/leaveout)*size(data,1));
  ii = permuted_indices(1:validation_size);
  iinot = permuted_indices(validation_size+1:end);

  % calculate covariance from the training data (variables x variables)
  c = cov(data(iinot,:));

  % calculate the mean from the training data (1 x variables)
  mn = mean(data(iinot,:),1);

end

% calc indices of the diagonal elements
diagix = find(eye(size(c,1)));

% try different shrinkage levels
nll = zeros(1,length(shrinklevels),class(data));           % mean negative log likelihood
covs = zeros([size(c) length(shrinklevels)],class(data));  % various covariance matrices
for p=1:length(shrinklevels)

  % shrink the covariance (off-diagonal elements mix with 0)
  c2 = c*shrinklevels(p);
  c2(diagix) = c(diagix);  % preserve the diagonal elements
  if 0
    figure;imagesc(c2,[-.5 .5]);colormap(cmapsign4);title(sprintf('%.3f',shrinklevels(p)));drawnow;
  end

  % record
  covs(:,:,p) = c2;

  % evaluate the PDF on the validation data, obtaining log likelihoods
  if size(data,3) > 1

    % handle the regular case of all valid trials
    if ~isuneven
      [pr,err] = calcmvgaussianpdf(squish(permute(zeromean(data(:,:,ii),1),[1 3 2]),2),mn,c2,1);  % manually deal with mean

    % handle very tricky case
    else
      datastore = cast([],class(data));  % observations x instances
      for q=1:length(ii)
        temp = data(:,:,ii(q));  % observations x variables
        validix = ~any(isnan(temp),2);  % observations x 1 indicating valid data
        if sum(validix) > 1  % if we have at least two, let's use for the likelihood
          datastore = cat(1,datastore,zeromean(temp(validix,:),1));
        end
      end
      assert(~isempty(datastore),'validation data did not have any conditions with at least two observations');
      [pr,err] = calcmvgaussianpdf(datastore,mn,c2,1);  % manually deal with mean
    end
    
  else
    [pr,err] = calcmvgaussianpdf(data(ii,:),mn,c2,1);
  end
  if err ~= 0
    nll(p) = NaN;
    continue;
  end

  % calculate mean negative log likelihood (lower values mean higher probabilities)
  nll(p) = mean(-pr);
  
end

% which achieves the minimum?
[min0,min0ix] = min(nll);
shrinklevel = shrinklevels(min0ix);

% do some error checking (only in the case of multiple shrinkage levels)
if length(nll) > 1
  if all(isnan(nll))
    warning('all covariance matrices were singular');
  elseif length(unique(nll(isfinite(nll)))) == 1 && size(c,1)>1
    warning('there was only one unique finite log-likelihood; something might be wrong?');
  elseif ~isfinite(min0)
    warning('selected likelihood is not finite; something might be wrong?');
  end
end

%% %%%%% PREPARE FINAL OUTPUT

% in this case, we have to re-estimate using the full dataset
if wantfull

  % handle special scenario
  if size(data,3) > 1
  
    % handle the regular case of all valid trials
    if ~isuneven

      % let's be efficient and re-use previous calculations
      c = c * (length(iinot) / size(data,3));
      for pp=1:length(ii)
        c = c + cov(data(:,:,ii(pp))) / size(data,3);
      end
    
    else

      % same idea, but tread with caution
      c = c * validcnt;
      for pp=1:length(ii)
        validix = ~any(isnan(data(:,:,ii(pp))),2);  % logical indicating rows that have valid data
        if sum(validix) > 1
          validcnt = validcnt + 1;
          c = c + cov(data(:,:,ii(pp)),'omitrows');
        end
      end
      c = c / validcnt;

    end
    
    % the mean just stays zero, no problem
  
  % handle regular case
  else
  
    % calculate mean and covariance from all the data
    c = cov(data);
    mn = mean(data,1);
    
  end

  % apply shrinkage
  c2 = c*shrinklevel;
  c2(diagix) = c(diagix);
  c = c2;

% otherwise, we just extract the desired shrunken estimate
else
  c = covs(:,:,min0ix);
end

%%%%% JUNK:

  % if singular, just give up (mvnpdf does not work on singular matrices)
%   if cond(c2) > condthresh
% 
%     nll(p) = NaN;
%   
%   else
  % note that nll(p) might be Inf (if there is a data point with 0 probability)
%   end
%or Inf (which indicates
%    that one or more data points had 0 probability). 

% if ~exist('condthresh','var') || isempty(condthresh)
%   condthresh = 1e6;  % any condition number greater than this is unacceptable
% end

% <condthresh> (optional) is the maximum condition number. Any covariance
%   matrix with a condition larger than this is deemed invalid and
%   is therefore skipped. Default: 1e6.


%   try
%     if size(data,3) > 1
%       pr = mvnpdfNOEXP(squish(permute(zeromean(data(:,:,ii),1),[1 3 2]),2),mn,c2);  % manually deal with mean
%     else
%       pr = mvnpdfNOEXP(data(ii,:),mn,c2);
%     end
%   catch ME
%     if isequal(ME.identifier,'stats:mvnpdf:BadMatrixSigma')
%       nll(p) = NaN;
%       continue;
%     else
%       rethrow(ME);
%     end
%   end
