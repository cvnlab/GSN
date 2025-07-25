function [c,rapprox] = constructnearestpsdcovariance(c)

% function [c,rapprox] = constructnearestpsdcovariance(c)
%
% <c> is a square matrix (N x N)
%
% Use the method of Higham 1988 to obtain the nearest symmetric
% positive semidefinite matrix to <c> (in the Frobenius norm).
%
% Return:
%  <c> as the approximating matrix
%  <rapprox> as the correlation between the original matrix
%    and the approximating matrix.
%
% example:
% c1 = cov(randn(100,10));
% c1(1,1) = -1;
% [c2,rapprox] = constructnearestpsdcovariance(c1);
% figure; imagesc(c1,[-2 2]);
% figure; imagesc(c2,[-2 2]);
% [T,err] = cholcov(c1); err
% [T,err] = cholcov(c2); err
% rapprox

% ensure symmetric
c = (c+c')/2;

% check if PSD
[T,err] = cholcov(c);

% if err is 0, we don't have to do anything!
if err == 0

  % just set
  rapprox = 1;

% if err is not 0, we have to do some work
else

  % construct nearest PSD matrix (with respect to Frobenius norm)
  try
    [u,s,v] = svd(c,0);
    c2 = (c + v*s*v')/2;  % average with symmetric polar factor
  catch  % in some rare cases, the svd fails to converge.
         % in that case, let's resort to this approach:
    [v,d] = eig(c);
    d(d<0) = 0;
    c2 = v*d*v';
  end
  c2 = (c2+c2')/2;  % ensure symmetric again

  % check that it is indeed PSD
  [T,err] = cholcov(c2);
  if err ~= 0
    c2 = c2 + eye(size(c2))*1e-10;           % add a small ridge
    c2 = constructnearestpsdcovariance(c2);  % recurse
  end

  % calculate how good the approximation is
  rapprox = corr(c(:),c2(:));
  
  % replace
  c = c2;
  
end
