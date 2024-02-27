function [W, H, nmfV] = NMF(V, parameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: NMF
% Date: May 2018
% Programmer: Christian Dittmar
%
% Given a non-negative matrix V, find non-negative templates W and activations
% H that approximate V.
%
% References:
% [1] Lee, DD & Seung, HS. "Algorithms for Non-negative Matrix Factorization"
%
% [2] Andrzej Cichocki, Rafal Zdunek, Anh Huy Phan, and Shun-ichi Amari
% "Nonnegative Matrix and Tensor Factorizations: Applications to
% Exploratory Multi-Way Data Analysis and Blind Source Separation"
% John Wiley and Sons, 2009.
%
% Input:  V               K x M non-negative matrix to be facorized%
%         parameter.
%           costFunc      Cost function used for the optimization, currently
%                         supported are:
%                         'EucDdist' for Euclidean Distance
%                         'KLDiv' for Kullback Leibler Divergence
%                         'ISDiv' for Itakura Saito Divergence
%           numIter       Number of iterations the algorithm will run.
%           numComp       the rank of the approximation
%
% Output: W               K x R non-negative templates
%         H               R x M non-negative activations
%         nmfV            cell-array with approximated component matrices
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If you use the 'NMF toolbox' please refer to:
% [3]  Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
%     Müller
%     NMF Toolbox: Music Processing Applications of Nonnegative Matrix
%     Factorization
%     In Proceedings of the International Conference on Digital Audio Effects
%     (DAFx), 2019.
%
% License:
% This file is part of 'NMF toolbox'.
% https://www.audiolabs-erlangen.de/resources/MIR/NMFtoolbox/
% 'NMF toolbox' is free software: you can redistribute it and/or modify it
% under the terms of the GNU General Public License as published by the
% the Free Software Foundation, either version 3 of the License, or (at
% your option) any later version.
%
% 'NMF toolbox' is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
% Public License for more details.
%
% You should have received a copy of the GNU General Public License along
% with 'NMF toolbox'. If not, see http://www.gnu.org/licenses/.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isfield(parameter,'costFunc')
  parameter.costFunc = 'KLDiv';
end
if ~isfield(parameter,'numIter')
  parameter.numIter = 30;
end
if ~isfield(parameter, 'fixW')
    parameter.fixW = 0;
end

% get important params
[K, M] = size(V);
R = parameter.numComp;
L = parameter.numIter;

% initialization of W and H
if iscell(parameter.initW)
    W = cell2mat(parameter.initW);
elseif isnumeric(parameter.initW)
    W = parameter.initW;
end;

H = parameter.initH;

% create helper matrix of all ones
onesMatrix = ones(K, M);

% normalize to unit sum
V = V./(eps+sum(V(:)));

% maiin iterations
for iter = 1:L

  % compute approximation
  Lambda = eps+(W*H);

  % switch between pre-defined update rules
  switch parameter.costFunc

    case 'EucDist' % euclidean update rules
        if ~parameter.fixW
            W = W .* (V * H' ./ ((Lambda * H') + eps));
        end
      H = H .* (W' * V ./ ((W' * Lambda) + eps));

    case 'KLDiv' % Kullback Leibler divergence update rules
        if ~parameter.fixW
            W = W .* ((V ./ Lambda) * H') ./ (onesMatrix*H'+eps);
        end
      H = H .* (W' * (V ./ Lambda)) ./ (W'*onesMatrix+eps);

    case 'ISDiv' % Itakura Saito divergence update rules
        if ~parameter.fixW
            W = W .* ((Lambda.^-2 .* V) * H') ./ (Lambda.^-1 * H' + eps);
        end
      H = H .* (W' * (Lambda.^-2 .* V)) ./ (W' * Lambda.^-1 + eps);

    otherwise
      error('Unknown cost function');

  end

  % normalize templates to unit sum
  if ~parameter.fixW
    normVec = sum(W,1);
    W = bsxfun(@times,W,1./(eps+normVec));
  end

end

% compute final output approximation
for r = 1:R
  nmfV{r} = W(:,r)*H(r,:);
end

end
