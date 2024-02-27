function [ W, H, cnmfY, costFunc ] = NMFconv( V, parameter )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: NMFconv
% Date: March 2018
% Programmer: Christian Dittmar
%
% Convolutive Non-Negative Matrix Factorization with Beta-Divergence and
% optional regularization parameters as described in chapter 3.7 of [1].
% The averaged activation updates are computed via the compact algorithm
% given in paragraph 3.7.3. For the sake of consistency, we use the notation
% from [2] instead of the one from the book.
%
% References:
% [1] Andrzej Cichocki, Rafal Zdunek, Anh Huy Phan, and Shun-ichi Amari
% "Nonnegative Matrix and Tensor Factorizations: Applications to
% Exploratory Multi-Way Data Analysis and Blind Source Separation"
% John Wiley and Sons, 2009.
%
% [2] Christian Dittmar and Meinard Müller "Reverse Engineering the Amen
% Break -- Score-informed Separation and Restoration applied to Drum
% Recordings" IEEE/ACM Transactions on Audio, Speech, and Language Processing,
% 24(9): 1531-1543, 2016.
%
% Input:  V                   matrix that shall be decomposed (typically a
%                             magnitude spectrogram of dimension
%                             numBins x numFrames)
%         parameter.
%           numComp           number of NMFD components (denoted as R in [2])
%           numIter           number of NMFD iterations (denoted as L in [2])
%           numTemplateFrames number of time frames for the 2D-template (denoted as T in [2])
%           initW             an initial estimate for the templates (denoted as W^(0) in [2])
%           initH             an initial estimate for the gains (denoted as H^(0) in [2])
%           beta              the beta parameter of the divergence:
%                             -1 -> equals Itakura Saito divergence
%                              0 -> equals Kullback Leiber divergence
%                              1 -> equals Euclidean distance
%           sparsityWeight    strength of the activation sparsity
%           uncorrWeight      strength of the template uncorrelatedness
%
% Output: W                   cell-array with the learned templates
%         H                   matrix with the learned activations
%         cnmfV               cell-array with approximated component
%                             spectrograms
%         costFunc            the approximation quality per iteration
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
if nargin < 2
  parameter = [];
end
if nargin < 1
  error('Please specify input matrix V.');
end

if ~isfield(parameter,'numComp')
  parameter.numComp = 3;
end
if ~isfield(parameter,'numIter')
  parameter.numIter = 30;
end
if ~isfield(parameter,'numTemplateFrames')
  parameter.numTemplateFrames = 8;
end
if ~isfield(parameter,'beta')
  parameter.beta = 0;
end
if ~isfield(parameter,'sparsityWeight')
  parameter.sparsityWeight = 0;
end
if ~isfield(parameter,'uncorrWeight')
  parameter.uncorrWeight = 0;
end

% use parameter nomenclature as in [2]
[K,M] = size(V);
T = parameter.numTemplateFrames;
R = parameter.numComp;
L = parameter.numIter;
beta = parameter.beta;
sparsityWeight = parameter.sparsityWeight;
uncorrWeight = parameter.uncorrWeight;

% use initial templates
if ~isfield(parameter,'initW')
  initW = initTemplates(parameter,'random');
else
  initW = parameter.initW;
end

% stack the templates into a tensor
for r = 1:R
  tensorW(:,r,:) = squeeze(initW{r});
end

% use initial activations
if ~isfield(parameter,'initH')
  initH = initActivations(parameter,'uniform');
else
  initH = parameter.initH;
end

% copy initial
H = initH;

% this is important to prevent initial jumps in the divergence measure
V = V./(eps+sum(V(:)));

for iter = 1:L

  % compute first approximation
  Lambda = convModel(tensorW,H);
  LambdaBeta = Lambda.^beta;
  Q = V.*LambdaBeta./Lambda;

  costMat = V.*(V.^beta - Lambda.^beta)/(eps+beta) - (V.^(beta+1) - Lambda.^(beta+1))/(eps+beta+1);
  costFunc(iter) = mean(costMat(:));

  for t = 1:T

    % respect zero index
    tau = t-1;

    % use tau for shifting and t for indexing
    transpH = shiftOperator(H,tau)';

    numeratorUpdateW = Q*transpH;
    denominatorUpdateW = eps + LambdaBeta*transpH + uncorrWeight*sum(squeeze(tensorW(:,:,setdiff(1:T,t))),3);
    tensorW(:,:,t) = tensorW(:,:,t) .* (numeratorUpdateW./denominatorUpdateW);
  end

  numeratorUpdateH = convModel(permute(tensorW, [2 1 3]),fliplr(Q));
  denominatorUpdateH = convModel(permute(tensorW,[2 1 3]),fliplr(LambdaBeta)) + sparsityWeight + eps;
  H = H .* fliplr(numeratorUpdateH./denominatorUpdateH);

  % normalize templates to unit sum
  normVec = sum(sum(tensorW,1),3);
  tensorW = bsxfun(@times,tensorW,1./(eps+normVec));

%   subplot(2,1,1)
%   imagesc(H);
%   axis xy
%   drawnow
%
%   subplot(2,1,2)
%   imagesc(sqrt(reshape(permute(tensorW,[1 3 2]),K,T*R)));
%   axis xy
%   drawnow
%   1;

end

% compute final output approximation
for r = 1:R
  W{r} = squeeze(tensorW(:,r,:));
  cnmfY{r} = convModel(tensorW(:,r,:), H(r,:));
end
