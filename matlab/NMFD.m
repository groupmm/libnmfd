function [W, H, nmfdV, costFunc, tensorW] = NMFD(V, parameter, paramConstr)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: NMFD
% Date: May 2015
% Programmer: Christian Dittmar
%
% Non-Negative Matrix Factor Deconvolution with Kullback-Leibler-Divergence
% and fixable components. The core algorithm was proposed in [1], the
% specific adaptions are used in [2].
%
% References:
% [1] Paris Smaragdis "Non-negative Matix Factor Deconvolution;
% Extraction of Multiple Sound Sources from Monophonic Inputs".
% International Congress on Independent Component Analysis and Blind Signal
% Separation (ICA), 2004
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
%         paramConstr.        if this is given, it should contain
%                             parameters for constraints
%
% Output: W                   cell-array with the learned templates
%         H                   matrix with the learned activations
%         nmfdV               cell-array with approximated component
%                             spectrograms
%         costFunc            the approximation quality per iteration
%         tensorW             if desired, we can also return the tensor
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If you use the 'NMF toolbox' please refer to:
% [3] Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
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
if ~exist('paramConstr') | isempty(paramConstr)
  paramConstr.type = 'none';
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

% use parameter nomenclature as in [2]
[K,M] = size(V);
T = parameter.numTemplateFrames;
R = parameter.numComp;
L = parameter.numIter;

% use initial templates
if ~isfield(parameter,'initW')
  initW = initTemplates(parameter,'random');
else
  initW = parameter.initW;
end

if ~isfield(parameter,'initH')
  initH = initActivations(parameter,'uniform');
else
  initH = parameter.initH;
end

if ~isfield(parameter, 'fixH')
    parameter.fixH = false;
end;

if ~isfield(parameter, 'fixW')
    parameter.fixW = false;
end;

% stack the templates into a tensor
for r = 1:R
  tensorW(:,r,:) = squeeze(initW{r});
end

% the activations are matrix shaped
H = initH;

% create helper matrix of all ones (denoted as J in eq (5,6) in [2])
onesMatrix = ones(K, M);

% this is important to prevent initial jumps in the divergence measure
V = V./(eps+sum(V(:)));

% repeat for the number of iterations
for iter = 1:L

  % if given from the outside, apply soft constraints
  if isfield(paramConstr,'funcPointerPreProcess')
   % [tensorW,H] = softConstraintsNMF(tensorW, H, iter, L, paramConstr);
    [tensorW,H] = paramConstr.funcPointerPreProcess(tensorW, H, iter, L, paramConstr);
  end

  % compute first approximation
  Lambda = convModel(tensorW,H);

  % store the divergence with respect to the target spectrogram
  costMat = V.*log(1+V./(Lambda+eps))-V+Lambda;
  costFunc(iter) = mean(costMat(:));

  % compute the ratio of the input to the model
  Q = V./(Lambda+eps);

  % accumulate activation updates here
  multH = zeros(R,M);

  % go through all template frames
  for t = 1:T

    % use tau for shifting and t for indexing
    tau = t-1;

    % The update rule for W as given in eq. (5) in [2]
    % pre-compute intermediate, shifted and transposed activation matrix
    transpH = shiftOperator(H,tau)';

    % multiplicative update for W
    multW = (Q*transpH)./(onesMatrix*transpH+eps);

    if ~parameter.fixW
        tensorW(:,:,t) = tensorW(:,:,t) .* multW;
    end;

    % The update rule for W as given in eq. (6) in [2]
    % pre-compute intermediate matrix for basis functions W
    transpW = tensorW(:,:,t)';

    % compute update term for this tau
    addW = transpW*shiftOperator(Q,-tau) ./ (transpW*onesMatrix+eps);

    % accumulate update term
    multH = multH + addW;

  end

  % multiplicative update for H, with the average over all T template frames
  if ~parameter.fixH
    H = H .* (multH./T);
  end;

  % if given from the outside, apply soft constraints
  if isfield(paramConstr,'funcPointerPostProcess')
    [tensorW,H] = paramConstr.funcPointerPostProcess(tensorW, H, iter, L, paramConstr);
  end


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
  %   colormap(1-gray)
  %   axis xy
  %   drawnow
  %   1;

end

% compute final output approximation
for r = 1:R
  W{r} = squeeze(tensorW(:,r,:));
  nmfdV{r} = convModel(tensorW(:,r,:), H(r,:));
end
