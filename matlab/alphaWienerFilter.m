function [ sourceX, softMasks ] = alphaWienerFilter( mixtureX, sourceA, alpha, binarize )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: alphaWienerFilter
% Date of Revision: 08 2016
% Programmer: Christian Dittmar
%
% Given a cell-array of spectrogram estimates as input, this function
% computes the alpha-related soft masks for extracting the sources. Details
% about this procedure are given in [2], further experimental studies in [2].
%
% [1] Antoine Liutkus and Roland Badeau: Generalized Wiener filtering with
% fractional power spectrograms, ICASPP 2015
%
% [2] Christian Dittmar et al.: An Experimental Approach to Generalized
% Wiener Filtering in Music Source Separation, EUSIPCO 2016
%
% Input:
%        mixtureX: the mixture spectrogram (numBins x numFrames) (may be real-
%              or complex-valued)
%        sourceA: a cell-array holding the equally sized spectrogram estimates
%              of single sound sources (aka components)
%        alpha: the fractional power in rand [0 ... 2]
%        binarize: if this is set to true, we binarize the masks
%
% Output:
%        sourceX: a cell-array of extracted source spectrograms
%        softMasks: a cell-array with the extracted masks
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~exist('alpha') | isempty(alpha)
  alpha = 1.2; %% this is the optimal value reported in [1]
end

if ~exist('binarize') | isempty(binarize)
  binarize = false;
end

%% to do: check size correspondence
[numBins,numFrames] = size(mixtureX);

%% get number of estimated sources / components
numComp = length(sourceA);

%% initialize the mixture of the sources / components with a small constant
mixtureA = eps+zeros(numBins,numFrames);

%% make superposition
for k = 1:numComp
  mixtureA = mixtureA + sourceA{k}.^alpha;
end

%% compute soft masks and spectrogram estimates
for k = 1:numComp
  currSoftMask = sourceA{k}.^alpha ./ mixtureA;
  softMasks{k} = currSoftMask;

  %% if desired, make this a binar mask
  if binarize
    softMasks{k} = double(softMasks{k} > (1/numComp));
  end

  %% and apply it to the mixture
  sourceX{k} = mixtureX.*currSoftMask;
end
