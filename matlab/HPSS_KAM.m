function [kamX,Kern,KernOrd] = HPSS_KAM_Fitzgerald(X, numIter, kernDim, useMedian, alphaParam)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: HPSS_KAM
% Date: July 2017
% Programmer: Christian Dittmar
%
% This re-implements the KAM-based HPSS-algorithm described in [1]. This is
% a generalization of the median-filter based algorithm first presented in [2].
% Our own variant of this algorithm [3] is also supported.
%
% References:
% [1] Derry FitzGerald, Antoine Liutkus, Zafar Rafii, Bryan Pardo,
% and Laurent Daudet, �Harmonic/percussive separation
% using Kernel Additive Modelling,� in Irish Signals
% and Systems Conference (IET), Limerick, Ireland,
% 2014, pp. 35�40.
%
% [2] Derry FitzGerald, �Harmonic/percussive separation using
% median filtering,� in Proceedings of the International
% Conference on Digital Audio Effects (DAFx),
% Graz, Austria, 2010, pp. 246�253.
%
% [3] Christian Dittmar, Jonathan Driedger, Meinard M�ller,
% and Jouni Paulus, �An experimental approach to generalized
% wiener filtering in music source separation,� EUSIPCO 2016.
%
% Input:  X                   input mixture magnitude spectrogram
%         numIter             the number of iterations
%         kernDim             the kernel dimensions
%         useMedian           reverts to FitzGeralds old method
%         alphaParam          the alpha-Wiener filter exponent
%
% Output: kamX                cell-array containing the percussive and
%                             harmonic estimate
%         Kern                the kernels used for enhancing percussive and
%                             harmonic part
%         KernOrd             the order of the kernels

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If you use the 'NMF toolbox' please refer to:
% [4]  Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
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
% Check parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get dimensions
[numBins,numFrames] = size(X);

% this defaults to Fitzgeralds method
if ~exist('numIter') | isempty(numIter)
  numIter = 1;
end

% this is the value from [2]
if ~exist('kernDim') | isempty(kernDim)
  kernDim = 17;
end

% check whether to use median filter
if ~exist('useMedian') | isempty(useMedian)
  useMedian = false;
end

% check whether alpha param was given
if ~exist('alphaParam') | isempty(alphaParam)
  alphaParam = 1;
end

% prepare data for the KAM iterations
kamX = [];
K = [];

KernOrd = round(kernDim/2);

% construct median filter kernel
Kern = false(kernDim);
Kern(KernOrd,:) = true;

% construct low-pass filter kernel
K{1} = hann(kernDim);
K{1} = K{1}/sum(K{1});

K{2} = hann(kernDim)';
K{2} = K{2}/sum(K{2});

% intitalize first version with copy of original
kamX{1} = X;
kamX{2} = X;

for iter = 1:numIter

  if useMedian

    % update estimates via method from [1]
    kamX{1} = ordfilt2(kamX{1},KernOrd,Kern');
    kamX{2} = ordfilt2(kamX{2},KernOrd,Kern);

  else

    % update estimates via method from [2]
    kamX{1} = conv2(kamX{1},K{1},'same');
    kamX{2} = conv2(kamX{2},K{2},'same');

  end

  % apply alpha Wiener filtering
  kamX = alphaWienerFilter(X, kamX, alphaParam );

end

end
