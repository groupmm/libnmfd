function [ percWeight ] = percussivenessEstimation( W )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: percussivenessEstimation
% Date: May 2018
% Programmer: Christian Dittmar
%
% This function takes a matrix or tensor of NMF templates and estimates the
% percussivness by assuming that the lower part explains percussive and the
% upper part explains harmonic components. This is explained in sec. 2.4,
% especially eq. (4) in [1].
%
% References:
% [1] Christian Dittmar, Patricio L�pez-Serrano, Meinard M�ller: "Unifying
% Local and Global Methods for Harmonic-Percussive Source Separation"
% In Proceedings of the IEEE International Conference on Acoustics,
% Speech, and Signal Processing (ICASSP), 2018.
%
% Input:  W           a K x R matrix (or K x R x T tensor) of NMF (NMFD) templates
%
% Output: percWeight  the resulting percussiveness estimate per component
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If you use the 'NMF toolbox' please refer to:
% [1] Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
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
% get dimensions of templates
[K,R,T] = size(W);

% this assumes that the matrix (tensor) is formed in the way we need it
numBins = K/2;

% do the calculation, which is essentially a ratio
percWeight = [];
for c = 1:R
  percPart = squeeze(W(1:numBins,c,:));
  harmPart = squeeze(W(1:end,c,:));
  percWeight(c) = sum(percPart(:))/sum(harmPart(:));
end
