function [ W, H ] = semiFixedComponentConstraintsNMF(W,H,iter,numIter,parameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: semiFixedComponentConstraintsNMF
% Date: Jun 2018
% Programmer: Christian Dittmar and Patricio Lopez-Serrano
%
% Implements a simplified version of the soft constraints in [1].
%
% References:
% [1] Patricio Lopez-Serrano, Christian Dittmar, Jonathan Driedger, and
%     Meinard Müller.
%     Towards modeling and decomposing loop-based
%     electronic music.
%     In Proceedings of the International Conference
%     on Music Information Retrieval (ISMIR), pages 502–508,
%     New York City, USA, August 2016.
%
% [2] Christian Dittmar and Daniel Gärtner
%     Real-time transcription and separation of drum recordings based on
%     NMF decomposition.
%     In Proceedings of the International Conference on Digital Audio
%     Effects (DAFx): 187–194, 2014.
%
%
% Input:  W                   NMF templates given in matrix/tensor form
%         H                   NMF activations given as matrix
%         iter                current iteration count
%         numIter             target number of iterations
%         parameter.
%           initW             initial version of the NMF templates
%           initH             initial version of the NMF activations
%           adaptDegree       0 = fixed, 1 = semi-fixed, 2 = adaptive
%           adaptTiming       only in combination with semi-fixed adaptDegree
%
%
% Output: W                   processed NMF templates
%         H                   processed NMF activations
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

% roughly equal to Driedger
% mix with originally estimated sources

[numRows] = size(W, 1);

weight = iter / numIter;

if ~iscell(parameter.initW)
    newWeight = repmat(weight .* parameter.adaptDegree .^ parameter.adaptTiming, numRows, 1);
    initW = parameter.initW;
else
    R = length(parameter.initW);

    newWeight = zeros(size(W));

    % stack the templates into a tensor
    for r = 1:R
      initW(:,r,:) = squeeze(parameter.initW{r});

      cMat = repmat(weight .* parameter.adaptDegree(r) .^ parameter.adaptTiming(r), ...
                    numRows, ...
                    size(W, 3));

      newWeight(:, r, :) = cMat;

    end
end;

newWeight(:, parameter.adaptDegree > 1.0) = 1;

%% constrain the range of values
newWeight = min(newWeight,1);
newWeight = max(newWeight,0);

%% compute counter part
initWeight = 1 - newWeight;
W = W .* newWeight + initW .* initWeight;

end
