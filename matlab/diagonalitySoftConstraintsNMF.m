function [ W, H ] = diagonalitySoftConstraintsNMF(W,H,iter,numIter,parameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: diagonalitySoftConstraintsNMF
% Date: May 2018
% Programmer: Christian Dittmar
%
% Implements a simplified version of the soft constraints in [1].
%
% References:
% [1] Jonathan Driedger, Thomas Praetzlich, and Meinard Mueller
% Let It Bee � Towards NMF-Inspired Audio Mosaicing
% In Proceedings of the International Conference on Music Information
% Retrieval (ISMIR): 350�356, 2015.
%
% Input:  W                   NMF templates given in matrix/tensor form
%         H                   NMF activations given as matrix
%         iter                current iteration count
%         numIter             target number of iterations
%         parameter.
%           KernOrd           order of smoothing operation
%           initW             initial version of the NMF templates
%
% Output: W                   processed NMF templates
%         H                   processed NMF activations
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If you use the 'NMF toolbox' please refer to:
% [1]  Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
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
H = conv2(H,eye(parameter.KernOrd),'same');

end
