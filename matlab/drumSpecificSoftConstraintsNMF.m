function [ W, H ] = drumSpecificSoftConstraintsNMF(W,H,iter,numIter,parameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: drumSpecificSoftConstraintsNMF
% Date: May 2018
% Programmer: Christian Dittmar
%
% Implements the drum specific soft constraints that can be applied during NMF or
% NMFD iterations. These constraints affect the activation vectors only and
% are described in sec.23 of [1].
%
% References:
% [1] Christian Dittmar, Patricio Lopez-Serrano, Meinard Mueller
% Unifying Local and Global Methods for Harmonic-Percussive Source Separation
% In Proceedings of the IEEE International Conference on Acoustics,
% Speech, and Signal Processing (ICASSP), 2018.
%
% Input:  W                   NMF templates given in matrix/tensor form
%         H                   NMF activations given as matrix
%         iter                current iteration count
%         numIter             target number of iterations
%         parameter.
%           KernOrd           order of smoothing operation
%           Kern              concrete smoothing kernel
%           initH             initial version of the NMF activations
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

% this assumes that the templates are constructed as described in sec.
% 2.4 of [1]
[ percWeight ] = percussivenessEstimation( W );

% promote harmonic sustained gains
Hh = ordfilt2(H,parameter.KernOrd,parameter.Kern);

% promote decaying impulses gains
Hp = NEMA(H,parameter.decay);

% make weighted sum according to percussiveness measure
H = bsxfun(@times,Hh,1-percWeight') + bsxfun(@times,Hp,percWeight');

end
