function [ A ] = shiftOperator( A, shiftAmount )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: shiftOperator
% Date: March 2018
% Programmer: Christian Dittmar
%
% Shift operator as described in eq. (5) from [1]. It shifts the columns
% of a matrix to the left or the right and fills undefined elements with
% zeros.
%
% References:
% [1] Paris Smaragdis "Non-negative Matix Factor Deconvolution;
% Extraction of Multiple Sound Sources from Monophonic Inputs".
% International Congress on Independent Component Analysis and Blind Signal
% Separation (ICA), 2004
%
% Input:  A            arbitrary matrix to undergo the shifting operation
%         shiftAmount  positive numbers shift to the right, negative numbers
%                      shift to the left, zero leaves the matrix unchanged
%
% Output: A            result of this operation
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
% get dimensions
[numRows, numCols] = size(A);

% limit shift range
shiftAmount = sign(shiftAmount) * min(abs(shiftAmount), numCols);

% apply circular shift along the column dimension
A = circshift(A,shiftAmount,2);

% discriminate between positive and negative shift
% to apply appropriate zero padding
if shiftAmount < 0
  A(:,(numCols+(shiftAmount+1)):numCols) = 0;
elseif shiftAmount > 0
  A(:,1:shiftAmount) = 0;
end
