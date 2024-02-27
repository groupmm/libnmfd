function [ A ] = NEMA( A, lambda )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: NEMA
% Date: April 2018
% Programmer: Christian Dittmar
%
% This function takes a matrix of row-wise time series and applies a
% non-linear exponential moving average (NEMA) to each row. This filter
% introduces exponentially decaying slopes and is defined in eq. (3) from [1].
%
% The difference equation of that filter would be:
% y(n) = max( x(n), y(n-1)*(decay) + x(n)*(1-decay) )
%
% References:
% [1] Christian Dittmar, Patricio López-Serrano, Meinard Müller: "Unifying
% Local and Global Methods for Harmonic-Percussive Source Separation"
% In Proceedings of the IEEE International Conference on Acoustics,
% Speech, and Signal Processing (ICASSP), 2018.
%
% Input:  A         the matrix with time series in its rows
%         lambda    the decay parameter in the range [0 ... 1], this can be
%                   given as a column-vector with individual decays per row
%                   or as a scalar
%
% Output: A         the result after application of the NEMA filter
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If you use the 'NMF toolbox' please refer to:
% [2] Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
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
if ~exist('lambda') | isempty(lambda)
  lambda = 0.9;
end

% prevent instable filter
lambda = max(0,min(0.9999999,lambda));

% get input dimensions
[numRows, numCols] = size(A);

% start recursive processing at second time frame
for k = 2:numCols

  storeRow = A(:,k);
  A(:,k) = lambda.*A(:,k-1) + A(:,k).*(1-lambda);
  A(:,k) = max(A(:,k),storeRow);

end
