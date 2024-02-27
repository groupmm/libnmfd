function [ logFreqLogMagA, logFreqAxis ] = logFreqLogMag( A, deltaF, binsPerOctave, upperFreq, lowerFreq, logComp )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: logFreqLogMag
% Date: April 2018
% Programmer: Christian Dittmar
%
% Given a magnitude spectrogram, this function maps it onto a compact
% representation with logarithmically spaced frequency axis and logarithmic
% magnitude compression.
%
% Input: A              the real-valued magnitude spectrogram oriented as
%                       numBins x numFrames, it can also be given as a
%                       cell-array containing mutiple spectrograms
%        deltaF         the spectral resolution of the spectrogram
%        binsPerOctave  the spectral selectivity of the log-freq axis
%        upperFreq      the lower frequency border
%        lowerFreq      the upper frequency border
%
% Output:
%        logFreqLogMagA the log-magnitude spectrogram on logarithmically
%                       spaced frequency axis
%        logFreqAxis    a vector giving the center frequencies of each bin
%                       along the logarithmically spaced frequency axis
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~exist('binsPerOctave') | isempty(binsPerOctave)
  binsPerOctave = 36;
end

if ~exist('upperFreq') | isempty(upperFreq)
  upperFreq = 22050; % should be checked
end

if ~exist('lowerFreq') | isempty(lowerFreq)
  lowerFreq = midi2freq(24);
end

if ~exist('logComp', 'var') || isempty(logComp)
    logComp = 1;
end

% convert to cell array if necessary
if ~iscell(A)
  wasMatInput = true;
  A = mat2cell(A,size(A,1),size(A,2));
else
  wasMatInput = false;
end

% get number of components
numComp = length(A);

for k = 1:numComp
  % get component spectrogram
  compA = A{k};

  % get dimensions
  [numLinBins, numFrames] = size(compA);

  % set up linear frequency axis
  linFreqAxis = [0:numLinBins-1]*deltaF;

  % get upper limit
  upperFreq = linFreqAxis(end);

  % set up logarithmic frequency axis
  numLogBins = ceil( binsPerOctave*log2(  upperFreq / lowerFreq ));
  logFreqAxis = [0:numLogBins-1]';
  logFreqAxis = lowerFreq * power(2.0, logFreqAxis /  binsPerOctave);

  % map to logarithmic axis by means of linear interpolation
  logBinAxis = logFreqAxis / deltaF;

  % do not use the interp1 function, since it requires the Signal
  % Processing Toolbox
  floorBinAxis = floor(logBinAxis);
  ceilBinAxis = floorBinAxis+1;
  fraction = logBinAxis-floorBinAxis;

  % get magnitude values
  floorMag = compA(floorBinAxis,:);
  ceilMag = compA(ceilBinAxis,:);

  % compute weighted sum
  logFreqA = bsxfun(@times,floorMag,(1-fraction)) + bsxfun(@times,ceilMag,fraction);

  % apply magnitude compression
  logFreqLogMagA{k} = log(1 + (logComp .* logFreqA));

end

% revert back to matrix if necessary
if wasMatInput
  logFreqLogMagA = cell2mat(logFreqLogMagA);
end

end
