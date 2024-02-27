function [X,A,P] = forwardSTFT(x, parameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: forwardSTFT
% Date of Revision: 11 2014
% Programmer: Christian Dittmar
%
% Given a time signal as input, this computes the spectrogram by means of
% the Short-time fourier transform
%
% Input: x                the time signal oriented as numSamples x 1
%        parameter.
%          blockSize      the blocksize to use during analysis
%          hopSize        the hopsize to use during analysis
%          winFunc        the analysis window
%          reconstMirror  this switch decides whether to discard the mirror
%                         spectrum or not
%          appendFrame    this switch decides if we use silence in the
%                         beginning and the end
%
% Output: X               the complex valued spectrogram in numBins x numFrames
%         A               the magnitude spectrogram
%         P               the phase spectrogram (wrapped in -pi ... +pi)
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
if nargin < 1
  error('Please specify input time-domain signal x.');
end
if nargin < 2
  parameter = [];
end

if ~isfield(parameter,'blockSize')
  parameter.blockSize = 2048;
end
if ~isfield(parameter,'hopSize')
  parameter.hopSize = 512;
end
if ~isfield(parameter,'winFunc')
  parameter.winFunc = hann(parameter.blockSize);
end
if ~isfield(parameter,'reconstMirror')
  parameter.reconstMirror = true;
end
if ~isfield(parameter,'appendFrame')
  parameter.appendFrame = true;
end

% parse parameters to internal variables for better readability
blockSize = parameter.blockSize;
halfBlockSize = round(blockSize/2);
hopSize = parameter.hopSize;
winFunc = parameter.winFunc;
reconstMirror = parameter.reconstMirror;
appendFrame = parameter.appendFrame;

% the number of bins needs to be corrected
% if we want to discard the mirror spectrum
if reconstMirror
  numBins = blockSize/2+1;
else
  numBins = blockSize;
end

% append safety space in the beginning and end
if appendFrame
  x = [zeros(halfBlockSize,1);x;zeros(halfBlockSize,1)];
end

% pre-compute the number of frames
numFrames = round((length(x)-blockSize)/hopSize);

% initialize with correct size
X = zeros(numBins, numFrames);

counter = 0;
for k = 1:hopSize:length(x)-blockSize

  % where to pick
  ind = k:k+blockSize-1;

  % pick signal frame
  snip = x(ind);

  % apply windowing
  snip = snip.*winFunc;

  % do fft
  f = fft(snip,blockSize);

  % if required, remove upper half of spectrum
  if reconstMirror
    f = f(1:blockSize/2+1);
  end

  % store into STFT matrix
  counter = counter+1;
  X(:,counter) = f;

end

% compute derived matrices
A = abs(X);
P = angle(X);
