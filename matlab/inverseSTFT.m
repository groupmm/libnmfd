function [y, env] = inverseSTFT(X, parameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: inverseSTFT
% Date of Revision: 06 2016
% Programmer: Christian Dittmar
%
% Given a valid STFT spectrogram as input, this reconstructs the corresponding
% time-domain signal by  means of the frame-wise inverse FFT and overlap-add
% method described as LSEE-MSTFT in:
% [1] Daniel W. Griffin and Jae S. Lim, "Signal estimation
% from modified short-time fourier transform," IEEE
% Transactions on Acoustics, Speech and Signal Processing,
% vol. 32, no. 2, pp. 236-243, Apr 1984.
%
% Input: X                the complex-valued spectrogram matrix oriented
%                         with dimensions numBins x numFrames
%        parameter.
%          blockSize      the blocksize to use during synthesis
%          hopSize        the hopsize to use during synthesis
%          anaWinFunc     the analysis window function
%          reconstMirror  this switch decides whether the mirror spectrum
%                         should be reconstructed or not
%          appendFrame    this switch decides whethter to compensate for
%                         zero padding or not
%          synWinFunc     the synthesis window function (per default the
%                         same as analysis window)
%          analyticSig    if this is set to true, we want the analytic signal
%          numSamples     the original number of samples
%
% Output: y         the resynthesized signal
%         env       the envelope used for normalization of the synthesis window
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If you use the 'NMF toolbox' please refer to:
% [2]  Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
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
  error('Please specify STFT matrix X.');
end
[numBins,numFrames] = size(X);

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

% this controls if the upper part of the spectrum is given or should be
% reconctructed by 'mirroring' (flip and conjugate) of the lower spectrum
if ~isfield(parameter,'reconstMirror')
  if numBins == parameter.blockSize
    parameter.reconstMirror = false;
  elseif (numBins < parameter.blockSize)
    parameter.reconstMirror = true;
  end
end
if ~isfield(parameter,'appendFrame')
  parameter.appendFrame = true;
end
if ~isfield(parameter,'analyticSig')
  parameter.analyticSig = false;
end

% for better readability, use internal variables
reconstMirror = parameter.reconstMirror;
appendFrame = parameter.appendFrame;
analyticSig = parameter.analyticSig;
blockSize = parameter.blockSize;
halfBlockSize = round(blockSize/2);
hopSize = parameter.hopSize;

% for simplicity, we assume the analysis and synthesis windows to be equal
anaWinFunc = parameter.winFunc;
synWinFunc = parameter.winFunc;

% this needs to be changed if we are interested in the analytic signal
if (analyticSig)
  reconstMirror = false;
  scale = 2;
else
  scale = 1;
end

%% set up zero vectors
sigLen = numFrames*hopSize+blockSize;
y = zeros(sigLen,1);

%% construct normalization function for the synthesis window
%% that equals to the denominator in eq. (6) in [1]
winFuncProd = anaWinFunc.*synWinFunc;
redundancy = round(blockSize/hopSize);

%% we construct the hopSize-periodic normalization
%% function that will be used applied to the synthesis window
env = zeros(blockSize,1);
winFuncProd = anaWinFunc.*synWinFunc;

%% we begin construction of this hopSize-periodic normalization
%% function already before the valid part of the signal
for k = -(redundancy-1):(redundancy-1)
  envInd = (hopSize*(k));
  winInd = [1:blockSize]';
  envInd = envInd+winInd;

  valid = find((envInd > 0) & envInd <= blockSize);
  envInd = envInd(valid);
  winInd = winInd(valid);

  %% add product of analysis and synthesis window
  env(envInd) = env(envInd) + winFuncProd(winInd);
end

%% apply normalization
synWinFunc = synWinFunc./env;

%% this is neeed for the output construction
winInd = [1:blockSize]';

%% then begin frame-wise reconstruction
for k = 1:numFrames

  %% pick one spectral frame
  currSpec = X(:,k);

  %% introduce artificial mirror spectrum
  if (reconstMirror)
    %% usually, we take the conjugate of the lower spectrum and flip it
    if ~analyticSig
      currSpec = [currSpec;conj(flipud(currSpec(2:end-1)))];
    else %% in case of a desired analytic signal, we simply put zeros
      currSpec = [currSpec;zeros( (blockSize/2)-1 ,1)];
    end
  end

  %% go back to time domain
  snip = ifft(currSpec, blockSize);

  %% if we want the analytic signal, we can skip this treatment
  if ~analyticSig
    snip = real(snip);
  end

  %% apply reconstruction window
  snip = scale*snip.*synWinFunc;

  %% find out, where to put the samples
  envInd = (hopSize*(k-1));
  envInd = envInd+winInd;

  %% and overlap add, the normalization of the synthesis window is already
  %% factored in
  y(envInd) = y(envInd) + snip;

end

%% if required, remove safety spaces again
if appendFrame
  y = y(halfBlockSize+1:end-halfBlockSize);
end

% if this was given, then revert number of samples
if isfield(parameter,'numSamples')
  y = y(1:parameter.numSamples);
end
