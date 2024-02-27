function [ Xout, Pout, res ] = LSEE_MSTFTM_GriffinLim( X, parameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: LSEE_MSTFTMGriffinLim
% Date of Revision: 06 2016
% Programmer: Christian Dittmar
%
% Performs one iteration of the phase reconstruction algorithm as
% described in:
% [1] Daniel W. Griffin and Jae S. Lim, "Signal estimation
% from modified short-time fourier transform," IEEE
% Transactions on Acoustics, Speech and Signal Processing,
% vol. 32, no. 2, pp. 236-243, Apr 1984.
%
% The operation performs an iSTFT (LSEE-MSTFT) followed by STFT on the
% resynthesized signal.
%
% Input: X: the STFT spectrogram to iterate upon
%        blockSize: the used blocksize
%        hopSize: the used hopsize (denoted as S in [1])
%        anaWinFunc: the window used for analysis (denoted w in [1])
%        synWinFunc: the window used for synthesis (denoted w in [1])
%        reconstMirror: if this is enabled, we have to generate the
%                       mirror spectrum by means of conjugation and flipping
%        appendFrames: if this is enabled, safety spaces have to be removed
%                      after the iSTFT
%        targetEnv: if desired, we can define a time-signal mask from the
%                   outside for better restoration of transients
%
% Output:
%        Xout: the spectrogram after iSTFT->STFT processing
%        Aout: the magnitude spectrogram after iSTFT->STFT processing
%        Pout: the phase spectrogram after iSTFT->STFT processing
%        res: reconstructed time-domain signal obtained via iSTFT
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
[numBins, ~] = size(X);

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
if ~isfield(parameter,'appendFrames')
  parameter.appendFrames = true;
end
if ~isfield(parameter,'analyticSig')
  parameter.analyticSig = false;
end

if ~isfield(parameter,'numIterGriffinLim')
  parameter.numIterGriffinLim = 50;
end

Xout = X;
A = abs(Xout);

for k = 1 : parameter.numIterGriffinLim

%% perform inverse STFT
res = inverseSTFT(Xout, parameter);

%% perfom forward STFT
[~,~,Pout] = forwardSTFT(res, parameter);

Xout = A .* exp(1i * Pout);

end


end
