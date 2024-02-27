%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: demoDrumSoundSeparationNMF
% Date: April 2018
% Programmer: Christian Dittmar
%
% This is the demo script which illustrates the main functionalities of the
% 'NMF toolbox'. For a detailed description we refer to [1,2] (see
% References below).
%
% The script proceeds in the following steps:
%   1. It loads an example audio file containing a drum mixture recording.
%   2. It computes the STFT of the audio data.
%   3. It applies different NMF variants to decompose the recording into
%      the constituent drum sound events.
%   4. It visualizes the NMF decomposition results.
%   5. It resynthesizes the separated audio streams and saves them as wav
%      files to the hard drive.
%
% References:
% [1] Christian Dittmar, Meinard Müller
%     Reverse Engineering the Amen Break  Score-informed Separation and
%     Restoration applied to Drum Recordings
%     IEEE/ACM Transactions on Audio, Speech, and Language Processing,
%     24(9): 15311543, 2016.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear variables
clc


inpPath = '../data/';
outPath = 'output/';

% create output directory, if it doesn't exist
if ~exist(outPath, 'dir')
  mkdir(outPath);
end

filename = 'runningExample_AmenBreak.wav';

 % 1. load the audio signal
[x,fs] = audioread([inpPath filename]);
x = mean(x,2);

 % 2. compute STFT
% spectral parameters
paramSTFT.blockSize = 2048;
paramSTFT.hopSize = 512;
paramSTFT.winFunc = hann(paramSTFT.blockSize);
paramSTFT.reconstMirror = true;
paramSTFT.appendFrame = true;
paramSTFT.numSamples = length(x);

 % STFT computation
[X,A,P] = forwardSTFT(x,paramSTFT);

 % get dimensions and time and freq resolutions
[numBins,numFrames] = size(X);
deltaT = paramSTFT.hopSize / fs;
deltaF = fs / paramSTFT.blockSize;

 % 3. apply NMF variants to STFT magnitude
% set common parameters
numComp = 3;
numIter = 30;
numTemplateFrames = 8;

 % generate initial guess for templates
paramTemplates.deltaF = deltaF;
paramTemplates.numComp = numComp;
paramTemplates.numBins = numBins;
paramTemplates.numTemplateFrames = numTemplateFrames;
initW = initTemplates(paramTemplates,'drums');

 % generate initial activations
paramActivations.numComp = numComp;
paramActivations.numFrames = numFrames;
initH = initActivations(paramActivations,'uniform');

 % NMFD parameters
paramNMFD.numComp = numComp;
paramNMFD.numFrames = numFrames;
paramNMFD.numIter = numIter;
paramNMFD.numTemplateFrames = numTemplateFrames;
paramNMFD.initW = initW;
paramNMFD.initH = initH;

 % NMFD core method
[nmfdW, nmfdH, nmfdV, divKL] = NMFD(A, paramNMFD);

 % alpha-Wiener filtering
nmfdA = alphaWienerFilter(A,nmfdV,1);

 % visualize
paramVis.deltaT = deltaT;
paramVis.deltaF = deltaF;
paramVis.endeSec = 3.8;
paramVis.fontSize = 14;
visualizeComponentsNMF(A, nmfdW, nmfdH, nmfdA, paramVis);

% resynthesize
for k = 1:numComp
  Y = nmfdA{k} .* exp(j * P);

  % re-synthesize, omitting the Griffin Lim iterations
  y = inverseSTFT(Y, paramSTFT);

  % save result
  audiowrite([outPath,'Winstons_AmenBreak_NMFD_component_',...
    num2str(k), '.wav'],y,fs);
end
