%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: demoEDMDecompositionFourComp
% Date: May 2018
% Programmer: Patricio López-Serrano
%
% This demo illustrates decomposition of a loop-based electronic dance
% music track, following [1,2].
%
% The script proceeds as follows:
%
% 1. Load audio files for the complete, downmixed track, as well as for the
%    individual loops that the track contains.
% 2. Compute STFTs for all audio data.
% 3. Each loop becomes a fixed template ("page") in the tensor W.
%    The track spectrogram is the target to approximate, V.
%    We wish to learn the activation matrix H, which answers the question
%    "Where was each loop activated throughout the track?"
% 4. Visualize results.
%
% References:
% [1] Patricio López-Serrano, Christian Dittmar, Jonathan Driedger, and
%     Meinard Müller.
%     Towards modeling and decomposing loop-based
%     electronic music.
%     In Proceedings of the International Conference
%     on Music Information Retrieval (ISMIR), pages 502–508,
%     New York City, USA, August 2016.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If you use the 'NMF toolbox' please refer to:
% [2] Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
%     Müller.
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

% initialization
inpPath = '../data/';
outPath = 'output/';

% create output directory, if it doesn't exist
if ~exist(outPath, 'dir')
  mkdir(outPath);
end

filename = 'LSDDM_EM_track.wav';
filenameEffects = 'LSDDM_EM_Effects.wav';
filenameBass = 'LSDDM_EM_bass.wav';
filenameMelody = 'LSDDM_EM_melody.wav';
filenameDrums = 'LSDDM_EM_drums.wav';

% 1. load the audio signal
[xTr,fs] = audioread([inpPath filename]);

[xEffects, fsEffects] = audioread([inpPath filenameEffects]);
[xBass, fsBass] = audioread([inpPath filenameBass]);
[xMelody, fsMelody] = audioread([inpPath filenameMelody]);
[xDrums, fsDrums] = audioread([inpPath filenameDrums]);
% make monaural if necessary
xTr = mean(xTr, 2);
xEffects = mean(xEffects, 2);
xBass = mean(xBass, 2);
xMelody = mean(xMelody, 2);
xDrums = mean(xDrums, 2);

% 2. compute STFT
paramSTFT.blockSize = 4096;
paramSTFT.hopSize = 2048;
paramSTFT.winFunc = hann(paramSTFT.blockSize);
paramSTFT.reconstMirror = true;
paramSTFT.appendFrame = true;

% STFT computation
[XTr, ATr, PTr] = forwardSTFT(xTr, paramSTFT);

% get dimensions and time and freq resolutions
[numBinsTr, numFramesTr] = size(XTr);
deltaT = paramSTFT.hopSize / fs;
deltaF = fs / paramSTFT.blockSize;

% STFT computation
[XEffects, AEffects, PEffects] = forwardSTFT(xEffects, paramSTFT);
[XBass, ABass, PBass] = forwardSTFT(xBass, paramSTFT);
[XMel, AMel, PMel] = forwardSTFT(xMelody, paramSTFT);
[XDrums, ADrums, PDrums] = forwardSTFT(xDrums, paramSTFT);
[numBinsBass, numFramesBass] = size(XBass);

% 3. apply NMF variants to STFT magnitude
numComp = 4;
numIter = 30;

initW = [];
initW{1} = ABass;
initW{2} = AMel;
initW{3} = ADrums;
initW{4} = AEffects;
paramNMFD.initW = initW;
numTemplateFrames = numFramesBass;

% generate initial activations
paramActivations.numComp = numComp;
paramActivations.numFrames = numFramesTr;
initH = initActivations(paramActivations, 'uniform');
paramNMFD.initH = initH;

% NMFD parameters
paramNMFD.numComp = numComp;
paramNMFD.numFrames = numFramesTr;
paramNMFD.numIter = numIter;
paramNMFD.numTemplateFrames = numTemplateFrames;
paramNMFD.numBins = numBinsTr;
paramNMFD.fixW = 1;

% NMFD core method
[nmfdW, nmfdH, nmfdV, divKL] = NMFD(ATr, paramNMFD);

%% visualize
paramVis.deltaT = deltaT;
paramVis.deltaF = deltaF;
paramVis.logComp = 1e5;
fh1 = visualizeComponentsNMF(ATr, nmfdW, nmfdH, nmfdV, paramVis);

%% save result
saveas(fh1,[outPath,'LSDDM_EM.png']);
