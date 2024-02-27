%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: test_initTemplates
% Date: Jun 2019
% Programmer: Christian Dittmar, Yiğitcan Özer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear all;
clc;

addpath('../matlab/');
inpPath = '../data/';
outPath = 'matrices/initTemplates/';
% create directory if necessary
mkdir(outPath);
filename = 'runningExample_IGotYouMixture.wav';

warning('OFF','MATLAB:audiovideo:audiowrite:dataClipped');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. load the audio signal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[x,fs] = audioread([inpPath filename]);

% make monaural if necessary
x = mean(x,2);

% read corresponding transcription files
melodyTranscription = textread([inpPath, 'runningExample_IGotYouMelody.txt']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. compute STFT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
deltaF = fs / paramSTFT.blockSize;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. initTemplates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set common parameters
numCompDrum = 3;
numTemplateFrames = 8;

paramTemplates.deltaF = deltaF;
paramTemplates.numBins = numBins;
paramTemplates.numTemplateFrames = numTemplateFrames;
paramTemplates.pitches = melodyTranscription(:,2);
pitchedW = initTemplates(paramTemplates,'pitched');

% generate audio-informed templates for the drum part
paramTemplates.numComp = numCompDrum;
drumsW = initTemplates(paramTemplates ,'drums');

% generate uniform templates
uniformW = initTemplates(paramTemplates ,'uniform');

% convert cell arrays into matrices
uniformW=cell2mat(uniformW);
drumsW=cell2mat(drumsW);
pitchedW=cell2mat(pitchedW);

% save matrices
save([outPath 'uniformW.mat'], 'uniformW')
save([outPath 'drumsW.mat'], 'drumsW')
save([outPath 'pitchedW.mat'], 'pitchedW')
