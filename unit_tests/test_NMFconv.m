%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: test_NMFconv
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
outPath = 'matrices/NMFconv/';
% create directory if necessary
mkdir(outPath);

filename = 'runningExample_AmenBreak.wav';

warning('OFF','MATLAB:audiovideo:audiowrite:dataClipped');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. load the audio signal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[x,fs] = audioread([inpPath filename]);

% make monaural if necessary
x = mean(x,2);

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
deltaT = paramSTFT.hopSize / fs;
deltaF = fs / paramSTFT.blockSize;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. apply NMF variants to STFT magnitude
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set common parameters
numComp = 3;
numIter = 3;
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

% NMFconv parameters
paramNMFconv.numComp = numComp;
paramNMFconv.numFrames = numFrames;
paramNMFconv.numIter = numIter;
paramNMFconv.numTemplateFrames = numTemplateFrames;
paramNMFconv.initW = initW;
paramNMFconv.initH = initH;
paramNMFconv.beta = 0;

% NMFconv core method
[nmfconvW, nmfconvH, nmfconvV, divBeta] = NMFconv(A, paramNMFconv);

% convert cell arrays into matrices
nmfconvW=cell2mat(nmfconvW);
nmfconvV=cell2mat(nmfconvV);

% save matrices
save([outPath 'nmfconvW.mat'], 'nmfconvW')
save([outPath 'nmfconvH.mat'], 'nmfconvH')
save([outPath 'nmfconvV.mat'], 'nmfconvV')
save([outPath 'divBeta.mat'], 'divBeta')
