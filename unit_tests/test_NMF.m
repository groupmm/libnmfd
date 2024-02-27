%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: test_NMF
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
outPath = 'matrices/NMF/';
% create directory if necessary
mkdir(outPath);

filename = 'runningExample_AmenBreak.wav';

warning('OFF','MATLAB:audiovideo:audiowrite:dataClipped');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[x,fs] = audioread([inpPath filename]);
x = mean(x,2);

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
% apply NMF variants to STFT magnitude
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
[nmfconvW, ~, nmfconvV, ~] = NMFconv(A, paramNMFconv);

% alpha-Wiener filtering
nmfconvA = alphaWienerFilter(A,nmfconvV,1);

W0 = horzcat(nmfconvW{:});
initW = W0;

% set common parameters
numComp = size(W0, 2);
numIter = 3;

% generate random initialization for activations
paramActivations = [];
paramActivations.numComp = numComp;
paramActivations.numFrames = numFrames;
initH = initActivations(paramActivations, 'uniform');

% store common parameters
paramNMF = [];
paramNMF.numComp = numComp;
paramNMF.numFrames = numFrames;
paramNMF.numIter = numIter;
paramNMF.initW = initW;
paramNMF.initH = initH;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NMF with Euclidean Distance cost function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
paramNMF.costFunc = 'EucDist';

% call NMF iterations
[nmfEucDistW, nmfEucDistH, nmfEucDistV] = NMF(A, paramNMF);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NMF with Euclidean KLDiv cost function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
paramNMF.costFunc = 'KLDiv';
[nmfKLDivW, nmfKLDivH, nmfKLDivV] = NMF(A, paramNMF);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NMF with Euclidean ISDiv cost function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
paramNMF.costFunc = 'ISDiv';
[nmfISDivW, nmfISDivH, nmfISDivV] = NMF(A, paramNMF);


% convert cell arrays into matrices
nmfEucDistV=cell2mat(nmfEucDistV);
nmfKLDivV=cell2mat(nmfKLDivV);
nmfISDivV=cell2mat(nmfISDivV);

% save matrices
save([outPath 'nmfEucDistW.mat'], 'nmfEucDistW')
save([outPath 'nmfEucDistH.mat'], 'nmfEucDistH')
save([outPath 'nmfEucDistV.mat'], 'nmfEucDistV')
save([outPath 'nmfKLDivW.mat'], 'nmfKLDivW')
save([outPath 'nmfKLDivH.mat'], 'nmfKLDivH')
save([outPath 'nmfKLDivV.mat'], 'nmfKLDivV')
save([outPath 'nmfISDivW.mat'], 'nmfISDivW')
save([outPath 'nmfISDivH.mat'], 'nmfISDivH')
save([outPath 'nmfISDivV.mat'], 'nmfISDivV')

