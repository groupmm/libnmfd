%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: test_NMFdiag
%
% Saves the resulting matrices of NMFdiag into local directory for test
% purposes.
%
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
outPath = 'matrices/NMFdiag/';

% create directory if necessary
mkdir(outPath);

filenameSource = 'Bees_Buzzing.wav';
filenameTarget = 'Beatles_LetItBe.wav';

warning('OFF','MATLAB:audiovideo:audiowrite:dataClipped');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. load the source and target signal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read signals
[xs,fs] = audioread([inpPath filenameSource]);
[xt,fs] = audioread([inpPath filenameTarget]);

% make monaural if necessary
xs = mean(xs,2);
xt = mean(xt,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. compute STFT of both signals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% spectral parameters
paramSTFT.blockSize = 2048;
paramSTFT.hopSize = 1024;
paramSTFT.winFunc = hann(paramSTFT.blockSize);
paramSTFT.reconstMirror = true;
paramSTFT.appendFrame = true;
paramSTFT.numSamples = length(xt);

% STFT computation
[Xs,As,Ps] = forwardSTFT(xs,paramSTFT);
[Xt,At,Pt] = forwardSTFT(xt,paramSTFT);

% get dimensions and time and freq resolutions
[~,numTargetFrames] = size(Xt);
[~,numSourceFrames] = size(Xs);
deltaT = paramSTFT.hopSize / fs;
deltaF = fs / paramSTFT.blockSize;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. apply continuity NMF variants to mosaicing pair
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize activations randomly
rng(0);
H0 = rand(numSourceFrames,numTargetFrames);
save([outPath 'H0.mat'], 'H0')

% init templates by source frames
W0 = bsxfun(@times,As,1./(eps+sum(As)));
Xs = bsxfun(@times,Xs,1./(eps+sum(As)));

% parameters taken from Jonathan Driedger's toolbox
paramNMFdiag.numOfIter = 3;
paramNMFdiag.continuity.polyphony = 10;
paramNMFdiag.continuity.length = 7;
paramNMFdiag.continuity.grid = 1;
paramNMFdiag.continuity.sparsen = [1 7];

% fixW = 0 for test purposes 
paramNMFdiag.fixW = true;

% call the reference implementation as provided by Jonathan Driedger
% with divergence update rules
[nmfdiagW_div, nmfdiagH_div] = NMFdiag(At, W0, H0, paramNMFdiag);

% save matrices
save([outPath 'nmfdiagW_div.mat'], 'nmfdiagW_div')
save([outPath 'nmfdiagH_div.mat'], 'nmfdiagH_div')



