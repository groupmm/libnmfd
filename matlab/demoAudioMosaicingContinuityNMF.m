%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: demoAudioMosaicingContinuityNMF
% Date: May 2018
% Programmer: Christian Dittmar, Jonathan Driedger
%
% This is a demo script which illustrates the main functionalities of the
% 'NMF toolbox'. For a detailed description we refer to [1,2] (see References below).
%
% The script proceeds in the following steps:
%
% 1. It loads an target audio file containing the intro of the song "Let it be", by "The Beatles".
% 2. It loads a source audio file containing the sound of buzzing bees in different pitches.
% 3. It computes the STFT of all audio data.
% 4. It applies the diagonal NMF as described in [1], in order to approximate the target with the timbral content of the source.
% 5. It visualizes the NMF results.
% 6. It resynthesizes the audio mosaic.
%
% References:
% [1] Jonathan Driedger, Thomas Prätzlich, and Meinard Müller
%     Let It Bee — Towards NMF-Inspired Audio Mosaicing
%     In Proceedings of the International Conference on Music Information Retrieval (ISMIR): 350–356, 2015.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If you use the 'NMF toolbox' please refer to
% [2] Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard Müller
%     NMF Toolbox: Music Processing Applications of Nonnegative Matrix Factorization
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

filenameSource = 'Bees_Buzzing.wav';
filenameTarget = 'Beatles_LetItBe.wav';

% 1. load the source and target signal
% read signals
[xs,fs] = audioread([inpPath filenameSource]);
[xt,fs] = audioread([inpPath filenameTarget]);
% make monaural if necessary
xs = mean(xs,2);
xt = mean(xt,2);

% 2. compute STFT of both signals
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
[numBins,numTargetFrames] = size(Xt);
[numBins,numSourceFrames] = size(Xs);
deltaT = paramSTFT.hopSize / fs;
deltaF = fs / paramSTFT.blockSize;

% 3. apply continuity NMF variants to mosaicing pair
% initialize activations randomly
H0 = rand(numSourceFrames,numTargetFrames);

% init templates by source frames
W0 = bsxfun(@times,As,1./(eps+sum(As)));
Xs = bsxfun(@times,Xs,1./(eps+sum(As)));

% parameters taken from Jonathan Driedger's toolbox
paramNMFdiag.fixW = 1;
paramNMFdiag.numOfIter = 20;
paramNMFdiag.continuity.polyphony = 10;
paramNMFdiag.continuity.length = 7;
paramNMFdiag.continuity.grid = 5;
paramNMFdiag.continuity.sparsen = [1 7];

% reference implementation by Jonathan Driedger
[nmfdiagW, nmfdiagH] = NMFdiag(At, W0, H0, paramNMFdiag);

% create mosaic, replace magnitude by complex frames
contY = Xs*nmfdiagH;

% visualize
paramVis = [];
paramVis.deltaF = deltaF;
paramVis.deltaT = deltaT;
fh1 = visualizeComponentsNMF(At, nmfdiagW, nmfdiagH, [], paramVis);

% save result
saveas(fh1,[outPath,'LetItBee_NMFdiag.png']);

% resynthesize using Griffin-Lim, 50 iterations by default
[Xout, Pout, res] = LSEE_MSTFTM_GriffinLim(contY, paramSTFT);

% save result
audiowrite([outPath,'LetItBee_NMFdiag_with_target_',filenameTarget],res,fs);
