%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: demoDrumExtractionKAM_NMF_percThreshold
% Date: May 2018
% Programmer: Christian Dittmar
%
% This is the demo script which illustrates the main functionalities of the
% 'NMF toolbox'. For a detailed description we refer to [1,2] (see
% References below).
%
% The script proceeds in the following steps:
%   1. It loads an example audio file containing drums and melodic
%      instruments
%   2. It computes the STFT of the audio data.
%   3. It applies KAM and NMF as described in [2], with random
%      initialization of the NMF components. The final classification into
%      harmonic and percussive is done according to the percussiveness
%      threshold p_thresh = 0.25 as given in [2].
%   4. It visualizes the decomposition results.
%   5. It resynthesizes the separated audio streams and saves them as wav
%      files to the hard drive.
%
% References:%
% [1] Christian Dittmar, Meinard Müller
%     Reverse Engineering the Amen Break "Score-informed Separation and
%     Restoration applied to Drum Recordings
%     IEEE/ACM Transactions on Audio, Speech, and Language Processing,
%     24(9): 1531-1543, 2016.
% [2] Christian Dittmar, Patricio López-Serrano, Meinard Müller
%     Unifying Local and Global Methods for Harmonic-Percussive Source Separation
%     In Proceedings of the IEEE International Conference on Acoustics,
%     Speech, and Signal Processing (ICASSP), 2018.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If you use the 'NMF toolbox' please refer to:
% [3] Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
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

inpPath = '../data/';
outPath = 'output/';

% create output directory, if it doesn't exist
if ~exist(outPath, 'dir')
  mkdir(outPath);
end

filename = 'runningExample_IGotYouMixture.wav';

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

% get logarithmically-spaced frequency axis version for visualization
% purposes
[logFreqLogMagA,logFreqAxis] = logFreqLogMag( A, deltaF );
[numLogBins] = length(logFreqAxis);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. apply KAM-based Harmonic Percussive Separation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set common parameters
numIterKAM = 30;
[kamA,Kern,KernOrd] = HPSS_KAM(A, numIterKAM, 13);

% visualize
paramVis = [];
paramVis.deltaT = deltaT;
paramVis.deltaF = deltaF;
fh1 = visualizeComponentsKAM(kamA, paramVis);

% save result
saveas(fh1,[outPath,'demoDrumExtractionKAM_NMF_percThreshold_KAM.png']);

% resynthesize KAM results
for k = 1:2
  Y = kamA{k}.*exp(j*P);
  y = inverseSTFT(Y, paramSTFT);
  % save result
  audiowrite([outPath,'demoDrumExtractionKAM_NMF_percThreshold_KAM_component_',...
    num2str(k),'_extracted_from_',filename],y,fs);
end

% concatenate new NMF target
V = cell2mat(kamA');
numDoubleBins = size(V,1);

% prepare matrix to revert concatenation,
AccuMat = [eye(numBins),eye(numBins)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4. apply NMF with drum-specific soft constraints to KAM-based target
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set common parameters
numIterNMF = 60;
numComp = 30;
numTemplateFrames = 1;

% generate random templates covering 2 times the original frequency range
paramTemplates.numComp = numComp;
paramTemplates.numBins = numDoubleBins;
paramTemplates.numTemplateFrames = numTemplateFrames;
initW = initTemplates(paramTemplates,'random');

% generate uniform activations
paramActivations.numComp = numComp;
paramActivations.numFrames = numFrames;
initH = initActivations(paramActivations,'uniform');

% set NMFD parameters
paramNMFD.numComp = numComp;
paramNMFD.numFrames = numFrames;
paramNMFD.numIter = numIterNMF;
paramNMFD.numTemplateFrames = numTemplateFrames;
paramNMFD.numBins = numDoubleBins;
paramNMFD.initW = initW;
paramNMFD.initH = initH;

% set soft constraint parameters
paramConstr = [];
paramConstr.funcPointerPreProcess = @drumSpecificSoftConstraintsNMF;
paramConstr.Kern = Kern;
paramConstr.KernOrd = KernOrd;
paramConstr.decay = 0.75;
paramConstr.numBinsDrum = numBins;

% NMFD core method
[nmfdW, nmfdH, ~, ~, tensorW] = NMFD(V, paramNMFD, paramConstr);

% get final percussiveness estimate
[ percWeight ] = percussivenessEstimation( tensorW );

% re-order components, only for visualization
[ord,sortInd] = sort(percWeight,'descend');
tensorW = tensorW(:,sortInd,:);
nmfdH = nmfdH(sortInd,:);
nmfdW = nmfdW(sortInd);
percWeight = percWeight(sortInd);

% perform final thresholding
percWeight = double(percWeight > 0.25);

% compute separate models for percussive and harmonic part
% in the case of numTemplateFrames=1, this step equals eq. (1) in [3]
Vp = convModel(tensorW,diag(percWeight)*nmfdH);
Vh = convModel(tensorW,diag(1-percWeight)*nmfdH);

% accumulate back to original spectrum, reverting the stacking
% this step is described in the last paragraph of sec. 2.4 in [3]
Ap = AccuMat * Vp;
Ah = AccuMat * Vh;

% alpha-Wiener filtering
nmfdA = alphaWienerFilter(A,{Ap Ah},1);

% create reduced version of templates for visualization
nmfdW = cellfun(@(x) AccuMat*x,nmfdW,'UniformOutput',false);
fh2 = visualizeComponentsNMF(A, nmfdW, nmfdH, nmfdA, paramVis);

% save result
saveas(fh2,[outPath,'demoDrumExtractionKAM_NMF_percThreshold_NMF.png']);

% resynthesize NMF with soft constraints results
for k = 1:2
  Y = nmfdA{k}.*exp(j*P);
  y = inverseSTFT(Y, paramSTFT);
  % save result
  audiowrite([outPath,'demoDrumExtractionKAM_NMF_percThreshold_NMF_component_',...
    num2str(k),'_extracted_from_',filename],y,fs);
end
