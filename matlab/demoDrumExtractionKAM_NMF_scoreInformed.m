%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: demoDrumExtractionKAM_NMF_scoreInformed
% Date: April 2018
% Programmer: Christian Dittmar
%
% This is the demo script which illustrates the main functionalities of the
% 'NMF toolbox'. For a detailed description we refer to [1, 2] (see
% References below).
%
% The script proceeds in the following steps:
%   1. It loads an example audio file containing drums and melodic
%      instruments
%   2. It computes the STFT of the audio data.
%   3. It applies KAM and NMF as described in [2], with score-informed
%      initialization of the components [1]
%   4. It visualizes the decomposition results.
%   5. It resynthesizes the separated audio streams and saves them as wav
%      files to the hard drive.
%
% References:
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

% read corresponding transcription files
melodyTranscription = textread([inpPath, 'runningExample_IGotYouMelody.txt']);
drumsTranscription = textread([inpPath, 'runningExample_IGotYouDrums.txt']);

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
saveas(fh1,[outPath,'demoDrumExtractionKAM_NMF_scoreInformed_KAM.png']);

% resynthesize KAM results
for k = 1:2
  Y = kamA{k}.*exp(j*P);
  y = inverseSTFT(Y, paramSTFT);
  % save result
  audiowrite([outPath,'demoDrumExtractionKAM_NMF_scoreInformed_KAM_component_',...
    num2str(k),'_extracted_from_',filename],y,fs);
end

% concatenate new NMF target
V = cell2mat(kamA');
numDoubleBins = size(V,1);

% prepare matrix to revert concatenation
AccuMat = [eye(numBins),eye(numBins)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4. apply score-informed NMF to KAM-based target
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set common parameters
numIterNMF = 30;   % in the score-informed case, less iterations are necessary
numTemplateFrames = 1; % this can also be adjusted upwards

% generate score-informed templates for the melodic part
paramTemplates.deltaF = deltaF;
paramTemplates.numBins = numBins;
paramTemplates.numTemplateFrames = numTemplateFrames;
paramTemplates.pitches = melodyTranscription(:,2);
pitchedW = initTemplates(paramTemplates,'pitched');

% generate score-informed activations for the melodic part
paramActivations.deltaT = deltaT;
paramActivations.numFrames = numFrames;
paramActivations.pitches = melodyTranscription(:,2);
paramActivations.onsets = melodyTranscription(:,1);
paramActivations.durations = melodyTranscription(:,3);
pitchedH = initActivations(paramActivations,'pitched');

% generate score-informed activations for the drum part
paramActivations.drums = drumsTranscription(:,2);
paramActivations.onsets = drumsTranscription(:,1);
paramActivations.decay = 0.75;
drumsH = initActivations(paramActivations,'drums');
numCompDrum = size(drumsH,1);

% generate audio-informed templates for the drum part
paramTemplates.numComp = numCompDrum;
drumsW = initTemplates(paramTemplates,'drums');

% join drum and pitched initialization
initH = [drumsH;pitchedH];

% now get the total number of components
numComp = size(initH,1);

% fill remaining parts of templates with random values
% create joint templates
initW = [];
informedWeight = 5;

% fill missing template parts with noise
for k = 1:numComp
  if k <= numCompDrum
    initW{k} = [informedWeight*drumsW{k}./max(drumsW{k}); ...
      rand(numBins,numTemplateFrames)];
  else
    initW{k} = [rand(numBins,numTemplateFrames); ...
      informedWeight*pitchedW{k-numCompDrum}./max(pitchedW{k-numCompDrum})];
  end

  % normalize to unit sum
  initW{k} = initW{k} / sum(initW{k}(:));

end

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

% set percussiveness to ground truth information
percWeight = [ones(1,numel(drumsW)), zeros(1,numel(pitchedW))];

% compute separate models for percussive and harmonic part
Vp = convModel(tensorW,diag(percWeight)*nmfdH);
Vh = convModel(tensorW,diag(1-percWeight)*nmfdH);

% accumulate back to original spectrum, reverting the stacking
% this step is described in the last paragraph of sec. 2.4 in [2]
Ap = AccuMat * Vp;
Ah = AccuMat * Vh;

% alpha-Wiener filtering
nmfdA = alphaWienerFilter(A,{Ap Ah},1.0);

% visualize results
% create reduced version of templates for visualization
nmfdW = cellfun(@(x) AccuMat*x,nmfdW,'UniformOutput',false);
fh2 = visualizeComponentsNMF(A, nmfdW, nmfdH, nmfdA, paramVis);

% save result
saveas(fh2,[outPath,'demoDrumExtractionKAM_NMF_scoreInformed_NMF.png']);

% resynthesize results of NMF with soft constraints and score information
for k = 1:2
  Y = nmfdA{k}.*exp(j*P);
  y = inverseSTFT(Y, paramSTFT);
  % save result
  audiowrite([outPath,'demoDrumExtractionKAM_NMF_scoreInformed_NMF_component_',...
    num2str(k),'_extracted_from_',filename],y,fs);
end
