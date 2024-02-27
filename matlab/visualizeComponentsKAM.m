function fh = visualizeComponentsKAM(compA, parameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: visualizeComponentsKAM
% Date: June 2018
% Programmer: Christian Dittmar
%
% Given a cell array of component spectrograms, this function provides
% a visualization.
%
% Input:  compA           cell array with R individual component magnitude
%                         spectrograms
%         parameter.
%           deltaT        temporal resolution
%           deltaF        spectral resolution
%           startSec      where to zoom in on the time axis
%           endeSec       where to zoom in on the time axis
%
% Output: fh              the figure handle
%
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
% check parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get number of components
R = numel(compA);

% get spectrogram dimensions
[numLinBins,numFrames] = size(compA{1});

if ~isfield(parameter,'startSec')
  parameter.startSec = 1*parameter.deltaT;
end

if ~isfield(parameter,'endeSec')
  parameter.endeSec = numFrames*parameter.deltaT;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot MMF / NMFD components
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% map template spectrograms to a logarithmically-spaced frequency
% and logarithmic magnitude compression
[logFreqLogMagCompA,logFreqAxis] = logFreqLogMag( compA, parameter.deltaF );
numLogBins = numel(logFreqAxis);

timeAxis = [1:numFrames]*parameter.deltaT;
freqAxis = [1:numLogBins];

% subsample freq axis
subSamp = find(mod(logFreqAxis,55) < 0.001);
subSampFreqAxis = logFreqAxis(subSamp);

% further plot params
plotHeight = 0.4;
setFontSize = 11;
startSec = parameter.startSec;
endeSec = parameter.endeSec;

% make new figure
fh = figure;

% plot the component spectrogram matrix
subplot('Position',[0.38 0.08 0.6 plotHeight]);
image(timeAxis,freqAxis,coloredComponents(logFreqLogMagCompA));axis xy
xlim([startSec endeSec]);
set(gca,'YTick',[]);
colormap(flip(gray));
title('A = A_p + A_h');
xlabel('Time in seconds');

% set font size and other properties
set(findall(fh,'-property','FontSize'),'FontSize',setFontSize);
set(fh,'Color',[1 1 1]);
set(0,'Units','pixels')
scnsize = get(0,'ScreenSize');
outerPos = scnsize;
set(fh,'OuterPosition',outerPos.*[scnsize(3)*(1-0.9) scnsize(4)*(1-0.9) 0.85 0.85]);
drawnow
