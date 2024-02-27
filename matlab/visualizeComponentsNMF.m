function fh = visualizeComponentsNMF(V, W, H, compV, parameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: visualizeComponentsNMF
% Date: June 2018
% Programmer: Christian Dittmar
%
% Given a non-negative matrix V, and its non non-negative NMF or NMFD components,
% this function provides a visualization.
%
% Input:  V               K x M non-negative target matrix, in our case,
%                         this is usually a magnitude spectrogram
%         W               cell array with R indiviual K X T learned template matrices
%         H               R X M matrix of learned activations
%         compV           cell array with R individual component magnitude
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
% get spectrogram dimensions
[numLinBins, numFrames] = size(V);

% get number of components
R = size(H,1);

% check other parameters
if ~isfield(parameter,'compColVec')
  if R == 2
    parameter.compColVec = [1 0 0;0 0.5 0.5];
  elseif R == 3
    parameter.compColVec = [1 0 0;0 1 0; 0 0 1];
  elseif R == 4
    parameter.compColVec = [1,0,1;1,0.5,0; 0 1 0;0,0.5,1];
  else
    parameter.compColVec = ones(R,1)*[0.5 0.5 0.5];
  end
end

if ~isfield(parameter,'startSec')
  parameter.startSec = 1*parameter.deltaT;
end

if ~isfield(parameter,'endeSec')
  parameter.endeSec = numFrames*parameter.deltaT;
end

if ~isfield(parameter, 'logComp')
    parameter.logComp = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot MMF / NMFD components
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% map the target and the templates to a logarithmically-spaced frequency
% and logarithmic magnitude compression
[logFreqLogMagV,logFreqAxis] = logFreqLogMag( V, parameter.deltaF, [], [], [], parameter.logComp);
[numLogBins] = length(logFreqAxis);

[logFreqLogMagW,logFreqAxis] = logFreqLogMag( W, parameter.deltaF, [], [], [], parameter.logComp);

if ~isempty(compV)
  [logFreqLogMagCompV,logFreqAxis] = logFreqLogMag( compV, parameter.deltaF, [], [], [], parameter.logComp);
else
  logFreqLogMagCompV = {logFreqLogMagV}; % simulate one component
end

timeAxis = [1:numFrames]*parameter.deltaT;
freqAxis = [1:numLogBins];

% subsample freq axis
subSamp = find(mod(logFreqAxis,55) < 0.001);
subSampFreqAxis = logFreqAxis(subSamp);

% further plot params
plotHeight = 0.4;
if ~isfield(parameter, 'fontSize')
    setFontSize = 11;
else
    setFontSize = parameter.fontSize;
end;

startSec = parameter.startSec;
endeSec = parameter.endeSec;

% normalize NMF / NMFD activations to unit maximum
H = bsxfun(@times,H,1./(eps+max(H')'));

fh = figure;

% first, plot the component spectrogram matrix
subplot('Position',[0.38 0.08 0.6 plotHeight]);
if R <= 4
  image(timeAxis,freqAxis,coloredComponents(logFreqLogMagCompV));axis xy  
elseif numel(logFreqLogMagCompV) == 2 % special case for HPSS
  image(timeAxis,freqAxis,coloredComponents(logFreqLogMagCompV));axis xy
else
  imagesc(timeAxis,freqAxis,logFreqLogMagV);axis xy
end
xlim([startSec endeSec]);
set(gca,'YTick',[]);
colormap(flip(gray));
%title('V \approx W \cdot H');
xlabel('Time in seconds');

% second, plot the activations as polygons
subplot('Position',[0.38 0.55 0.6 plotHeight]);
hold off
plot(0);

% decide between different visualizations
if R > 10

  imagesc(timeAxis,[1:R],H);
  axis xy;
  colormap(flip(gray));

  ax = gca;
  ax.YTick = [1 R];
  ax.YTickLabels = [0 R-1];

  ylabel('Template');

else
  hold on
  for r = 1:R
    currActivation = 0.95*H(r,:); % put some headroom
    xcoord = [timeAxis,fliplr(timeAxis)];
    ycoord = r+[zeros(1,numFrames),fliplr(currActivation)];
    fill(xcoord,ycoord,parameter.compColVec(r,:));

  end

  ylim([1 R+1]);
  set(gca,'YTick',0.5+[1:R],'YTickLabel',[1:R]);

end

%set(gca,'XTick',[]);
xlim([startSec endeSec]);
%title('H');

% third, plot the templates
subplot('Position',[0.06 0.08 0.28 plotHeight]);
set(gca,'YTick',[],'XTick',[]);
%title('W');

if R > 10

  numTemplateFrames = 1;
  if iscell(logFreqLogMagW)
    numTemplateFrames = size(logFreqLogMagW{1},2);
    normW = cell2mat(logFreqLogMagW);
  else
    normW = logFreqLogMagW;
  end

  normW = bsxfun(@times,normW,1./(eps+max(normW)));

  imagesc(normW);axis xy
  colormap(flip(gray));

  set(gca,'YTick',subSamp,'YTickLabel',subSampFreqAxis);

  ax = gca;
  ax.XTick = [1 R*numTemplateFrames];
  ax.XTickLabels = [0 R-1];
  xlabel('Template');

  ylabel('Frequency in Hz');

else

  for r = 1:R

    axes('Position',[0.06+(r-1)*0.28/R 0.08 0.28/R plotHeight]);

    if iscell(logFreqLogMagW)
      currTemplate = logFreqLogMagW{r};
    else
      currTemplate = logFreqLogMagW(:,r);
    end

    if R <= 4
      % make a trick to color code the template spectrograms
      tempCellArray = [];
      for g = 1:R
        tempCellArray{g} = zeros(size(currTemplate));
      end
      tempCellArray{r} = currTemplate;

      image(coloredComponents(tempCellArray));axis xy
    else
      currTemplate = currTemplate./max(currTemplate);

      imagesc(currTemplate);axis xy
      colormap(flip(gray));
    end

    if size(currTemplate,2) > 1
      set(gca,'XTick',0.5*size(currTemplate,2));
    else
       set(gca,'XTick',1);
    end

    set(gca,'XTickLabel',r);

    if (r > 1)
      set(gca,'YTick',[]);
    else
      set(gca,'YTick',subSamp,'YTickLabel',subSampFreqAxis);
      ylabel('Frequency in Hz');
    end
  end
end

% set font size and other properties
set(findall(fh,'-property','FontSize'),'FontSize',setFontSize);
set(fh,'Color',[1 1 1]);
set(0,'Units','pixels')
scnsize = get(0,'ScreenSize');
outerPos = scnsize;
set(fh,'OuterPosition',outerPos.*[scnsize(3)*(1-0.9) scnsize(4)*(1-0.9) 0.85 0.85]);
drawnow
