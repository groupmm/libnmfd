function [ rgbA ] = coloredComponents( compA, colVec )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: coloredComponents
% Date: April 2018
% Programmer: Christian Dittmar
%
% Maps a cell-array with parallel component spectrograms into a color-coded
% spectrogram image, similar to Fig. 10 in [1]. Works best for three
% components corresponding to RGB.
%
% References:
% [1] Christian Dittmar and Meinard M�ller "Reverse Engineering the Amen
% Break � Score-informed Separation and Restoration applied to Drum
% Recordings" IEEE/ACM Transactions on Audio, Speech, and Language Processing,
% 24(9): 1531�1543, 2016.
%
% Input:  compA               cell-array with the component spectrograms,
%                             all should have the same dimensions
%
% Output: rgbA                color-coded spectrogram
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numComp = length(compA);
[numBins,numFrames] = size(compA{1});

if nargin < 2
    % set up color table
    colVec = hsv(numComp);
else
    colVec = rgb2hsv(colVec);
end;

% initialize as empty
rgbA = zeros(numBins,numFrames,3);

for k = 1:numComp
  maxVal = max(compA{k}(:));

  if maxVal < eps
    maxVal = 1;
  end

  intensity = compA{k}./maxVal;
  intensity = 1-intensity;

  for g = 1:3
    colorSlice(:,:,g) = colVec(k,g)*intensity;
  end
  rgbA = rgbA + colorSlice;

end

% convert to HSV space
hsvA = rgb2hsv(rgbA);

% invert luminance
hsvA(:,:,3) = (hsvA(:,:,3) / max(max(squeeze(hsvA(:,:,3)))));

% shift hue circularly
hsvA(:,:,1) = mod((1/(numComp-1))+hsvA(:,:,1),1);

% convert to RGB space
rgbA = hsv2rgb(hsvA);

%image(rgbA);axis xy

end
