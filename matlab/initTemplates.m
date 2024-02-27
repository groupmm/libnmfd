function [ initW ] = initTemplates(parameter,strategy)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: initTemplates
% Date: March 2018
% Programmer: Christian Dittmar
%
% Implements different initialization strategies for NMF templates. The
% strategies 'random' and 'uniform' are self-explaining. The strategy
% 'pitched' uses comb-filter templates as described in [1]. The strategy
% 'drums' uses pre-extracted, averaged spectra of desired drum types [2].
%
% References:
% [1] Jonathan Driedger, Harald Grohganz, Thomas Prätzlich, Sebastian Ewert
% and Meinard Müller "Score-informed audio decomposition and applications"
% In Proceedings of the ACM International Conference on Multimedia (ACM-MM)
% Barcelona, Spain, 2013.
%
% [2] Christian Dittmar and Meinard M�ller "Reverse Engineering the Amen
% Break -- Score-informed Separation and Restoration applied to Drum
% Recordings" IEEE/ACM Transactions on Audio, Speech, and Language Processing,
% 24(9): 1531-1543, 2016.
%
% Input:  parameter.
%           numComp           number of NMF components
%           numBins           number of frequency bins
%           numTemplateFrames number of time frames for 2D-templates
%           pitches           optional vector of MIDI pitch values
%           drumTypes         optional cell-array of drum type strings
%         strategy            string describing the intialization strategy
%
% Output: initW               cell array with the desired templates
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If you use the 'NMF toolbox' please refer to:
% [3]  Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
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
if ~exist('strategy') | isempty(strategy)
  strategy = 'random';
end
if ~isfield(parameter,'pitchTolDown')
  parameter.pitchTolDown = 0.75;
end
if ~isfield(parameter,'pitchTolUp')
  parameter.pitchTolUp = 0.75;
end
if ~isfield(parameter,'numHarmonics')
  parameter.numHarmonics = 25;
end
if ~isfield(parameter,'numTemplateFrames')
  parameter.numTemplateFrames = 1;
end

initW = [];
switch strategy
  case 'random'
    % fix random seed
    rng(0);

    for k = 1:parameter.numComp
      initW{k} = rand(parameter.numBins,parameter.numTemplateFrames);
    end
  case 'uniform'
    for k = 1:parameter.numComp
      initW{k} = ones(parameter.numBins,parameter.numTemplateFrames);
    end
  case 'pitched'

    uniquePitches = unique(parameter.pitches);

    % needs to be overwritten
    parameter.numComp = numel(uniquePitches);

    for k = 1:numel(uniquePitches)
      % initialize as zeros
      initW{k} = eps+zeros(parameter.numBins,parameter.numTemplateFrames);

      % then insert non-zero entries in bands around hypothetic harmonics
      curPitchFreqLower_Hz = midi2freq(uniquePitches(k)-parameter.pitchTolDown);
      curPitchFreqUpper_Hz = midi2freq(uniquePitches(k)+parameter.pitchTolUp);

      for g = 1:parameter.numHarmonics

        currPitchFreqLower_Bins = g*curPitchFreqLower_Hz/parameter.deltaF;
        currPitchFreqUpper_Bins = g*curPitchFreqUpper_Hz/parameter.deltaF;

        binRange = round(currPitchFreqLower_Bins):round(currPitchFreqUpper_Bins);
        binRange(binRange < 1) = [];
        binRange(binRange > parameter.numBins) = [];

        % insert 1/f intensity
        initW{k}(binRange,:) = 1/g;

      end
    end

  case 'drums'
    s = load('../data/dictW.mat');

    % sanity check
    if parameter.numBins == size(s.dictW,1)
      for k = 1:size(s.dictW,2)
        initW{k} = s.dictW(:,k)*linspace(1,0.1,parameter.numTemplateFrames);
      end
    end

    % needs to be overwritten
    parameter.numComp = numel(initW);
end

% do final normalization
for k = 1:parameter.numComp
  initW{k} = initW{k} / (eps+sum(initW{k}(:)));
end

end
