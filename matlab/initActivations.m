function [ initH ] = initActivations(parameter,strategy)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: initActivations
% Date: March 2018
% Programmer: Christian Dittmar
%
% Implements different initialization strategies for NMF activations. The
% strategies 'random' and 'uniform' are self-explaining. The strategy
% 'pitched' places gate-like activations at the frames, where certain notes
% are active in the ground truth transcription [1]. The strategy
% 'drums' places decaying impulses at the frames where drum onsets are given
% in the ground truth transcription [2].
%
% References:
% [1] Jonathan Driedger, Harald Grohganz, Thomas Prätzlich, Sebastian Ewert
% and Meinard Müller "Score-informed audio decomposition and applications"
% In Proceedings of the ACM International Conference on Multimedia (ACM-MM)
% Barcelona, Spain, 2013.
%
% [2] Christian Dittmar and Meinard Müller "Reverse Engineering the Amen
% Break -- Score-informed Separation and Restoration applied to Drum
% Recordings" IEEE/ACM Transactions on Audio, Speech, and Language Processing,
% 24(9): 1531-1543, 2016.
%
% Input:  parameter.
%           numComp           number of NMF components
%           numFrames         number of time frames
%           deltaT            the temporal resolution
%           pitches           optional vector of MIDI pitch values
%           onsets            optional vector of note onsets (in seconds)
%           durations         optional vector of note durations (in seconds)
%           drums             optional vector of drum type indices
%           decay             optional vector of decay values per component
%           onsetOffsetTol    optional parameter giving the onset / offset
%                             tolerance (in seconds)
%         strategy            string describing the intialization strategy
%
% Output: initH               matrix with intial activation functions
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if ~isfield(parameter,'decay');
  parameter.decay = 0.75;
end

if ~isfield(parameter,'onsetOffsetTol');
  parameter.onsetOffsetTol = 0.025;
end

initH = [];
switch strategy
  case 'random'
    % fix random seed
    rng(0);
    initH = rand(parameter.numComp,parameter.numFrames);

  case 'uniform'
    initH = ones(parameter.numComp,parameter.numFrames);

  case 'pitched'
    uniquePitches = unique(parameter.pitches);

    % overwrite
    parameter.numComp = numel(uniquePitches);

    % initialize activations with very small values
    initH = eps + zeros(parameter.numComp,parameter.numFrames);

    for k = 1:numel(uniquePitches)

      % find corresponding note onsets and durations
      ind = find(parameter.pitches == uniquePitches(k));

      % insert activations
      for g = 1:length(ind)

        currInd = ind(g);

        noteStartInSeconds = parameter.onsets(currInd);
        noteEndeInSeconds = noteStartInSeconds + parameter.durations(currInd);

        noteStartInSeconds = noteStartInSeconds-parameter.onsetOffsetTol;
        noteEndeInSeconds = noteEndeInSeconds+parameter.onsetOffsetTol;

        noteStartInFrames = round(noteStartInSeconds/parameter.deltaT);
        noteEndeInFrames = round(noteEndeInSeconds/parameter.deltaT);
        frameRange = [noteStartInFrames:noteEndeInFrames];
        frameRange(frameRange < 1) = [];
        frameRange(frameRange > parameter.numFrames) = [];

        % insert gate-like activation
        initH(k,frameRange) = 1;

      end
    end

  case 'drums'
    uniqueDrums = unique(parameter.drums);

    % overwrite
    parameter.numComp = numel(uniqueDrums);

    % initialize activations with very small values
    initH = eps + zeros(parameter.numComp,parameter.numFrames);

    % sanity check
    if numel(uniqueDrums) == parameter.numComp

      % insert impulses at onset positions
      for k = 1:length(uniqueDrums)
        currOns = find(parameter.drums == uniqueDrums(k));
        currOns = parameter.onsets(currOns);
        currOns = round(currOns/parameter.deltaT);

        currOns(currOns == 0) = 1;

        valid = find(currOns > 0 & currOns < parameter.numFrames);
        initH(uniqueDrums(k),currOns(valid)) = 1;
      end

      % add exponential decay
      initH = NEMA(initH,parameter.decay);

    end
end

end
