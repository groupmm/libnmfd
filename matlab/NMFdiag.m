function [W, H] = NMFdiag(V, W0, H0, parameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: NMFdiag
% Date of Revision: 2015-02
% Programmer: Jonathan Driedger
%
% Given a non-negative matrix V, find non-negative matrix factors W and H
% such that V ~ WH. Possibly also enforce continuity constraints.
%
% REFERENCES
%	Lee, DD & Seung, HS. "Algorithms for Non-negative Matrix Factorization"
%   Ewert, S. & Mueller, M. "Using Score-Informed Constraints For NMF-Based
%   Source Separation"
%
% Input
%       V               NxM matrix to be facorized
%       W0              Initialized W matrix
%       H0              Initialized H matrix
%
%       parameter.
%         distMeas      Distance measure which is used for the
%                       optimization. Values are 'euclidean' for Euclidean,
%                       or 'divergence' for KL-divergence.
%         numOfIter     Number of iterations the algorithm will run.
%         fixW          Set to 1 if Templates W should be fixed during the
%                       update process.
%                       divergence cost function update rules
%         continuity.   Set of parameters related to the enforced
%                       continuity constraints.
%           length      Number of templates which should be activated
%                       successively.
%           decay       Parameter for specifying the decaying gain of
%                       successively activated templates.
%           grid        This value indicates in wich iterations of the NMF
%                       update procedure the continuity constraints should
%                       be enforced.
%
% Output
%       W               NxK non-negative matrix factor
%       H               KxM non-negative matrix factor
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


if nargin < 4
    parameter = [];
end

if ~isfield(parameter,'distMeas')
    parameter.distMeas = 'divergence';
end
if ~isfield(parameter,'numOfIter')
    parameter.numOfIter = 50;
end
if ~isfield(parameter,'fixW')
    parameter.fixW = 0;
end
if ~isfield(parameter,'continuity')
    parameter.continuity.length = 10;
    parameter.continuity.grid = 5;
    parameter.continuity.sparsen = [1 1];
    parameter.continuity.polyphony = 5;
end
if ~isfield(parameter,'vis')
    parameter.vis = 0;
end

numOfSimulAct = parameter.continuity.polyphony;
vis = parameter.vis;


%% matrices dimensions consistency check

[N, M]   = size(V);  % V matrix dimensions
[WN, WK] = size(W0); % W matrix dimensions
[HK, HM] = size(H0); % H matrix dimensions
K = WK;


% check W matrix dimensions are consistent
if WN ~= N
    error('W matrix has incosistent dimensions.');
end

% check H matrix dimensions are consistent
if HK ~= K || HM ~= M
    error('H matrix has incosistent dimensions.');
end



%% matrices non-negativity check

% check V matrix is non-negative
if find(V < 0)
    error('V matrix must not contain negative values.');
end

% check W matrix is non-negative
if find(W0 < 0)
    error('W matrix must not contain negative values.');
end

% check H matrix is non-negative
if find(H0 < 0)
    error('H matrix must not contain negative values.');
end


%% V matrix factorization

% initialization of W and H
W = W0;
H = H0;

fixW = parameter.fixW;

energyInW = sum(W.^2);
energyScaler = repmat(energyInW',1,size(H,2));

% prepare the max neighborhood kernel
s = parameter.continuity.sparsen;
if mod(s(1),2)~=1 || mod(s(2),2)~=1
    error('sparsity parameter needs to be odd!');
end
maxFiltKernel = zeros(s);
maxFiltKernel(:,ceil(s(2)/2)) = 1;
maxFiltKernel(ceil(s(1)/2),:) = 1;
numOfEntriesInKernel = sum(sum(maxFiltKernel));

for k = 0 : parameter.numOfIter - 1

    if vis
        figure;
        imagesc(H);
        axis xy;
        colormap(1-gray)
        title(['Activation Matrix H in Iteration ' num2str(k+1)]);
    end

    % in every 'grid' iteration of the update...
    if mod(k,parameter.continuity.grid)==0

        % sparsen the activations...
        if max(s) > 1
            % should in principle also include the energyScaler...
            H_filt = ordfilt2(H,numOfEntriesInKernel,maxFiltKernel); % find max values in neighborhood
            H(H ~= H_filt) = H(H ~= H_filt) * (1 - (k+1)/parameter.numOfIter);
        end

        % ...restrict polyphony...
        if numOfSimulAct < size(H,2)
            [~,sortVec] = sort(H.*energyScaler,1,'descend');
            for j = 1:size(H,2)
                H(sortVec(numOfSimulAct+1:end,j),j) = ...
                    H(sortVec(numOfSimulAct+1:end,j),j) * ...
                    (1 - (k+1)/parameter.numOfIter);
            end
        end

        % ... and enforce continuity
        filt = eye(parameter.continuity.length);
        H = conv2(H,filt,'same');

    end

    switch parameter.distMeas

        case 'euclidean' % euclidean update rules
            H = H .* (W' * V ./ ((W' * W * H) + eps));
            if ~fixW
                W = W .* (V * H' ./ (((W * H) * H') + eps));
            end

        case 'divergence' % divergence update rules
            H = H .* (W' * (V ./ ((W * H)+eps))) ./ (sum(W, 1)' * ones(1, M) + eps);
            if ~fixW
                W = W .* ((V ./ ((W * H)+eps)) * H') ./ (ones(N, 1) * sum(H, 2)' + eps);
            end

        otherwise
            error('Unknown distance measure');

    end
end

if vis
    figure;
    imagesc(H);
    axis xy;
    colormap(1-gray)
    title('Final Activation Matrix H');
end

end
