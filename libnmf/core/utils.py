import numpy as np
from scipy.ndimage import median_filter
from typing import Union, List

from libnmf.dsp.utils import conv2
from libnmf.dsp.filters import nema
from libnmf.utils import EPS, load_matlab_dict, midi2freq


def diagonality_soft_constraints_nmf(H: np.ndarray,
                                     kern_ord: int) -> np.ndarray:
    """Implements a simplified version of the soft constraints in [1]

    References
    ----------
    [1] Jonathan Driedger, Thomas Prätzlich, and Meinard Müller
    Let It Bee -- Towards NMF-Inspired Audio Mosaicing
    In Proceedings of the International Conference on Music Information Retrieval (ISMIR): 350-356, 2015.

    Parameters
    ----------
    H: np.ndarray
        NMF activations given as matrix
    kern_ord: int
        Order of smoothing operation

    Returns
    -------
    H: np.ndarray
        Processed NMF activations
    """

    H = conv2(H, np.eye(kern_ord), 'same')

    return H

def percussiveness_estimation(W: np.ndarray) -> np.ndarray:
    """This function takes a matrix or tensor of NMF templates and estimates the percussiveness by assuming that the
    lower part explains percussive and the upper part explains harmonic components. This is explained in sec. 2.4,
    especially eq. (4) in [1].

    References
    ----------
    [1] Christian Dittmar, Patricio López-Serrano, Meinard Müller:
    "Unifying Local and Global Methods for Harmonic-Percussive Source Separation"
    In Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2018.

    Parameters
    ----------
    W: np.ndarray
        K x R matrix (or K x R x T tensor) of NMF (NMFD) templates

    Returns
    -------
    perc_weight: np.ndarray
        The resulting percussiveness estimate per component
    """
    # get dimensions of templates
    K, R, T = W.shape

    # this assumes that the matrix (tensor) is formed in the way we need it
    num_bins = int(K/2)

    # do the calculation, which is essentially a ratio
    perc_weight = np.zeros(R)

    for c in range(R):
        perc_part = W[:num_bins, c, :]
        harm_part = W[:, c, :]
        perc_weight[c] = perc_part.sum() / harm_part.sum()

    return perc_weight




def drum_specific_soft_constraints_nmf(W: np.ndarray,
                                       H: np.ndarray,
                                       iter,
                                       num_iter,
                                       parameter):
    """Implements the drum specific soft constraints that can be applied during
    NMF or NMFD iterations. These constraints affect the activation vectors only and
    are described in sec.23 of [2].

    References
    ----------
    [2] Christian Dittmar, Patricio Lopez-Serrano, Meinard Müller
    Unifying Local and Global Methods for Harmonic-Percussive Source Separation
    In Proceedings of the IEEE International Conference on Acoustics,
    Speech, and Signal Processing (ICASSP), 2018.

    Parameters
    ----------
    W: array-like
        NMF templates given in matrix/tensor form

    H: array-like
        NMF activations given as matrix

    iter: int
        Current iteration count

    num_iter: int
        Target number of iterations

    parameter: dict
        Kern     Order of smoothing operation
        Kern     Concrete smoothing kernel
        initH    Initial version of the NMF activations
        initW    Initial version of the NMF templates

    Returns
    -------
    W: array-like
        Processed NMF templates

    H_out: array-like
        Processed NMF activations
    """
    # this assumes that the templates are constructed as described in sec. 2.4 of [2]
    percWeight = percussiveness_estimation(W).reshape(1, -1)

    # promote harmonic sustained gains
    Hh = median_filter(H, footprint=parameter['Kern'], mode='constant')

    # promote decaying impulses gains
    Hp = nema(H, decay)

    # make weighted sum according to percussiveness measure
    H_out = Hh * (1 - percWeight.T) + Hp * percWeight.T

    return W, H_out

def init_templates(num_comp: int,
                   num_bins: int,
                   pitches: Union[List[int], None],
                   strategy='random',
                   pitch_tol_up: float = 0.75,
                   pitch_tol_down: float = 0.75,
                   num_harmonics: int = 25,
                   num_template_frames: int = 1,
                   delta_F = None) -> np.ndarray:
    """Implements different initialization strategies for NMF templates. The strategies 'random' and 'uniform' are
    self-explaining. The strategy 'pitched' uses comb-filter templates as described in [1]. The strategy 'drums' uses
     pre-extracted, averaged spectra of desired drum types [2].

    References
    ----------
    [1] Jonathan Driedger, Harald Grohganz, Thomas Prätzlich, Sebastian Ewert and Meinard Mueller
    "Score-informed audio decomposition and applications"
    In Proceedings of the ACM International Conference on Multimedia (ACM-MM) Barcelona, Spain, 2013.

    [2] Christian Dittmar and Meinard Müller
    "Reverse Engineering the Amen Break - Score-informed Separation and Restoration applied to Drum Recordings"
    IEEE/ACM Transactions on Audio, Speech, and Language Processing, 24(9): 1531-1543, 2016.

    Parameters
    ----------
    parameter: dict
        numComp           Number of NMF components
        numBins           Number of frequency bins
        numTemplateFrames Number of time frames for 2D-templates
        pitches           Optional array of MIDI pitch values
        drumTypes         Optional list of drum type strings

    strategy: str
        String describing the initialization strategy

    Returns
    -------
    initW: array-like
        List with the desired templates
    """
    # check parameters
    init_W = list()

    if strategy == 'random':
        # fix random seed
        np.random.seed(42)

        for k in range(num_comp):
            init_W.append(np.random.rand(num_bins, num_template_frames))

    elif strategy == 'uniform':
        for k in range(num_comp):
            init_W.append(np.ones((num_bins, num_template_frames)))

    elif strategy == 'pitched':
        unique_pitches = np.unique(pitches)

        # needs to be overwritten
        num_comp = unique_pitches.size

        for k in range(unique_pitches.size):
            # initialize as zeros
            init_W.append(EPS + np.zeros((num_bins, num_template_frames)))

            # then insert non-zero entries in bands around hypothetic harmonics
            cur_pitch_freq_lower_hz = midi2freq(unique_pitches[k] - pitch_tol_down)
            curPitchFreqUpper_Hz = midi2freq(unique_pitches[k] + pitch_tol_up)

            for g in range(num_harmonics):
                curr_pitch_freq_lower_bins = (g + 1) * cur_pitch_freq_lower_hz / delta_F
                curr_pitch_freq_upper_bins = (g + 1) * curPitchFreqUpper_Hz / delta_F

                bin_range = np.arange(int(round(curr_pitch_freq_lower_bins)) - 1, int(round(curr_pitch_freq_upper_bins)))
                bin_range = bin_range[0:num_bins]

                # insert 1/f intensity
                init_W[k][bin_range, :] = 1/(g+1)

    elif strategy == 'drums':
        dict_W = load_matlab_dict('../data/dictW.mat', 'dictW')

        if num_bins == dict_W.shape[0]:
            for k in range(dict_W.shape[1]):
                init_W.append(dict_W[:, k].reshape(-1, 1) * np.linspace(1, 0.1, num_template_frames))

        # needs to be overwritten
        num_comp = len(init_W)

    else:
        raise ValueError('Invalid strategy.')

    # do final normalization
    for k in range(num_comp):
        init_W[k] /= (EPS + init_W[k].sum())

    return init_W


def init_activations(num_comp: int,
                     num_frames: int,
                     strategy: str,
                     delta_T: float,
                     pitches: Union[List[int], None],
                     decay: float = 0.75,
                     onset_offset_tol: float = 0.025,
                     onsets: List = None,
                     durations: List = None,
                     drums: List = None):
    """Implements different initialization strategies for NMF activations. The
    strategies 'random' and 'uniform' are self-explaining. The strategy
    'pitched' places gate-like activations at the frames, where certain notes
    are active in the ground truth transcription [2]. The strategy
    'drums' places decaying impulses at the frames where drum onsets are given
    in the ground truth transcription [3].

    References
    ----------
    [2] Jonathan Driedger, Harald Grohganz, Thomas Prätzlich, Sebastian Ewert
    and Meinard Müller "Score-informed audio decomposition and applications"
    In Proceedings of the ACM International Conference on Multimedia (ACM-MM)
    Barcelona, Spain, 2013.

    [3] Christian Dittmar and Meinard Müller -- Reverse Engineering the Amen
    Break - Score-informed Separation and Restoration applied to Drum
    Recordings" IEEE/ACM Transactions on Audio, Speech, and Language Processing,
    24(9): 1531-1543, 2016.

    Parameters
    ----------
    parameter: dict
        numComp           Number of NMF components
        numFrames         Number of time frames
        deltaT            The temporal resolution
        pitches           Optional array of MIDI pitch values
        onsets            Optional array of note onsets (in seconds)
        durations         Optional array of note durations (in seconds)
        drums             Optional array of drum type indices
        decay             Optional array of decay values per component
        onsetOffsetTol    Optional parameter giving the onset / offset

    strategy: str
        String describing the intialization strategy

    Returns
    -------
    initH: array-like
        Array with initial activation functions
    """

    if strategy == 'random':
        np.random.seed(42)
        init_H = np.random.rand(num_comp, num_frames)

    elif strategy == 'uniform':
        init_H = np.ones((num_comp, num_frames))

    elif strategy == 'pitched':
        uniquePitches = np.unique(pitches)

        # overwrite
        num_comp = uniquePitches.size

        # initialize activations with very small values
        init_H = EPS + np.zeros((num_comp, num_frames))

        for k in range(uniquePitches.size):

            # find corresponding note onsets and durations
            ind = np.nonzero(pitches == uniquePitches[k])[0]

            # insert activations
            for g in range(len(ind)):
                curr_ind = ind[g]

                note_start_in_seconds = onsets[curr_ind]
                note_end_in_seconds = note_start_in_seconds + durations[curr_ind]

                note_start_in_seconds -= onset_offset_tol
                note_end_in_seconds += onset_offset_tol

                note_start_in_frames = int(round(note_start_in_seconds / delta_T))
                note_ende_in_frames = int(round(note_end_in_seconds / delta_T))

                frameRange = np.arange(note_start_in_frames, note_ende_in_frames + 1)
                frameRange = frameRange[frameRange >= 0]
                frameRange = frameRange[frameRange <= num_frames]

                # insert gate-like activation
                init_H[k, frameRange-1] = 1

    elif strategy == 'drums':
        uniqueDrums = np.unique(drums)

        # overwrite
        num_comp = uniqueDrums.size

        # initialize activations with very small values
        init_H = EPS + np.zeros((num_comp, num_frames))

        # sanity check
        if uniqueDrums.size == num_comp:

            # insert impulses at onset positions
            for k in range(len(uniqueDrums)):
                currOns = np.nonzero(drums == uniqueDrums[k])[0]
                currOns = onsets[currOns]
                currOns = np.round(currOns/delta_T).astype(np.int)
                currOns = currOns[currOns >= 0]
                currOns = currOns[currOns <= num_frames]

                init_H[uniqueDrums[k].astype(int)-1, currOns-1] = 1

            # add exponential decay
            init_H = nema(init_H, decay)

    else:
        raise ValueError('Invalid strategy.')

    return init_H


def shift_operator(A: np.ndarray,
                   shift_amount: int) -> np.ndarray:
    """Shift operator as described in eq. (5) from [1]. It shifts the columns of a matrix to the left or the right and
    fills undefined elements with zeros.

    References
    ----------
    [1] Paris Smaragdis
    "Non-negative Matrix Factor Deconvolution; Extraction of Multiple Sound Sources from Monophonic Inputs".
    International Congress on Independent Component Analysis and Blind Signal Separation (ICA), 2004

    Parameters
    ----------
    A: np.ndarray
        Arbitrary matrix to undergo the shifting operation

    shift_amount: int
        Positive numbers shift to the right, negative numbers shift to the left, zero leaves the matrix unchanged

    Returns
    -------
    shifted: np.ndarray
        Result of this operation
    """
    # Get dimensions
    num_rows, num_cols = A.shape

    # Limit shift range
    shift_amount = np.sign(shift_amount) * min(abs(shift_amount), num_cols)

    # Apply circular shift along the column dimension
    shifted = np.roll(A, shift_amount, axis=-1)

    if shift_amount < 0:
        shifted[:, num_cols + shift_amount: num_cols] = 0

    elif shift_amount > 0:
        shifted[:, 0: shift_amount] = 0

    else:
        pass

    return shifted
