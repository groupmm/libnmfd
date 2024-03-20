import numpy as np
from tqdm import tnrange
from typing import List, Tuple, Union

from libnmfd.dsp.filters import nema
from libnmfd.utils import EPS, load_matlab_dict, midi2freq
from libnmfd.utils.core_utils import drum_specific_soft_constraints_nmf


def nmf_conv(V:np.ndarray,
             num_comp: int = 3,
             num_iter: int = 30,
             num_template_frames: int = 8,
             beta: float = 0,
             init_W: np.ndarray = None,
             init_H: np.ndarray = None,
             sparsity_weight: float = 0,
             uncorr_weight: float = 0,
             num_bins: int = None,
             num_frames : int = None,
             **kwargs) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Convolutive Non-Negative Matrix Factorization with Beta-Divergence and optional regularization parameters as
     described in chapter 3.7 of [1]. The averaged activation updates are computed via the compact algorithm given
     in paragraph 3.7.3. For the sake of consistency, we use the notation from [2] instead of the one from the book.

    References
    ----------
    [1] Andrzej Cichocki, Rafal Zdunek, Anh Huy Phan, and Shun-ichi Amari
    Nonnegative Matrix and Tensor Factorizations: Applications to Exploratory Multi-Way Data Analysis and Blind Source
    Separation
    John Wiley and Sons, 2009.

    [2] Christian Dittmar and Meinard Müller
    Reverse Engineering the Amen Break — Score-Informed Separation and Restoration Applied to Drum Recordings
    IEEE/ACM Transactions on Audio, Speech, and Language Processing, 24(9): 1531–1543, 2016.

    Parameters
    ----------
    V: np.ndarray
        Matrix that shall be decomposed (typically a magnitude spectrogram of dimension num_bins x num_frames)

    num_comp: int
        Number of NMFD components (denoted as R in [2])

    num_iter: int
        Number of NMFD iterations (denoted as L in [2])

    num_template_frames: int
        Number of time frames for the 2D-template (denoted as T in [2])

    init_W: np.ndarray
        An initial estimate for the templates (denoted as W^(0) in [2])

    init_H: np.ndarray
        An initial estimate for the gains (denoted as H^(0) in [2])

    beta: float
        The beta parameter of the divergence:
            -1 -> equals Itakura Saito divergence
             0 -> equals Kullback Leiber divergence
             1 -> equals Euclidean distance

    sparsity_weight: float
        Strength of the activation sparsity

    uncorr_weight: float
        Strength of the template uncorrelatedness

    Returns
    -------
    W: np.ndarray
        List with the learned templates

    H: np.ndarray
        Matrix with the learned activations

    cnmfY: np.ndarray
        List with approximated component spectrograms

    cost_func: np.ndarray
        The approximation quality per iteration
    """
    # use parameter nomenclature as in [2]
    K, M = V.shape
    num_bins = K if num_bins is None else num_bins
    num_frames = M if num_frames is None else num_frames
    T = num_template_frames
    R = num_comp
    L = num_iter


    init_W = init_templates(num_comp=num_comp,
                            num_bins=num_bins,
                            strategy='random',
                            **kwargs) if init_W is None else init_W

    tensor_W = np.zeros((init_W[0].shape[0], R, T))

    # stack the templates into a tensor
    for r in range(R):
        tensor_W[:, r, :] = init_W[r].copy()

    init_H = init_activations(num_comp=num_comp,
                              num_frames=num_frames,
                              strategy='uniform',
                              **kwargs) if init_H is None else init_H

    # copy initial
    H = init_H.copy()

    # this is important to prevent initial jumps in the divergence measure
    V_tmp = V / (EPS + V.sum())

    cost_func = np.zeros(L)

    for iter in tnrange(L, desc='Processing'):
        # compute first approximation
        Lambda = conv_model(tensor_W, H)
        LambdaBeta = Lambda ** beta
        Q = V_tmp * LambdaBeta / Lambda

        costMat = V_tmp * (V_tmp ** beta - Lambda ** beta) / (EPS + beta) - (
                    V_tmp ** (beta + 1) - Lambda ** (beta + 1)) / (EPS +
                                                                   beta + 1)
        cost_func[iter] = costMat.mean()

        for t in range(T):
            # respect zero index
            tau = t

            # use tau for shifting and t for indexing
            transp_H = shift_operator(H, tau).T

            numerator_update_W = Q @ transp_H

            denominator_update_W = EPS + LambdaBeta @ transp_H + uncorr_weight * \
                                 np.sum(tensor_W[:, :, np.setdiff1d(np.arange(T), np.array([t]))], axis=2)

            tensor_W[:, :, t] *= numerator_update_W / denominator_update_W

        numerator_update_H = conv_model(np.transpose(tensor_W, (1, 0, 2)), np.fliplr(Q))
        denominator_update_H = conv_model(np.transpose(tensor_W, (1, 0, 2)), np.fliplr(LambdaBeta)) + sparsity_weight\
                               + EPS
        H *= np.fliplr(numerator_update_H / denominator_update_H)

        # normalize templates to unit sum
        norm_vec = np.sum(np.sum(tensor_W, axis=0), axis=1).reshape(-1, 1)
        tensor_W = tensor_W * 1 / (EPS + norm_vec)

    W = list()
    cnmfY = list()

    for r in range(R):
        W.append(tensor_W[:, r, :])
        cnmfY.append(conv_model(np.expand_dims(tensor_W[:, r, :], axis=1), np.expand_dims(H[r, :], axis=0)))

    return W, H, cnmfY, cost_func


def nmfd(V: np.ndarray,
         num_comp: int = 3,
         num_iter: int = 30,
         num_template_frames: int = 8,
         init_W: np.ndarray = None,
         init_H: np.ndarray = None,
         func_preprocess=drum_specific_soft_constraints_nmf,
         func_postprocess=None,
         fix_W: bool = False,
         fix_H: bool = False,
         num_bins: int = None, # TODO: Are these really needed?
         num_frames: int = None,
         **kwargs) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """Non-Negative Matrix Factor Deconvolution with Kullback-Leibler-Divergence and fixable components. The core
     algorithm was proposed in [1], the specific adaptions are used in [2].

    References
    ----------
    [1] Paris Smaragdis
    Non-negative Matrix Factor Deconvolution; Extraction of Multiple Sound Sources from Monophonic Inputs".
    International Congress on Independent Component Analysis and Blind Signal Separation (ICA), 2004

    [2] Christian Dittmar and Meinard Müller
    Reverse Engineering the Amen Break — Score-Informed Separation and Restoration Applied to Drum Recordings
    IEEE/ACM Transactions on Audio, Speech, and Language Processing, 24(9): 1531–1543, 2016.

    Parameters
    ----------
    V: np.ndarray
        Matrix that shall be decomposed (typically a magnitude spectrogram of dimension numBins x numFrames)

    num_comp: int
        Number of NMFD components (denoted as R in [2])

    num_iter: int
        Number of NMFD iterations (denoted as L in [2])

    num_template_frames: int
        Number of time frames for the 2D-template (denoted as T in [2])

    init_W: np.ndarray
        An initial estimate for the templates (denoted as W^(0) in [2])

    init_H: np.ndarray
        An initial estimate for the gains (denoted as H^(0) in [2])

    fix_W: bool
        TODO

    fix_H: bool
        TODO

    func_preprocess: function
        Call for preprocessing

    func_postprocess: function
        Call for postprocessing

    Returns
    -------
    W: List[np.ndarray]
        List with the learned templates

    H: np.ndarray
        Matrix with the learned activations

    nmfd_V: List[np.ndarray]
        List with approximated component spectrograms

    cost_func: np.ndarray
        The approximation quality per iteration

    tensor_W: np.ndarray
        If desired, we can also return the tensor
    """
    # use parameter nomenclature as in [2]
    K, M = V.shape
    T = num_template_frames
    R = num_comp
    L = num_iter

    num_bins = K if num_bins is None else num_bins
    num_frames = M if num_frames is None else num_frames

    tensor_W = np.zeros((init_W[0].shape[0], R, T))
    cost_func = np.zeros(L)

    init_W = init_templates(num_comp=num_comp,
                            num_bins=num_bins,
                            strategy='random',
                            **kwargs) if init_W is None else init_W

    init_H = init_activations(num_comp=num_comp,
                              num_frames=num_frames,
                              strategy='uniform',
                              **kwargs) if init_H is None else init_H

    # stack the templates into a tensor
    for r in range(R):
        tensor_W[:, r, :] = init_W[r]

    # the activations are matrix shaped
    H = init_H.copy()

    # create helper matrix of all ones (denoted as J in eq (5,6) in [2])
    ones_matrix = np.ones((K, M))

    # this is important to prevent initial jumps in the divergence measure
    V_tmp = V / (EPS + V.sum())

    for iter in tnrange(L, desc='Processing'):
        # if given from the outside, apply soft constraints
        if func_preprocess is not None:
            tensor_W, H = func_preprocess(tensor_W, H, iter, L, **kwargs)

        # compute first approximation
        Lambda = conv_model(tensor_W, H)

        # store the divergence with respect to the target spectrogram
        cost_mat = V_tmp * np.log(1.0 + V_tmp/(Lambda+EPS)) - V_tmp + Lambda
        cost_func[iter] = cost_mat.mean()

        # compute the ratio of the input to the model
        Q = V_tmp / (Lambda + EPS)

        # accumulate activation updates here
        mult_H = np.zeros((R, M))

        # go through all template frames
        for t in range(T):
            # use tau for shifting and t for indexing
            tau = t

            # The update rule for W as given in eq. (5) in [2]
            # pre-compute intermediate, shifted and transposed activation matrix
            transpH = shift_operator(H, tau).T

            # multiplicative update for W
            multW = Q @ transpH / (ones_matrix @ transpH + EPS)

            if not fix_W:
                tensor_W[:, :, t] *= multW

            # The update rule for W as given in eq. (6) in [2]
            # pre-compute intermediate matrix for basis functions W
            transp_W = tensor_W[:, :, t].T

            # compute update term for this tau
            add_W = (transp_W @ shift_operator(Q, -tau)) / (transp_W @ ones_matrix + EPS)

            # accumulate update term
            mult_H += add_W

        # multiplicative update for H, with the average over all T template frames
        if not fix_H:
            H *= mult_H / T

        # if given from the outside, apply soft constraints
        if func_postprocess in kwargs:
            tensor_W, H = func_postprocess(tensor_W, H, iter, L, **kwargs)

        # normalize templates to unit sum
        norm_vec = tensor_W.sum(axis=2).sum(axis=0)
        tensor_W *= 1.0 / (EPS + np.expand_dims(norm_vec, axis=1))

    W = list()
    nmfd_V = list()

    # compute final output approximation
    for r in range(R):
        W.append(tensor_W[:, r, :])
        nmfd_V.append(conv_model(np.expand_dims(tensor_W[:, r, :], axis=1), np.expand_dims(H[r, :], axis=0)))

    return W, H, nmfd_V, cost_func, tensor_W



def conv_model(W: np.ndarray,
               H: np.ndarray) -> np.ndarray:
    """Convolutive NMF model implementing the eq. (4) from [1]. Note that it can also be used to compute the standard
     NMF model in case the number of time frames of the templates equals one.

    References
    ----------
    [1] Christian Dittmar and Meinard Müller
    Reverse Engineering the Amen Break — Score-Informed Separation and Restoration Applied to Drum Recordings
    IEEE/ACM Transactions on Audio, Speech, and Language Processing, 24(9): 1531–1543, 2016.

    Parameters
    ----------
    W: np.ndarray
        Tensor holding the spectral templates which can be interpreted as a set of
        spectrogram snippets with dimensions: num_bins x num_comp x num_template_frames

    H: np.ndarray
        Corresponding activations with dimensions: num_comp x num_target_frames

    Returns
    -------
    lamb: np.ndarray
        Approximated spectrogram matrix

    """
    # the more explicit matrix multiplication will be used
    num_bins, num_comp, num_template_frames = W.shape
    num_comp, num_frames = H.shape

    # initialize with zeros
    lamb = np.zeros((num_bins, num_frames))

    # this is doing the math as described in [1], eq (4)
    # the alternative conv2() method does not show speed advantages
    for k in range(num_template_frames):
        mult_result = W[:, :, k] @ shift_operator(H, k)
        lamb += mult_result

    lamb += EPS

    return lamb


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


def init_templates(num_comp: int,
                   num_bins: int,
                   num_template_frames: int = 1,
                   strategy: str = 'random',
                   pitches: List[int] = None,
                   pitch_tol_up: float = 0.75,
                   pitch_tol_down: float = 0.75,
                   num_harmonics: int = 25,
                   freq_res: float = None) -> List[np.ndarray]:
    """Implements different initialization strategies for NMF templates. The strategies 'random' and 'uniform' are
    self-explaining. The strategy 'pitched' uses comb-filter templates as described in [1]. The strategy 'drums' uses
    pre-extracted, averaged spectra of desired drum types [2].

    References
    ----------
    [1] Jonathan Driedger, Harald Grohganz, Thomas Prätzlich, Sebastian Ewert, and Meinard Müller
    Score-Informed Audio Decomposition and Applications
    In Proceedings of the ACM International Conference on Multimedia (ACM-MM): 541–544, 2013.

    [2] Christian Dittmar and Meinard Müller
    Reverse Engineering the Amen Break — Score-Informed Separation and Restoration Applied to Drum Recordings
    IEEE/ACM Transactions on Audio, Speech, and Language Processing, 24(9): 1531–1543, 2016.

    Parameters
    ----------
    num_comp: int
        Number of NMF components

    num_bins: int
        Number of frequency bins

    num_template_frames: int
        Number of time frames for 2D-templates

    strategy: str
        String describing the initialization strategy

    pitches: list
        Optional list of MIDI pitch values

    pitch_tol_up: float
        TODO

    pitch_tol_down: float
        TODO

    num_harmonics: int
        Number of harmonics

    freq_res: float
        Spectral resolution

    Returns
    -------
    initW: List[np.ndarray]
        List with the desired templates
    """
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
        if pitches is None:
            assert ValueError('pitches must be specified.')
        unique_pitches = np.unique(pitches)

        # needs to be overwritten
        num_comp = unique_pitches.size

        for k in range(unique_pitches.size):
            # initialize as zeros
            init_W.append(EPS + np.zeros((num_bins, num_template_frames)))

            # then insert non-zero entries in bands around hypothetic harmonics
            cur_pitch_freq_lower_hz = midi2freq(unique_pitches[k] - pitch_tol_down)
            cur_pitch_freq_upper_hz = midi2freq(unique_pitches[k] + pitch_tol_up)

            for g in range(num_harmonics):
                curr_pitch_freq_lower_bins = (g + 1) * cur_pitch_freq_lower_hz / freq_res
                curr_pitch_freq_upper_bins = (g + 1) * cur_pitch_freq_upper_hz / freq_res

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
                     time_res: float = None,
                     pitches: List[int] = None,
                     decay: Union[np.ndarray, float] = None,
                     onsets: List[float] = None,
                     durations: List[float] = None,
                     drums: List[str] = None,
                     onset_offset_tol: float = 0.025):
    """Implements different initialization strategies for NMF activations. The strategies 'random' and 'uniform' are
    self-explaining. The strategy pitched' places gate-like activations at the frames, where certain notes are active
    in the ground truth transcription [1]. The strategy drums' places decaying impulses at the frames where drum onsets
    are given in the ground truth transcription [2].

    References
    ----------
    [1] Jonathan Driedger, Harald Grohganz, Thomas Prätzlich, Sebastian Ewert, and Meinard Müller
    Score-Informed Audio Decomposition and Applications
    In Proceedings of the ACM International Conference on Multimedia (ACM-MM): 541–544, 2013.

    [2] Christian Dittmar and Meinard Müller
    Reverse Engineering the Amen Break — Score-Informed Separation and Restoration Applied to Drum Recordings
    IEEE/ACM Transactions on Audio, Speech, and Language Processing, 24(9): 1531–1543, 2016.

    Parameters
    ----------
    num_comp: int
        Number of NMF components

    num_frames: int
        Number of time frames

    strategy: str
        String describing the initialization strategy

    time_res: float
        The temporal resolution

    pitches: list or None
        Optional list of MIDI pitch values

    decay: np.ndarray or float
        The decay parameter in the range [0 ... 1], this can be given as a column-vector with individual decays per row
        or as a scalar

    onsets: list
        Optional list of note onsets (in seconds)

    durations: list
        Optional list of note durations (in seconds)

    drums: list
        Optional list of drum type indices

    onset_offset_tol: float
        Optional parameter giving the onset / offset

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
        if pitches is None:
            assert ValueError('pitches must be specified.')
        unique_pitches = np.unique(pitches)

        # overwrite
        num_comp = unique_pitches.size

        # initialize activations with very small values
        init_H = EPS + np.zeros((num_comp, num_frames))

        for k in range(unique_pitches.size):
            # find corresponding note onsets and durations
            ind = np.nonzero(pitches == unique_pitches[k])[0]

            # insert activations
            for g in range(len(ind)):
                curr_ind = ind[g]

                note_start_in_seconds = onsets[curr_ind]
                note_end_in_seconds = note_start_in_seconds + durations[curr_ind]

                note_start_in_seconds -= onset_offset_tol
                note_end_in_seconds += onset_offset_tol

                note_start_in_frames = int(round(note_start_in_seconds / time_res))
                note_ende_in_frames = int(round(note_end_in_seconds / time_res))

                frame_range = np.arange(note_start_in_frames, note_ende_in_frames + 1)
                frame_range = frame_range[frame_range >= 0]
                frame_range = frame_range[frame_range <= num_frames]

                # insert gate-like activation
                init_H[k, frame_range-1] = 1

    elif strategy == 'drums':
        unique_drums = np.unique(drums)

        # overwrite
        num_comp = unique_drums.size

        # initialize activations with very small values
        init_H = EPS + np.zeros((num_comp, num_frames))

        # sanity check
        if unique_drums.size == num_comp:

            # insert impulses at onset positions
            for k in range(len(unique_drums)):
                curr_ons = np.nonzero(drums == unique_drums[k])[0]
                curr_ons = onsets[curr_ons]
                curr_ons = np.round(curr_ons/time_res).astype(np.int)
                curr_ons = curr_ons[curr_ons >= 0]
                curr_ons = curr_ons[curr_ons <= num_frames]

                init_H[unique_drums[k].astype(int)-1, curr_ons-1] = 1

            # add exponential decay
            init_H = nema(init_H, decay)

    else:
        raise ValueError('Invalid strategy.')

    return init_H

