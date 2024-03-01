import numpy as np
from tqdm import tnrange
from typing import List, Tuple

from .utils import drum_specific_soft_constraints_nmf, init_templates, init_activations
from libnmf.utils import EPS


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
    "Nonnegative Matrix and Tensor Factorizations: Applications to
    Exploratory Multi-Way Data Analysis and Blind Source Separation"
    John Wiley and Sons, 2009.

    [2] Christian Dittmar and Meinard Müller
    "Reverse Engineering the Amen Break - Score-informed Separation and Restoration applied to Drum Recordings"
    IEEE/ACM Transactions on Audio, Speech, and Language Processing, 24(9): 1531-1543, 2016.

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
         num_bins: int = None,
         num_frames: int = None,
         **kwargs) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """Non-Negative Matrix Factor Deconvolution with Kullback-Leibler-Divergence and fixable components. The core
     algorithm was proposed in [1], the specific adaptions are used in [2].

    References
    ----------
    [1] Paris Smaragdis "Non-negative Matrix Factor Deconvolution;
    Extraction of Multiple Sound Sources from Monophonic Inputs".
    International Congress on Independent Component Analysis and Blind Signal
    Separation (ICA), 2004

    [2] Christian Dittmar and Meinard Müller
    "Reverse Engineering the Amen Break - Score-informed Separation and Restoration applied to Drum Recordings"
    IEEE/ACM Transactions on Audio, Speech, and Language Processing, 24(9): 1531-1543, 2016.

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

    fix_H: bool

    func_preprocess: function
        Call for preprocessing

    func_postprocess: function
        Call for postprocessing

    Returns
    -------
    W: List
        List with the learned templates
    H: np.ndarray
        Matrix with the learned activations
    nmfd_V: List
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
    "Reverse Engineering the Amen Break " Score-informed Separation and Restoration applied to Drum Recordings"
    IEEE/ACM Transactions on Audio, Speech, and Language Processing, 24(9): 15311543, 2016.


    Parameters
    ----------
    W: np.ndarray
        Tensor holding the spectral templates which can be interpreted as a set of
        spectrogram snippets with dimensions: numBins x numComp x numTemplateFrames
    H: np.ndarray
        Corresponding activations with dimensions: numComponents x numTargetFrames

    Returns
    -------
    lamb: np.ndarray
        Approximated spectrogram matrix

    """
    # the more explicit matrix multiplication will be used
    numBins, numComp, numTemplateFrames = W.shape
    numComp, numFrames = H.shape

    # initialize with zeros
    lamb = np.zeros((numBins, numFrames))

    # this is doing the math as described in [2], eq (4)
    # the alternative conv2() method does not show speed advantages

    for k in range(numTemplateFrames):
        multResult = W[:, :, k] @ shift_operator(H, k)
        lamb += multResult

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