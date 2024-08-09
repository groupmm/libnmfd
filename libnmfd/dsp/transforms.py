import numpy as np
from scipy.fftpack import fft, ifft
from typing import List, Tuple, Union

from libnmfd.utils import midi2freq


def forward_stft(x: np.ndarray,
                 block_size: int = 2048,
                 hop_size: int = 512,
                 win: np.ndarray = None,
                 reconst_mirror: bool = True,
                 append_frames: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given a time signal as input, this computes the spectrogram by means of the Short-time fourier transform

    Parameters
    ----------
    x: np.ndarray
        The time signal oriented as numSamples x 1

    block_size: int
        The block size to use during analysis

    hop_size: int
        The hop size to use during analysis

    win: np.ndarray
        The analysis window

    reconst_mirror: bool
       This switch decides whether to discard the mirror spectrum or not

    append_frames: bool
        This switch decides if we prepend/append silence in the beginning and the end

    Returns
    -------
    X: np.ndarray
        The complex valued spectrogram in num_bins x num_frames

    A: np.ndarray
        The magnitude spectrogram

    P: np.ndarray
        The phase spectrogram (wrapped in -pi ... +pi)
    """
    if win is None:
        win = np.hanning(block_size)

    half_block_size = round(block_size / 2)

    # the number of bins needs to be corrected
    # if we want to discard the mirror spectrum
    num_bins = round(block_size/ 2) + 1 if reconst_mirror else block_size

    # append safety space in the beginning and end
    if append_frames:
        x = np.concatenate((np.zeros(half_block_size), x, np.zeros(half_block_size)), axis=0)

    num_samples = len(x)

    # pre-compute the number of frames
    num_frames = round(num_samples / hop_size)

    # initialize with correct size
    X = np.zeros((int(num_bins), num_frames), dtype=np.complex64)

    counter = 0

    for k in range(0, len(x)-block_size, hop_size):
        # where to pick
        ind = range(k, k+block_size)

        # pick signal frame
        snip = x[ind]

        # apply windowing
        snip *= win

        # do FFT
        f = fft(snip, axis=0)

        # if required, remove the upper half of spectrum
        if reconst_mirror:
            f = np.delete(f, range(num_bins, block_size), axis=0)

        # store into STFT matrix
        X[:, counter] = f
        counter += 1

    # after constructing the STFT array, remove excessive frames
    X = np.delete(X, range(counter, num_frames), axis=1)

    # compute derived matrices
    # get magnitude
    A = np.abs(X)

    # get phase
    P = np.angle(X)

    # return complex-valued STFT, magnitude STFT, and phase STFT
    return X, A, P


def inverse_stft(X: np.ndarray,
                 block_size: int = 2048,
                 hop_size: int = 512,
                 ana_win_func: np.ndarray = None,
                 syn_win_func: np.ndarray = None,
                 reconst_mirror: bool = True,
                 append_frames: bool = True,
                 analytic_sig: bool = False,
                 num_samp: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Given a valid STFT spectrogram as input, this reconstructs the corresponding time-domain signal by  means of
    the frame-wise inverse FFT and overlap-add method described as LSEE-MSTFT in [1].

    References
    ----------
    [1] Daniel W. Griffin and Jae S. Lim
    "Signal estimation from modified short-time fourier transform",
    IEEE Transactions on Acoustics, Speech and Signal Processing, vol. 32, no. 2, pp. 236-243, Apr 1984.

    Parameters
    ----------
    X: np.ndarray
        The complex-valued spectrogram matrix oriented with dimensions
        num_bins x num_frames

    block_size: int
        The block size to use during synthesis

    hop_size: int
        The hop size to use during synthesis

    ana_win_func: np.ndarray
        The analysis window function

    syn_win_func: np.ndarray
        The synthesis window function (per default set same as analysis window)

    reconst_mirror: bool
        This switch decides whether the mirror spectrum should be reconstructed or not

    append_frames: bool
        This switch decides whether to compensate for zero padding or not

    analytic_sig: bool
        If this is set to True, we want the analytic signal

    num_samp:int
         The original number of samples

    Returns
    -------
    y: np.ndarray
        The resynthesized signal
        
    syn_win_func: np.ndarray
        The envelope used for normalization of the synthesis window
    """

    # get dimensions of STFT array and prepare corresponding output
    num_bins, num_frames = X.shape
    num_pad_bins = block_size - num_bins
    num_samples = num_frames * hop_size + block_size

    # for simplicity, we assume the analysis and synthesis windows to be equal
    if ana_win_func is None:
        ana_win_func = np.hanning(block_size)

    if syn_win_func is None:
        syn_win_func = np.hanning(block_size)

    # prepare helper variables
    half_block_size = round(block_size / 2)

    # we need to change the signal scaling in case of the analytic signal
    scale = 2.0 if analytic_sig else 1.0

    # decide between analytic and real output
    y = np.zeros(num_samples, dtype=np.complex64) if analytic_sig else np.zeros(num_samples, dtype=np.float32)

    # construct normalization function for the synthesis window
    # that equals the denominator in eq. (6) in [1]
    win_prod = ana_win_func * syn_win_func
    redundancy = round(block_size / hop_size)

    # construct hop_size-periodic normalization function that will be
    # applied to the synthesis window
    nrm_func = np.zeros(block_size)

    # begin with construction outside the support of the window
    for k in range(-redundancy + 1, redundancy):
        nrm_func_ind = hop_size * k
        win_ind = np.arange(0, block_size)
        nrm_func_ind += win_ind

        # check which indices are inside the defined support of the window
        valid_index = np.where((nrm_func_ind >= 0) & (nrm_func_ind < block_size))
        nrm_func_ind = nrm_func_ind[valid_index]
        win_ind = win_ind[valid_index]

        # accumulate product of analysis and synthesis window
        nrm_func[nrm_func_ind] += win_prod[win_ind]

    # apply normalization function
    syn_win_func /= nrm_func

    # prepare index for output signal construction
    frame_ind = np.arange(0, block_size)

    # then begin frame-wise reconstruction
    for k in range(num_frames):

        # pick spectral frame
        curr_spec = X[:, k].copy()

        # if desired, construct artificial mirror spectrum
        if reconst_mirror:
            # if the analytic signal is wanted, put zeros instead
            pad_mirror_spec = np.zeros(num_pad_bins)

            if not analytic_sig:
                pad_mirror_spec = np.conjugate(np.flip(curr_spec[1:-1], axis=0))

            # concatenate mirror spectrum to base spectrum
            curr_spec = np.concatenate((curr_spec, pad_mirror_spec), axis=0)

        # transform to time-domain
        snip = ifft(curr_spec)

        # treat differently if analytic signal is desired
        if not analytic_sig:
            snip = np.real(snip)

        # apply scaling and synthesis window
        snip *= syn_win_func * scale

        # determine overlap-add position
        overlap_add_ind = k * hop_size + frame_ind

        # and do the overlap add, with synthesis window and scaling factor included
        y[overlap_add_ind] += snip

    # check if borders need to be removed
    if append_frames:
        y = y[half_block_size:len(y) - half_block_size]

    # check if number of samples was defined from outside
    if num_samp is not None:
        y = y[0:num_samp]

    return y.reshape(-1, 1), syn_win_func




def log_freq_log_mag(A: Union[np.ndarray, List[np.ndarray]],
                     freq_res: float,
                     bins_per_octave: int = 36,
                     lower_freq: float = midi2freq(24),
                     log_comp: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Given a magnitude spectrogram, this function maps it onto a compact representation with logarithmically spaced
    frequency axis and logarithmic magnitude compression.

    Parameters
    ----------
    A: np.ndarray or List of np.ndarray
        The real-valued magnitude spectrogram oriented as num_bins x num_frames, it can also be given as a list of
        multiple spectrograms
        
    freq_res: float
        The spectral resolution of the spectrogram
        
    bins_per_octave: np.ndarray
        The spectral selectivity of the log-freq axis
        
    lower_freq: float
        The lower frequency border
        
    log_comp: float
        Factor to control the logarithmic magnitude compression

    Returns
    -------
    log_freq_log_mag_A: np.ndarray
        The log-magnitude spectrogram on logarithmically spaced frequency axis

    log_freq_axis: np.ndarray
        An array giving the center frequencies of each bin along the logarithmically spaced frequency axis
    """

    # convert to list if necessary
    if not isinstance(A, list):
        was_arr_input = True
        A = [A]
    else:
        was_arr_input = False

    # get number of components
    num_comp = len(A)
    log_freq_log_mag_A = list()
    log_freq_axis = None

    for k in range(num_comp):
        # get component spectrogram
        comp_A = A[k]

        # get dimensions
        num_lin_bins, num_frames = comp_A.shape

        # set up linear frequency axis
        lin_freq_axis = np.arange(0, num_lin_bins) * freq_res

        # get upper limit
        upper_freq = lin_freq_axis[-1]

        # set up logarithmic frequency axis
        num_log_bins = np.ceil(bins_per_octave * np.log2(upper_freq / lower_freq))
        log_freq_axis = np.arange(0, num_log_bins)
        log_freq_axis = lower_freq * np.power(2.0, log_freq_axis / bins_per_octave)

        # map to logarithmic axis by means of linear interpolation
        log_bin_axis = log_freq_axis / freq_res

        # compute linear interpolation for the logarithmic mapping
        floor_bin_axis = np.floor(log_bin_axis).astype(np.int32) - 1
        ceil_bin_axis = floor_bin_axis + 1
        fraction = log_bin_axis - floor_bin_axis - 1

        # get magnitude values
        floor_mag = comp_A[floor_bin_axis, :]
        ceil_mag = comp_A[ceil_bin_axis, :]

        # compute weighted sum
        log_freq_A = floor_mag * (1 - fraction).reshape(-1, 1) + ceil_mag * fraction.reshape(-1, 1)

        # apply magnitude compression
        log_freq_log_mag_A.append(np.log(1 + (log_comp * log_freq_A)))

    # revert back to matrix if necessary
    if was_arr_input:
        log_freq_log_mag_A = np.array(log_freq_log_mag_A[0])

    return log_freq_log_mag_A, log_freq_axis.reshape(-1, 1)

