import numpy as np
from scipy.ndimage.filters import convolve


def conv2(x: np.ndarray, y:np.ndarray, mode:str = 'same') -> np.ndarray:
    """Emulate the function conv2 from Mathworks.
    Usage:
    z = conv2(x,y,mode='same')

    Parameters
    ----------
    x: np.ndarray
        The sequence / matrix to be convolved with the kernel y

    y: np.ndarray
        The convolution kernel

    mode: str
        The mode of convolution, only 'same' is supported

    Returns
    -------
    z: np.ndarray
        The result of the conv2 operation

    """
    # We must provide an offset for each non-singleton dimension to reproduce the results of Matlab's conv2.
    # A simple implementation supporting the 'same' option, only, could be made like below
    # source: https://stackoverflow.com/questions/3731093/is-there-a-python-equivalent-of-matlabs-conv2-function

    if not mode == 'same':
        raise NotImplementedError("Mode not supported")

    # Add singleton dimensions
    if len(x.shape) < len(y.shape):
        dim = x.shape
        for i in range(len(x.shape), len(y.shape)):
            dim = (1,) + dim
        x = x.reshape(dim)

    elif len(y.shape) < len(x.shape):
        dim = y.shape
        for i in range(len(y.shape), len(x.shape)):
            dim = (1,) + dim
        y = y.reshape(dim)

    origin = ()

    # Apparently, the origin must be set in a special way to reproduce
    # the results of scipy.signal.convolve and Matlab
    for i in range(len(x.shape)):
        if ((x.shape[i] - y.shape[i]) % 2 == 0 and
                x.shape[i] > 1 and
                y.shape[i] > 1):
            origin = origin + (-1,)
        else:
            origin = origin + (0,)

    z = convolve(x, y, mode='constant', origin=origin)

    return z
