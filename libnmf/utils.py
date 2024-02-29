import numpy as np


def freq2midi(freq):
    """Converts a given frequency in Hz to the corresponding MIDI pitch, thereby
    applying a quantization to semi-tone steps on the equal tempered scale.
    No sanity checks on the validity of the input are performed.

    Parameters
    ----------
    freq: float
        The given frequency in Hz

    Returns
    -------
    midi: float
        The MIDI pitch, quantized to equal temp. scale
    """
    midi = round(69 + 12 * np.log2(freq/440))
    return midi


def midi2freq(midi):
    """Converts a given MIDI pitch to the corresponding frequency in Hz. No
    sanity checks on the validity of the input are performed.

    Parameters
    ----------
    midi: array-like / float
        The MIDI pitch, can also be floating value

    Returns
    -------
    freq: array-like / float
        The frequency in Hz
    """
    freq = (440.0 / 32) * 2 ** ((midi - 9) / 12)

    return freq