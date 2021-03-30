import warnings
import numpy as np
from scipy import signal

def butter_design(
    fs: float, cutoffs: Tuple[float], order: int = 4, btype: str = 'bandpass'
):
    '''Get Butterworth filter design with params specified

    Args:
        cutoffs: should have ascending order e.g. (1.0, 15.0)

    Returns:
        b, a: Numerator (b) and denominator (a) polynomials of the IIR filter

    implementation taken from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    '''
    nyq = 0.5 * fs
    normal = tuple(cut / nyq for cut in cutoffs)
    return signal.butter(order, normal, btype=btype)

def slice_epochs(eeg, starts, epoch_start: int, epoch_end: int):
    '''Slices long eeg recording into epochs

    Args:
        epoch_start: index of an epoch start relative to `starts`
        epoch_end: index of an epoch end relative to `starts`
    '''
    if np.any(starts + epoch_start < 0) or np.any(starts + epoch_end >= eeg.shape[1]):
        warnings.warn('Epochs boundaries are out of eeg array indices')
    return np.stack(
        [eeg[:, (start + epoch_start) : (start + epoch_end)] for start in starts]
    )