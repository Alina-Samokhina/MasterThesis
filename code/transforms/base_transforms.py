import numpy as np
from scipy import interpolate, signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from typing import Optional

from .utils import butter_design


class Transformer(BaseEstimator, TransformerMixin):
    '''
    Base class for transforms providing dummy implementation
        of the methods expected by sklearn
    '''

    def fit(self, x, y=None):
        return self


class Identical(Transformer):
    def transform(self, x):
        return x


class ButterFilter(Transformer):
    '''Applies Scipy's Butterworth filter'''

    def __init__(self, rate: float, order: int, high: float, low: float):
        self.rate = rate
        self.order = order
        self.high = high
        self.low = low

        self.design = butter_design(self.rate, (self.high, self.low), self.order)

    def transform(self, batch):
        out = np.empty(len(batch), np.object)
        out[:] = [signal.filtfilt(*self.design, item) for item in batch]
        return out


class Decimator(Transformer):
    def __init__(self, factor: int, ftype='iir'):
        '''
        factor: downsampling factor, shouldn't be more than 13,
            see :py:funct:`scipy.signal.decimate` for more info
        '''
        self.factor = factor
        self.ftype = ftype

    def transform(self, batch):
        '''
        Args:
            batch: iterable of np.ndarrays
        Returns:
            np.ndarray of np.objects shaped (len(batch), )
                In other words it outputs ndarray of objects each of which is
                result of decimation of items from batch
        '''
        out = np.empty(len(batch), np.object)
        out[:] = [signal.decimate(item, self.factor, ftype=self.ftype) for item in batch]
        return out


class ChannelwiseScaler(Transformer):
    '''Performs channelwise scaling according to given scaler
    '''

    def __init__(self, scaler: Optional[Transformer] = None):
        '''Args:
            scaler: instance of one of sklearn.preprocessing classes
                StandardScaler or MinMaxScaler or analogue
        '''
        self.scaler = scaler or StandardScaler()

    def fit(self, batch: np.ndarray, y=None):
        '''
        Args:
            batch: array of eegs,
                batch shaped (n_eegs, n_channels, n_ticks)
                    or first dim may be separated to list/array
        '''
        for signals in batch:
            self.scaler.partial_fit(signals.T)
        return self

    def transform(self, batch):
        '''Scales each channel

        Args:
            batch: iterable of records (n_channels, n_samples)
        Returns the same format as input
        '''
        scaled = np.empty_like(batch)
        for i, signals in enumerate(batch):
            # double T for scaling each channel separately
            scaled[i] = self.scaler.transform(signals.T).T
        return scaled


class Clipper(Transformer):
    normalization = 1e-6  # for microvolts

    def __init__(self, minv: float, maxv: Optional[float] = None):
        '''
        Args:
            minv: bottom level of clipping, microvolts = 1e-6 volts
            maxv: top level of clipping
                if None: clipping from -minv to minv
        '''
        self.minv = -minv if maxv is None else minv
        self.maxv = minv if maxv is None else maxv

    def transform(self, batch):
        for item in batch:
            np.clip(
                item,
                self.minv * self.normalization,
                self.maxv * self.normalization,
                out=item,
            )
        return batch


class Resampler(Transformer):
    def __init__(self, in_rate: float, out_rate: float):
        self.in_rate = in_rate
        self.out_rate = out_rate

    def transform(self, batch):
        if self.in_rate == self.out_rate:
            return batch
        out = np.empty(len(batch), np.object)
        signal_lengths = [item.shape[1] for item in batch]
        new_lengths = [
            int(round(signal_length / self.in_rate * self.out_rate))
            for signal_length in signal_lengths
        ]
        functs = []
        functs[:] = [
            interpolate.interp1d(
                np.linspace(0.0, 1.0, item.shape[1]),
                item,
                axis=1,
                copy=False,
                assume_sorted=True,
            )
            for item in batch
        ]
        out[:] = [
            funct(np.linspace(0.0, 1.0, new_len))
            for (funct, new_len) in zip(functs, new_lengths)
        ]
        return out
    
    
def make_eeg_pipe(
    sampling_rate: float,
    decimation_factor: int,
    lowfreq: float = 0.5,
    highfreq: float = 12.0,
    clip: float = 100.0,
):
    transfs = (
        Decimator(decimation_factor),
        ButterFilter(sampling_rate // decimation_factor, 4, lowfreq, highfreq),
        Resampler(sampling_rate // decimation_factor, highfreq * 2 / 0.8),
        Clipper(clip),
        ChannelwiseScaler(),
    )
    return make_pipeline(*transfs)
