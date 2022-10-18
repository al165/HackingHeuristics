import os
import sys
from typing import Tuple, List, Dict, Callable

print(sys.version)

import librosa
import numpy as np

import scipy.signal
from scipy.fft import fft


class Processor:
    def __init__(self, sr: int = 256):
        self.sr = sr

    def __call__(self, y: np.ndarray):
        raise NotImplementedError


class NormaliseProcessor(Processor):
    def __init__(
        self,
        sr: int = 256,
        mean: float = 0,
        update: bool = True,
        max_hist: int = 4096,
        inrange: Tuple[float, float] = (-1, 1),
    ):
        super(NormaliseProcessor, self).__init__(sr=sr)

        self.mean = mean
        self.update = update
        self.max_hist = max_hist
        self.inrange = inrange

        self._buffer = None

    def __call__(self, y: np.ndarray) -> np.ndarray:
        if self.update:
            if self._buffer is not None:
                self._buffer = np.concatenate([self._buffer, y])[-self.max_hist :]
            else:
                self._buffer = np.copy(y)[-self.max_hist :]

            self.mean = self._buffer.mean()

        norm = y - self.mean
        norm /= float(self.inrange[1] - self.inrange[0])

        return norm


class FilterProcessor(Processor):
    def __init__(
        self,
        sr: int = 256,
        Wn: Tuple[float, float] = (25, 45),
        order: int = 4,
        btype: str = "bandpass",
    ):
        super(FilterProcessor, self).__init__(sr=sr)

        self.filter = scipy.signal.butter(
            N=order,
            Wn=Wn,
            btype=btype,
            fs=sr,
            analog=False,
            output="sos",
        )

    def __call__(self, y: np.ndarray) -> np.ndarray:
        data_filtered = scipy.signal.sosfilt(self.filter, y)
        return data_filtered


class MovingAverageProcessor(Processor):
    def __init__(
        self,
        sr: int = 256,
        window: int = 5,
    ):
        super(MovingAverageProcessor, self).__init__(sr=sr)

        self.window = window

    def __call__(self, y: np.ndarray) -> np.ndarray:

        avg = np.convolve(y, np.ones(self.window), "valid") / self.window
        return avg


class RMSProcessor(Processor):
    def __init__(
        self,
        sr: int = 256,
        frame_length: int = 2048,
        hop_length: int = 512,
    ):
        super(RMSProcessor, self).__init__(sr=sr)

        self.frame_length = frame_length
        self.hop_length = hop_length
        self._buffer = None

    def __call__(self, y: np.ndarray) -> np.ndarray:
        if self._buffer is not None:
            self._buffer = np.concatenate([self._buffer[-self.hop_length :], y])
        else:
            self._buffer = np.copy(y)

        rms = librosa.feature.rms(
            y=self._buffer, hop_length=self.hop_length, frame_length=self.frame_length
        )[0][1:]

        return rms


class RecordProcessor(Processor):
    def __init__(self, fn: str, sr: int = 256,):
        super(RecordProcessor, self).__init__(sr=sr)

        self.fn = fn

        with open(fn, "w") as f:
            f.write(f"sr = {sr}\n")

    def __call__(self, y: np.ndarray) -> np.ndarray:

        with open(self.fn, "a") as f:
            for value in y:
                f.write(f"{value}\n")

        return y



class ProcessorList:
    def __init__(self, *processors: List[Processor]):
        self.processors = processors

    def __call__(self, y: np.ndarray) -> np.ndarray:
        for processor in self.processors:
            y = processor(y)

        return y

    def add(self, processor: Processor):
        self.processors.append(processor)

    def extend(self, processors: List[Processor]):
        self.processors.extend(processor)


class FeatureExtractor:
    def __init__(self, sr: int = 256, units: str = "samples", period: int = None):
        """
        Parameters
        ----------
        sr : int
            Sample rate
        units : str (optional)
            Units of time. Defualt "samples".
        period : int (optional)
            Length of buffer to calculate feature over. Default `sr`.

        """

        self.sr = sr
        self.units = units
        if period is None:
            self.period = sr
        else:
            self.period = period

    def __call__(self, buffer: np.ndarray) -> Dict[str, float]:
        raise NotImplementedError()


class MeanFeature(FeatureExtractor):
    def __call__(self, buffer: np.ndarray) -> Dict[str, float]:
        result = {
            "mean": np.mean(buffer[-self.period :]),
        }
        return result


class StdDevFeature(FeatureExtractor):
    def __call__(self, buffer: np.ndarray) -> Dict[str, float]:
        result = {
            "std": np.std(buffer[-self.period :]),
        }
        return result


class DeltaFeature(FeatureExtractor):
    def __init__(self, *args, fn: Callable[np.ndarray, float] = np.mean, **kwargs):
        super(DeltaFeature, self).__init__(*args, **kwargs)
        self.fn = fn

    def __call__(self, buffer: np.ndarray) -> Dict[str, float]:
        delta = 0

        if len(buffer) > self.period:
            a = buffer[-2 * self.period : -self.period]
            b = buffer[-self.period :]
            delta = self.fn(a) - self.fn(b)

        result = {
            "delta": delta,
        }

        return result


class DerivativeFeature(FeatureExtractor):
    def __call__(self, buffer: np.ndarray) -> Dict[str, float]:
        d = np.diff(buffer[-self.period :])
        result = {
            "derivative_mean": np.mean(d),
            "derivative_abs_mean": np.mean(np.abs(d)),
            "derivative_max": np.max(d),
        }


class PeakActivityFeature(FeatureExtractor):
    def __init__(
        self,
        *args,
        sensitivity: float = 0.1,
        peaks_kwargs: Dict[str, float] = {},
        **kwargs,
    ):
        super(PeakActivityFeature, self).__init__(*args, **kwargs)

        self.peaks_kwargs = peaks_kwargs
        self.sensitivity = sensitivity

    def __call__(self, buffer: np.ndarray) -> Dict[str, float]:
        data = buffer[-self.period :]
        avg = data.mean()
        pos = np.where(data - avg > self.sensitivity)[0]
        neg = np.where(data - avg < -self.sensitivity)[0]

        data_p = data[pos]
        data_n = data[neg]

        peaks_p = scipy.signal.find_peaks(data_p, **self.peaks_kwargs)[0]
        peaks_n = scipy.signal.find_peaks(-data_n, **self.peaks_kwargs)[0]

        p_rate = len(peaks_p) / (self.period / self.sr)
        n_rate = len(peaks_n) / (self.period / self.sr)
        all_rate = p_rate + n_rate

        result = {
            "p_rate": p_rate,
            "n_rate": n_rate,
            "all_rate": all_rate,
        }

        return result


class FeatureExtractorCollection:
    def __init__(self, *extractors: List[FeatureExtractor]):
        self.extractors = extractors

    def __call__(self, buffer: np.ndarray) -> Dict[str, float]:
        results = dict()
        for extractor in self.extractors:
            res = extractor(np.copy(buffer))
            results |= res

        return results

    def add(self, extractor):
        self.extractors.append(extractor)




def main():
    print("Test for data processors")

    y = np.random.randn(8000)
    print(y.shape)

    p = ProcessorList()
    p.add(NormaliseProcessor())
    p.add(FilterProcessor())
    p.add(RMSProcessor())

    y = p(y)
    print(y)
    print(y.shape)


if __name__ == "__main__":
    main()
