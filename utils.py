import pandas as pd
import glob
import warnings
from typing import Union, Optional, Tuple, List, Iterator, Iterable
from pyannote.core.utils.types import Alignment
from pyannote.core.segment import Segment
import numpy as np
from pyannote.core import Timeline
from pyannote.core import Annotation
import torch


import re


from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def comp_eer_sklearn(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)

    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    print(f'EER is {eer*100} and threshold is {thresh}')
    return eer*100, thresh


def get_embed_from_mfcc(mfcc_path, model):
    mfcc_files = natural_sort(glob.glob(f"{mfcc_path}/*.npz"))
    # breakpoint()
    if len(mfcc_files) == 0:
        breakpoint()
    embeds = []
    for mfcc_file in mfcc_files:
        mfcc = torch.tensor(np.load(mfcc_file)["mfcc"][None, :]).cuda()
        with torch.no_grad():
            embeds.append(model.extract(mfcc).cpu().numpy())
    return np.concatenate(embeds)

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def read_sad_from_rttm(rttm_file):
    df = pd.read_csv(rttm_file, sep=" ", usecols=[3, 4], names=["start", "duration"])
    df["end"] = df["start"] + df["duration"]
    df["speech"] = "speech"
    return df


def load_rttm_single(file_rttm):
    names = [
        "NA1",
        "uri",
        "NA2",
        "start",
        "duration",
        "NA3",
        "NA4",
        "speaker",
        "NA5",
        "NA6",
    ]
    dtype = {"uri": str, "start": float, "duration": float, "speaker": str}
    data = pd.read_csv(
        file_rttm,
        names=names,
        dtype=dtype,
        delim_whitespace=True,
        keep_default_na=False,
    )

    annotation = Annotation()
    for i, turn in data.iterrows():
        segment = Segment(turn.start, turn.start + turn.duration)
        annotation[segment, '_'] = turn.speaker
    return annotation


def my_load_rttm(file_rttm):
    names = [
        "NA1",
        "uri",
        "NA2",
        "start",
        "duration",
        "NA3",
        "NA4",
        "speaker",
        "NA5",
        "NA6",
    ]
    dtype = {"uri": str, "start": float, "duration": float, "speaker": str}
    data = pd.read_csv(
        file_rttm,
        names=names,
        dtype=dtype,
        delim_whitespace=True,
        keep_default_na=False,
    )

    annotations = dict()
    for uri, turns in data.groupby("uri"):
        annotation = Annotation(uri=uri)
        for i, turn in turns.iterrows():
            segment = Segment(turn.start, turn.start + turn.duration)
            annotation[segment, '_'] = turn.speaker
        annotations[uri] = annotation

    return annotations


def greedy_merge(tl, min_dur):
    tls = []
    tl_new = Timeline()
    for seg in tl:
        tl_new = tl_new.add(seg)
        if tl_new.duration() > min_dur:
            tls.append(tl_new)
            tl_new = Timeline()
    return tls


def read_sad(file):
    df = pd.read_csv(file, sep="\t", names=["starts", "ends"], usecols=[0, 1])
    return Timeline([Segment(row["starts"], row["ends"]) for _, row in df.iterrows()])


def split_time_line(tl, max_len, min_len, duration, step):
    new_tl = []
    for seg in tl:
        if seg.duration > max_len:
            for seg in MySlidingWindow(start=seg.start, end=seg.end, duration=duration, step=step):
                if seg.duration < min_len:
                    continue
                new_tl.append(seg)
            # new_tl.append([seg for seg in MySlidingWindow(start=seg.start, end=seg.end, duration=duration, step=step)])
        elif seg.duration > min_len:
            new_tl.append(seg)
    return Timeline(new_tl)


class MySlidingWindow:
    def __init__(self, duration=0.030, step=0.010, start=0.000, end=None):

        # duration must be a float > 0
        if duration <= 0:
            raise ValueError("'duration' must be a float > 0.")
        self.__duration = duration

        # step must be a float > 0
        if step <= 0:
            raise ValueError("'step' must be a float > 0.")
        self.__step: float = step

        # start must be a float.
        self.__start: float = start

        # if end is not provided, set it to infinity
        if end is None:
            self.__end: float = np.inf
        else:
            # end must be greater than start
            if end <= start:
                raise ValueError("'end' must be greater than 'start'.")
            self.__end: float = end

        # current index of iterator
        self.__i: int = -1

    @property
    def start(self) -> float:
        """Sliding window start time in seconds."""
        return self.__start

    @property
    def end(self) -> float:
        """Sliding window end time in seconds."""
        return self.__end

    @property
    def step(self) -> float:
        """Sliding window step in seconds."""
        return self.__step

    @property
    def duration(self) -> float:
        """Sliding window duration in seconds."""
        return self.__duration

    def closest_frame(self, t: float) -> int:
        """Closest frame to timestamp.

        Parameters
        ----------
        t : float
            Timestamp, in seconds.

        Returns
        -------
        index : int
            Index of frame whose middle is the closest to `timestamp`

        """
        return int(np.rint(
            (t - self.__start - .5 * self.__duration) / self.__step
        ))

    def samples(self, from_duration: float, mode: Alignment = 'strict') -> int:
        """Number of frames

        Parameters
        ----------
        from_duration : float
            Duration in seconds.
        mode : {'strict', 'loose', 'center'}
            In 'strict' mode, computes the maximum number of consecutive frames
            that can be fitted into a segment with duration `from_duration`.
            In 'loose' mode, computes the maximum number of consecutive frames
            intersecting a segment with duration `from_duration`.
            In 'center' mode, computes the average number of consecutive frames
            where the first one is centered on the start time and the last one
            is centered on the end time of a segment with duration
            `from_duration`.

        """
        if mode == 'strict':
            return int(np.floor((from_duration - self.duration) / self.step)) + 1

        elif mode == 'loose':
            return int(np.floor((from_duration + self.duration) / self.step))

        elif mode == 'center':
            return int(np.rint((from_duration / self.step)))

    def crop(self, focus: Union[Segment, 'Timeline'],
             mode: Alignment = 'loose',
             fixed: Optional[float] = None,
             return_ranges: Optional[bool] = False) -> \
            Union[np.ndarray, List[List[int]]]:
        """Crop sliding window

        Parameters
        ----------
        focus : `Segment` or `Timeline`
        mode : {'strict', 'loose', 'center'}, optional
            In 'strict' mode, only indices of segments fully included in
            'focus' support are returned. In 'loose' mode, indices of any
            intersecting segments are returned. In 'center' mode, first and
            last positions are chosen to be the positions whose centers are the
            closest to 'focus' start and end times. Defaults to 'loose'.
        fixed : float, optional
            Overrides `Segment` 'focus' duration and ensures that the number of
            returned frames is fixed (which might otherwise not be the case
            because of rounding erros).
        return_ranges : bool, optional
            Return as list of ranges. Defaults to indices numpy array.

        Returns
        -------
        indices : np.array (or list of ranges)
            Array of unique indices of matching segments
        """

        from pyannote.core.timeline import Timeline

        if not isinstance(focus, (Segment, Timeline)):
            msg = '"focus" must be a `Segment` or `Timeline` instance.'
            raise TypeError(msg)

        if isinstance(focus, Timeline):

            if fixed is not None:
                msg = "'fixed' is not supported with `Timeline` 'focus'."
                raise ValueError(msg)

            if return_ranges:
                ranges = []

                for i, s in enumerate(focus.support()):
                    rng = self.crop(s, mode=mode, fixed=fixed,
                                    return_ranges=True)

                    # if first or disjoint segment, add it
                    if i == 0 or rng[0][0] > ranges[-1][1]:
                        ranges += rng

                    # if overlapping segment, update last range
                    else:
                        ranges[-1][1] = rng[0][1]

                return ranges

            # concatenate all indices
            indices = np.hstack([
                self.crop(s, mode=mode, fixed=fixed, return_ranges=False)
                for s in focus.support()])

            # remove duplicate indices
            return np.unique(indices)

        # 'focus' is a `Segment` instance

        if mode == 'loose':

            # find smallest integer i such that
            # self.start + i x self.step + self.duration >= focus.start
            i_ = (focus.start - self.duration - self.start) / self.step
            i = int(np.ceil(i_))

            if fixed is None:
                # find largest integer j such that
                # self.start + j x self.step <= focus.end
                j_ = (focus.end - self.start) / self.step
                j = int(np.floor(j_))
                rng = (i, j + 1)

            else:
                n = self.samples(fixed, mode='loose')
                rng = (i, i + n)

        elif mode == 'strict':

            # find smallest integer i such that
            # self.start + i x self.step >= focus.start
            i_ = (focus.start - self.start) / self.step
            i = int(np.ceil(i_))

            if fixed is None:

                # find largest integer j such that
                # self.start + j x self.step + self.duration <= focus.end
                j_ = (focus.end - self.duration - self.start) / self.step
                j = int(np.floor(j_))
                rng = (i, j + 1)

            else:
                n = self.samples(fixed, mode='strict')
                rng = (i, i + n)

        elif mode == 'center':

            # find window position whose center is the closest to focus.start
            i = self.closest_frame(focus.start)

            if fixed is None:
                # find window position whose center is the closest to focus.end
                j = self.closest_frame(focus.end)
                rng = (i, j + 1)
            else:
                n = self.samples(fixed, mode='center')
                rng = (i, i + n)

        else:
            msg = "'mode' must be one of {'loose', 'strict', 'center'}."
            raise ValueError(msg)

        if return_ranges:
            return [list(rng)]

        return np.array(range(*rng), dtype=np.int64)

    def segmentToRange(self, segment: Segment) -> Tuple[int, int]:
        warnings.warn("Deprecated in favor of `segment_to_range`",
                      DeprecationWarning)
        return self.segment_to_range(segment)

    def segment_to_range(self, segment: Segment) -> Tuple[int, int]:
        """Convert segment to 0-indexed frame range

        Parameters
        ----------
        segment : Segment

        Returns
        -------
        i0 : int
            Index of first frame
        n : int
            Number of frames

        Examples
        --------

            >>> window = SlidingWindow()
            >>> print window.segment_to_range(Segment(10, 15))
            i0, n

        """
        # find closest frame to segment start
        i0 = self.closest_frame(segment.start)

        # number of steps to cover segment duration
        n = int(segment.duration / self.step) + 1

        return i0, n

    def rangeToSegment(self, i0: int, n: int) -> Segment:
        warnings.warn("This is deprecated in favor of `range_to_segment`",
                      DeprecationWarning)
        return self.range_to_segment(i0, n)

    def range_to_segment(self, i0: int, n: int) -> Segment:
        """Convert 0-indexed frame range to segment

        Each frame represents a unique segment of duration 'step', centered on
        the middle of the frame.

        The very first frame (i0 = 0) is the exception. It is extended to the
        sliding window start time.

        Parameters
        ----------
        i0 : int
            Index of first frame
        n : int
            Number of frames

        Returns
        -------
        segment : Segment

        Examples
        --------

            >>> window = SlidingWindow()
            >>> print window.range_to_segment(3, 2)
            [ --> ]

        """

        # frame start time
        # start = self.start + i0 * self.step
        # frame middle time
        # start += .5 * self.duration
        # subframe start time
        # start -= .5 * self.step
        start = self.__start + (i0 - .5) * self.__step + .5 * self.__duration
        duration = n * self.__step
        end = start + duration

        # extend segment to the beginning of the timeline
        if i0 == 0:
            start = self.start

        return Segment(start, end)

    def samplesToDuration(self, nSamples: int) -> float:
        warnings.warn("This is deprecated in favor of `samples_to_duration`",
                      DeprecationWarning)
        return self.samples_to_duration(nSamples)

    def samples_to_duration(self, n_samples: int) -> float:
        """Returns duration of samples"""
        return self.range_to_segment(0, n_samples).duration

    def durationToSamples(self, duration: float) -> int:
        warnings.warn("This is deprecated in favor of `duration_to_samples`",
                      DeprecationWarning)
        return self.duration_to_samples(duration)

    def duration_to_samples(self, duration: float) -> int:
        """Returns samples in duration"""
        return self.segment_to_range(Segment(0, duration))[1]

    def __getitem__(self, i: int) -> Segment:
        start = self.__start + i * self.__step
        if start >= self.__end:
            return None
        if start + self.__duration > self.__end:
            return Segment(start=start, end=self.__end)
        return Segment(start=start, end=start + self.__duration)

    def next(self) -> Segment:
        return self.__next__()

    def __next__(self) -> Segment:
        self.__i += 1
        window = self[self.__i]

        if window:
            return window
        else:
            raise StopIteration()

    def __iter__(self) -> 'SlidingWindow':
        """Sliding window iterator

        Use expression 'for segment in sliding_window'

        Examples
        --------

        >>> window = SlidingWindow(end=0.1)
        >>> for segment in window:
        ...     print(segment)
        [ 00:00:00.000 -->  00:00:00.030]
        [ 00:00:00.010 -->  00:00:00.040]
        [ 00:00:00.020 -->  00:00:00.050]
        [ 00:00:00.030 -->  00:00:00.060]
        [ 00:00:00.040 -->  00:00:00.070]
        [ 00:00:00.050 -->  00:00:00.080]
        [ 00:00:00.060 -->  00:00:00.090]
        [ 00:00:00.070 -->  00:00:00.100]
        [ 00:00:00.080 -->  00:00:00.110]
        [ 00:00:00.090 -->  00:00:00.120]
        """

        # reset iterator index
        self.__i = -1
        return self

    def __len__(self) -> int:
        """Number of positions

        Equivalent to len([segment for segment in window])

        Returns
        -------
        length : int
            Number of positions taken by the sliding window
            (from start times to end times)

        """
        if np.isinf(self.__end):
            raise ValueError('infinite sliding window.')

        # start looking for last position
        # based on frame closest to the end
        i = self.closest_frame(self.__end)

        while (self[i]):
            i += 1
        length = i

        return length

    def copy(self) -> 'SlidingWindow':
        """Duplicate sliding window"""
        duration = self.duration
        step = self.step
        start = self.start
        end = self.end
        sliding_window = self.__class__(
            duration=duration, step=step, start=start, end=end
        )
        return sliding_window

    def __call__(self,
                 support: Union[Segment, 'Timeline'],
                 align_last: bool = False) -> Iterable[Segment]:
        """Slide window over support

        Parameter
        ---------
        support : Segment or Timeline
            Support on which to slide the window.
        align_last : bool, optional
            Yield a final segment so that it aligns exactly with end of support.

        Yields
        ------
        chunk : Segment

        Example
        -------
        >>> window = SlidingWindow(duration=2., step=1.)
        >>> for chunk in window(Segment(3, 7.5)):
        ...     print(tuple(chunk))
        (3.0, 5.0)
        (4.0, 6.0)
        (5.0, 7.0)
        >>> for chunk in window(Segment(3, 7.5), align_last=True):
        ...     print(tuple(chunk))
        (3.0, 5.0)
        (4.0, 6.0)
        (5.0, 7.0)
        (5.5, 7.5)
        """

        from pyannote.core import Timeline
        if isinstance(support, Timeline):
            segments = support

        elif isinstance(support, Segment):
            segments = Timeline(segments=[support])

        else:
            msg = (
                f'"support" must be either a Segment or a Timeline '
                f'instance (is {type(support)})'
            )
            raise TypeError(msg)

        for segment in segments:

            if segment.duration < self.duration:
                continue

            window = SlidingWindow(duration=self.duration,
                                   step=self.step,
                                   start=segment.start,
                                   end=segment.end)

            for s in window:
                # ugly hack to account for floating point imprecision
                if s in segment:
                    yield s
                    last = s

            if align_last and last.end < segment.end:
                yield Segment(start=segment.end - self.duration,
                              end=segment.end)


if __name__ == '__main__':
    # rttm2sad("/home12a/wwlin/corpus/vox_diar/rttm", "/home12a/wwlin/corpus/vox_diar/sad")
    sw = MySlidingWindow(start=0, end=10, duration=3, step=3)
    for x in sw:
        print(x)
