# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemad, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from ._core.utils import TimeUnits as _TimeUnits

__all__ = [
    "DataConfig",
    "HDF5Source",
    "SampledChannel",
    "SpectrumConfig",
    "TimestampOptions",
    "TimestampedChannel",
]


def _normalize_integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    """Accept Python and NumPy integers without accepting booleans or coercing other types."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer.")

    if isinstance(value, np.integer):
        value = int(value)
    elif not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")

    return value


def normalize_real(value: Any, *, name: str, positive: bool = False) -> float:
    """Accept finite Python and NumPy real numbers without accepting strings or booleans."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite real number.")

    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a finite real number.")

    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number.") from exc

    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")

    if positive and normalized <= 0:
        raise ValueError(f"{name} must be positive.")

    return normalized


def _normalize_observation_bound(value: Any, *, name: str) -> int | float | None:
    """Normalize NumPy scalars while preserving large integral origins exactly."""

    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite real number.")

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, int):
        return value

    if isinstance(value, (float, np.floating)):
        normalized = float(value)

        if not math.isfinite(normalized):
            raise ValueError(f"{name} must be finite.")

        return normalized

    raise TypeError(f"{name} must be a finite real number.")


def _require_bool(value: Any, *, name: str) -> bool:
    """Accept only actual Python booleans."""

    if type(value) is not bool:
        raise TypeError(f"{name} must be a Boolean.")

    return value


def _require_choice(value: Any, *, name: str, choices: tuple[str, ...]) -> str:
    """Require an exact string from a fixed set."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")

    if value not in choices:
        allowed = ", ".join(repr(choice) for choice in choices)
        raise ValueError(f"{name} must be one of {allowed}.")

    return value


def _normalize_selector_integer(value: Any, *, name: str) -> int | None:
    """Normalize one optional HDF5 slice component."""

    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer or None.")

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, int):
        return value

    raise TypeError(f"{name} must be an integer or None.")


def _normalize_hdf5_selection(value: Any) -> tuple[int | slice, ...]:
    """Validate HDF5 selection syntax and normalize NumPy integers."""

    if not isinstance(value, (list, tuple)):
        raise TypeError("selection must be a list or tuple.")

    if not value:
        raise ValueError("selection cannot be empty.")

    normalized: list[int | slice] = []

    for item in value:
        if isinstance(item, (bool, np.bool_)):
            raise TypeError("HDF5 selection entries must be integers or slices.")

        if isinstance(item, np.integer):
            normalized.append(int(item))
            continue

        if isinstance(item, int):
            normalized.append(item)
            continue

        if not isinstance(item, slice):
            raise TypeError("HDF5 selection entries must be integers or slices.")

        start = _normalize_selector_integer(item.start, name="HDF5 slice start")
        stop = _normalize_selector_integer(item.stop, name="HDF5 slice stop")
        step = _normalize_selector_integer(item.step, name="HDF5 slice step")

        if step not in (None, 1):
            raise ValueError("HDF5 slice steps other than 1 are not supported.")

        normalized.append(slice(start, stop, step))

    return tuple(normalized)


def _normalize_device(value: Any) -> str:
    """Validate device syntax without requiring that the device is available."""

    if not isinstance(value, str):
        raise TypeError("device must be a string.")

    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            "device must be 'cpu', 'mps', 'cuda', 'cuda:N', 'xpu', or 'xpu:N', where N is a "
            "nonnegative integer."
        ) from exc

    if device.type not in {"cpu", "cuda", "mps", "xpu"}:
        raise ValueError(
            f"Unsupported device type {device.type!r}; use 'cpu', 'mps', 'cuda', 'cuda:N', 'xpu', "
            "or 'xpu:N'."
        )

    if device.type in {"cpu", "mps"} and device.index is not None:
        raise ValueError(f"{device.type!r} does not support a numbered device index.")

    return str(device)


@dataclass(frozen=True, slots=True)
class HDF5Source:
    """HDF5-backed storage configuration for one measurement channel.

    An :class:`HDF5Source` can be used as :attr:`SampledChannel.data` or
    :attr:`TimestampedChannel.timestamps`. When the channel is active in a calculation, the selected
    dataset is opened read-only and read in chunks rather than loaded into memory all at once. The
    selected values are flattened in C order to form one logical channel. Dataset-dependent
    validation is performed when the source is opened.

    Attributes
    ----------
    file : Path
        Path to the HDF5 file. User-directory markers are expanded and the path is resolved when the
        file is opened.
    dataset : str
        Path of the dataset inside the HDF5 file.
    selection : tuple[Any, ...]
        Dataset selection containing one integer or slice for each dataset dimension. Integer
        selectors fix a dimension; one or two dimensions must remain unfixed. Slice steps must be
        one. The resulting real numeric or Boolean values are flattened in C order. For example,
        ``(slice(None), slice(None), 0)`` selects one channel from the final dataset axis.
    """

    file: Path
    dataset: str
    selection: tuple[int | slice, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.file, (str, Path)):
            raise TypeError("file must be a string or pathlib.Path.")

        if not isinstance(self.dataset, str):
            raise TypeError("dataset must be a string.")

        if not self.dataset:
            raise ValueError("dataset cannot be empty.")

        object.__setattr__(self, "file", Path(self.file))
        object.__setattr__(self, "selection", _normalize_hdf5_selection(self.selection))


def _validate_stored_data(
    value: Any,
    *,
    label: str,
    allow_empty: bool,
    allow_boolean: bool,
) -> np.ndarray | torch.Tensor | HDF5Source:
    """Validate storage type, shape, device, and intrinsic dtype."""

    if isinstance(value, HDF5Source):
        return value

    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(
                f"{label} must be stored on the CPU; "
                "the calculation device is selected by SpectrumConfig."
            )

        if value.layout != torch.strided:
            raise TypeError(f"{label} must use PyTorch's strided tensor layout.")

        if value.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional.")

        if not allow_empty and value.numel() == 0:
            raise ValueError(f"{label} cannot be empty.")

        try:
            torch.iinfo(value.dtype)
            is_integer = True
        except TypeError:
            is_integer = False

        is_boolean = value.dtype == torch.bool
        is_supported = value.is_floating_point() or is_integer or (allow_boolean and is_boolean)

        if value.is_complex() or not is_supported:
            expected = "real numeric or Boolean" if allow_boolean else "real numeric"
            raise TypeError(f"{label} must contain {expected} data.")

        return value

    if not isinstance(value, np.ndarray):
        raise TypeError(f"{label} must be a NumPy array, CPU PyTorch tensor, or HDF5Source.")

    if value.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional.")

    if not allow_empty and value.size == 0:
        raise ValueError(f"{label} cannot be empty.")

    is_boolean = np.issubdtype(value.dtype, np.bool_)
    is_real_numeric = np.issubdtype(value.dtype, np.number) and not np.issubdtype(
        value.dtype, np.complexfloating
    )

    if not is_real_numeric and not (allow_boolean and is_boolean):
        expected = "real numeric or Boolean" if allow_boolean else "real numeric"
        raise TypeError(f"{label} must contain {expected} data.")

    return value


@dataclass(frozen=True, slots=True, eq=False)
class SampledChannel:
    """Configuration for one sampled measurement channel.

    A :class:`SampledChannel` represents values recorded at a constant sampling interval. During a
    spectrum calculation, active sampled channels are split into windows and transformed with an
    FFT. All active sampled channels must have the same sampling interval and logical length.

    In-memory data is retained without copying and must not be mutated during a calculation.

    Attributes
    ----------
    data : numpy.ndarray | torch.Tensor | HDF5Source
        Sample values for the channel. In-memory input must be a nonempty one-dimensional NumPy
        array or CPU PyTorch tensor containing real numeric or Boolean values. An
        :class:`HDF5Source` is read lazily and flattened into one logical channel.
    dt : float
        Positive time interval between consecutive samples, in units of :attr:`DataConfig.t_unit`.
    """

    data: np.ndarray | torch.Tensor | HDF5Source
    dt: float

    def __post_init__(self) -> None:
        validated_data = _validate_stored_data(
            self.data,
            label="SampledChannel data",
            allow_empty=False,
            allow_boolean=True,
        )
        normalized_dt = normalize_real(self.dt, name="SampledChannel dt", positive=True)

        object.__setattr__(self, "data", validated_data)
        object.__setattr__(self, "dt", normalized_dt)


@dataclass(frozen=True, slots=True, eq=False)
class TimestampedChannel:
    """Configuration for one timestamped measurement channel.

    A :class:`TimestampedChannel` represents discrete events by their occurrence times. Active
    timestamped channels are transformed directly at the required frequencies, with event
    amplitudes determined by :class:`TimestampOptions`. Their timestamps must lie within the
    explicit observation interval configured by :class:`DataConfig`.

    In-memory timestamps are retained without copying and must not be mutated during a calculation.

    Attributes
    ----------
    timestamps : numpy.ndarray | torch.Tensor | HDF5Source
        Event times in units of :attr:`DataConfig.t_unit`. In-memory input must be a one-dimensional
        NumPy array or CPU PyTorch tensor containing finite, nondecreasing real numbers. Empty
        inputs and duplicate timestamps are valid. An :class:`HDF5Source` is read lazily and
        flattened in C order; the flattened timestamps must satisfy the same ordering constraints.
    """

    timestamps: np.ndarray | torch.Tensor | HDF5Source

    def __post_init__(self) -> None:
        timestamps = _validate_stored_data(
            self.timestamps,
            label="TimestampedChannel timestamps",
            allow_empty=True,
            allow_boolean=False,
        )

        if isinstance(timestamps, torch.Tensor):
            if not bool(torch.isfinite(timestamps).all().item()):
                raise ValueError("TimestampedChannel timestamps must contain only finite values.")

            if timestamps.numel() > 1 and bool(torch.any(timestamps[1:] < timestamps[:-1]).item()):
                raise ValueError("TimestampedChannel timestamps must be nondecreasing.")

        elif isinstance(timestamps, np.ndarray):
            if not np.all(np.isfinite(timestamps)):
                raise ValueError("TimestampedChannel timestamps must contain only finite values.")

            if timestamps.size > 1 and np.any(timestamps[1:] < timestamps[:-1]):
                raise ValueError("TimestampedChannel timestamps must be nondecreasing.")

        object.__setattr__(self, "timestamps", timestamps)


@dataclass(frozen=True, slots=True, eq=False)
class DataConfig:
    """Configuration for sampled and timestamped measurement channels.

    :class:`DataConfig` groups the input channels and defines their shared time coordinate. Channel
    positions become the indices used in ``requested_spectra``. Only channels required by the
    requested spectra are opened and included when the runtime calculation is planned.

    The observation bounds describe one common half-open interval. Timestamped calculations require
    both bounds explicitly. For sampled-only calculations, the start can default to zero and the
    stop can be inferred from the active channel length and sampling interval.

    Attributes
    ----------
    channels : tuple[SampledChannel | TimestampedChannel, ...]
        Ordered, nonempty collection of measurement channels. Their tuple positions define the
        channel indices used to request auto- and cross-spectra.
    observation_start : int | float | None = None
        Start of the common observation interval, in ``t_unit``. Defaults to zero for sampled-only
        calculations. Must be specified when any active channel is timestamped.
    observation_stop : int | float | None = None
        Exclusive end of the common observation interval, in ``t_unit``. For sampled-only
        calculations, the default is ``observation_start + sample_count * dt``. Must be specified
        when any active channel is timestamped.
    t_unit : Literal["s", "ms", "us", "ns", "ps"] = "s"
        Time unit used by sampling intervals, timestamps, and observation bounds. It determines the
        corresponding result frequency unit: ``"Hz"``, ``"kHz"``, ``"MHz"``, ``"GHz"``, or
        ``"THz"``.
    """

    channels: tuple[SampledChannel | TimestampedChannel, ...]
    observation_start: int | float | None = None
    observation_stop: int | float | None = None
    t_unit: _TimeUnits = "s"

    def __post_init__(self) -> None:
        if not isinstance(self.channels, (list, tuple)):
            raise TypeError("channels must be a list or tuple of explicit channel objects.")

        if not self.channels:
            raise ValueError("channels must contain at least one channel.")

        normalized_channels = tuple(self.channels)

        for index, channel in enumerate(normalized_channels):
            if not isinstance(channel, (SampledChannel, TimestampedChannel)):
                raise TypeError(
                    f"Channel {index} must be a SampledChannel or TimestampedChannel; "
                    f"received {type(channel).__name__}."
                )

        observation_start = _normalize_observation_bound(
            self.observation_start,
            name="observation_start",
        )
        observation_stop = _normalize_observation_bound(
            self.observation_stop,
            name="observation_stop",
        )
        t_unit = _require_choice(self.t_unit, name="t_unit", choices=("s", "ms", "us", "ns", "ps"))

        if (
            observation_start is not None
            and observation_stop is not None
            and observation_start >= observation_stop
        ):
            raise ValueError("observation_start must be less than observation_stop.")

        object.__setattr__(self, "channels", normalized_channels)
        object.__setattr__(self, "observation_start", observation_start)
        object.__setattr__(self, "observation_stop", observation_stop)
        object.__setattr__(self, "t_unit", t_unit)


@dataclass(frozen=True, slots=True)
class TimestampOptions:
    """Statistical weighting options for active timestamped channels.

    Controls the amplitudes assigned to timestamped events before their Fourier coefficients are
    calculated. Unit weighting assigns every event an amplitude of one.
    Exponential weighting generates independent positive amplitudes for multiple realizations and
    averages the resulting spectral estimates. The same options apply to every active timestamped
    channel in a calculation.

    Attributes
    ----------
    weighting : Literal["unit", "exponential"]
        Event-amplitude model. ``"unit"`` performs one deterministic realization and does not
        accept any of the optional fields. ``"exponential"`` requires ``scale`` and
        ``repetitions``.
    scale : float | None = None
        Scale of the exponential amplitude distribution. Must be positive. Used only with
        exponential weighting.
    repetitions : int | None = None
        Number of independent exponential-amplitude realizations to average. Must be positive. Used
        only with exponential weighting.
    repetitions_per_batch : int | None = None
        Maximum number of amplitude realizations calculated in parallel. Larger values may improve
        throughput while increasing memory use. If omitted, at most 10 realizations are calculated
        per batch. Used only with exponential weighting.
    seed : int | None = None
        Nonnegative random seed for reproducible exponential amplitudes. If omitted, a random seed
        is chosen for each calculation. Results for an explicit seed do not depend on the
        configured repetition batch size. Used only with exponential weighting.
    """

    weighting: Literal["unit", "exponential"]
    scale: float | None = None
    repetitions: int | None = None
    repetitions_per_batch: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        weighting = _require_choice(
            self.weighting,
            name="weighting",
            choices=("unit", "exponential"),
        )

        scale = (
            None if self.scale is None else normalize_real(self.scale, name="scale", positive=True)
        )
        repetitions = (
            None
            if self.repetitions is None
            else _normalize_integer(self.repetitions, name="repetitions", minimum=1)
        )
        repetitions_per_batch = (
            None
            if self.repetitions_per_batch is None
            else _normalize_integer(
                self.repetitions_per_batch,
                name="repetitions_per_batch",
                minimum=1,
            )
        )
        seed = None if self.seed is None else _normalize_integer(self.seed, name="seed", minimum=0)

        if weighting == "unit":
            if any(
                value is not None for value in (scale, repetitions, repetitions_per_batch, seed)
            ):
                raise ValueError(
                    "Unit timestamp weighting does not accept scale, repetitions, "
                    "repetitions_per_batch, or seed."
                )
        elif scale is None or repetitions is None:
            raise ValueError("Exponential timestamp weighting requires scale and repetitions.")

        object.__setattr__(self, "weighting", weighting)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "repetitions", repetitions)
        object.__setattr__(self, "repetitions_per_batch", repetitions_per_batch)
        object.__setattr__(self, "seed", seed)


@dataclass(frozen=True, slots=True)
class SpectrumConfig:
    """Spectrum configuration for polyspectra calculations.

    :class:`SpectrumConfig` describes what the user asks the calculation to use: frequency spacing
    and bounds, window count per spectral estimate, backend torch device, and compatibility options.
    These settings are later resolved together with :class:`DataConfig` into the internal runtime
    configuration used by :func:`~signalsnap_pytorch.calculate_spectra`.

    ``df`` is the requested frequency spacing. For active sampled channels, the discrete Fourier
    transform cannot produce arbitrary frequency spacings for a given common sampling interval
    ``dt``. They are related via:

        window_points = round(1 / (dt * df)),

    where ``window_points`` is the number of samples used for each DFT. The calculation uses the
    closest available frequency spacing. Check the frequency axis of the
    :class:`~signalsnap_pytorch.results.SpectrumResult` to see the true frequencies.

    The specified frequency bounds and step use the reciprocal unit corresponding to
    :attr:`DataConfig.t_unit`.

    Attributes
    ----------
    df : float | None
        Requested frequency spacing. Must be positive. For calculations containing sampled channels,
        this may be omitted, in which case ``window_points`` defaults to 1000. Timestamp-only
        calculations require an explicit ``df``.
    f_min : float = 0.0
        Lower frequency bound. If omitted, zero is used.
    f_max : float | None = None
        Upper frequency bound. For calculations containing sampled channels, this may be omitted, in
        which case the Nyquist frequency determined by the active sampled channels' common ``dt`` is
        used. Timestamp-only calculations require an explicit ``f_max``.
    timestamp_options : TimestampOptions | None = None
        Statistical weighting applied to every active timestamped channel. Planning requires this
        for calculations containing timestamped channels and rejects it for sampled-only
        calculations.
    m : int = 10
        Number of windows used per spectral estimate. This may be reduced at runtime if the signal
        is too short. Must be at least as high as the highest requested order. Must be positive.
    uncertainty_estimation : Literal["global", "short_term"] = "global"
        Method used to estimate spectrum uncertainty. ``"global"`` computes the standard error of
        the mean from every spectral estimate. ``"short_term"`` divides consecutive estimates into
        complete batches of ``m_var``, calculates a variance-of-mean estimate for each batch,
        averages those variances, and takes their component-wise square root. Incomplete trailing
        batches contribute to the final spectrum but not its uncertainty.
    m_var : int = 10
        Number of consecutive spectral estimates in each short-term uncertainty batch. Must be at
        least two. Ignored when ``uncertainty_estimation="global"``. If fewer unshifted estimates
        are available, this value may be reduced at runtime.
    device : str = "cpu"
        Torch device requested for calculation. Can be ``"cpu"``, ``"cuda"``, ``"cuda:N"``,
        ``"mps"``, ``"xpu"``, or ``"xpu:N"``.
    precision : Literal["auto", "single", "double"] = "auto"
        Floating point precision. ``"single"`` uses ``float32`` and ``complex64``; ``"double"``
        uses ``float64`` and ``complex128``. ``"auto"`` chooses ``"single"`` for MPS or XPU and
        ``"double"`` for CPU or CUDA.
    spectral_estimates_max : int | None = int(1e6)
        Maximum number of unshifted spectral estimates. If ``None``, as many estimates as possible
        are calculated based on the data. The true number of spectral estimates may be lower if the
        data does not have enough samples. If ``interlacing=True``, up to the same number of
        additional shifted estimates are calculated. The number of shifted estimates may also be one
        less than the number of unshifted estimates if the final shifted windows do not fit. Must be
        positive.
    spectral_estimates_per_batch : int = 1
        Maximum number of spectral estimates calculated in parallel. This will speed up the
        calculation but increase the memory demands on the specified torch device. The final
        calculation batch may contain fewer estimates. In short-term mode, a batch that can hold at
        least one complete uncertainty group is reduced to a multiple of ``m_var``. Must be a
        positive integer.
    interlacing : bool = False
        Compute additional spectral estimates for windows shifted by half a window size, to
        compensate the low weight of data points produced by the window function near the original
        window edges. Uncertainty estimates are calculated separately for unshifted and shifted
        spectra; when both are available, the reported uncertainty is the component-wise maximum of
        the two placement-group uncertainty arrays.
    old_window : bool = False
        Compatibility option to reproduce legacy results. If set to ``True``, the old window
        function from the old API is used as a window function.
    """

    df: float | None = None
    f_min: float = 0.0
    f_max: float | None = None
    timestamp_options: TimestampOptions | None = None
    m: int = 10
    uncertainty_estimation: Literal["global", "short_term"] = "global"
    m_var: int = 10
    device: str = "cpu"
    precision: Literal["auto", "single", "double"] = "auto"
    spectral_estimates_max: int | None = int(1e6)
    spectral_estimates_per_batch: int = 1
    interlacing: bool = False
    old_window: bool = False

    def __post_init__(self) -> None:
        df = None if self.df is None else normalize_real(self.df, name="df", positive=True)
        f_min = normalize_real(self.f_min, name="f_min")
        f_max = None if self.f_max is None else normalize_real(self.f_max, name="f_max")

        if self.timestamp_options is not None and not isinstance(
            self.timestamp_options, TimestampOptions
        ):
            raise TypeError("timestamp_options must be a TimestampOptions object or None.")

        m = _normalize_integer(self.m, name="m", minimum=1)
        uncertainty_estimation = _require_choice(
            self.uncertainty_estimation,
            name="uncertainty_estimation",
            choices=("global", "short_term"),
        )
        m_var = _normalize_integer(self.m_var, name="m_var", minimum=2)
        device = _normalize_device(self.device)
        precision = _require_choice(
            self.precision,
            name="precision",
            choices=("auto", "single", "double"),
        )
        spectral_estimates_max = (
            None
            if self.spectral_estimates_max is None
            else _normalize_integer(
                self.spectral_estimates_max,
                name="spectral_estimates_max",
                minimum=1,
            )
        )
        spectral_estimates_per_batch = _normalize_integer(
            self.spectral_estimates_per_batch,
            name="spectral_estimates_per_batch",
            minimum=1,
        )
        interlacing = _require_bool(self.interlacing, name="interlacing")
        old_window = _require_bool(self.old_window, name="old_window")

        if f_max is not None and f_min >= f_max:
            raise ValueError(f"f_min ({f_min}) must be less than f_max ({f_max}).")

        object.__setattr__(self, "df", df)
        object.__setattr__(self, "f_min", f_min)
        object.__setattr__(self, "f_max", f_max)
        object.__setattr__(self, "m", m)
        object.__setattr__(self, "uncertainty_estimation", uncertainty_estimation)
        object.__setattr__(self, "m_var", m_var)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "spectral_estimates_max", spectral_estimates_max)
        object.__setattr__(self, "spectral_estimates_per_batch", spectral_estimates_per_batch)
        object.__setattr__(self, "interlacing", interlacing)
        object.__setattr__(self, "old_window", old_window)
