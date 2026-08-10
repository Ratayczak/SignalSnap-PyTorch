# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemad, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    ValidationInfo,
    field_validator,
    model_validator,
)

from ._core.utils import TimeUnits as _TimeUnits

__all__ = [
    "DataConfig",
    "HDF5Source",
    "PhotonOptions",
    "SampledChannel",
    "SpectrumConfig",
    "TimestampedChannel",
]

_SHARED_CONFIG = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class HDF5Source(BaseModel):
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

    model_config = _SHARED_CONFIG

    file: Path
    dataset: str
    selection: tuple[Any, ...]

    @field_validator("dataset")
    @classmethod
    def _validate_dataset(cls, value: str) -> str:
        """Reject an empty HDF5 dataset path."""
        if not value:
            raise ValueError("dataset cannot be empty.")
        return value

    @field_validator("selection")
    @classmethod
    def _validate_selection(cls, value: tuple[Any, ...]) -> tuple[Any, ...]:
        """Validate selector syntax and normalize NumPy integer components."""
        if not value:
            raise ValueError("selection cannot be empty.")

        normalized = []

        for item in value:
            if isinstance(item, (bool, np.bool_)):
                raise TypeError("HDF5 selection entries must be integers or slices.")

            if isinstance(item, np.integer):
                item = int(item)

            if not isinstance(item, (int, slice)):
                raise TypeError("HDF5 selection entries must be integers or slices.")

            if isinstance(item, slice):
                if item.start is None:
                    start = None
                else:
                    if isinstance(item.start, (bool, np.bool_)):
                        raise TypeError("HDF5 slice start must be an integer or None.")
                    if not isinstance(item.start, (int, np.integer)):
                        raise TypeError("HDF5 slice start must be an integer or None.")
                    start = int(item.start)

                if item.stop is None:
                    stop = None
                else:
                    if isinstance(item.stop, (bool, np.bool_)):
                        raise TypeError("HDF5 slice stop must be an integer or None.")
                    if not isinstance(item.stop, (int, np.integer)):
                        raise TypeError("HDF5 slice stop must be an integer or None.")
                    stop = int(item.stop)

                if item.step is None:
                    step = None
                else:
                    if isinstance(item.step, (bool, np.bool_)):
                        raise TypeError("HDF5 slice step must be an integer or None.")
                    if not isinstance(item.step, (int, np.integer)):
                        raise TypeError("HDF5 slice step must be an integer or None.")
                    step = int(item.step)

                if step not in (None, 1):
                    raise ValueError("HDF5 slice steps other than 1 are not supported.")

                normalized.append(slice(start, stop, step))
            else:
                normalized.append(item)

        return tuple(normalized)


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


class SampledChannel(BaseModel):
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

    model_config = _SHARED_CONFIG

    data: Any
    dt: Annotated[float, Field(gt=0)]

    @field_validator("data")
    @classmethod
    def _validate_data(cls, data: Any) -> Any:
        """Validate sampled storage, shape, device, and dtype."""

        return _validate_stored_data(
            data,
            label="SampledChannel data",
            allow_empty=False,
            allow_boolean=True,
        )

    @field_validator("dt", mode="before")
    @classmethod
    def _reject_boolean_dt(cls, value: Any) -> Any:
        """Reject Boolean sampling intervals before numeric coercion."""

        if isinstance(value, (bool, np.bool_)):
            raise TypeError("SampledChannel dt must be a positive finite number.")

        return value


class TimestampedChannel(BaseModel):
    """Configuration for one timestamped measurement channel.

    A :class:`TimestampedChannel` represents discrete events by their occurrence times. Active
    timestamped channels are transformed directly at the required frequencies, with event
    amplitudes determined by :class:`PhotonOptions`. Their timestamps must lie within the explicit
    observation interval configured by :class:`DataConfig`.

    In-memory timestamps are retained without copying and must not be mutated during a calculation.

    Attributes
    ----------
    timestamps : numpy.ndarray | torch.Tensor | HDF5Source
        Event times in units of :attr:`DataConfig.t_unit`. In-memory input must be a one-dimensional
        NumPy array or CPU PyTorch tensor containing finite, nondecreasing real numbers. Empty
        inputs and duplicate timestamps are valid. An :class:`HDF5Source` is read lazily and
        flattened in C order; the flattened timestamps must satisfy the same ordering constraints.
    """

    model_config = _SHARED_CONFIG

    timestamps: Any

    @field_validator("timestamps")
    @classmethod
    def _validate_timestamps(cls, timestamps: Any) -> Any:
        """Validate timestamp storage, shape, device, dtype, and ordering."""

        timestamps = _validate_stored_data(
            timestamps,
            label="TimestampedChannel timestamps",
            allow_empty=True,
            allow_boolean=False,
        )

        if isinstance(timestamps, HDF5Source):
            return timestamps

        if isinstance(timestamps, torch.Tensor):
            if not bool(torch.isfinite(timestamps).all().item()):
                raise ValueError("TimestampedChannel timestamps must contain only finite values.")

            if timestamps.numel() > 1 and bool(torch.any(timestamps[1:] < timestamps[:-1]).item()):
                raise ValueError("TimestampedChannel timestamps must be nondecreasing.")

            return timestamps

        if not np.all(np.isfinite(timestamps)):
            raise ValueError("TimestampedChannel timestamps must contain only finite values.")

        if timestamps.size > 1 and np.any(timestamps[1:] < timestamps[:-1]):
            raise ValueError("TimestampedChannel timestamps must be nondecreasing.")

        return timestamps


class DataConfig(BaseModel):
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

    model_config = _SHARED_CONFIG

    channels: Annotated[tuple[SampledChannel | TimestampedChannel, ...], Field(min_length=1)]
    observation_start: StrictInt | StrictFloat | None = None
    observation_stop: StrictInt | StrictFloat | None = None
    t_unit: _TimeUnits = "s"

    @field_validator("channels", mode="before")
    @classmethod
    def _require_explicit_channels(cls, channels: Any) -> Any:
        """Reject bare arrays, tensors, HDF5 sources, and other implicit channels."""

        if not isinstance(channels, (list, tuple)):
            raise TypeError("channels must be a list or tuple of explicit channel objects.")

        for index, channel in enumerate(channels):
            if not isinstance(channel, (SampledChannel, TimestampedChannel)):
                raise TypeError(
                    f"Channel {index} must be a SampledChannel or TimestampedChannel; "
                    f"received {type(channel).__name__}."
                )

        return channels

    @field_validator("observation_start", "observation_stop", mode="before")
    @classmethod
    def _normalize_observation_bound(cls, value: Any) -> Any:
        """Preserve integral origins while normalizing NumPy scalar bounds."""

        if value is None:
            return None

        if isinstance(value, (bool, np.bool_)):
            raise TypeError("Observation bounds must be finite real numbers.")

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        return value

    @model_validator(mode="after")
    def _validate_observation_interval(self) -> DataConfig:
        """Require ordered bounds when both observation limits are explicit."""

        if (
            self.observation_start is not None
            and self.observation_stop is not None
            and self.observation_start >= self.observation_stop
        ):
            raise ValueError("observation_start must be less than observation_stop.")

        return self


class PhotonOptions(BaseModel):
    """Statistical weighting options for active timestamped channels.

    :class:`PhotonOptions` controls the amplitudes assigned to timestamped events before their
    Fourier coefficients are calculated. Unit weighting assigns every event an amplitude of one.
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

    model_config = _SHARED_CONFIG

    weighting: Literal["unit", "exponential"]
    scale: Annotated[float, Field(gt=0)] | None = None
    repetitions: Annotated[int, Field(gt=0)] | None = None
    repetitions_per_batch: Annotated[int, Field(gt=0)] | None = None
    seed: Annotated[StrictInt, Field(ge=0)] | None = None

    @field_validator("scale", "repititions", "repititions_per_batch", mode="before")
    @classmethod
    def _reject_boolean_numeric_fields(
        cls,
        value: Any,
        info: ValidationInfo,
    ) -> Any:
        """Reject Booleans before numeric coercion."""
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{info.field_name} cannot be Boolean.")
        return value
    

    @model_validator(mode="after")
    def _validate_weighting_fields(self) -> PhotonOptions:
        """Require only the fields belonging to the selected weighting."""

        exponential_fields = (self.scale, self.repetitions, self.repetitions_per_batch, self.seed)

        if self.weighting == "unit":
            if any(value is not None for value in exponential_fields):
                raise ValueError(
                    "Unit photon weighting does not accept scale, repetitions, "
                    "repetitions_per_batch, or seed."
                )

            return self

        if self.scale is None or self.repetitions is None:
            raise ValueError("Exponential photon weighting requires scale and repetitions.")

        return self


class SpectrumConfig(BaseModel):
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
    photon_options : PhotonOptions | None = None
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

    model_config = _SHARED_CONFIG

    df: Annotated[float, Field(gt=0)] | None = None
    f_min: float = 0.0
    f_max: float | None = None
    photon_options: PhotonOptions | None = None
    m: Annotated[int, Field(gt=0)] = 10
    uncertainty_estimation: Literal["global", "short_term"] = "global"
    m_var: Annotated[int, Field(ge=2)] = 10
    device: str = "cpu"
    precision: Literal["auto", "single", "double"] = "auto"
    spectral_estimates_max: Annotated[int, Field(gt=0)] | None = int(1e6)
    spectral_estimates_per_batch: Annotated[int, Field(ge=1)] = 1
    interlacing: bool = False
    old_window: bool = False

    @model_validator(mode="after")
    def _validate_limits(self) -> SpectrumConfig:
        """Require the lower frequency bound to precede an explicit upper bound."""
        if self.f_max is not None and self.f_min >= self.f_max:
            raise ValueError(f"f_min ({self.f_min}) must be less than f_max ({self.f_max}).")

        return self

    @field_validator(
        "df",
        "f_min",
        "f_max",
        "m",
        "m_var",
        "spectral_estimates_max",
        "spectral_estimates_per_batch",
        mode="before",
    )
    @classmethod
    def _reject_boolean_numeric_fields(
        cls,
        value: Any,
        info: ValidationInfo,
    ) -> Any:
        """Reject Booleans before numeric coercion."""
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{info.field_name} cannot be Boolean.")
        return value

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        """Validate device syntax without checking hardware availability."""
        try:
            device = torch.device(value)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                "device must be 'cpu', 'mps', 'cuda', 'cuda:N', 'xpu', or 'xpu:N', where N "
                "is a nonnegative integer."
            ) from exc

        if device.type not in {"cpu", "cuda", "mps", "xpu"}:
            raise ValueError(
                f"Unsupported device type {device.type!r}; use 'cpu', 'mps', 'cuda', 'cuda:N', "
                "'xpu', or 'xpu:N'."
            )

        if device.type in {"cpu", "mps"} and device.index is not None:
            raise ValueError(f"{device.type!r} does not support a numbered device index.")

        return str(device)
