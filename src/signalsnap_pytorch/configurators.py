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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ._core.utils import TimeUnits as _TimeUnits

_SHARED_CONFIG = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class DataConfig(BaseModel):
    """Configuration for data used in polyspectra calculations.

    These settings are later resolved together with :class:`SpectrumConfig` into the internal
    runtime configuration used by :func:`~signalsnap_pytorch.calculate_spectra`.

    Together with ``df`` from :class:`SpectrumConfig`, ``dt`` will be used to determine the number
    of data points (``window_points``) used for each Fourier transform:

        window_points = round(1 / (dt * df))

    and to determine the Nyquist frequency:

        f_nyquist = 1 / (2 * dt)

    If ``df`` is not specified in :class:`SpectrumConfig`, ``window_points`` is set to ``1000``.

    Attributes
    ----------
    channels : tuple[Any, ...]
        Tuple of data channels. Each channel is recorded (real) signal data and can either be a
        one-dimensional, nonempty, real-valued numeric or Boolean array with a shape and dtype
        attribute or a :class:`HDF5Channel`.
    dt : float
        The time interval between two consecutive data points. Must be positive.
    t_unit : Literal["s", "ms", "us", "ns", "ps"]
        Unit of the time step. Defaults to ``"s"``.
    """

    model_config = _SHARED_CONFIG

    channels: Annotated[tuple[Any, ...], Field(min_length=1)]
    dt: Annotated[float, Field(gt=0)]
    t_unit: _TimeUnits = "s"

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, channels: tuple[Any, ...]) -> tuple[Any, ...]:
        """Validate the shape and dtype of in-memory channels."""
        for index, channel in enumerate(channels):
            if isinstance(channel, HDF5Channel):
                continue

            if channel is None:
                raise ValueError(f"Channel {index} cannot be None.")

            if not hasattr(channel, "shape"):
                raise TypeError(f"Array channel {index} must provide a shape attribute.")

            if len(channel.shape) != 1:
                raise ValueError(f"Array channel {index} must be one-dimensional.")

            if channel.shape[0] == 0:
                raise ValueError(f"Array channel {index} cannot be empty.")

            try:
                channel_dtype = np.dtype(channel.dtype)
            except TypeError:
                channel_dtype = np.asarray(channel).dtype

            if np.issubdtype(channel_dtype, np.complexfloating):
                raise TypeError(f"Array channel {index} cannot be complex.")

            is_numeric = np.issubdtype(channel_dtype, np.number)
            is_boolean = np.issubdtype(channel_dtype, np.bool_)

            if not is_numeric and not is_boolean:
                raise TypeError(
                    f"Array channel {index} must be numeric; received dtype {channel_dtype}."
                )

        return channels


class HDF5Channel(BaseModel):
    """Configuration for HDF5 input channels.

    The specified data channel inside the HDF5 file is loaded lazily to allow for inputs exceeding
    system memory.

    Attributes
    ----------
    file : Path
        File path of the HDF5 file.
    dataset : str
        Dataset path inside the HDF5 file.
    selection : tuple[Any, ...]
        Selection of the data channel in the dataset. Can include integer and slice selectors. Slice
        step must be 1. The selection must contain a non-empty, real-valued numeric or Boolean
        output and can have at most two unfixed dimensions. The selected values are flattened in
        C-order. Example: (slice(None), slice(None), 0).
    """

    model_config = _SHARED_CONFIG

    file: Path
    dataset: str
    selection: tuple[Any, ...]

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, value: str) -> str:
        """Reject an empty HDF5 dataset path."""
        if not value:
            raise ValueError("dataset cannot be empty.")
        return value

    @field_validator("selection")
    @classmethod
    def validate_selection(cls, value: tuple[Any, ...]) -> tuple[Any, ...]:
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


class SpectrumConfig(BaseModel):
    """Spectrum configuration for polyspectra calculations.

    :class:`SpectrumConfig` describes what the user asks the calculation to use: frequency spacing
    and bounds, window count per spectral estimate, backend torch device, and compatibility options.
    These settings are later resolved together with :class:`DataConfig` into the internal runtime
    configuration used by :func:`~signalsnap_pytorch.calculate_spectra`.

    ``df`` is the requested frequency spacing. The discrete Fourier transform cannot result in
    arbitrary frequency spacings with a given sample spacing ``dt`` from :class:`DataConfig`. They
    are related via:

        window_points = round(1 / (dt * df)),

    where ``window_points`` is the number of samples used for each DFT. The calculation uses the
    closest available frequency spacing. Check the frequency axis of the
    :class:`~signalsnap_pytorch.results.SpectrumResult` to see the true frequencies.

    Attributes
    ----------
    df : float | None
        Requested frequency spacing in the specified frequency range. Must be positive. If omitted,
        ``window_points`` is set to 1000.
    f_min : float = 0.0
        Lower frequency bound. If omitted, zero is used.
    f_max : float | None = None
        Upper frequency bound. If omitted, the Nyquist frequency based on :class:`DataConfig`'s
        ``dt`` is used.
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
    def validate_limits(self) -> SpectrumConfig:
        """Require the lower frequency bound to precede an explicit upper bound."""
        if self.f_max is not None and self.f_min >= self.f_max:
            raise ValueError(f"f_min ({self.f_min}) must be less than f_max ({self.f_max}).")

        return self

    @field_validator("spectral_estimates_per_batch", mode="before")
    @classmethod
    def reject_boolean_spectral_estimates_per_batch(cls, value: Any) -> Any:
        """Reject Booleans before Pydantic coerces them to integers."""
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("spectral_estimates_per_batch must be a positive integer.")
        return value

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
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
