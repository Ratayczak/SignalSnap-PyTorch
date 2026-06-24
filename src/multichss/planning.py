# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from .results import SpectrumResult, SpectrumResultStore
from .utils import FrequencyUnits, TimeUnits, unit_conversion_time_to_freq

if TYPE_CHECKING:
    from .configurators import CrossConfig, DataConfig, SpectrumConfig
    

@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Resolved calculation settings derived from user configuration.

    ``SpectrumConfig`` and ``DataConfig`` describe what the user asked for.
    ``RuntimeConfig`` describes what the calculation will actually use 
    after defaults, data-size constraints, frequency axes, and device
    details have been resolved.

    Parameters
    ----------
    selected : tuple[int, ...]
        Data-channel indices used by the calculation.
    orders : tuple[int, ...]
        Spectrum orders to calculate.
    dt : float
        Sampling interval shared by all selected data channels.
    fs : float
        Sampling frequency.
    t_unit : Literal["s", "ms", "us", "ns", "ps"]
        Unit of time step.
    f_unit : Literal["Hz", "kHz", "MHz", "GHz", "THz"]
        Unit of sampling frequency.
    f_max_allowed : float
        Nyquist frequency.
    f_min, f_max : float
        Effective frequency bounds used for index selection.
    t_window : float
        Window duration in time units.
    window_points : int
        Number of samples per window.
    m : int
        Effective window count per spectrum
    n_data_points : int
        Number of samples in each selected data channel.
    n_windows : int
        Number of window groups processed by the calculation.
    freq_all : np.ndarray
        Full generated frequency axis before selecting ``f_min:f_max``.
    f_min_idx, f_max_idx : int
        Slice indices selecting the configured frequency band.
    use_full_fft : bool
        Whether negative frequencies require full FFT handling.
    use_float32 : bool
        Whether host data should be converted to float32 before device
        upload.
    device : torch.device
        Torch device used for calculation.
    s3_calc: Literal["1/4", "1/2"]
        
    """

    selected: tuple[int, ...]
    orders: tuple[int, ...]
    dt: float
    fs: float
    t_unit: TimeUnits
    f_unit: FrequencyUnits
    f_max_allowed: float
    f_min: float
    f_max: float
    t_window: float
    window_points: int
    m: int  # covers m and m_var
    n_data_points: int
    n_windows: int
    freq_all: np.ndarray
    f_min_idx: int
    f_max_idx: int
    use_full_fft: bool
    use_float32: bool
    device: torch.device
    s3_calc: Literal["1/4", "1/2"]


@dataclass(frozen=True, slots=True)
class SpectrumTask:
    """Description of one spectrum that should be calculated.

    A task is the normalized representation of a user request after
    :class:`SpectrumConfig` and :class:`CrossConfig` have been expanded.
    It identifies one polyspectrum order and the channel tuple that should
    be used for that calculation.

    Parameters
    ----------
    order : int
        The polyspectrum order to calculate.
    channels : tuple[int, ...]
        The channel indices used by this calculation. Auto-spectra repeat
        the same channel once per order, e.g. ``(0, 0)`` for the
        second-order spectrum of channel 0 and ``(0, 0, 0)`` for the
        third-order spectrum. Cross-spectra are represented by the
        configured channel tuple, e.g. ``(0, 1)`` for a second-order
        cross-spectrum.
    """

    order: int
    channels: tuple[int, ...]


def normalize_selected(
    data_config_list: list[DataConfig],
    selected: list[int] | None = None,
) -> tuple[int, ...]:
    """Resolve selected data-channel indices."""
    if selected is None:
        return tuple(range(len(data_config_list)))

    if not selected:
        raise ValueError("At least one data channel must be selected.")

    n_data_configs = len(data_config_list)
    for channel in selected:
        if channel < 0 or channel >= n_data_configs:
            raise IndexError(
                f"Selected channel {channel} is outside available data "
                f"channels 0..{n_data_configs - 1}."
            )

    return tuple(selected)


def validate_data_configs(
    data_config_list: list[DataConfig],
    selected: tuple[int, ...],
) -> tuple[int, float, TimeUnits]:
    """Validate selected data and return ``(n_data_points, dt, t_unit)``."""
    if not data_config_list:
        raise ValueError("At least one DataConfig is required.")

    if not selected:
        raise ValueError("At least one data channel must be selected.")

    first_config = data_config_list[selected[0]]
    if first_config.data is None:
        raise ValueError(f"Selected channel {selected[0]} does not contain data.")

    try:
        n_data_points = first_config.data.shape[0]
    except AttributeError as exc:
        raise TypeError("DataConfig.data must provide a shape attribute.") from exc

    dt = first_config.dt
    t_unit = first_config.t_unit

    for channel in selected:
        data_config = data_config_list[channel]
        if data_config.data is None:
            raise ValueError(f"Selected channel {channel} does not contain data.")
        if data_config.data.shape[0] != n_data_points:
            raise ValueError("Imported data must have same length!")
        if data_config.dt != dt or data_config.t_unit != t_unit:
            raise ValueError("Selected data channels must use the same dt and t_unit.")

    return n_data_points, dt, t_unit


def build_runtime_config(
    spectrum_config: SpectrumConfig,
    data_config_list: list[DataConfig],
    selected: list[int] | None = None,
) -> RuntimeConfig:
    """Resolve immutable user configuration into calculation runtime values.

    This absorbs the non-mutating parts of the old calculator ``__init__``
    and ``setup_calc_spec`` logic: selected-channel normalization, data 
    validation, Nyquist-derived ``f_max`` defaulting, window sizing, 
    effective ``m`` value, FFT mode, device precision behavior, and 
    frequency-band indices.
    """
    selected_channels = normalize_selected(data_config_list, selected)
    n_data_points, dt, t_unit = validate_data_configs(data_config_list, 
                                                      selected_channels)

    device = torch.device(spectrum_config.backend)
    fs = 1 / dt
    f_max_allowed = 1 / (2 * dt)
    f_max = spectrum_config.f_max
    if f_max is None:
        f_max = f_max_allowed

    window_len_factor = f_max_allowed / (f_max - spectrum_config.f_min)
    t_window = (spectrum_config.spectrum_size - 1) * (2 * dt * window_len_factor)
    window_points = int(np.round(t_window / dt))
    if window_points <= 0:
        raise ValueError("Calculated window_points must be greater than zero.")

    orders = (
        [1, 2, 3, 4]
        if spectrum_config.order_in == "all"
        else list(spectrum_config.order_in)
    )
    if spectrum_config.f_min < 0 and 3 in orders:
        print(
            "For negative frequencies in order 3 use s3_calc and positive frequencies\n"
        )
        print("Example: f_min=0, f_max=5, s3_calc='1/2'")
        orders.remove(3)

    if not orders:
        raise ValueError(
            "No spectrum orders remain after applying runtime constraints."
        )

    if not window_points * spectrum_config.m + window_points // 2 < n_data_points:
        m = (n_data_points - window_points // 2) // window_points
        if m < max(orders):
            raise ValueError("Not enough data points")
        print(
            "Values have been changed, because not enough data points were available."
            f"Old m: {spectrum_config.m}, new m: {m}"
        )
    else:
        m = spectrum_config.m

    n_windows = int(np.floor(n_data_points / (m * window_points)))
    use_full_fft = spectrum_config.f_min < 0
    if use_full_fft:
        freq_all = np.fft.fftfreq(window_points, dt)
        freq_all = np.fft.fftshift(freq_all)
    else:
        freq_all = np.fft.rfftfreq(window_points, dt)

    f_max_idx = int(np.sum(freq_all <= f_max))
    f_min_idx = int(np.sum(freq_all < spectrum_config.f_min))

    return RuntimeConfig(
        selected=selected_channels,
        orders=tuple(orders),
        dt=dt,
        fs=fs,
        t_unit=t_unit,
        f_unit=unit_conversion_time_to_freq(t_unit),
        f_max_allowed=f_max_allowed,
        f_min=spectrum_config.f_min,
        f_max=f_max,
        t_window=t_window,
        window_points=window_points,
        m=m,
        n_data_points=n_data_points,
        n_windows=n_windows,
        freq_all=freq_all,
        f_min_idx=f_min_idx,
        f_max_idx=f_max_idx,
        use_full_fft=use_full_fft,
        use_float32=spectrum_config.backend == "mps",
        device=device,
        s3_calc=spectrum_config.s3_calc
    )


def build_spectrum_tasks(
    runtime_config: RuntimeConfig,
    cross_config: CrossConfig,
) -> list[SpectrumTask]:
    """Build the concrete spectrum tasks requested by the configuration.

    Expands the high-level configuration into one :class:`SpectrumTask` per
    spectrum that should be calculated. Auto-correlation tasks are 
    generated for each selected channel when ``cross_config.auto_corr`` is
    enabled. Cross tasks are generated from ``cross_corr_2``,
    ``cross_corr_3``, and ``cross_corr_4`` when their corresponding orders
    are requested.

    Parameters
    ----------
    runtime_config : RuntimeConfig
        Configuration for spectrum order, frequency bounds, and numerical
        calculation settings.
    cross_config : CrossConfig
        Configuration describing whether auto-spectra and which cross
        spectra should be calculated.

    Returns
    -------
    list[SpectrumTask]
        Ordered list of concrete spectrum calculations to perform.
    """
    tasks: list[SpectrumTask] = []

    if cross_config.auto_corr:
        for channel in runtime_config.selected:
            for order in runtime_config.orders:
                channels = (channel,) * order
                tasks.append(SpectrumTask(order=order, channels=channels))

    cross_specs = {
        2: cross_config.cross_corr_2 or [],
        3: cross_config.cross_corr_3 or [],
        4: cross_config.cross_corr_4 or [],
    }

    for order, channel_groups in cross_specs.items():
        if order not in runtime_config.orders:
            continue
        for channels in channel_groups:
            channels = tuple(channels)
            if len(channels) != order:
                raise ValueError(
                    f"Order {order} spectra require {order} channels, got {channels}."
                )
            tasks.append(SpectrumTask(channels=channels, order=order))

    return tasks


def initialize_result_store(tasks: list[SpectrumTask]) -> SpectrumResultStore:
    """Create an empty result store for a list of spectrum tasks.

    Each task is converted into a :class:`SpectrumResult` with matching 
    order and channels. The returned store contains result containers only;
    device buffers, frequency axes, accumulated spectra, and error 
    estimates are filled later by the calculation pipeline.

    Parameters
    ----------
    tasks : list[SpectrumTask]
        Spectrum tasks that should receive corresponding result containers.

    Returns
    -------
    SpectrumResultStore
        Store containing one empty :class:`SpectrumResult` per task.
    """
    store = SpectrumResultStore()
    for task in tasks:
        store.add(SpectrumResult(order=task.order, channels=task.channels))
    return store
