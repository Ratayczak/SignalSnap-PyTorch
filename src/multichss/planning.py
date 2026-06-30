# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from .results import SpectrumResult, SpectrumResultStore
from .utils import S3Calcs

if TYPE_CHECKING:
    from .configurators import CrossConfig, DataConfig, SpectrumConfig


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Resolved calculation settings derived from user configuration.

    :class:`SpectrumConfig` and :class:`DataConfig` describe what the user asked for
    :class:`RuntimeConfig` describes what the calculation will actually use after defaults,
    data-size constraints, frequency axes, and device details have been resolved.

    Attributes
    ----------
    selected_channels : tuple[int, ...]
        Data-channel indices used by the calculation.
    orders : tuple[int, ...]
        Spectrum orders to calculate.
    dt : float
        Sampling interval shared by all selected data channels.
    window_points : int
        Number of samples per window.
    m : int
        Number of windows used per spectral estimate. This may be reduced at runtime if the signal
        is too short. Must be positive.
    n_data_points : int
        Number of samples in each selected data channel.
    n_windows : int
        Number of window groups processed by the calculation.
    freq_band : np.ndarray
        Selected frequency axis.
    f_min_idx, f_max_idx : int
        Slice indices selecting the configured frequency band.
    use_full_fft : bool
        Whether negative frequencies require full FFT handling.
    real_dtype : torch.dtype
        Sets the dtype of floats.
    complex_dtype : torch.dtype
        Sets the dtype of complex numbers.
    device : torch.device
        Torch device used for calculation.
    s3_calc : Literal["1/4", "1/2"]
        Method used for third-order spectrum calculation.
    spectral_estimates_max : int | None
        Maximum number of spectral estimates. If ``None``, as many estimates as possible are
        calculated based on the data. The true number of spectral estimates may be lower if the data
        does not have enough samples. Must be positive. # TODO
    old_window : bool
        Compatibility option. If set to ``True``, the approximated confined Gaussian window from the
        old API is used as a window function.
    """

    selected_channels: tuple[int, ...]
    orders: tuple[int, ...]
    dt: float
    window_points: int
    m: int
    n_data_points: int
    n_windows: int
    freq_band: np.ndarray
    f_min_idx: int
    f_max_idx: int
    use_full_fft: bool
    real_dtype: torch.dtype
    complex_dtype: torch.dtype
    device: torch.device
    s3_calc: S3Calcs
    spectral_estimates_max: int | None
    old_window: bool


@dataclass(frozen=True, slots=True)
class SpectrumTask:
    """Description of one spectrum that should be calculated.

    A task is the normalized representation of a user request after :class:`SpectrumConfig` and
    :class:`CrossConfig` have been expanded. It identifies one polyspectrum order and the channel
    tuple that should be used for that calculation.

    Parameters
    ----------
    order : int
        The polyspectrum order to calculate.
    channels : tuple[int, ...]
        The channel indices used by this calculation. Auto-spectra repeat the same channel once per
        order, e.g. ``(0, 0)`` for the second-order spectrum of channel 0 and ``(0, 0, 0)`` for the
        third-order spectrum. Cross-spectra are represented by the configured channel tuple, e.g.
        ``(0, 1)`` for a second-order cross-spectrum.
    """

    order: int
    channels: tuple[int, ...]


def _normalize_selected(
    data_config_list: list[DataConfig], selected: list[int] | None = None
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


def _validate_data_configs(
    data_config_list: list[DataConfig], selected: tuple[int, ...]
) -> tuple[int, float]:
    """Validate selected data and return ``(n_data_points, dt)``."""

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

    return n_data_points, dt


def build_runtime_config(
    spectrum_config: SpectrumConfig,
    data_config_list: list[DataConfig],
    selected: list[int] | None = None,
) -> RuntimeConfig:
    """Resolve user configuration into immutable runtime calculation settings.

    Validates the selected data channels, derives the frequency axis and frequency-band indices,
    checks Nyquist-frequency bounds, resolves the effective window size and window count, and
    selects torch dtypes and device settings used by the spectrum calculation.

    Parameters
    ----------
    spectrum_config : :class:`SpectrumConfig`
        User configuration for spectrum orders, frequency bounds, precision, device, windowing, and
        related calculation options.
    data_config_list : list[:class:`DataConfig`]
        Data configurations containing the input data and sampling metadata.
    selected : list[int] | None, optional
        Data-channel indices to use. If ``None``, all data configurations are selected.
    """
    selected_channels = _normalize_selected(data_config_list, selected)
    n_data_points, dt = _validate_data_configs(data_config_list, selected_channels)
    f_max_allowed = 1 / (2 * dt)
    f_max = spectrum_config.f_max

    if f_max is None:
        f_max = f_max_allowed

        if f_max <= spectrum_config.f_min:
            raise ValueError("f_min is larger than the Nyquist frequency.")

    if f_max > f_max_allowed:
        raise ValueError("f_max is larger than the Nyquist frequency.")

    if spectrum_config.f_min < -f_max_allowed:
        raise ValueError("f_min outside of Nyquist frequency bounds.")

    window_T = (spectrum_config.frequency_points - 1) / (f_max - spectrum_config.f_min)
    window_points = int(np.round(window_T / dt))
    if window_points <= 0:
        raise ValueError("Calculated window_points must be greater than zero.")

    orders = [1, 2, 3, 4] if spectrum_config.orders == "all" else list(spectrum_config.orders)
    if spectrum_config.f_min < 0 and 3 in orders:
        raise ValueError(
            "For negative frequencies in order 3 use s3_calc='1/2' and positive frequencies.\n"
            "Example: f_min=0, f_max=5, s3_calc='1/2'"
        )

    if not orders:
        raise ValueError("No spectrum orders remain after applying runtime constraints.")

    required_points = window_points * spectrum_config.m + window_points // 2
    if not required_points < n_data_points:
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

    if spectrum_config.precision == "single":
        real_dtype = torch.float32
        complex_dtype = torch.complex64
    elif spectrum_config.precision == "double":
        real_dtype = torch.float64
        complex_dtype = torch.complex128
    else:
        if spectrum_config.device == "mps":
            real_dtype = torch.float32
            complex_dtype = torch.complex64
        else:
            real_dtype = torch.float64
            complex_dtype = torch.complex128

    return RuntimeConfig(
        selected_channels=selected_channels,
        orders=tuple(orders),
        dt=dt,
        window_points=window_points,
        m=m,
        n_data_points=n_data_points,
        n_windows=n_windows,
        freq_band=freq_all[f_min_idx:f_max_idx],
        f_min_idx=f_min_idx,
        f_max_idx=f_max_idx,
        use_full_fft=use_full_fft,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        device=torch.device(spectrum_config.device),
        s3_calc=spectrum_config.s3_calc,
        spectral_estimates_max=spectrum_config.spectral_estimates_max,
        old_window=spectrum_config.old_window,
    )


def build_spectrum_tasks(
    runtime_config: RuntimeConfig, cross_config: CrossConfig
) -> list[SpectrumTask]:
    """Build the concrete spectrum tasks requested by the configuration.

    Expands the high-level configuration into one :class:`SpectrumTask` per spectrum that should be
    calculated. Auto-correlation tasks are generated for each selected channel when
    ``cross_config.auto_corr`` is enabled. Cross tasks are generated from ``cross_corr_2``,
    ``cross_corr_3``, and ``cross_corr_4`` when their corresponding orders are requested.

    Parameters
    ----------
    runtime_config : :class:`RuntimeConfig`
        Configuration for spectrum order, frequency bounds, and numerical calculation settings.
    cross_config : :class:`CrossConfig`
        Configuration describing whether auto-spectra and which cross spectra should be calculated.

    Returns
    -------
    list[:class:`SpectrumTask`]
        Ordered list of concrete spectrum calculations to perform.
    """

    tasks: list[SpectrumTask] = []

    if cross_config.auto_corr:
        for channel in runtime_config.selected_channels:
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
                raise ValueError(f"Order {order} spectra require {order} channels, got {channels}.")
            for channel in channels:
                if channel not in runtime_config.selected_channels:
                    raise ValueError(
                        f"Cross spectrum {channels} references channel {channel}, "
                        f"which is not in selected channels {runtime_config.selected_channels}."
                    )
            tasks.append(SpectrumTask(channels=channels, order=order))

    if not tasks:
        raise ValueError("No spectrum tasks were requested.")

    task_keys = [(task.channels, task.order) for task in tasks]
    if len(task_keys) != len(set(task_keys)):
        raise ValueError("Duplicate spectrum tasks were requested.")

    return tasks


def initialize_result_store(
    tasks: list[SpectrumTask], runtime: RuntimeConfig
) -> SpectrumResultStore:
    """Create an initialized result store for a list of spectrum tasks.

    Each task is converted into a :class:`SpectrumResult` with matching order and channels.

    Parameters
    ----------
    tasks : list[:class:`SpectrumTask`]
        Spectrum tasks that should receive corresponding result containers.
    runtime : :class:`RuntimeConfig`
        :class:`RuntimeConfig` that contains all necessary information to initialize result arrays.

    Returns
    -------
    SpectrumResultStore
        Store containing one initialized :class:`SpectrumResult` per task.
    """
    
    store = SpectrumResultStore()
    for task in tasks:
        store.add(SpectrumResult(order=task.order, channels=task.channels))
    store.initialize_arrays(runtime)
    return store
