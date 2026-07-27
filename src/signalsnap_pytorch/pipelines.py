# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemad, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import warnings

from tqdm.auto import tqdm

from ._core import accumulation as _accumulation
from ._core import data_access as _data_access
from ._core import fft as _fft
from ._core import planning as _planning
from ._core import spectra as _spectra
from .configurators import DataConfig, SpectrumConfig
from .results import SpectrumResultStore

__all__ = ["calculate_spectra"]


def calculate_spectra(
    data_config: DataConfig,
    spectrum_config: SpectrumConfig,
    *,
    requested_spectra: list[tuple[int, ...]] | None = None,
    show_progress: bool = True,
) -> SpectrumResultStore:
    """Calculate requested auto- and cross-polyspectra for one or more data channels.

    Builds the runtime configuration, expands the requested spectrum tasks, iterates over windowed
    signal chunks, computes Fourier coefficients, accumulates spectra, and finalizes mean spectra
    and uncertainty estimates.

    Per-spectrum calculation and finalization failures emit a RuntimeWarning and the corresponding
    result is omitted.

    Parameters
    ----------
    data_config : :class:`DataConfig`
        Input signal channels and sampling metadata.
    spectrum_config : :class:`SpectrumConfig`
        Frequency, windowing, precision, and device options.
    requested_spectra : list[tuple[int, ...]] | None
        Specifies which (multi-channel) spectra will be calculated. Each tuple represents one auto-
        or cross-correlation spectrum. Each tuple entry is a channel index which matches the index
        in ``data_config.channels``. Each tuple must contain one through four entries. Duplicate
        tuples and invalid channels are rejected. If ``None``, the auto-spectra of orders
        1 to 4 will be calculated for all available data channels.
    show_progress : bool
        Display a progress bar with elapsed time and an estimated time remaining. Progress is
        measured in spectral estimates, each of which includes reading the required channel data,
        computing Fourier coefficients, and accumulating every requested spectrum. Defaults to
        ``True``.

    Returns
    -------
    SpectrumResultStore
        Finalized spectra indexed by ``channels``.

    Warns
    -----
    RuntimeWarning
        If calculating or finalizing an individual spectrum fails. The failed spectrum is omitted,
        while other requested spectra continue processing. Failures in shared setup, input reading,
        or FFT processing are not isolated and propagate to the caller.
    """

    spectra_channels, active_data_channels = _planning.resolve_channels(
        requested_spectra,
        channel_count=len(data_config.channels),
    )

    with _data_access.open_channels(data_config, active_data_channels) as channels:
        # Resolve user inputs and initialize reusable calculation state.
        runtime = _planning.build_runtime_config(
            data_config=data_config,
            opened_channels=channels,
            spectrum_config=spectrum_config,
            spectra_channels=spectra_channels,
        )
        window_buffer = _fft.prepare_window(runtime)
        third_order_cache = (
            _spectra.build_third_order_cache(runtime) if 3 in runtime.orders else None
        )
        accumulator_store = _accumulation.initialize_accumulator_store(runtime)

        failed_spectra: set[tuple[int, ...]] = set()

        # Each data slice contains estimate_count groups of runtime.m windows and produces that
        # many estimates for every requested spectrum.
        with tqdm(
            total=_planning.window_slice_count(runtime),
            desc="Calculating spectra",
            unit="estimate",
            disable=not show_progress,
        ) as progress:
            for start, end, estimate_count, shifted in _planning.iter_window_slices(runtime):
                coeffs_by_channel = {}

                # Compute Fourier coefficients for each active channel.
                for channel_index in runtime.active_data_channels:
                    data = _data_access.read_channel(channels[channel_index], start, end)
                    chunk = _fft.reshape_window_chunk(data, runtime, estimate_count)
                    chunk = _fft.to_device(chunk, runtime)
                    coeffs_by_channel[channel_index] = _fft.compute_fft(
                        chunk=chunk,
                        window=window_buffer.window,
                        runtime=runtime,
                    )

                intermediate_buffer = _spectra.build_intermediate_slice_buffer(
                    runtime=runtime,
                    coeffs_by_channel=coeffs_by_channel,
                    third_order_cache=third_order_cache,
                )

                # Compute and accumulate every requested spectrum for this data slice.
                for spectrum_channels in runtime.spectra_channels:
                    if spectrum_channels in failed_spectra:
                        continue

                    accumulator = accumulator_store.get(spectrum_channels)

                    # Isolate calculation failures to the affected spectrum so the remaining spectrum
                    # requests can continue.
                    try:
                        spectral_estimates = _spectra.compute_spectral_estimates(
                            channels=spectrum_channels,
                            intermediate_buffer=intermediate_buffer,
                            window_buffer=window_buffer,
                            runtime=runtime,
                        )
                        _accumulation.accumulate_spectral_estimates(
                            accumulator=accumulator,
                            spectral_estimates=spectral_estimates,
                            shifted=shifted,
                        )
                    except Exception as exc: # noqa: BLE001 -- intentional per-spectrum fault boundary
                        failed_spectra.add(spectrum_channels)
                        warnings.warn(
                            f"Calculation failed for spectrum {spectrum_channels}: "
                            f"{type(exc).__name__}: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )

                progress.update(estimate_count)

        # Finalize accumulated spectra and their uncertainty estimates.
        result_store = SpectrumResultStore()
        for accumulator in accumulator_store:
            if accumulator.channels in failed_spectra:
                continue

            # Isolate finalization failures so other completed spectra can still be returned.
            try:
                result = _accumulation.finalize_result(accumulator)
            except Exception as exc: # noqa: BLE001 -- intentional per-spectrum fault boundary
                warnings.warn(
                    f"Could not finalize spectrum for channels {accumulator.channels}: "
                    f"{type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            result_store.add(result)

    return result_store
