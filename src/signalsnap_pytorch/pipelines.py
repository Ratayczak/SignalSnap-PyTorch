# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemad, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import warnings

import torch
from tqdm.auto import tqdm

from ._core import accumulation as _accumulation
from ._core import data_access as _data_access
from ._core import fft as _fft
from ._core import planning as _planning
from ._core import spectra as _spectra
from ._core import timestamps as _timestamps
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
        has_sampled_channels = any(
            isinstance(channel_plan, _planning.SampledChannelPlan)
            for channel_plan in runtime.channel_plans.values()
        )
        has_timestamped_channels = any(
            isinstance(channel_plan, _planning.TimestampedChannelPlan)
            for channel_plan in runtime.channel_plans.values()
        )

        if has_sampled_channels and has_timestamped_channels:
            raise NotImplementedError("Mixed sampled/timestamped execution is not enabled yet.")

        selected_frequency_plans = tuple(runtime.spectrum_frequency_plans.values())
        common_frequency_plan = selected_frequency_plans[0]

        if any(plan is not common_frequency_plan for plan in selected_frequency_plans[1:]):
            raise NotImplementedError(
                "Multiple spectrum frequency views are not connected to "
                "coefficient preparation yet."
            )

        has_third_order = any(len(channels) == 3 for channels in runtime.spectrum_frequency_plans)
        timestamp_cursors: dict[int, _timestamps.TimestampCursor] = {}

        if has_timestamped_channels:
            if not isinstance(common_frequency_plan, _planning.TimestampFrequencyPlan):
                raise TypeError(
                    "Timestamp coefficient preparation requires a TimestampFrequencyPlan."
                )

            window_buffer = _fft.prepare_timestamp_window(runtime)
            third_order_cache = (
                _spectra.build_timestamp_third_order_cache(runtime, common_frequency_plan)
                if has_third_order
                else None
            )
            timestamp_cursors = {
                channel: _timestamps.TimestampCursor(
                    channels[channel],
                    runtime.window_plan.observation_start,
                )
                for channel in runtime.active_data_channels
            }
        else:
            if not isinstance(common_frequency_plan, _planning.SampledFrequencyPlan):
                raise TypeError("Sampled coefficient preparation requires a SampledFrequencyPlan.")

            first_channel = runtime.active_data_channels[0]
            first_channel_plan = runtime.channel_plans[first_channel]
            assert isinstance(first_channel_plan, _planning.SampledChannelPlan)

            window_buffer = _fft.prepare_window(
                runtime,
                dt=first_channel_plan.dt,
                window_points=common_frequency_plan.window_points,
            )
            third_order_cache = (
                _spectra.build_third_order_cache(runtime, common_frequency_plan)
                if has_third_order
                else None
            )
        accumulator_store = _accumulation.initialize_accumulator_store(runtime)

        failed_spectra: set[tuple[int, ...]] = set()

        # Each window batch contains estimate_count groups of plan.windows_per_estimate windows and
        # produces that many estimates for every requested spectrum.
        plan = runtime.window_plan

        with tqdm(
            total=_planning.physical_estimate_count(plan),
            desc="Calculating spectra",
            unit="estimate",
            disable=not show_progress,
        ) as progress:
            for batch in _planning.iter_window_batches(plan):
                prepared_timestamp_channels: dict[int, _timestamps.PreparedTimestampBatch] = {}
                coefficient_batch: _spectra.CoefficientBatch | None = None

                if has_timestamped_channels:
                    assert isinstance(common_frequency_plan, _planning.TimestampFrequencyPlan)
                    assert isinstance(
                        window_buffer,
                        (_fft.DefaultTimestampWindow, _fft.LegacyTimestampWindow),
                    )
                    assert third_order_cache is None or isinstance(
                        third_order_cache,
                        _spectra.TimestampThirdOrderFrequencyCache,
                    )

                    prepared_timestamp_channels = {
                        channel: _timestamps.prepare_timestamp_batch(
                            timestamp_cursors[channel],
                            batch,
                        )
                        for channel in runtime.active_data_channels
                    }
                else:
                    assert isinstance(common_frequency_plan, _planning.SampledFrequencyPlan)
                    assert isinstance(window_buffer, _fft.WindowBuffer)
                    assert third_order_cache is None or isinstance(
                        third_order_cache, _spectra.ThirdOrderIndexCache
                    )
                    coeffs_by_channel = {}

                    for channel_index in runtime.active_data_channels:
                        channel_plan = runtime.channel_plans[channel_index]
                        assert isinstance(channel_plan, _planning.SampledChannelPlan)

                        channel_window_points = round(batch.duration / channel_plan.dt)
                        start = round(float(batch.relative_starts[0, 0]) / channel_plan.dt)
                        end = (
                            start
                            + batch.estimate_count
                            * runtime.window_plan.windows_per_estimate
                            * channel_window_points
                        )
                        data = _data_access.read_source(channels[channel_index], start, end)
                        chunk = _fft.reshape_window_chunk(
                            chunk=data,
                            estimate_count=batch.estimate_count,
                            windows_per_estimate=plan.windows_per_estimate,
                            window_points=channel_window_points,
                        )
                        chunk = _fft.to_device(chunk, runtime)
                        coeffs_by_channel[channel_index] = _fft.compute_fft(
                            chunk=chunk,
                            window=window_buffer.window,
                            dt=channel_plan.dt,
                        )

                    coefficient_batch = _spectra.build_coefficient_batch(
                        frequency_plan=common_frequency_plan,
                        coeffs_by_channel=coeffs_by_channel,
                        third_order_cache=third_order_cache,
                    )
                    del coeffs_by_channel

                realization_sums: dict[tuple[int, ...], torch.Tensor] = {}

                for realization_ids in runtime.repetition_plan.iter_batches():
                    if has_timestamped_channels:
                        assert isinstance(common_frequency_plan, _planning.TimestampFrequencyPlan)
                        assert isinstance(
                            window_buffer,
                            (_fft.DefaultTimestampWindow, _fft.LegacyTimestampWindow),
                        )
                        assert third_order_cache is None or isinstance(
                            third_order_cache,
                            _spectra.TimestampThirdOrderFrequencyCache,
                        )
                        timestamp_coefficients = {}

                        for channel in runtime.active_data_channels:
                            channel_plan = runtime.channel_plans[channel]
                            assert isinstance(channel_plan, _planning.TimestampedChannelPlan)

                            timestamp_coefficients[channel] = (
                                _timestamps.materialize_timestamp_channel_coefficients(
                                    prepared_timestamp_channels[channel],
                                    channel_index=channel,
                                    channel_plan=channel_plan,
                                    realization_ids=realization_ids,
                                    frequency_plan=common_frequency_plan,
                                    timestamp_window=window_buffer,
                                    runtime=runtime,
                                    third_order_cache=third_order_cache,
                                )
                            )

                        coefficient_batch = _spectra.CoefficientBatch(
                            by_channel=timestamp_coefficients,
                        )

                    if coefficient_batch is None:
                        raise RuntimeError("No channel coefficients were prepared for this batch.")

                    realization_count = next(iter(coefficient_batch.by_channel.values())).dc.shape[
                        0
                    ]

                    if realization_count != len(realization_ids):
                        raise RuntimeError(
                            "The coefficient realization axis does not match the current "
                            "repetition batch."
                        )

                    for spectrum_channels in runtime.spectrum_frequency_plans:
                        if spectrum_channels in failed_spectra:
                            continue

                        try:
                            spectral_estimates = _spectra.compute_spectral_estimates(
                                channels=spectrum_channels,
                                coefficient_batch=coefficient_batch,
                                window_buffer=window_buffer,
                                runtime=runtime,
                            )
                            chunk_sum = spectral_estimates.sum(dim=0)

                            if spectrum_channels in realization_sums:
                                realization_sums[spectrum_channels] += chunk_sum
                            else:
                                realization_sums[spectrum_channels] = chunk_sum
                        except Exception as exc:  # noqa: BLE001
                            failed_spectra.add(spectrum_channels)
                            realization_sums.pop(spectrum_channels, None)
                            warnings.warn(
                                f"Calculation failed for spectrum {spectrum_channels}: "
                                f"{type(exc).__name__}: {exc}",
                                RuntimeWarning,
                                stacklevel=2,
                            )

                for spectrum_channels, realization_sum in realization_sums.items():
                    if spectrum_channels in failed_spectra:
                        continue

                    accumulator = accumulator_store.get(spectrum_channels)

                    try:
                        spectral_estimates = realization_sum / runtime.repetition_plan.count
                        _accumulation.accumulate_spectral_estimates(
                            accumulator=accumulator,
                            spectral_estimates=spectral_estimates,
                            shifted=batch.shifted,
                        )
                    except Exception as exc:  # noqa: BLE001
                        failed_spectra.add(spectrum_channels)
                        warnings.warn(
                            f"Calculation failed for spectrum {spectrum_channels}: "
                            f"{type(exc).__name__}: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )

                progress.update(batch.estimate_count)

        # Finalize accumulated spectra and their uncertainty estimates.
        result_store = SpectrumResultStore()
        for accumulator in accumulator_store:
            if accumulator.channels in failed_spectra:
                continue

            # Isolate finalization failures so other completed spectra can still be returned.
            try:
                result = _accumulation.finalize_result(accumulator)
            except Exception as exc:  # noqa: BLE001 -- intentional per-spectrum fault boundary
                warnings.warn(
                    f"Could not finalize spectrum for channels {accumulator.channels}: "
                    f"{type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            result_store.add(result)

    return result_store
