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

    Builds the runtime configuration, iterates over windowed signal chunks, computes Fourier
    coefficients, accumulates spectra, and finalizes mean spectra and uncertainty estimates.

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

    # Validate and normalize the channel tuples of the requested spectra.
    resolved_requested_spectra, active_data_channels = _planning.resolve_requested_spectra(
        requested_spectra,
        channel_count=len(data_config.channels),
    )

    # Open all the sources (e.g. load numpy array or open HDF5 file) for the data channels used in
    # the calculations.
    with _data_access.open_channels(data_config, active_data_channels) as channels:
        # Resolve user inputs into runtime calculation settings.
        runtime = _planning.build_runtime_config(
            data_config=data_config,
            opened_channels=channels,
            spectrum_config=spectrum_config,
            spectra_channels=resolved_requested_spectra,
        )

        # Group the channels into sampled and timestamped channels, since they need to undergo
        # different preparation and Fourier transforms.
        sampled_data_channels = tuple(
            channel
            for channel, channel_plan in runtime.channel_plans.items()
            if isinstance(channel_plan, _planning.SampledChannelPlan)
        )
        timestamped_data_channels = tuple(
            channel
            for channel, channel_plan in runtime.channel_plans.items()
            if isinstance(channel_plan, _planning.TimestampedChannelPlan)
        )
        has_sampled_channels = bool(sampled_data_channels)
        has_timestamped_channels = bool(timestamped_data_channels)

        # Sampled channels use FFT coefficients, while timestamped channels are transformed
        # directly, including on an FFT-derived grid in mixed spectra. The
        # CoefficientPreparationPlan objects group spectrum requests by frequency grid and record
        # which channels need output-band or third-order closing coefficients.
        coefficient_preparation_plans_by_type = _planning.build_coefficient_preparation_plans(
            runtime
        )
        fft_coefficient_preparation_plan = coefficient_preparation_plans_by_type.get(
            _planning.FFTFrequencyPlan
        )
        fft_frequency_plan = runtime.fft_frequency_plan
        sampled_third_order_closing_channels = (
            fft_coefficient_preparation_plan.third_order_closing_frequency_channels
            & set(sampled_data_channels)
            if fft_coefficient_preparation_plan is not None
            else set()
        )

        # Prepare reusable window buffers and mappings from output-frequency pairs to third-order
        # closing coefficients. Timestamp cursors retain each source's sequential read position
        # across batches.
        timestamp_cursors: dict[int, _timestamps.TimestampCursor] = {}

        sampled_window_buffer: _fft.WindowBuffer | None = None
        timestamp_window_buffer: _fft.TimestampWindow | None = None
        fft_third_order_cache: _spectra.ThirdOrderIndexCache | None = None
        timestamp_third_order_caches: dict[
            type[_planning.FFTFrequencyPlan | _planning.DirectFrequencyPlan],
            _spectra.TimestampThirdOrderFrequencyCache,
        ] = {}

        if has_sampled_channels:
            if not isinstance(fft_frequency_plan, _planning.FFTFrequencyPlan):
                raise TypeError("Sampled coefficient preparation requires an FFTFrequencyPlan.")

            first_channel = sampled_data_channels[0]
            first_channel_plan = runtime.channel_plans[first_channel]
            assert isinstance(first_channel_plan, _planning.SampledChannelPlan)

            sampled_window_buffer = _fft.prepare_window(
                runtime,
                dt=first_channel_plan.dt,
                window_points=fft_frequency_plan.window_points,
            )
            fft_third_order_cache = (
                _spectra.build_third_order_cache(runtime, fft_frequency_plan)
                if sampled_third_order_closing_channels
                else None
            )

        if has_timestamped_channels:
            timestamp_window_buffer = _fft.prepare_timestamp_window(runtime)
            timestamp_third_order_caches = {
                plan_type: _spectra.build_timestamp_third_order_cache(
                    runtime,
                    preparation_plan.frequency_plan,
                )
                for plan_type, preparation_plan in coefficient_preparation_plans_by_type.items()
                if any(
                    channel in preparation_plan.third_order_closing_frequency_channels
                    for channel in preparation_plan.direct_transform_channels
                )
            }
            timestamp_cursors = {
                channel: _timestamps.TimestampCursor(
                    channels[channel],
                    runtime.window_plan.observation_start,
                )
                for channel in timestamped_data_channels
            }

        normalization_windows = _spectra.build_normalization_windows(
            runtime,
            sampled_window=sampled_window_buffer,
            timestamp_window=timestamp_window_buffer,
        )

        # Initialize storage for accumulating spectral estimates across window batches.
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
            # Process input in batches of physical windows. Sampled coefficients are computed once
            # per batch, while timestamp coefficients are materialized per amplitude-realization
            # batch. The resulting coefficients are reused to compute all requested spectral.
            for batch in _planning.iter_window_batches(plan):
                sampled_coefficients_by_channel: dict[int, _spectra.ChannelCoefficients] | None = (
                    None
                )

                if has_sampled_channels:
                    assert isinstance(fft_frequency_plan, _planning.FFTFrequencyPlan)
                    assert sampled_window_buffer is not None

                    sampled_coefficients_by_channel = {}

                    for channel in sampled_data_channels:
                        channel_plan = runtime.channel_plans[channel]
                        assert isinstance(channel_plan, _planning.SampledChannelPlan)

                        third_order_cache = (
                            fft_third_order_cache
                            if channel in sampled_third_order_closing_channels
                            else None
                        )

                        sampled_coefficients_by_channel[channel] = (
                            _spectra.prepare_sampled_channel_coefficients(
                                channel_index=channel,
                                source=channels[channel],
                                channel_plan=channel_plan,
                                batch=batch,
                                frequency_plan=fft_frequency_plan,
                                window_buffer=sampled_window_buffer,
                                runtime=runtime,
                                third_order_cache=third_order_cache,
                            )
                        )

                prepared_timestamp_channels: dict[int, _timestamps.PreparedTimestampBatch] = {}

                if has_timestamped_channels:
                    prepared_timestamp_channels = {
                        channel: _timestamps.prepare_timestamp_batch(
                            timestamp_cursors[channel],
                            batch,
                        )
                        for channel in timestamped_data_channels
                    }

                realization_sums: dict[tuple[int, ...], torch.Tensor] = {}

                for realization_ids in runtime.repetition_plan.iter_batches():
                    event_amplitudes_by_channel = {}

                    if has_timestamped_channels:
                        for channel in timestamped_data_channels:
                            channel_plan = runtime.channel_plans[channel]
                            assert isinstance(channel_plan, _planning.TimestampedChannelPlan)

                            event_amplitudes_by_channel[channel] = (
                                _timestamps.materialize_timestamp_event_amplitudes(
                                    prepared_timestamp_channels[channel],
                                    channel_index=channel,
                                    channel_plan=channel_plan,
                                    realization_ids=realization_ids,
                                    runtime=runtime,
                                )
                            )

                    coefficients_by_type: dict[
                        type[_planning.FFTFrequencyPlan | _planning.DirectFrequencyPlan],
                        dict[int, _spectra.ChannelCoefficients],
                    ] = {}

                    # Build one channel-to-coefficients mapping for each frequency-plan type.
                    for (
                        plan_type,
                        preparation_plan,
                    ) in coefficient_preparation_plans_by_type.items():
                        frequency_plan = preparation_plan.frequency_plan
                        coefficients_by_channel: dict[
                            int,
                            _spectra.ChannelCoefficients,
                        ] = {}

                        # Sampled coefficients are deterministic, so expand their
                        # realization axis to match the current repetition batch.
                        if (
                            isinstance(frequency_plan, _planning.FFTFrequencyPlan)
                            and sampled_coefficients_by_channel is not None
                        ):
                            expanded_sampled_coefficients = (
                                _spectra.expand_deterministic_coefficients(
                                    sampled_coefficients_by_channel,
                                    realization_count=len(realization_ids),
                                )
                            )
                            coefficients_by_channel.update(expanded_sampled_coefficients)

                        # Timestamp coefficients depend on the current realization IDs.
                        if preparation_plan.direct_transform_channels:
                            if timestamp_window_buffer is None:
                                raise RuntimeError("Timestamp window was not prepared.")

                            for channel in preparation_plan.direct_transform_channels:
                                timestamp_third_order_cache = (
                                    timestamp_third_order_caches.get(plan_type)
                                    if channel
                                    in preparation_plan.third_order_closing_frequency_channels
                                    else None
                                )

                                coefficients_by_channel[channel] = (
                                    _timestamps.materialize_timestamp_coefficients(
                                        prepared_timestamp_channels[channel],
                                        frequency_plan,
                                        timestamp_window_buffer,
                                        runtime,
                                        timestamp_third_order_cache,
                                        event_amplitudes_by_channel[channel],
                                        needs_output=(
                                            channel in preparation_plan.band_coefficient_channels
                                        ),
                                    )
                                )

                        if not coefficients_by_channel:
                            raise RuntimeError(
                                f"No coefficients were prepared for {plan_type.__name__}."
                            )

                        for coefficients in coefficients_by_channel.values():
                            if coefficients.realization_count != len(realization_ids):
                                raise RuntimeError(
                                    "The coefficient realization axis does not "
                                    "match the current repetition batch."
                                )

                        coefficients_by_type[plan_type] = coefficients_by_channel

                    # Calculate every requested spectrum using the mapping belonging
                    # to its frequency-plan type.
                    for spectrum_channels in runtime.requested_spectra:
                        if spectrum_channels in failed_spectra:
                            continue

                        frequency_plan = runtime.frequency_plan_for(spectrum_channels)
                        coefficients_by_channel = coefficients_by_type[type(frequency_plan)]

                        try:
                            normalization_window = normalization_windows[spectrum_channels]

                            spectral_estimates = _spectra.compute_spectral_estimates(
                                channels=spectrum_channels,
                                coefficients_by_channel=coefficients_by_channel,
                                window_buffer=normalization_window,
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
