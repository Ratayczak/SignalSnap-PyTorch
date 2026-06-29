# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from .aggregator import accumulate_spectrum, finalize_result
from .configurators import CrossConfig, DataConfig, SpectrumConfig
from .fft import (
    compute_fft,
    iter_window_slices,
    prepare_windows,
    reshape_window_chunk,
    to_device,
)
from .planning import (
    build_runtime_config,
    build_spectrum_tasks,
    initialize_result_store,
)
from .spectra import build_third_order_cache, compute_single_spectrum


def calculate_spectra(
    spectrum_config: SpectrumConfig,
    cross_config: CrossConfig,
    data_config_list: list[DataConfig],
    selected: list[int] | None = None,
):
    runtime_config = build_runtime_config(
        spectrum_config=spectrum_config,
        data_config_list=data_config_list,
        selected=selected,
    )
    tasks = build_spectrum_tasks(runtime_config, cross_config)
    result_store = initialize_result_store(tasks, runtime_config)
    single_window, repeated_window = prepare_windows(runtime_config)
    third_order_cache = (
        build_third_order_cache(runtime_config)
        if 3 in runtime_config.orders
        else None
    )

    for chunk_index, (start, end) in enumerate(iter_window_slices(runtime_config)):
        if (
            runtime_config.spectral_estimates_max is not None
            and chunk_index >= runtime_config.spectral_estimates_max
        ):
            break

        coeffs_by_channel = {}

        for channel in runtime_config.selected_channels:
            data = data_config_list[channel].data[start:end]
            chunk = reshape_window_chunk(data, runtime_config)
            chunk = to_device(chunk, runtime_config)
            coeffs_by_channel[channel] = compute_fft(
                chunk,
                repeated_window,
                runtime_config,
            )

        for task in tasks:
            spectrum = compute_single_spectrum(
                task=task,
                coeffs_by_channel=coeffs_by_channel,
                single_window=single_window,
                runtime=runtime_config,
                third_order_cache=third_order_cache,
            )
            result = result_store.get(task.channels, task.order)
            accumulate_spectrum(result, spectrum, runtime_config)

    for result in result_store.results.values():
        finalize_result(result)

    return result_store
