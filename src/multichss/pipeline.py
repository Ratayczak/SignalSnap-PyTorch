# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from .planning import (
    build_runtime_config,
    build_spectrum_tasks,
    initialize_result_store,
)
from .configurators import SpectrumConfig, CrossConfig, DataConfig


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
    tasks = build_spectrum_tasks(
        runtime_config,
        cross_config
    )
    result_store = initialize_result_store(tasks)
    return runtime_config, tasks, result_store
