# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from typing import Any, Iterable

from multichss.configurators import DataConfig


def data_config_dic(
    data_config_list: Iterable["DataConfig"],
) -> dict[Any, "DataConfig"]:
    return {config.data: config for config in data_config_list}


def unit_conversion_freq_to_time(f_unit: str) -> str:
    mapping = {"Hz": "s", "kHz": "ms", "MHz": "us", "GHz": "ns", "THz": "ps"}

    try:
        return mapping[f_unit]
    except KeyError:
        raise ValueError(f"Unknown frequency unit: {f_unit}")
