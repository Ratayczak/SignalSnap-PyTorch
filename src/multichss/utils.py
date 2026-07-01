# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Literal, TypeAlias

if TYPE_CHECKING:
    from multichss.configurators import DataConfig

TimeUnits: TypeAlias = Literal["s", "ms", "us", "ns", "ps"]
FrequencyUnits: TypeAlias = Literal["Hz", "kHz", "MHz", "GHz", "THz"]
S3Calcs: TypeAlias = Literal["1/4", "1/2"]


def data_config_dic(data_config_list: Iterable["DataConfig"]) -> dict[Any, "DataConfig"]:
    """Return a lookup mapping each data object to its DataConfig."""
    return {config.data: config for config in data_config_list}


def unit_conversion_time_to_freq(t_unit: TimeUnits) -> FrequencyUnits:
    """Return the frequency unit corresponding to a time-step unit."""
    mapping: dict[TimeUnits, FrequencyUnits] = {
        "s": "Hz",
        "ms": "kHz",
        "us": "MHz",
        "ns": "GHz",
        "ps": "THz",
    }

    try:
        return mapping[t_unit]
    except KeyError:
        raise ValueError(f"Unknown time unit: {t_unit}")
