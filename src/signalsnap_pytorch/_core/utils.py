# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemad, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from typing import Literal

type TimeUnits = Literal["s", "ms", "us", "ns", "ps"]
type FrequencyUnits = Literal["Hz", "kHz", "MHz", "GHz", "THz"]
type PlotComponent = Literal["re", "im"]


def unit_conversion_time_to_freq(t_unit: TimeUnits) -> FrequencyUnits:
    """Return the frequency unit corresponding to a time-step unit.

    Parameters
    ----------
    t_unit : Literal["s", "ms", "us", "ns", "ps"]
        Unit used for the sampling interval.

    Returns
    -------
    Literal["Hz", "kHz", "MHz", "GHz", "THz"]
        Reciprocal frequency unit with the matching SI prefix.

    Raises
    ------
    ValueError
        If ``t_unit`` is unsupported at runtime.
    """
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
