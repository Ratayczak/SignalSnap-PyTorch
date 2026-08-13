# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemad, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from .config import (
    DataConfig,
    HDF5Source,
    SampledChannel,
    SpectrumConfig,
    TimestampedChannel,
    TimestampOptions,
)
from .metadata import CalculationMetadata, SpectrumMetadata
from .pipelines import calculate_spectra
from .results import SpectrumResult, SpectrumResultStore

__all__ = [
    "CalculationMetadata",
    "DataConfig",
    "HDF5Source",
    "SampledChannel",
    "SpectrumConfig",
    "SpectrumMetadata",
    "SpectrumResult",
    "SpectrumResultStore",
    "TimestampOptions",
    "TimestampedChannel",
    "calculate_spectra",
]
