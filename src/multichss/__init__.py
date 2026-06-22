# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

import warnings

from .MultiChSS_SpectrumCalculator import SpectrumCalculator
from .MultiChSS_SpectrumPlotter import SpectrumPlotter

from .configurators import CrossConfig, DataConfig, PlotConfig, SpectrumConfig
from .planning import RuntimeConfig, build_runtime_config
from .utils import data_config_dic

__all__ = [
    "CrossConfig",
    "DataConfig",
    "PlotConfig",
    "SpectrumConfig",
    "RuntimeConfig",
    "SpectrumCalculator",
    "SpectrumPlotter",
    "build_runtime_config",
    "data_config_dic"
]

def __getattr__(name):
    if name == "DataImportConfig":
        warnings.warn(
            "DataImportConfig is deprecated and reworked into DataConfig. Please update your code.",
            DeprecationWarning,
            stacklevel=2
        )
        return DataConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
