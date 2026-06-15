# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from typing import Literal

class InvalidConfigError(Exception):
    pass

class PlotConfig:
    VALID_PLOT_FORMAT_SETS = {
        frozenset(['re']),
        frozenset(['im']),
        frozenset(['re', 'im'])
    }

    def __init__(self, f_min, f_max, display_orders=[1, 2, 3, 4], significance=1,
                 arcsinh_scale=(False, 0.02), plot_format=['re', 'im'],
                 insignif_transparency=0.8, output: Literal["show", "save"] = "show", output_path: str = "plot.png"):

        self.display_orders = display_orders
        self.plot_lims = (f_min, f_max)
        self.significance = significance
        self.arcsinh_scale = arcsinh_scale
        self.plot_format = plot_format
        self.insignif_transparency = insignif_transparency
        self.output = output
        self.output_path = output_path

        self.validate()

    def validate(self):
        f_min, f_max = self.plot_lims

        if not isinstance(f_min, (int, float)) or not isinstance(f_max, (int, float)):
            raise InvalidConfigError("f_min and f_max must be numeric values (int or float).")

        if f_min >= f_max:
            raise InvalidConfigError(f"f_min ({f_min}) must be less than f_max ({f_max}).")

        if not isinstance(self.display_orders, list) or not all(
            isinstance(i, int) and 1 <= i <= 4 for i in self.display_orders):
            raise InvalidConfigError("display_orders must be a list of integers between 1 and 4.")

        if not isinstance(self.significance, int) or self.significance <= 0:
            raise InvalidConfigError(f"significance must be a positive int, got: {self.significance}")

        if (not isinstance(self.arcsinh_scale, tuple) or
            len(self.arcsinh_scale) != 2 or
            not isinstance(self.arcsinh_scale[0], bool) or
            not isinstance(self.arcsinh_scale[1], (int, float)) or
            self.arcsinh_scale[1] < 0):
            raise InvalidConfigError(f"arcsinh_scale must be a tuple of (bool, non-negative float), got: {self.arcsinh_scale}")

        if not isinstance(self.plot_format, list):
            raise InvalidConfigError(f"plot_format must be a list, got {type(self.plot_format).__name__}")

        fmt_set = frozenset(self.plot_format)
        if fmt_set not in self.VALID_PLOT_FORMAT_SETS:
            raise InvalidConfigError(f"plot_format must be one of ['re'], ['im'], or ['re', 'im']. Got: {self.plot_format}")

        if not isinstance(self.insignif_transparency, (int, float)) or not (0 <= self.insignif_transparency <= 1):
            raise InvalidConfigError(f"insignif_transparency must be a float between 0 and 1, got: {self.insignif_transparency}")