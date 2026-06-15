# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

class InvalidConfigError(Exception):
    pass

class CrossConfig:
    def __init__(self, auto_corr: bool = True, cross_corr_2=None, cross_corr_3=None, cross_corr_4=None):
        self.auto_corr = auto_corr
        self.cross_corr_2 = cross_corr_2
        self.cross_corr_3 = cross_corr_3
        self.cross_corr_4 = cross_corr_4

        self.validate()

    def validate(self):
        if not isinstance(self.auto_corr, (bool)):
            raise InvalidConfigError(f"Invalid 'auto_corr': {self.auto_corr}.\n"
                                     f"Must be boolean.")
