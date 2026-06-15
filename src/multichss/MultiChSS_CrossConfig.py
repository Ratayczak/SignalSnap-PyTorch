# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

import os

from pydantic import BaseModel, ConfigDict


os.environ['PYDANTIC_ERRORS_INCLUDE_URL'] = '0'

class CrossConfig(BaseModel):
    model_config = ConfigDict(strict=True)

    auto_corr: bool = True
    cross_corr_2: list[tuple[int, int]] | None = None
    cross_corr_3: list[tuple[int, int, int]] | None = None
    cross_corr_4: list[tuple[int, int, int, int]] | None = None