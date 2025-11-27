# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from typing import Optional, List, Tuple

class InvalidConfigError(Exception):
    pass

class CrossConfig:
    auto_corr: bool
    cross_corr_2: Optional[List[Tuple[int, int]]]
    cross_corr_3: Optional[List[Tuple[int, int, int]]]
    cross_corr_4: Optional[List[Tuple[int, int, int, int]]]

    def __init__(
        self,
        auto_corr: bool = True,
        cross_corr_2: Optional[List[Tuple[int, int]]] = None,
        cross_corr_3: Optional[List[Tuple[int, int, int]]] = None,
        cross_corr_4: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> None:

        self.auto_corr = auto_corr
        self.cross_corr_2 = cross_corr_2
        self.cross_corr_3 = cross_corr_3
        self.cross_corr_4 = cross_corr_4

        self.validate()

    def validate(self) -> None:
        if not isinstance(self.auto_corr, bool):
            raise InvalidConfigError(
                f"Invalid 'auto_corr': {self.auto_corr}. Must be boolean."
            )

        # validate cross_corr_2
        if self.cross_corr_2 is not None:
            if not (
                isinstance(self.cross_corr_2, list)
                and all(
                    isinstance(t, tuple)
                    and len(t) == 2
                    and all(isinstance(i, int) for i in t)
                    for t in self.cross_corr_2
                )
            ):
                raise InvalidConfigError(
                    "'cross_corr_2' must be list of 2-tuples of ints"
                )

        # validate cross_corr_3
        if self.cross_corr_3 is not None:
            if not (
                isinstance(self.cross_corr_3, list)
                and all(
                    isinstance(t, tuple)
                    and len(t) == 3
                    and all(isinstance(i, int) for i in t)
                    for t in self.cross_corr_3
                )
            ):
                raise InvalidConfigError(
                    "'cross_corr_3' must be list of 3-tuples of ints"
                )

        # validate cross_corr_4
        if self.cross_corr_4 is not None:
            if not (
                isinstance(self.cross_corr_4, list)
                and all(
                    isinstance(t, tuple)
                    and len(t) == 4
                    and all(isinstance(i, int) for i in t)
                    for t in self.cross_corr_4
                )
            ):
                raise InvalidConfigError(
                    "'cross_corr_4' must be list of 4-tuples of ints"
                )
