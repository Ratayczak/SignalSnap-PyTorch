# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from typing import Any, Dict, Iterable, List, Optional

class InvalidConfigError(Exception):
    """Raised when configuration is invalid."""
    pass

class SpectrumConfig:
    VALID_FREQ_UNITS: set[str] = {'Hz', 'kHz', 'MHz', 'GHz', 'THz'}
    VALID_S3_CALC: set[str] = {'1/4', '1/2'}
    VALID_BACKENDS: set[str] = {'cpu', # cpu
                                'mps', # Apple metal
                                'cuda' # Nvidia
                                }

    def __init__(
        self,
        dt: float,
        f_unit: str = 'Hz',
        f_max: Optional[float] = None,
        f_min: float = 0.0,
        s3_calc: str = '1/4', #TODO s3_calc can be either '1/4' or '1/2' for now. I will add '1' to it too. 
        f_lists: Optional[Any] = None,
        backend: str = 'mps',
        spectrum_size: int = 100,
        order_in: str | List[int] = 'all',
        m: int = 10,
        m_var: int = 10,
        show_first_frame: bool = True,
        break_after: int = int(1e6),
    ) -> None:

        self.dt = dt
        self.f_unit = f_unit
        self.f_max = f_max
        self.f_min = f_min
        self.s3_calc = s3_calc
        self.f_lists = f_lists
        self.backend = backend
        self.spectrum_size = spectrum_size
        self.order_in = order_in
        self.m = m
        self.m_var = m_var
        self.show_first_frame = show_first_frame
        self.break_after = break_after

        self.validate()

    def validate(self) -> None:
        if not isinstance(self.dt, float) or self.dt <= 0:
            raise InvalidConfigError("'dt' must be positive float.")

        if self.f_unit not in self.VALID_FREQ_UNITS:
            raise InvalidConfigError(
                f"Invalid frequency unit '{self.f_unit}'. Valid: {self.VALID_FREQ_UNITS}"
            )

        if self.s3_calc not in self.VALID_S3_CALC:
            raise InvalidConfigError(
                f"Invalid s3_calc '{self.s3_calc}'. Valid: {self.VALID_S3_CALC}"
            )

        if not isinstance(self.f_min, (float, int)):
            raise InvalidConfigError("'f_min' must be float or int.")

        if self.f_max is not None:
            if not isinstance(self.f_max, (float, int)):
                raise InvalidConfigError("'f_max' must be float or int.")
            if self.f_min >= self.f_max:
                raise InvalidConfigError("'f_min' must be < 'f_max'.")

        if self.backend not in self.VALID_BACKENDS:
            raise InvalidConfigError(
                f"Invalid backend '{self.backend}'. Valid: {self.VALID_BACKENDS}"
            )

        if self.order_in != 'all':
            if not isinstance(self.order_in, list) or \
               not all(isinstance(i, int) and 1 <= i <= 4 for i in self.order_in):
                raise ValueError(
                    "order_in must be 'all' or list of integers 1 - 4."
                )

        if not isinstance(self.show_first_frame, bool):
            raise InvalidConfigError("'show_first_frame' must be bool.")

        if not isinstance(self.m, int) or self.m <= 0:
            raise InvalidConfigError("'m' must be positive int.")

        if not isinstance(self.m_var, int) or self.m_var <= 0:
            raise InvalidConfigError("'m_var' must be positive int.")

        if not isinstance(self.spectrum_size, int) or self.spectrum_size <= 0:
            raise InvalidConfigError("'spectrum_size' must be positive int.")


class DataImportConfig:
    def __init__(
        self,
        data: Optional[Any] = None,
        path: Optional[str] = None,
        group_key: Optional[str] = None,
        dataset: Optional[str] = None,
        dt: Optional[float] = None
    ) -> None:
        self.data = data
        self.path = path
        self.group_key = group_key
        self.dataset = dataset
        self.dt = dt

    @staticmethod
    def data_config_dic(
        data_config_list: Iterable["DataImportConfig"]
    ) -> Dict[Any, "DataImportConfig"]:
        return {config.data: config for config in data_config_list}