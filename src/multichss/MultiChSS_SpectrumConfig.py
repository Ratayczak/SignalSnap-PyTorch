# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

class InvalidConfigError(Exception):
    """Raised when configuration is invalid."""
    pass

class SpectrumConfig:
    VALID_FREQ_UNITS = {'Hz', 'kHz', 'MHz', 'GHz', 'THz'}

    VALID_S3_CALC = {'1/4', '1/2'}

    VALID_BACKENDS = {'cpu', # cpu
                      'mps', # apple silicon metal gpu
                      'cuda' # nvidia gpus
                     }


    def __init__(self, dt, f_unit='Hz', f_max=None, f_min=0, s3_calc='1/4', f_lists=None,
                 backend='mps', spectrum_size=100, order_in='all',
                 m=10, m_var=10, show_first_frame=True, break_after=int(1e6)):

        self.dt = dt
        self.f_unit = f_unit
        self.f_max = f_max
        self.f_min = f_min
        self.s3_calc = s3_calc #TODO s3_calc can be either '1/4' or '1/2' for now. I will add '1' to it too. 
        self.f_lists = f_lists
        self.backend = backend
        self.spectrum_size = spectrum_size
        self.order_in = order_in
        self.m = m
        self.m_var = m_var
        self.show_first_frame = show_first_frame
        self.break_after = break_after

        self.validate() # validating data rightly after creating the object. 

    def validate(self):
        """
        Helper function to validate user input. 
        """
        if not isinstance(self.dt, (float)) or self.dt <= 0:
            raise InvalidConfigError(f"Invalid 'dt': {self.dt}.\n"
                                     f"Must be float and positive.")

        if self.f_unit not in self.VALID_FREQ_UNITS:
            raise InvalidConfigError(f"Invalid frequency unit '{self.f_unit}.\n'"
                                     f"Valid options: {self.VALID_FREQ_UNITS}")

        if self.s3_calc not in self.VALID_S3_CALC:
            raise InvalidConfigError(f"Invalid s3_calc '{self.s3_calc}'.\n"
                                     f"Valid options: {self.VALID_S3_CALC}")

        if not isinstance(self.f_min, (float, int)):
            raise InvalidConfigError(f"Invalid 'f_min': {self.f_min}.\n"
                                     f"Must be either float or int.")

        if self.f_max is not None:
            if not isinstance(self.f_max, (float, int)):
                raise InvalidConfigError(f"Invalid 'f_max': {self.f_max}.\n"
                                         f"Must be either float or int.")
            if self.f_min >= self.f_max:
                raise InvalidConfigError(f"'f_min' ({self.f_min}) must be less than 'f_max' ({self.f_max})")

        if self.backend not in self.VALID_BACKENDS:
            raise InvalidConfigError(f"Invalid frequency unit '{self.f_unit}.\n'"
                                     f"Valid options: {self.VALID_BACKENDS}.")

        if self.order_in != 'all' and (not isinstance(self.order_in, list) or not all(1 <= i <= 4 for i in self.order_in)):
            raise ValueError("order_in must be 'all' or a list containing one or more numbers between 1 and 4.")

        if not isinstance(self.show_first_frame, (bool)):
            raise InvalidConfigError(f"Invalid 'show_first_frame': {self.show_first_frame}.\n"
                                     f"Must be boolian.")

        if not isinstance(self.m, (int)) or self.m <= 0:
            raise InvalidConfigError(f"Invalid 'm': {self.m}.\n"
                                     f"Must be a positive int.")

        if not isinstance(self.m_var, (int)) or self.m_var <= 0:
            raise InvalidConfigError(f"Invalid 'm_var': {self.m_var}.\n"
                                     f"Must be a positive int.")

        if not isinstance(self.spectrum_size, int) or self.spectrum_size <= 0:
            raise InvalidConfigError(f"Invalid 'spectrum_size': {self.spectrum_size}.\n"
                                     f"Must be a positive int.")

class DataImportConfig:
    def __init__(self, data=None, path=None, group_key=None, dataset=None, dt=None):
        """
        data can be imported in two ways. 
        First way is to give the data directly. if the array is in the script itself
        Second way is to import the hdf data by giving
        path
        group key
        data set
        and the dt in the file
        """
        self.data = data
        self.path = path
        self.group_key = group_key
        self.dataset = dataset
        self.dt = dt

    @staticmethod
    def data_config_dic(data_config_list):
        return {config.data: config for config in data_config_list}