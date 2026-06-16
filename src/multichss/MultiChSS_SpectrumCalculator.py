# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

import h5py
import numpy as np
import pickle
from tqdm.auto import tqdm
import torch
from numba import njit
from scipy.fft import rfftfreq
import pandas as pd
from tabulate import tabulate
from torch import Tensor
from typing import Dict, Optional, Tuple

from multichss.config import CrossConfig, SpectrumConfig, DataConfig

def load_spec(path):
    f = open(path, mode='rb')
    obj = pickle.load(f)
    f.close()
    return obj

def to_hdf(dt, data, path, group_name, dataset_name):
    with h5py.File(path, "w") as f:
        grp = f.create_group(group_name)
        d = grp.create_dataset(dataset_name, data=data)
        d.attrs['dt'] = dt

def unit_conversion(f_unit: str) -> str:

    mapping = {
        'Hz':  's',
        'kHz': 'ms',
        'MHz': 'us',
        'GHz': 'ns',
        'THz': 'ps',
    }

    try:
        return mapping[f_unit]
    except KeyError:
        raise ValueError(f'Unknown frequency unit: {f_unit}')

def gaussian_window(x: Tensor,
                    n_windows: int,
                    l: int,
                    sigma_t: float
) -> Tensor:
    """
    Approx. confined Gaussian window (see DOI:10.1016/j.sigpro.2014.03.033)
    """

    center = n_windows * 0.5
    denom  = 2.0 * l * sigma_t

    t = (x - center) / denom
    return torch.exp(-t * t)

def calc_window(x: Tensor, 
                n_windows: int, 
                l: int, 
                sigma_t: float
) -> Tensor:
    """
    Helper function to calculate the approx. confined gaussian window
    as defined in https://doi.org/10.1016/j.sigpro.2014.03.033
    """
    
    h: Tensor = x.new_tensor(-0.5)

    term_x = gaussian_window(x, n_windows, l, sigma_t)
    term_h = gaussian_window(h, n_windows, l, sigma_t)
    term_x_p_l = gaussian_window(x + l, n_windows, l, sigma_t)
    term_x_m_l = gaussian_window(x - l, n_windows, l, sigma_t)
    term_h_p_l = gaussian_window(h + l, n_windows, l, sigma_t)
    term_h_m_l = gaussian_window(h - l, n_windows, l, sigma_t)

    denom = term_h_p_l + term_h_m_l
    win = term_x - (term_h * (term_x_p_l + term_x_m_l)) / denom

    return win

def cg_window(n_windows: int, fs: float) -> Tuple[Tensor, float]:
    """
    Helper function to calculate the approx. confined gaussian window
    as defined in https://doi.org/10.1016/j.sigpro.2014.03.033
    """

    x = torch.linspace(0, n_windows, n_windows)
    l = n_windows + 1
    sigma_t = 0.14

    window = calc_window(x, n_windows, l, sigma_t)
    norm_t = (window * window).sum() / fs
    
    window_full = window / torch.sqrt(norm_t)
    norm = float(norm_t.item())

    return window_full, norm

# ------------------------------------------------------------------

class SpectrumCalculator:
    """
    Handling the calculation of polyspectra and preparing them for
    saving, visulization or any other assessment. The data and
    configuration of the calculation are set in SpectrumConfig, CrossConfig.
    These configurations are passed here for final calculation.

    Attributes
    ----------
    sconfig: Obj
        configuration of the parameters needed to calculate higher order spectra
        generally
    cconfing: Obj
        configuration of the parameters needed to calculate cross higher order
        spectra
    diconfig_list: list
        list of the data
    selected: list
        list of indices of the data to be selected for further calculation and
        evaluation
    """
    def __init__(self, sconfig: SpectrumConfig, cconfig: CrossConfig, 
                 diconfig_list: list[DataConfig], selected=None):
        self.sconfig = sconfig
        self.cconfig = cconfig
        self.diconfig_list = diconfig_list
        self.selected = selected if selected is not None else list(range(len(diconfig_list)))
        
        self.cross2_selected = (self.cconfig.cross_corr_2 
                                if hasattr(self.cconfig, 'cross_corr_2') and isinstance(self.cconfig.cross_corr_2, list)
                                else [])
        if self.sconfig.f_min < 0:
            self.cross3_selected = []
        else:
            if hasattr(self.cconfig, 'cross_corr_3') and isinstance(self.cconfig.cross_corr_3, list):
                self.cross3_selected = self.cconfig.cross_corr_3
            else:
                self.cross3_selected = []
        self.cross4_selected = (self.cconfig.cross_corr_4 
                                if hasattr(self.cconfig, 'cross_corr_4') and isinstance(self.cconfig.cross_corr_4, list)
                                else [])

        self.device = torch.device(self.sconfig.backend)
        self.t_unit = unit_conversion(sconfig.f_unit)
        self.fs = 1 / self.sconfig.dt

        self.f_max_allowed = 1 / (2 * self.sconfig.dt)
        if self.sconfig.f_max is None:
            # even if a maximum frequency is not given, the program does not crash
            # using Nyquist frequency which is half the sampling rate of a 
            # discrete system
            self.sconfig.f_max = self.f_max_allowed
        window_len_factor = self.f_max_allowed / (self.sconfig.f_max - self.sconfig.f_min)
        self.t_window = (self.sconfig.spectrum_size - 1) * (2 * self.sconfig.dt * window_len_factor)
        self.window_points = int(np.round(self.t_window / self.sconfig.dt))
        
        # Initialize various dictionaries in one go
        self._init_dicts()
        
        self.import_data()
        self.validate_shapes() # Crash if data shapes mismatch

        # Flag to use full FFT (for negative frequencies)
        self.use_full_fft = (self.sconfig.f_min < 0)

        # MPS backend precision support
        if self.sconfig.backend == 'mps':
            self.use_float32 = True  
        else:
            self.use_float32 = False

    def _init_dicts(self):
        """
        Initialize all bookkeeping dictionaries for selected and cross datasets.
        """

        # Map each group of keys to the orders they support
        order_map = {
            tuple(self.selected): [1, 2, 3, 4],
            tuple(self.cross2_selected): [2],
            tuple(self.cross3_selected): [3],
            tuple(self.cross4_selected): [4]
        }

        self.n_chunks = {}
        for keys in order_map:
            for key in keys:
                self.n_chunks[key] = 0

        self.m = {o: None for o in [1, 2, 3, 4]}
        self.m_var = {o: None for o in [1, 2, 3, 4]}

        self.freq = {}
        for keys, orders in order_map.items():
            for key in keys:
                self.freq[key] = {o: None for o in orders}

        self.s = {}
        self.s_gpu = {}
        self.s_err = {}
        self.s_err_gpu = {}
        self.s_errs = {}
        self.err_counter = {}
        self.n_error_estimates = {}

        for keys, orders in order_map.items():
            for key in keys:
                self.s[key] = {o: None for o in orders}
                self.s_gpu[key] = {o: None for o in orders}
                self.s_err[key] = {o: None for o in orders}
                self.s_err_gpu[key] = {o: None for o in orders}

                self.s_errs[key] = {o: [] for o in orders}
                self.err_counter[key] = {o: 0 for o in orders}
                self.n_error_estimates[key] = {o: 0 for o in orders}


    def validate_shapes(self):
        """
        helper function to make sure that data are the same length
        """
        expected_shape = self.diconfig_list[self.selected[0]].data.shape[0]
        for data_config in self.diconfig_list:
            if data_config.data.shape[0] != expected_shape:
                raise ValueError('Imported data must have same length!')

    def import_data(self):
        """
        cheks if the data is being imported from the script or externally.
        If the data is external the data_config.data must be None
        and the path to this external file must be given to data_config.path
        """
        for data_config in self.diconfig_list:
            if data_config.data is None and data_config.path is not None:
                with h5py.File(data_config.path, 'r') as main:
                    if not data_config.group_key:
                        main_data = main[data_config.dataset]
                    else:
                        main_data = main[data_config.group_key][data_config.dataset]
                    if data_config.dt is None:
                        data_config.dt = main_data.attrs.get('dt', None)
                    data_config.data = main_data[()]
                    print(f"Data loaded from {data_config.path}")

    def c1(self, a_w: Tensor) -> Tensor:
        """
        First order cumulant
        """
        s1 = torch.mean(a_w, dim=0)
        if self.use_full_fft:
            dc_index = s1.shape[0] // 2
            result = s1[dc_index]
        else:
            result = s1[0]

        return result
            

    def c2(self, 
           a_w1: Tensor, 
           a_w2: Tensor
    ) -> Tensor:
        """
        second order cumulant is the covariance
        """
        a_w2_star = torch.conj(a_w2)
        term_1 = torch.mean(a_w1 * a_w2_star, dim=0)
        
        # The number of windows `m` used for the calculation is obtained from `self.config.m`
        factor = self.sconfig.m / (self.sconfig.m - 1)
        
        term_2 = torch.mean(a_w1, dim=0) * torch.mean(a_w2_star, dim=0)
        s2 = factor * (term_1 - term_2)
        return s2.squeeze(-1)

    def a_w3_gen(self, f_max_idx, m):
        """
        generates an initialization tensor which will be used to calculate c3.
        here sconfig.s3_calc can either be 1/2 or 1/4, which is chosen by the user.
        1/2 to calculate positive and negative frequencies for x axis.
        1/4 to calculate positive frequencies only.
        """
        if self.sconfig.s3_calc == '1/2': 
            n = 2 * (f_max_idx // 2) - 1
            a_w3 = torch.ones((f_max_idx // 2, n, m), dtype=torch.complex64) *1j
        elif self.sconfig.s3_calc == '1/4':
            a_w3 = torch.ones((f_max_idx // 2, f_max_idx // 2, m), dtype=torch.complex64) *1j
        return a_w3

    def index_generation_to_aw_3(self, f_max_idx):
        """
        Constructs an index matrix to correctly place elements of the Fourier coefficients
        of the signal in the desired order, accounting for potential spectrum symmetry.
        here sconfig.s3_calc can either be 1/2 or 1/4, which is chosen by the user.
        1/2 to calculate positive and negative frequencies for x axis.
        1/4 to calculate positive frequencies only.
        """
        if self.sconfig.s3_calc == '1/2':
            n = f_max_idx // 2
            indices = torch.arange(n).unsqueeze(1) + torch.arange(-(n - 1), n)
        elif self.sconfig.s3_calc == '1/4':
            indices = torch.arange(f_max_idx // 2).unsqueeze(1) + torch.arange(f_max_idx // 2)
        return indices

    def calc_a_w3(self, a_w_all, f_max_idx, m, a_w3, indices):
    # the complex type must be unified to prevent mismatch errors
        # match dtype
        if a_w3.dtype != a_w_all.dtype:
            a_w3 = a_w3.to(a_w_all.dtype)
        # same device
        if a_w3.device != a_w_all.device:
            a_w3 = a_w3.to(a_w_all.device)

        row_ids = torch.arange(f_max_idx // 2, device=a_w_all.device)
        a_w3[row_ids, :, :] = a_w_all[indices, 0, :m]
        return a_w3.conj()


    def c3(self, a_w1, a_w2, a_w3):
        """
        third order cumulant
        C_3 = m^2 / [(m - 1)(m - 2)] . (< a_w1 . a_w2 . a_w3 >
               - < a_w1 >< a_w2 . a_w3 > - < a_w1 . a_w2 >< a_w3 > - < a_w1 . a_w3 >< a_w2 >
               + 2 < a_w1 >< a_w2 >< a_w3 >)
        with w3 = - w1 - w2 and as before <...> denotes the mean
        the factor m^2 / (m - 1)(m - 2) is the unbiased estimator for the third order cumulant
        (see arXiv:1904.12154)
        """

        # The number of windows `m` used for the calculation is obtained from `self.config.m`
        m = self.sconfig.m

        a_w1_modified = a_w1.transpose(-1, -2)
        a_w1_modified_stacked = a_w1_modified.expand(a_w1_modified.size(0), a_w2.size(1),
                                                     a_w1_modified.size(2))

        a_w2_modified_stacked = a_w2.expand((a_w2.size(0), a_w2.size(1), a_w1.size(1)))

        a_w3_modified = a_w3.permute(2, 0, 1)

        d_12 = a_w1_modified_stacked * a_w2_modified_stacked
        d_13 = a_w1_modified_stacked * a_w3_modified
        d_23 = a_w2_modified_stacked * a_w3_modified
        d_123 = d_12 * a_w3_modified

        d_means = [torch.mean(d, dim=0) for d in [a_w1_modified_stacked, a_w2_modified_stacked,
                                                  a_w3_modified, d_12, d_13, d_23, d_123]]
        
        d_1_mean, d_2_mean, d_3_mean, d_12_mean, d_13_mean, d_23_mean, d_123_mean = d_means
        s3 = m ** 2 / ((m - 1) * (m - 2)) * (d_123_mean - d_12_mean * d_3_mean -
                                             d_13_mean * d_2_mean - d_23_mean * d_1_mean +
                                             2 * d_1_mean * d_2_mean * d_3_mean)

        return s3


    def c4(self, a_w1, a_w2, a_w3, a_w4):
        """
        fourth order cumulant
        C_4 = m^2 / [(m - 1)(m - 2)(m - 3)] . 
              {(m + 1)<(a_w1 - <a_w1>)(a_w2 - <a_w2>)(a_w3 - <a_w3>)(a_w3 - <a_w3>) >
               - (m + 1)[<(a_w1 - <a_w1>)(a_w2 - <a_w2>)> <(a_w3 - <a_w3>)(a_w3 - <a_w3>)>
                         + 2 o.p.]}
         <...> denotes the mean
        see arXiv:1904.12154 for more information
        """

        # The number of windows `m` used for the calculation is obtained from `self.config.m`
        m = self.sconfig.m

        # --- for a better readability ---
        x = a_w1
        y = torch.conj(a_w2)
        z = a_w3
        w = torch.conj(a_w4)
        # --------------------------------

        x_mean = x - x.mean(dim=0, keepdim=True)
        y_mean = y - y.mean(dim=0, keepdim=True)
        z_mean = z - z.mean(dim=0, keepdim=True)
        w_mean = w - w.mean(dim=0, keepdim=True)

        # Compute product and various partial means
        xyzw = torch.matmul((x_mean * y_mean), (z_mean * w_mean).transpose(-1, -2))
        xyzw_mean = xyzw.mean(dim=0)

        xy_mean = (x_mean * y_mean).mean(dim=0)
        zw_mean = (z_mean * w_mean).mean(dim=0)
        xy_zw_mean = torch.matmul(xy_mean, zw_mean.transpose(-1, -2))

        xz_mean = torch.matmul(x_mean, z_mean.transpose(-1, -2)).mean(dim=0)
        yw_mean = torch.matmul(y_mean, w_mean.transpose(-1, -2)).mean(dim=0)
        xz_yw_mean = xz_mean * yw_mean

        xw_mean = torch.matmul(x_mean, w_mean.transpose(-1, -2)).mean(dim=0)
        yz_mean = torch.matmul(y_mean, z_mean.transpose(-1, -2)).mean(dim=0)
        xw_yz_mean = xw_mean * yz_mean

        # Final combination
        s4 = (m**2 / ((m - 1)*(m - 2)*(m - 3))) * (
                (m + 1)*xyzw_mean - (m - 1)*(xy_zw_mean + xz_yw_mean + xw_yz_mean)
            )
        return s4

    # ------------------------------------------------------------------

    def store_sum_single_spectrum(self, single_spectrum, order, dataset_idx):
        """
        Helper function to store the spectra of single frames afterwards used 
        for calculation of errors and overlaps.
        """
        if self.s_gpu[dataset_idx][order] is None:
            self.s_gpu[dataset_idx][order] = single_spectrum
        else:
            self.s_gpu[dataset_idx][order] += single_spectrum

        if order == 1:
            self.s_errs[dataset_idx][order][0, self.err_counter[dataset_idx][order]] = single_spectrum
        elif order == 2:
            self.s_errs[dataset_idx][order][:, self.err_counter[dataset_idx][order]] = single_spectrum
        else:
            self.s_errs[dataset_idx][order][:, :, self.err_counter[dataset_idx][order]] = single_spectrum

        self.err_counter[dataset_idx][order] += 1

        if self.err_counter[dataset_idx][order] % self.sconfig.m_var == 0:
            dim = 1 if order in [1, 2] else 2
            self.n_error_estimates[dataset_idx][order] += 1
            factor = self.sconfig.m_var / (self.sconfig.m_var - 1)
            
            # FIX: compute variance-of-mean separately for Re and Im,
            # and store as complex error: s_err = SEM(Re) + 1j*SEM(Im)
            x = self.s_errs[dataset_idx][order]
            xr = x.real
            xi = x.imag

            # Var(mean(Re)) = (sample-var(Re) / m_var)
            mean_xr2 = torch.mean(xr ** 2, dim=dim)
            mean_xr = torch.mean(xr, dim=dim)
            var_mean_re = factor * (mean_xr2 - mean_xr ** 2) / self.sconfig.m_var

            # Var(mean(Im)) = (sample-var(Im) / m_var)
            mean_xi2 = torch.mean(xi ** 2, dim=dim)
            mean_xi = torch.mean(xi, dim=dim)
            var_mean_im = factor * (mean_xi2 - mean_xi ** 2) / self.sconfig.m_var

            # accumulate Var(mean) estimates across error batches
            var_mean_re_np = var_mean_re.cpu().numpy()
            var_mean_im_np = var_mean_im.cpu().numpy()

            if self.s_err[dataset_idx][order] is None:
                self.s_err[dataset_idx][order] = var_mean_re_np + 1j * var_mean_im_np
            else:
                self.s_err[dataset_idx][order] += var_mean_re_np + 1j * var_mean_im_np
            

            self.err_counter[dataset_idx][order] = 0

    def store_final_spectrum(self, orders, n_chunks, dataset_idx):
        """
        helper function to store the final result after avereging all the results from the chunks
        """
        for order in orders:
            if self.s_gpu.get(dataset_idx, {}).get(order) is not None:
                self.s_gpu[dataset_idx][order] /= n_chunks
                self.s[dataset_idx][order] = self.s_gpu[dataset_idx][order].cpu().resolve_conj().numpy()
                n_est = self.n_error_estimates[dataset_idx][order]
                if n_est and self.s_err[dataset_idx][order] is not None:
                    # s_err currently stores SUM_over_batches( Var_mean_re + 1j*Var_mean_im )
                    # Convert to mean over batches, then SEM = sqrt(var_mean)
                    var_mean = self.s_err[dataset_idx][order] / n_est
                    var_re = np.maximum(np.real(var_mean), 0.0)
                    var_im = np.maximum(np.imag(var_mean), 0.0)
                    sem_re = np.sqrt(var_re) / 2  # interlaced correction
                    sem_im = np.sqrt(var_im) / 2
                    self.s_err[dataset_idx][order] = sem_re + 1j * sem_im
                else:
                    self.s_err[dataset_idx][order] = None

    def fourier_coeffs_to_spectra(self, orders, coeffs_gpu, f_min_idx, f_max_idx, single_window, dataset_idx):
        """
        Helper function to calculate the (1,2,3,4)-order cumulant from the Fourier coefficients of the windows in
        one frame.
        """
        for order in orders:
            if order == 1:
                a_w = coeffs_gpu[:, f_min_idx:f_max_idx, :]
                single_spectrum = self.c1(a_w) / (self.sconfig.dt * single_window.mean() * single_window.shape[0])
            elif order == 2:
                a_w = coeffs_gpu[:, f_min_idx:f_max_idx, :]
                single_spectrum = self.c2(a_w, a_w) / (self.sconfig.dt * (single_window ** 2).sum())
            elif order == 3:
                a_w1 = coeffs_gpu[:, f_min_idx:f_max_idx//2, :]
                a_w2 = a_w1
                
                coeffs_gpu_p = coeffs_gpu.permute((1, 2, 0))

                if self.sconfig.s3_calc == '1/2':
                    a_w1 = torch.cat((a_w1[:, 1:, :].flip([1]).conj(), a_w1), dim=1)
                    coeffs_gpu_p = torch.cat((coeffs_gpu_p, torch.conj(coeffs_gpu_p[1:, :, :].flip([0]))), dim=0)

                a_w3 = self.calc_a_w3(coeffs_gpu_p, f_max_idx, self.sconfig.m, self.a_w3_init, self.indi)
                single_spectrum = self.c3(a_w1, a_w2, a_w3) / (self.sconfig.dt * (single_window ** 3).sum())

            elif order == 4:
                a_w = coeffs_gpu[:, f_min_idx:f_max_idx, :]
                single_spectrum = self.c4(a_w, a_w, a_w, a_w) / (self.sconfig.dt * (single_window ** 4).sum())

            self.store_sum_single_spectrum(torch.conj(single_spectrum), order, dataset_idx)

    def fourier_coeffs_to_cross_spectra(self, orders, coeffs_gpu_dict, f_min_idx, f_max_idx, single_window, *keys):
        """
        Helper function to calculate the (1,2,3,4)-order cumulant from the Fourier coefficients of the windows in
        one frame. This function is essentially a more general way of fourier_coeffs_to_spectra function which
        can also calculate cross-spectra.
        """
        for order in orders:
            if len(keys) < order:
                raise ValueError(f"Need at least {order} keys for order {order}")
            
            if order == 2:
                key1, key2 = keys[0], keys[1]
                a_w1 = coeffs_gpu_dict[key1][:, f_min_idx:f_max_idx, :]
                a_w2 = coeffs_gpu_dict[key2][:, f_min_idx:f_max_idx, :]
                single_spectrum = self.c2(a_w1, a_w2) / (self.sconfig.dt * (single_window ** 2).sum())

            if order == 3:
                key1, key2, key3 = keys[0], keys[1], keys[2]

                a_w1 = coeffs_gpu_dict[key1][:, f_min_idx:f_max_idx//2, :]
                a_w2 = coeffs_gpu_dict[key2][:, f_min_idx:f_max_idx//2, :]
                coeffs_gpu_p = coeffs_gpu_dict[key3].permute((1, 2, 0))

                if self.sconfig.s3_calc == '1/2':
                    a_w1 = torch.cat((a_w1[:, 1:, :].flip([1]).conj(), a_w1), dim=1)
                    coeffs_gpu_p = torch.cat((coeffs_gpu_p, torch.conj(coeffs_gpu_p[1:, :, :].flip([0]))), dim=0)

                
                a_w3 = self.calc_a_w3(coeffs_gpu_p, f_max_idx, self.sconfig.m, self.a_w3_init, self.indi)
                single_spectrum = self.c3(a_w1, a_w2, a_w3) / (self.sconfig.dt * (single_window ** 3).sum())

            if order == 4:
                key1, key2, key3, key4 = keys[0], keys[1], keys[2], keys[3]

                a_w1 = coeffs_gpu_dict[key1][:, f_min_idx:f_max_idx, :]
                a_w2 = coeffs_gpu_dict[key2][:, f_min_idx:f_max_idx, :]
                a_w3 = coeffs_gpu_dict[key3][:, f_min_idx:f_max_idx, :]
                a_w4 = coeffs_gpu_dict[key4][:, f_min_idx:f_max_idx, :]
                single_spectrum = self.c4(a_w1, a_w2, a_w3, a_w4) / (self.sconfig.dt * (single_window ** 4).sum())
            
            # Store result keyed by the tuple of dataset indices
            self.store_sum_single_spectrum(torch.conj(single_spectrum), order, keys)

    def array_prep(self, orders, f_all_in, dataset_idx):
        """
        Helper function to initialize the arrays for errors.
        """
        f_max_idx = f_all_in.shape[0]
        for order in orders:
            if order == 3:
                half_size = int(f_max_idx//2)
                self.freq[dataset_idx][order] = f_all_in[:half_size]
                if self.sconfig.s3_calc == '1/2':
                    self.freq[dataset_idx][order] = np.concatenate((-self.freq[dataset_idx][order][:0:-1], 
                                                                     self.freq[dataset_idx][order]))
            else:
                self.freq[dataset_idx][order] = f_all_in
            if order == 1:
                self.s_errs[dataset_idx][order] = torch.ones((1, self.sconfig.m_var), 
                                                             device=self.sconfig.backend,
                                                             dtype=torch.complex64)
            elif order == 2:
                self.s_errs[dataset_idx][order] = torch.ones((f_max_idx, self.sconfig.m_var), 
                                                             device=self.sconfig.backend,
                                                             dtype=torch.complex64)
            elif order == 3:
                # for cross/cross3 we can have 2D freq
                if self.sconfig.s3_calc == '1/2': # non-negative x axis
                    n = 2 * (f_max_idx // 2) - 1
                    self.s_errs[dataset_idx][order] = torch.ones((f_max_idx//2, n,
                                                                  self.sconfig.m_var),
                                                                 device=self.sconfig.backend,
                                                                 dtype=torch.complex64)
                elif self.sconfig.s3_calc == '1/4':# negative and positive x axis
                    self.s_errs[dataset_idx][order] = torch.ones((f_max_idx//2, f_max_idx//2,
                                                                  self.sconfig.m_var),
                                                                 device=self.sconfig.backend,
                                                                 dtype=torch.complex64)

            elif order == 4:
                # for cross/cross4 we can have 2D freq
                self.s_errs[dataset_idx][order] = torch.ones((f_max_idx, f_max_idx, self.sconfig.m_var),
                                                             device=self.sconfig.backend,
                                                             dtype=torch.complex64)

    def process_order(self):
        """
        Helper function to turn the user input to meaningful array for functions.
        """
        orders_to_process = [1, 2, 3, 4] if self.sconfig.order_in == 'all' else self.sconfig.order_in
        if self.sconfig.f_min < 0 and 3 in orders_to_process:
            orders_to_process.remove(3)
            print('For negative frequencies in order 3 use s3_calc and positive frequencies\n')
            print("Example: f_min=0, f_max=5, s3_calc='1/2'")
        return orders_to_process

    def reset_variables(self, orders):
        """
        TODO: refactor
        """
        self.err_counter = {i: {1: 0, 2: 0, 3: 0, 4: 0} for i in self.selected}
        self.n_error_estimates = {i: {1: 0, 2: 0, 3: 0, 4: 0} for i in self.selected}

        self.err_counter.update({j: {2: 0} for j in self.cross2_selected})
        self.n_error_estimates.update({j: {2: 0} for j in self.cross2_selected})
        # for cross3
        self.err_counter.update({j: {3: 0} for j in self.cross3_selected})
        self.n_error_estimates.update({j: {3: 0} for j in self.cross3_selected})
        # for cross4
        self.err_counter.update({j: {4: 0} for j in self.cross4_selected})
        self.n_error_estimates.update({j: {4: 0} for j in self.cross4_selected})

        for dataset_idx in self.selected:
            for order in orders:
                self.freq[dataset_idx][order] = None
                self.s[dataset_idx][order] = None
                self.s_gpu[dataset_idx][order] = None
                self.s_err[dataset_idx][order] = None
                self.s_err_gpu[dataset_idx][order] = None
                self.s_errs[dataset_idx][order] = []
                self.m[order] = self.sconfig.m
                self.m_var[order] = self.sconfig.m_var

        for cross2_idx in self.cross2_selected:
            for order in orders:
                if order == 2:
                    self.freq[cross2_idx][order] = None
                    self.s[cross2_idx][order] = None
                    self.s_gpu[cross2_idx][order] = None
                    self.s_err[cross2_idx][order] = None
                    self.s_err_gpu[cross2_idx][order] = None
                    self.s_errs[cross2_idx][order] = []

        # Properly initialize cross4 sets
        for cross3_idx in self.cross3_selected:
            for order in orders:
                if order == 3:
                    self.freq[cross3_idx][order] = None
                    self.s[cross3_idx][order] = None
                    self.s_gpu[cross3_idx][order] = None
                    self.s_err[cross3_idx][order] = None
                    self.s_err_gpu[cross3_idx][order] = None
                    self.s_errs[cross3_idx][order] = []

        # Properly initialize cross4 sets
        for cross4_idx in self.cross4_selected:
            for order in orders:
                if order == 4:
                    self.freq[cross4_idx][order] = None
                    self.s[cross4_idx][order] = None
                    self.s_gpu[cross4_idx][order] = None
                    self.s_err[cross4_idx][order] = None
                    self.s_err_gpu[cross4_idx][order] = None
                    self.s_errs[cross4_idx][order] = []

    def reset(self):
        orders = self.process_order()
        self.orders = orders
        self.reset_variables(orders)
        return orders

    def setup_calc_spec(self, orders):
        """
        calculating the needed parameters to calculate spectra
        """
        n_data_points = self.diconfig_list[self.selected[0]].data.shape[0]

        if not self.window_points * self.sconfig.m + self.window_points // 2 < n_data_points:
            m = (n_data_points - self.window_points // 2) // self.window_points
            if m < max(orders):
                raise ValueError('Not enough data points')
            print(f'Values have been changed. Old m: {self.sconfig.m}, new m: {m}')
            self.sconfig.m = m
        else:
            m = self.sconfig.m

        denom_spec = self.window_points * m + self.window_points // 2
        n_spectra = n_data_points // denom_spec

        if n_spectra < self.sconfig.m:
            m_var = n_data_points // denom_spec
            if m_var < 2:
                raise ValueError('Not enough data points.')
            else:
                print(f'm_var values have been changed. Old: {self.sconfig.m_var}, new: {m_var}')
            self.m_var = m_var

        n_windows = int(np.floor(n_data_points / (m * self.window_points)))
        # Create frequency axis using full FFT if needed
        if self.use_full_fft:
            # if negative frequencies are needed
            freq_all_freq = np.fft.fftfreq(self.window_points, self.sconfig.dt)
            freq_all_freq = np.fft.fftshift(freq_all_freq)
        else:
            # for non-negative frequencies only
            freq_all_freq = np.fft.rfftfreq(self.window_points, self.sconfig.dt)

        # Determine indices for frequency band
        f_mask = freq_all_freq <= self.sconfig.f_max
        f_max_idx = np.sum(f_mask)
        f_mask = freq_all_freq < self.sconfig.f_min
        f_min_idx = np.sum(f_mask)

        return m, freq_all_freq, f_max_idx, f_min_idx, n_windows

    def _to_device(self, array):
        """
        Helper function that converts a NumPy array to a torch tensor on the proper device.
        """
        tensor = torch.from_numpy(array.astype(np.float32)) if self.use_float32 else torch.from_numpy(array)
        return tensor.to(self.device)

    def _compute_fft(self, window, chunk_gpu):
        """
        Helper function that computes the FFT (full or real) and applies scaling and shift if needed.
        """
        if self.use_full_fft:
            a_w = torch.fft.fft(window * chunk_gpu, dim=1)
            a_w *= self.sconfig.dt
            a_w = torch.fft.fftshift(a_w, dim=1)
        else:
            a_w = torch.fft.rfft(window * chunk_gpu, dim=1)
            a_w *= self.sconfig.dt
        return a_w

    def calc_spec(self):
        """
        main funtion that calculates spectra.
        """
        orders = self.reset()
        m, freq_all_freq, f_max_idx, f_min_idx, n_windows = self.setup_calc_spec(orders)

        for order in orders:
            self.m[order] = m

        if 3 in orders: # this might not optimal because the same thing is being generated multiple times.
            self.a_w3_init = self.a_w3_gen(f_max_idx, self.sconfig.m).to(self.device)
            self.indi = self.index_generation_to_aw_3(f_max_idx).to(self.device)

        single_window, _ = cg_window(int(self.window_points), self.fs)
        window = np.array(m * [single_window]).flatten().reshape((m, self.window_points, 1))
        window = self._to_device(window)

        # Prepare arrays for auto spectra
        for dataset_idx in self.selected:
            self.array_prep(orders, freq_all_freq[f_min_idx:f_max_idx], dataset_idx)


        # Prepare arrays for cross-spectra 2
        if self.cross2_selected:
            for pair in self.cross2_selected:
                self.array_prep([2], freq_all_freq[f_min_idx:f_max_idx], pair)

        # Prepare arrays for cross-spectra 3 
        if self.cross3_selected:
            for tpair in self.cross3_selected:
                self.array_prep([3], freq_all_freq[f_min_idx:f_max_idx], tpair)

        # Prepare arrays for cross-spectra 4 
        if self.cross4_selected:
            for fpair in self.cross4_selected:
                self.array_prep([4], freq_all_freq[f_min_idx:f_max_idx], fpair)

        

        for i in tqdm(range(n_windows), leave=False):
            for window_shift in [0, self.window_points // 2]:
                a_w_all_dict = {}
                for dataset_idx in self.selected:
                    data_config = self.diconfig_list[dataset_idx]
                    start = int(i * (self.window_points * m) + window_shift)
                    end = int((i + 1) * (self.window_points * m) + window_shift)
                    chunk = data_config.data[start:end]
                    if chunk.shape[0] == self.window_points * m:
                        chunk_r = chunk.reshape((m, self.window_points, 1))
                        chunk_gpu = self._to_device(chunk_r)
                        a_w_all_dict[dataset_idx] = self._compute_fft(window, chunk_gpu)
                    else:
                        a_w_all_dict[dataset_idx] = None

                # Auto-correlation
                if self.cconfig.auto_corr:
                    for dataset_idx, a_w_all_gpu in a_w_all_dict.items():
                        if a_w_all_gpu is None:
                            continue
                        self.n_chunks[dataset_idx] += 1
                        self.fourier_coeffs_to_spectra(orders, a_w_all_gpu, f_min_idx, f_max_idx, single_window, dataset_idx)
                        
                        if self.n_chunks[dataset_idx] == self.sconfig.break_after:
                            break

                for cross_group, order in [(self.cross2_selected, 2), (self.cross3_selected, 3), (self.cross4_selected, 4)]:
                    if cross_group:
                        for keys in cross_group:
                            if any(a_w_all_dict.get(k) is None for k in keys):
                                continue
                            self.n_chunks[keys] += 1
                            self.fourier_coeffs_to_cross_spectra(
                                [order],
                                a_w_all_dict,
                                f_min_idx,
                                f_max_idx,
                                single_window,
                                *keys
                            )
                            if self.n_chunks[keys] == self.sconfig.break_after:
                                break

        # Store final auto spectra
        for dataset_idx in self.selected:
            self.store_final_spectrum(orders, self.n_chunks[dataset_idx], dataset_idx)

        for cross_keys, order in [(self.cross2_selected, 2), (self.cross3_selected, 3), (self.cross4_selected, 4)]:
            if cross_keys:
                for keys in cross_keys:
                    self.store_final_spectrum([order], self.n_chunks[keys], keys)

        return self.freq, self.s, self.s_err