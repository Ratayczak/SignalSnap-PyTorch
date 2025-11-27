from torch import Tensor
import torch
from typing import Tuple

x = torch.linspace(0, 5, 5)
n_windows = 5
l = n_windows + 1
sigma_t = 0.14

def gaussian_window(x: Tensor,
                    n_windows: int,
                    l: int,
                    sigma_t: float
) -> Tensor:

    center = n_windows * 0.5
    denom  = 2.0 * l * sigma_t

    t = (x - center) / denom
    return torch.exp(-t * t)


def calc_window(x: Tensor, 
                n_windows: int, 
                l: int, 
                sigma_t: float
) -> Tensor:
    
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

    x = torch.linspace(0, n_windows, n_windows)
    l = n_windows + 1
    sigma_t = 0.14

    window = calc_window(x, n_windows, l, sigma_t)
    norm_t = (window * window).sum() / fs
    
    window_full = window / torch.sqrt(norm_t)
    norm = float(norm_t.item())

    return window_full, norm


# ------ outputs ------
out_1 = gaussian_window(x, n_windows, l, sigma_t)
out_2 = calc_window(x, n_windows, l, sigma_t)
out_3 = cg_window(n_windows, 0.5)
print(out_3)
