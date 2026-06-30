# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, DirectoryPath, Field, field_validator, model_validator

from .utils import S3Calcs, TimeUnits

os.environ["PYDANTIC_ERRORS_INCLUDE_URL"] = "0"
SHARED_CONFIG = ConfigDict(frozen=True, extra="forbid")


class CrossConfig(BaseModel):
    """Configuration to specify which single- or multi-channel spectra are to be computed at each
    order.

    Attributes
    ----------
        auto_corr: bool
            Determines whether single-channel (auto-correlation) spectra will be calculated.
        cross_corr_2: list[tuple[int, int]] | None
            Specifies which multi-channel (cross-correlation) spectra will be calculated at order 2.
            Each tuple represents one cross-correlation spectrum. Each tuple entry is a channel
            index.
        cross_corr_3: list[tuple[int, int, int]] | None
            Specifies which multi-channel (cross-correlation) spectra will be calculated at order 3.
            Each tuple represents one cross-correlation spectrum. Each tuple entry is a channel
            index.
        cross_corr_4: list[tuple[int, int, int, int]] | None
            Specifies which multi-channel (cross-correlation) spectra will be calculated at order 4.
            Each tuple represents one cross-correlation spectrum. Each tuple entry is a channel
            index.
    """

    model_config = SHARED_CONFIG

    auto_corr: bool = True
    cross_corr_2: list[tuple[int, int]] | None = None
    cross_corr_3: list[tuple[int, int, int]] | None = None
    cross_corr_4: list[tuple[int, int, int, int]] | None = None


class DataConfig(BaseModel):
    """Configuration for data used in polyspectra calculations.

    Attributes
    ----------
        data: array-like with a shape attribute
            The recorded signal data (e.g. a NumPy array).
        dt: float
            The time interval between two consecutive data points. Must be positive.
        t_unit: Literal["s", "ms", "us", "ns", "ps"]
            Unit of the time step. Defaults to "s".
    """

    model_config = SHARED_CONFIG

    data: Any
    dt: Annotated[float, Field(gt=0)]
    t_unit: TimeUnits = "s"


class PlotConfig(BaseModel):
    model_config = SHARED_CONFIG

    f_min: float
    f_max: float

    display_orders: list[Annotated[int, Field(ge=1, le=4)]] = [1, 2, 3, 4]
    significance: Annotated[int, Field(gt=0)] = 1
    arcsinh_scale: tuple[bool, Annotated[float, Field(ge=0)]] = (False, 0.02)
    plot_format: Annotated[list[Literal["re", "im"]], Field(min_length=1)] = ["re", "im"]
    insignif_transparency: Annotated[float, Field(ge=0.0, le=1.0)] = 0.8
    output: Literal["show", "save"] = "show"
    output_folder: DirectoryPath = Path(".").resolve()

    @field_validator("plot_format")
    @classmethod
    def ensure_unique_formats(cls, v: list[str]) -> list[str]:
        """Ensure plot_format does not contain duplicate components."""
        if len(v) != len(set(v)):
            raise ValueError("plot_format cannot contain duplicate elements.")
        return v

    @field_validator("output_folder")
    @classmethod
    def resolve_output_folder(cls, v: Path) -> Path:
        return v.resolve()

    @model_validator(mode="after")
    def validate_limits(self) -> "PlotConfig":
        if self.f_min >= self.f_max:
            raise ValueError(f"f_min ({self.f_min}) must be less than f_max ({self.f_max}).")
        return self


class SpectrumConfig(BaseModel):
    """Spectrum configuration for polyspectra calculations.

    ``SpectrumConfig`` describes what the user asks the calculation to use: frequency bounds, number
    of frequency points, spectrum orders, window count per spectral estimate, backend torch device,
    and compatibility options. These settings are later resolved together with ``DataConfig`` into a
    ``RuntimeConfig``.

    Attributes
    ----------
        f_min : float = 0.0
            Lower frequency bound. If omitted, zero is used.
        f_max : float | None = None
            Upper frequency bound. If omitted, the maximal allowed frequency is used.
        frequency_points : int = 100
            Number of frequency points in the specified frequency range. Must be positive.
        orders : Literal["all"] | list[int] = "all"
            Spectrum orders (between 1 and 4) to be calculated. ``all`` means orders
            ``[1, 2, 3, 4]``.
        m : int = 10
            Number of windows used per spectral estimate. This may be reduced at runtime if the
            signal is too short. Must be positive.
        s3_calc : Literal["1/4", "1/2"] = "1/4"
            Method used for third-order spectrum calculation.
        device : Literal["cpu", "mps", "cuda"]  = "cpu"
            Torch device requested for calculation.
        precision: Literal["auto", "single", "double"] = "auto"
            Floating point precision. ``single`` will result in ``float32`` and ``complex64``.
            ``double`` will result in ``float64`` and ``complex128``. ``auto`` will choose
            ``single`` if device is ``mps`` and ``double`` otherwise.
        spectral_estimates_max : int | None = int(1e6)
            Maximum number of spectral estimates. If ``None``, as many estimates as possible are
            calculated based on the data. The true number of spectral estimates may be lower if the
            data doesnot have enough samples. Must be positive.
        old_window : bool = False
            Compatibility option. If set to ``True``, the approximated confined Gaussian window from
            the old API is used as a window function.
    """

    model_config = SHARED_CONFIG

    f_min: float = 0.0
    f_max: float | None = None
    frequency_points: Annotated[int, Field(gt=0)] = 100
    orders: (
        Literal["all"] | Annotated[list[Annotated[int, Field(ge=1, le=4)]], Field(min_length=1)]
    ) = "all"
    m: Annotated[int, Field(gt=0)] = 10
    s3_calc: S3Calcs = "1/4"
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    precision: Literal["auto", "single", "double"] = "auto"
    spectral_estimates_max: Annotated[int, Field(gt=0)] | None = int(1e6)
    old_window: bool = False

    @property
    def _old_window(self) -> bool:
        return self.old_window

    @model_validator(mode="after")
    def validate_spectrum_request(self) -> "SpectrumConfig":
        if self.f_max is not None and self.f_min >= self.f_max:
            raise ValueError(f"f_min ({self.f_min}) must be less than f_max ({self.f_max}).")

        orders = [1, 2, 3, 4] if self.orders == "all" else self.orders
        if self.f_min < 0 and 3 in orders:
            raise ValueError(
                "Third-order spectra cannot be requested with f_min < 0. "
                "Use f_min=0 and s3_calc='1/2' for the third-order negative-frequency convention."
            )

        return self
