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

from pydantic import (
    BaseModel,
    ConfigDict,
    DirectoryPath,
    Field,
    field_validator,
    model_validator,
)

from .utils import S3Calcs, TimeUnits

os.environ["PYDANTIC_ERRORS_INCLUDE_URL"] = "0"
SHARED_CONFIG = ConfigDict(frozen=True, extra="forbid")


class CrossConfig(BaseModel):
    """Configuration to specify which single- or multi-channel spectra are
    to be computed at each order.

    Attributes
    ----------
        auto_corr:
            Determines wether single-channel (auto-correlation) spectra
            will be calculated.

        cross_corr_X:
            Specifies which multi-channel (cross-correlation) spectra will
            be calculated at order X.
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
        data: Any
            The recorded signal data (e.g. a NumPy array).
        dt: Annotated[float, Field(gt=0)]
            The time interval between two consecutive data points 
            (must be > 0).
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
    plot_format: Annotated[list[Literal["re", "im"]], Field(min_length=1)] = [
        "re",
        "im",
    ]
    insignif_transparency: Annotated[float, Field(ge=0.0, le=1.0)] = 0.8
    output: Literal["show", "save"] = "show"
    output_folder: DirectoryPath = Path(".").resolve()

    @field_validator("plot_format")
    @classmethod
    def ensure_unique_formats(cls, v: list[str]) -> list[str]:
        """
        Makes sure plot_format is one of
        ["re"], ["im"], ["re", "im"], or ["im", "re"]
        """
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
            raise ValueError(
                f"f_min ({self.f_min}) must be less than f_max ({self.f_max})."
            )
        return self


class SpectrumConfig(BaseModel):
    """Spectrum configuration for polyspectra calculations.

    ``SpectrumConfig`` describes what the user asks the calculation to use:
    frequency bounds, spectrum orders, backend, spectrum size, window count,
    and compatibility options. These settings are later resolved together
    with ``DataConfig`` into a ``RuntimeConfig``.

    Parameters
    ----------
    f_max : float | None
        Upper frequency bound. If omitted, the maximum allowed frequency is used.
    f_min : float
        Lower frequency bound.
    s3_calc : Literal["1/4", "1/2"]
        Method used for third-order spectrum calculation.
    backend : Literal["cpu", "mps", "cuda"]
        Torch backend requested for calculation.
    spectrum_size : int
        Number of frequency points in the spectrum.
    order_in : Literal["all"] | list[int]
        Spectrum orders to calculate.
    m : int
        Window count per spectrum.
    show_first_frame : bool
        Whether to display the first processed frame.
    break_after : int | None
        Maximum number of calculated spectra.
    _old_window : bool
        Compatibility option. If set to true, the wrong approximated
        confined gaussian window from the old API is used as a window
        function.
    """
    model_config = SHARED_CONFIG

    f_max: float | None = None
    f_min: float = 0.0
    s3_calc: S3Calcs = "1/4"  # TODO Add '1' here later when ready
    backend: Literal["cpu", "mps", "cuda"] = "mps"
    spectrum_size: Annotated[int, Field(gt=0)] = 100
    order_in: Literal["all"] | list[Annotated[int, Field(ge=1, le=4)]] = "all"
    m: Annotated[int, Field(gt=0)] = 10
    show_first_frame: bool = True
    break_after: Annotated[int, Field(gt=0)] | None = int(1e6)
    old_window: bool = False

    @property
    def _old_window(self) -> bool:
        return self.old_window

    @model_validator(mode="after")
    def validate_frequency_limits(self) -> "SpectrumConfig":
        if self.f_max is not None and self.f_min >= self.f_max:
            raise ValueError(
                f"f_min ({self.f_min}) must be less than f_max ({self.f_max})."
            )
        return self
