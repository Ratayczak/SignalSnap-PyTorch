# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

import os
from pathlib import Path

from typing import Annotated, Any, Literal
from pydantic import BaseModel, DirectoryPath, Field, field_validator, model_validator, ConfigDict


os.environ['PYDANTIC_ERRORS_INCLUDE_URL'] = '0'
SHARED_CONFIG = ConfigDict(frozen=True, extra="forbid")

class CrossConfig(BaseModel):
    model_config = SHARED_CONFIG

    auto_corr: bool = True
    cross_corr_2: list[tuple[int, int]] | None = None
    cross_corr_3: list[tuple[int, int, int]] | None = None
    cross_corr_4: list[tuple[int, int, int, int]] | None = None


class DataConfig(BaseModel):
    model_config = SHARED_CONFIG

    data: Any | None = None
    path: str | None = None
    group_key: str | None = None
    dataset: str | None = None
    dt: float | None = None


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
        """Makes sure plot_format is: ["re"], ["im"], ["re", "im"], or ["im", "re"]"""
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
    model_config = SHARED_CONFIG

    dt: Annotated[float, Field(gt=0)]
    
    f_unit: Literal['Hz', 'kHz', 'MHz', 'GHz', 'THz'] = 'Hz'
    f_max: float | None = None
    f_min: float = 0.0
    s3_calc: Literal['1/4', '1/2'] = '1/4'      #TODO Add '1' here later when ready
    backend: Literal['cpu', 'mps', 'cuda'] = 'mps'
    spectrum_size: Annotated[int, Field(gt=0)] = 100
    order_in: Literal['all'] | list[Annotated[int, Field(ge=1, le=4)]] = 'all'
    m: Annotated[int, Field(gt=0)] = 10
    m_var: Annotated[int, Field(gt=0)] = 10
    show_first_frame: bool = True
    break_after: int = int(1e6)

    @model_validator(mode="after")
    def validate_frequency_limits(self) -> "SpectrumConfig":
        if self.f_max is not None and self.f_min >= self.f_max:
            raise ValueError(f"f_min ({self.f_min}) must be less than f_max ({self.f_max}).")
        return self


