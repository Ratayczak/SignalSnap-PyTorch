# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

import os
from pathlib import Path

from typing import Annotated, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


os.environ['PYDANTIC_ERRORS_INCLUDE_URL'] = '0'

class CrossConfig(BaseModel):

    auto_corr: bool = True
    cross_corr_2: list[tuple[int, int]] | None = None
    cross_corr_3: list[tuple[int, int, int]] | None = None
    cross_corr_4: list[tuple[int, int, int, int]] | None = None


class PlotConfig(BaseModel):

    f_min: int | float
    f_max: int | float

    display_orders: list[Annotated[int, Field(ge=1, le=4)]] = [1, 2, 3, 4]
    significance: Annotated[int, Field(gt=0)] = 1
    arcsinh_scale: tuple[bool, Annotated[int | float, Field(ge=0)]] = (False, 0.02)
    plot_format: Annotated[list[Literal["re", "im"]], Field(min_length=1)] = ["re", "im"]
    insignif_transparency: Annotated[int | float, Field(ge=0.0, le=1.0)] = 0.8
    output: Literal["show", "save"] = "show"
    output_folder: Path = Path(".").resolve()
    
    @field_validator("plot_format")
    @classmethod
    def ensure_unique_formats(cls, v: list[str]) -> list[str]:
        """Makes sure plot_format is: ["re"], ["im"], ["re", "im"], or ["im", "re"]"""
        if len(v) != len(set(v)):
            raise ValueError("plot_format cannot contain duplicate elements.")
        return v
    
    @field_validator("output_folder", mode="before")
    @classmethod
    def resolve_output_folder(cls, v: str | Path) -> Path:
        """
        Automatically convert strings to Path objects to ensure output_folder is a Path object
        """
        return Path(v).resolve()
    
    @model_validator(mode="after")
    def validate_limits(self) -> "PlotConfig":
        if self.f_min >= self.f_max:
            raise ValueError(f"f_min ({self.f_min}) must be less than f_max ({self.f_max}).")
        return self
