# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemad, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from ._core.utils import FrequencyUnits as _FrequencyUnits
from .metadata import CalculationMetadata, SpectrumMetadata

__all__ = ["SpectrumResult", "SpectrumResultStore"]


@dataclass(frozen=True, slots=True)
class SpectrumResult:
    """Result of one requested spectral calculation.

    A result contains the calculated spectrum, its frequency axis, optional uncertainty information,
    and the metadata specific to this result. Results produced by a calculation pipeline share their
    :class:`CalculationMetadata` object with the containing :class:`SpectrumResultStore`.

    For a third-order result, ``spectrum[i, j]`` represents the frequency tuple
    ``(freq[i], freq[j], -(freq[i] + freq[j]))``. If the closing channel is sampled, entries whose
    implied closing frequency lies outside its FFT support are ``NaN``. A timestamped closing
    channel is evaluated by direct transform and is not restricted to sampled FFT support.

    A fourth-order result contains the diagonal slice
    ``(freq[i], -freq[i], freq[j], -freq[j])`` rather than the complete trispectrum.

    Attributes
    ----------
    channels : tuple[int, ...]
        Channel indices defining the spectrum. For example, ``(0, 0, 0)`` identifies a third-order
        auto-spectrum of channel 0, while ``(0, 1)`` identifies a second-order cross-spectrum
        between channels 0 and 1.
    freq : np.ndarray
        One-dimensional frequency axis associated with the spectrum. For a first-order spectrum,
        this is ``[0]``.
    freq_unit : Literal["Hz", "kHz", "MHz", "GHz", "THz"]
        Unit of ``freq``.
    spectrum : np.ndarray
        Final normalized spectral values transferred to the CPU. For ``F = len(freq)``, the shape is
        ``(1,)`` for first-order, ``(F,)`` for second-order, and ``(F, F)`` for third- and
        fourth-order results.
    spectrum_uncertainty : np.ndarray | None
        Component-wise uncertainty of ``spectrum``, or ``None`` when insufficient estimates are
        available. When present, it has the same shape as ``spectrum``.

        Its real and imaginary components independently contain the uncertainties of the
        corresponding spectrum components. The combined complex value has no statistical
        interpretation.

        With short-term uncertainty estimation, consecutive estimates are divided into complete
        groups of ``effective_m_var`` estimates. Incomplete trailing groups do not contribute.

        Shifted and unshifted estimates are evaluated separately. If both provide an uncertainty,
        their component-wise maximum is returned. If only the unshifted qualifies, that group's
        uncertainty is returned.
    calculation_metadata : CalculationMetadata | None
        Calculation-wide metadata shared with the containing result store and its other results.
        This is ``None`` only for manually constructed results that omit metadata.
    spectrum_metadata : SpectrumMetadata | None
        Metadata describing this specific requested spectrum. For pipeline-created results, this is
        the same object as ``store.spectra_metadata[channels]``. It is ``None`` only for manually
        constructed results that omit metadata.

    Notes
    -----
    ``calculation_metadata`` and ``spectrum_metadata`` must either both be provided or both be
    ``None``.
    """

    channels: tuple[int, ...]

    freq: np.ndarray
    freq_unit: _FrequencyUnits
    spectrum: np.ndarray
    spectrum_uncertainty: np.ndarray | None = None

    calculation_metadata: CalculationMetadata | None = None
    spectrum_metadata: SpectrumMetadata | None = None

    @property
    def order(self) -> int:
        """Return the order of the spectrum."""
        return len(self.channels)

    def __post_init__(self) -> None:
        """Validate the spectrum order, frequency axis, and result-array shapes."""
        if not 1 <= self.order <= 4:
            raise ValueError(f"Unsupported spectrum order {self.order}.")

        if self.freq.ndim != 1:
            raise ValueError("Frequency axis must be one-dimensional.")

        frequency_points = len(self.freq)

        expected_shape = {
            1: (1,),
            2: (frequency_points,),
            3: (frequency_points, frequency_points),
            4: (frequency_points, frequency_points),
        }[self.order]

        if self.spectrum.shape != expected_shape:
            raise ValueError(
                f"Order-{self.order} spectrum has shape "
                f"{self.spectrum.shape}; expected {expected_shape}."
            )

        if (
            self.spectrum_uncertainty is not None
            and self.spectrum_uncertainty.shape != expected_shape
        ):
            raise ValueError("Spectrum uncertainty must have the same shape as the spectrum.")

        if self.calculation_metadata is not None and not isinstance(
            self.calculation_metadata,
            CalculationMetadata,
        ):
            raise TypeError("calculation_metadata must be a CalculationMetadata object or None.")

        if self.spectrum_metadata is not None and not isinstance(
            self.spectrum_metadata,
            SpectrumMetadata,
        ):
            raise TypeError("spectrum_metadata must be a SpectrumMetadata object or None.")

        if (self.calculation_metadata is None) != (self.spectrum_metadata is None):
            raise ValueError(
                "calculation_metadata and spectrum_metadata must either both be "
                "provided or both be None."
            )

        if self.calculation_metadata is not None:
            assert self.spectrum_metadata is not None

            if self.channels not in self.calculation_metadata.requested_spectra:
                raise ValueError(
                    f"Spectrum {self.channels} is not described by its calculation metadata."
                )

            if self.spectrum_metadata.channels != self.channels:
                raise ValueError(
                    f"SpectrumMetadata.channels {self.spectrum_metadata.channels} "
                    f"does not match SpectrumResult.channels {self.channels}."
                )


@dataclass(slots=True)
class SpectrumResultStore:
    """Container for the results and metadata of one calculation pipeline.

    The ``results`` mapping contains one :class:`SpectrumResult` per successfully returned channel
    tuple. It can contain fewer entries than ``spectra_metadata`` because metadata is created for
    every planned spectrum before individual spectra are evaluated at their isolated failure
    boundaries.

    Selecting results creates another store that shares the original result and metadata objects.
    The complete planned metadata is retained by the selected store, including metadata for results
    that were not selected.

    This class handles collection-level storage and validation only. Numerical accumulation,
    uncertainty estimation, and result finalization are performed elsewhere.

    Attributes
    ----------
    results : dict[tuple[int, ...], SpectrumResult]
        Mapping from channel tuples to successfully returned spectrum results. For example,
        ``(0, 0)`` identifies the second-order auto-spectrum of channel 0, while ``(0, 1)``
        identifies a second-order cross-spectrum between channels 0 and 1.
    calculation_metadata : CalculationMetadata | None
        Calculation-wide metadata shared by every stored result. This is ``None`` only for manually
        constructed stores that omit metadata.
    spectra_metadata : Mapping[tuple[int, ...], SpectrumMetadata]
        Immutable mapping containing metadata for every spectrum planned by the calculation, keyed
        by the same channel tuples used by ``results``. Its iteration order is the resolved request
        order.

        This mapping may contain keys absent from ``results`` when an individual spectrum failed or
        when the store was created through result selection. It is empty when
        ``calculation_metadata`` is ``None``.

    Notes
    -----
    For pipeline-created stores, every result shares ``calculation_metadata`` by identity with the
    store and shares ``spectrum_metadata`` by identity with the corresponding value in
    ``spectra_metadata``.
    """

    results: dict[tuple[int, ...], SpectrumResult] = field(default_factory=dict)

    calculation_metadata: CalculationMetadata | None = None
    spectra_metadata: Mapping[tuple[int, ...], SpectrumMetadata] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        """Validate metadata and all initially supplied results."""
        if self.calculation_metadata is not None and not isinstance(
            self.calculation_metadata,
            CalculationMetadata,
        ):
            raise TypeError("calculation_metadata must be a CalculationMetadata object or None.")

        if not isinstance(self.spectra_metadata, Mapping):
            raise TypeError(
                "spectra_metadata must be a mapping from channel tuples "
                "to SpectrumMetadata objects."
            )

        if not isinstance(self.spectra_metadata, MappingProxyType):
            self.spectra_metadata = MappingProxyType(dict(self.spectra_metadata))

        for channels, metadata in self.spectra_metadata.items():
            if not isinstance(metadata, SpectrumMetadata):
                raise TypeError("spectra_metadata values must be SpectrumMetadata objects.")

            if channels != metadata.channels:
                raise ValueError(
                    f"Metadata key {channels} does not match "
                    f"SpectrumMetadata.channels {metadata.channels}."
                )
        if self.calculation_metadata is None:
            if self.spectra_metadata:
                raise ValueError(
                    "spectra_metadata must be empty when calculation_metadata is None."
                )
        else:
            described_spectra = tuple(self.spectra_metadata)
            if described_spectra != self.calculation_metadata.requested_spectra:
                raise ValueError(
                    "spectra_metadata must describe every requested spectrum "
                    "in resolved request order."
                )

        for channels, result in self.results.items():
            self._validate_result(result, channels)

    def __contains__(self, channels: object) -> bool:
        """Return whether a result exists for a channel tuple."""
        return channels in self.results

    def __iter__(self) -> Iterator[SpectrumResult]:
        """Iterate over results, rather than channel-tuple keys, in insertion order."""
        return iter(self.results.values())

    def __len__(self) -> int:
        """Return the number of stored results."""
        return len(self.results)

    def __getitem__(self, channels: tuple[int, ...]) -> SpectrumResult:
        """Return the result for a channel tuple.

        Raises
        ------
        KeyError
            If no result exists for ``channels``.
        """
        return self.results[channels]

    def _validate_result(
        self,
        result: SpectrumResult,
        channels: tuple[int, ...] | None = None,
    ) -> None:
        """Validate one result against the store and its shared metadata."""
        if not isinstance(result, SpectrumResult):
            raise TypeError(
                "SpectrumResultStore values must be SpectrumResult objects; "
                f"received {type(result).__name__}."
            )

        if channels is not None and channels != result.channels:
            raise ValueError(
                f"Result key {channels} does not match SpectrumResult.channels {result.channels}."
            )

        if result.calculation_metadata is not self.calculation_metadata:
            raise ValueError(
                f"Result {result.channels} does not share the store's calculation metadata."
            )

        if self.calculation_metadata is not None:
            try:
                expected_metadata = self.spectra_metadata[result.channels]
            except KeyError as exc:
                raise ValueError(
                    f"No SpectrumMetadata exists for result {result.channels}."
                ) from exc

            if result.spectrum_metadata is not expected_metadata:
                raise ValueError(
                    f"Result {result.channels} does not share its "
                    "SpectrumMetadata with the result store."
                )

    def add(self, result: SpectrumResult) -> None:
        """Add a result, replacing an existing result with the same channels.

        Parameters
        ----------
        result : SpectrumResult
            Result stored under its ``result.channels`` tuple.
        """

        self._validate_result(result)
        self.results[result.channels] = result

    def select(self, channels: Iterable[tuple[int, ...]]) -> SpectrumResultStore:
        """Return a new store containing the selected results.

        The new store shares its :class:`SpectrumResult` objects and their arrays with this store.

        Parameters
        ----------
        channels : Iterable[tuple[int, ...]]
            Channel tuples to include, in the desired output order.

        Returns
        -------
        SpectrumResultStore
            New store containing the selected results.

        Raises
        ------
        ValueError
            If any requested channel tuple is absent.
        """
        selected: dict[tuple[int, ...], SpectrumResult] = {}

        for channel_tuple in channels:
            try:
                selected[channel_tuple] = self.results[channel_tuple]
            except KeyError as exc:
                raise ValueError(
                    f"No spectrum result exists for channels {channel_tuple}."
                ) from exc

        return SpectrumResultStore(
            results=selected,
            calculation_metadata=self.calculation_metadata,
            spectra_metadata=self.spectra_metadata,
        )

    def select_by_order(self, order: int) -> SpectrumResultStore:
        """Return all results with the specified spectrum order.

        An empty store is returned when no matching results exist.

        Parameters
        ----------
        order : int
            Spectrum order from 1 through 4. NumPy integers are accepted; Booleans are rejected.

        Returns
        -------
        SpectrumResultStore
            New store containing matching results in their original order.

        Raises
        ------
        TypeError
            If ``order`` is not an integer.
        ValueError
            If ``order`` is outside the range 1 through 4.
        """
        if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)):
            raise TypeError("order must be an integer.")

        order = int(order)

        if not 1 <= order <= 4:
            raise ValueError("order must be between 1 and 4.")

        return self.select([result.channels for result in self if result.order == order])

    def select_by_channel(self, channel: int) -> SpectrumResultStore:
        """Return all results involving the specified data channel.

        A result matches when ``channel`` occurs anywhere in its channel tuple. An empty store is
        returned when no matching results exist.

        Parameters
        ----------
        channel : int
            Nonnegative channel index. NumPy integers are accepted; Booleans are rejected.

        Returns
        -------
        SpectrumResultStore
            New store containing matching results in their original order.

        Raises
        ------
        TypeError
            If ``channel`` is not an integer.
        ValueError
            If ``channel`` is negative.
        """
        if isinstance(channel, (bool, np.bool_)) or not isinstance(channel, (int, np.integer)):
            raise TypeError("channel must be an integer.")

        channel = int(channel)

        if channel < 0:
            raise ValueError("channel must be nonnegative.")

        return self.select([result.channels for result in self if channel in result.channels])
