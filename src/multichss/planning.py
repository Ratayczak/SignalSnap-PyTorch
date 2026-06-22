# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass
from .results import SpectrumResultStore, SpectrumResult
from .configurators import SpectrumConfig, CrossConfig


@dataclass(frozen=True, slots=True)
class SpectrumTask:
    """Description of one spectrum that should be calculated.

    A task is the normalized representation of a user request after
    :class:`SpectrumConfig` and :class:`CrossConfig` have been expanded.
    It identifies one polyspectrum order and the channel tuple that should
    be used for that calculation.

    Parameters
    ----------
    order : int
        The polyspectrum order to calculate.
    channels : tuple[int, ...]
        The channel indices used by this calculation. Auto-spectra repeat
        the same channel once per order, e.g. ``(0, 0)`` for the
        second-order spectrum of channel 0 and ``(0, 0, 0)`` for the
        third-order spectrum. Cross-spectra are represented by the
        configured channel tuple, e.g. ``(0, 1)`` for a second-order
        cross-spectrum.
    """

    order: int
    channels: tuple[int, ...]


def build_spectrum_tasks(
    spectrum_config: SpectrumConfig,
    cross_config: CrossConfig,
    selected: list[int],
) -> list[SpectrumTask]:
    """Build the concrete spectrum tasks requested by the configuration.

    Expands the high-level configuration into one :class:`SpectrumTask` per
    spectrum that should be calculated. If ``order_in`` is ``"all"``, orders
    1 through 4 are requested. Auto-correlation tasks are generated for each
    selected channel when ``cross_config.auto_corr`` is enabled. Cross tasks
    are generated from ``cross_corr_2``, ``cross_corr_3``, and
    ``cross_corr_4`` when their corresponding orders are requested.

    Third-order tasks are skipped when ``spectrum_config.f_min`` is negative.

    Parameters
    ----------
    spectrum_config : SpectrumConfig
        Configuration for spectrum order, frequency bounds, and numerical
        calculation settings.
    cross_config : CrossConfig
        Configuration describing whether auto-spectra and which cross-spectra
        should be calculated.
    selected : list[int]
        Channel indices for which auto-spectra should be generated.

    Returns
    -------
    list[SpectrumTask]
        Ordered list of concrete spectrum calculations to perform.
    """
    orders = (
        [1, 2, 3, 4]
        if spectrum_config.order_in == "all"
        else list(spectrum_config.order_in)
    )

    if spectrum_config.f_min < 0 and 3 in orders:
        print(
            "For negative frequencies in order 3 use s3_calc and positive frequencies\n"
        )
        print("Example: f_min=0, f_max=5, s3_calc='1/2'")
        orders.remove(3)

    tasks: list[SpectrumTask] = []

    if cross_config.auto_corr:
        for channel in selected:
            for order in orders:
                channels = (channel,) * order
                tasks.append(SpectrumTask(order=order, channels=channels))

    cross_specs = {
        2: cross_config.cross_corr_2 or [],
        3: cross_config.cross_corr_3 or [],
        4: cross_config.cross_corr_4 or [],
    }

    for order, channel_groups in cross_specs.items():
        if order not in orders:
            continue
        if spectrum_config.f_min < 0 and order == 3:
            continue
        for channels in channel_groups:
            channels = tuple(channels)
            if len(channels) != order:
                raise ValueError(
                    f"Order {order} spectra require {order} channels, got {channels}."
                )
            tasks.append(SpectrumTask(channels=channels, order=order))

    return tasks


def initialize_result_store(tasks: list[SpectrumTask]) -> SpectrumResultStore:
    """Create an empty result store for a list of spectrum tasks.

    Each task is converted into a :class:`SpectrumResult` with matching order
    and channels. The returned store contains result containers only; device
    buffers, frequency axes, accumulated spectra, and error estimates are
    filled later by the calculation pipeline.

    Parameters
    ----------
    tasks : list[SpectrumTask]
        Spectrum tasks that should receive corresponding result containers.

    Returns
    -------
    SpectrumResultStore
        Store containing one empty :class:`SpectrumResult` per task.
    """
    store = SpectrumResultStore()
    for task in tasks:
        store.add(SpectrumResult(order=task.order, channels=task.channels))
    return store
