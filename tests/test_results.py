import numpy as np
import pytest

from signalsnap_pytorch import SpectrumResult, SpectrumResultStore


def make_result(channels: tuple[int, ...]) -> SpectrumResult:
    frequency_points = 3
    order = len(channels)

    shape = {
        1: (1,),
        2: (frequency_points,),
        3: (frequency_points, frequency_points),
        4: (frequency_points, frequency_points),
    }[order]

    freq = np.array([0.0]) if order == 1 else np.arange(frequency_points)

    return SpectrumResult(
        channels=channels,
        freq=freq,
        freq_unit="Hz",
        spectrum=np.zeros(shape, dtype=complex),
    )


@pytest.mark.parametrize(
    ("freq", "spectrum", "spectrum_uncertainty", "message"),
    [
        pytest.param(
            np.zeros((2, 2)),
            np.zeros(2, dtype=complex),
            None,
            "Frequency axis must be one-dimensional",
            id="frequency-dimensions",
        ),
        pytest.param(
            np.zeros(2),
            np.zeros(3, dtype=complex),
            None,
            "spectrum has shape",
            id="spectrum-shape",
        ),
        pytest.param(
            np.zeros(2),
            np.zeros(2, dtype=complex),
            np.zeros(3, dtype=complex),
            "Spectrum uncertainty must have the same shape",
            id="uncertainty-shape",
        ),
    ],
)
def test_spectrum_result_rejects_inconsistent_array_shapes(
    freq,
    spectrum,
    spectrum_uncertainty,
    message,
):
    with pytest.raises(ValueError, match=message):
        SpectrumResult(
            channels=(0, 0),
            freq=freq,
            freq_unit="Hz",
            spectrum=spectrum,
            spectrum_uncertainty=spectrum_uncertainty,
        )


def test_result_store_selects_by_order():
    order_1 = make_result((0,))
    order_2_auto = make_result((0, 0))
    order_2_cross = make_result((0, 1))

    store = SpectrumResultStore()
    store.add(order_1)
    store.add(order_2_auto)
    store.add(order_2_cross)

    selected = store.select_by_order(2)

    assert list(selected) == [order_2_auto, order_2_cross]


def test_result_store_selects_by_channel():
    channel_0 = make_result((0, 0))
    cross = make_result((0, 1))
    channel_1 = make_result((1, 1))

    store = SpectrumResultStore()
    store.add(channel_0)
    store.add(cross)
    store.add(channel_1)

    selected = store.select_by_channel(0)

    assert list(selected) == [channel_0, cross]


def test_result_store_selection_can_be_empty():
    store = SpectrumResultStore()
    store.add(make_result((0, 0)))

    assert list(store.select_by_order(3)) == []
    assert list(store.select_by_channel(2)) == []


@pytest.mark.parametrize("order", [0, 5, -1, True, 2.0])
def test_result_store_rejects_invalid_order(order):
    store = SpectrumResultStore()

    with pytest.raises((TypeError, ValueError)):
        store.select_by_order(order)


@pytest.mark.parametrize("channel", [-1, True, 1.5])
def test_result_store_rejects_invalid_channel(channel):
    store = SpectrumResultStore()

    with pytest.raises((TypeError, ValueError)):
        store.select_by_channel(channel)
