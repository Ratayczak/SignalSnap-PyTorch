from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from signalsnap_pytorch import (
    CalculationMetadata,
    SpectrumMetadata,
    SpectrumResult,
    SpectrumResultStore,
)


def make_spectrum_metadata(channels: tuple[int, ...]) -> SpectrumMetadata:
    return SpectrumMetadata(
        channels=channels,
        frequency_view="sampled_fft",
        effective_f_min=0.0,
        effective_f_max=2.0,
        normalization_convention="sampled_discrete_default",
        closing_frequency_support=(
            "sampled_fft" if len(channels) == 3 else "not_applicable"
        ),
    )


def make_calculation_metadata(
    requested_spectra: tuple[tuple[int, ...], ...],
) -> CalculationMetadata:
    return CalculationMetadata(
        channel_kinds=("sampled", "timestamped"),
        active_channels=(0, 1),
        requested_spectra=requested_spectra,
        observation_start=1.0,
        observation_stop=17.0,
        time_unit="s",
        frequency_unit="Hz",
        requested_df=1.1,
        actual_df=1.0,
        requested_f_min=0.0,
        requested_f_max=2.0,
        window_duration=1.0,
        unshifted_offset=0.0,
        shifted_offset=0.5,
        window_convention="confined_gaussian",
        photon_weighting="exponential",
        exponential_scale=2.0,
        repetition_count=5,
        requested_repetition_batch_size=3,
        resolved_repetition_batch_size=3,
        user_seed=7,
        resolved_seed=7,
        requested_m=4,
        effective_m=4,
        requested_m_var=3,
        effective_m_var=3,
        uncertainty_estimation="short_term",
        unshifted_physical_estimate_count=4,
        shifted_physical_estimate_count=3,
        unshifted_coefficient_window_count=16,
        shifted_coefficient_window_count=12,
        real_dtype="float64",
        complex_dtype="complex128",
        requested_device="cpu",
        resolved_device="cpu",
    )


def make_result(
    channels: tuple[int, ...],
    calculation_metadata: CalculationMetadata | None = None,
    spectrum_metadata: SpectrumMetadata | None = None,
) -> SpectrumResult:
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
        calculation_metadata=calculation_metadata,
        spectrum_metadata=spectrum_metadata,
    )


def make_metadata_store() -> SpectrumResultStore:
    spectra_metadata = {
        channels: make_spectrum_metadata(channels)
        for channels in ((0,), (0, 1), (1, 1, 1))
    }
    calculation_metadata = make_calculation_metadata(
        tuple(spectra_metadata)
    )
    results = {
        channels: make_result(
            channels,
            calculation_metadata,
            metadata,
        )
        for channels, metadata in spectra_metadata.items()
    }
    return SpectrumResultStore(
        results=results,
        calculation_metadata=calculation_metadata,
        spectra_metadata=spectra_metadata,
    )


def test_metadata_value_objects_are_frozen_and_slotted():
    spectrum_metadata = make_spectrum_metadata((0, 1))
    calculation_metadata = make_calculation_metadata(((0, 1),))

    with pytest.raises(FrozenInstanceError):
        spectrum_metadata.effective_f_min = -1.0

    with pytest.raises(FrozenInstanceError):
        calculation_metadata.actual_df = 2.0

    with pytest.raises(TypeError):
        vars(spectrum_metadata)

    with pytest.raises(TypeError):
        vars(calculation_metadata)

    assert spectrum_metadata.order == 2


@pytest.mark.parametrize("field", ["calculation_metadata", "spectrum_metadata"])
def test_spectrum_result_requires_both_metadata_objects(field):
    calculation_metadata = make_calculation_metadata(((0, 0),))
    spectrum_metadata = make_spectrum_metadata((0, 0))
    values = {
        "calculation_metadata": calculation_metadata,
        "spectrum_metadata": spectrum_metadata,
    }
    values[field] = None

    with pytest.raises(ValueError, match="must either both be provided"):
        make_result((0, 0), **values)


def test_spectrum_result_rejects_metadata_for_different_spectrum():
    calculation_metadata = make_calculation_metadata(((0, 0), (0, 1)))

    with pytest.raises(ValueError, match="does not match SpectrumResult.channels"):
        make_result(
            (0, 0),
            calculation_metadata,
            make_spectrum_metadata((0, 1)),
        )


def test_metadata_store_shares_metadata_objects_with_results():
    store = make_metadata_store()

    for result in store:
        assert result.calculation_metadata is store.calculation_metadata
        assert result.spectrum_metadata is store.spectra_metadata[result.channels]


def test_result_store_spectra_metadata_is_immutable():
    store = make_metadata_store()

    with pytest.raises(TypeError):
        store.spectra_metadata[(0,)] = make_spectrum_metadata((0,))


@pytest.mark.parametrize(
    "selector",
    [
        lambda store: store.select([(0, 1)]),
        lambda store: store.select_by_order(2),
        lambda store: store.select_by_channel(1),
        lambda store: store.select_by_order(4),
    ],
)
def test_result_store_selections_preserve_metadata_identity(selector):
    store = make_metadata_store()
    selected = selector(store)

    assert selected.calculation_metadata is store.calculation_metadata
    assert selected.spectra_metadata is store.spectra_metadata


def test_result_store_rejects_equal_but_unrelated_calculation_metadata():
    store = make_metadata_store()
    unrelated_metadata = replace(store.calculation_metadata)
    spectrum_metadata = store.spectra_metadata[(0,)]
    unrelated_result = make_result(
        spectrum_metadata.channels,
        unrelated_metadata,
        spectrum_metadata,
    )

    with pytest.raises(ValueError, match="does not share.*calculation metadata"):
        store.add(unrelated_result)


def test_result_store_rejects_equal_but_unrelated_spectrum_metadata():
    store = make_metadata_store()
    unrelated_spectrum_metadata = replace(store.spectra_metadata[(0,)])
    unrelated_result = make_result(
        unrelated_spectrum_metadata.channels,
        store.calculation_metadata,
        unrelated_spectrum_metadata,
    )

    with pytest.raises(ValueError, match="does not share its SpectrumMetadata"):
        store.add(unrelated_result)


def test_result_store_rejects_incomplete_spectra_metadata():
    calculation_metadata = make_calculation_metadata(((0,), (0, 0)))

    with pytest.raises(ValueError, match="every requested spectrum"):
        SpectrumResultStore(
            calculation_metadata=calculation_metadata,
            spectra_metadata={(0,): make_spectrum_metadata((0,))},
        )


def test_manual_results_and_stores_do_not_require_metadata():
    result = make_result((0, 0))
    store = SpectrumResultStore()

    store.add(result)

    assert result.calculation_metadata is None
    assert result.spectrum_metadata is None
    assert store[(0, 0)] is result


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
