from types import SimpleNamespace

import numpy as np
import pytest
import torch

from signalsnap_pytorch._core.cumulants import (
    c2_factorized,
    c3_factorized,
    c4_factorized,
    gather_s3_third_factor,
)
from signalsnap_pytorch._core.fft import WindowBuffer
from signalsnap_pytorch._core.planning import (
    RepetitionPlan,
    SampledFrequencyPlan,
    TimestampFrequencyPlan,
)
from signalsnap_pytorch._core.spectra import (
    ChannelCoefficients,
    CoefficientBatch,
    ThirdOrderCoefficients,
    build_coefficient_batch,
    build_third_order_cache,
    build_timestamp_third_order_cache,
    compute_spectral_estimates,
)


def _center(values: torch.Tensor) -> torch.Tensor:
    """Center coefficients over the coefficient-window axis only."""

    return values - values.mean(dim=-2, keepdim=True)


@pytest.fixture
def centered_coefficients() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20260806)
    shape = (2, 3, 5, 4)
    return tuple(
        _center(torch.randn(shape, dtype=torch.float64, generator=generator))
        for _ in range(4)
    )


def _stack_legacy_calls(function, *factors: torch.Tensor) -> torch.Tensor:
    """Evaluate each realization and physical estimate through the legacy shape."""

    realization_results = []
    for realization in range(factors[0].shape[0]):
        estimate_results = []
        for estimate in range(factors[0].shape[1]):
            legacy_factors = tuple(
                factor[realization, estimate].unsqueeze(0) for factor in factors
            )
            estimate_results.append(function(*legacy_factors).squeeze(0))
        realization_results.append(torch.stack(estimate_results))
    return torch.stack(realization_results)


def test_c2_preserves_realization_and_physical_estimate_axes(centered_coefficients):
    x, y, _, _ = centered_coefficients
    m = x.shape[-2]

    actual = c2_factorized(m, x, y)
    expected = _stack_legacy_calls(lambda a, b: c2_factorized(m, a, b), x, y)

    assert actual.shape == (2, 3, 4)
    torch.testing.assert_close(actual, expected)


def test_c3_preserves_realization_and_physical_estimate_axes(centered_coefficients):
    x, y, z, _ = centered_coefficients
    m = x.shape[-2]
    closing = z[..., :, :, None].expand(-1, -1, -1, -1, z.shape[-1])

    actual = c3_factorized(m, x, y, closing)
    expected = _stack_legacy_calls(
        lambda a, b, c: c3_factorized(m, a, b, c),
        x,
        y,
        closing,
    )

    assert actual.shape == (2, 3, 4, 4)
    torch.testing.assert_close(actual, expected)


def test_c4_preserves_realization_and_physical_estimate_axes(centered_coefficients):
    x, y, z, w = centered_coefficients
    m = x.shape[-2]

    actual = c4_factorized(m, x, y, z, w)
    expected = _stack_legacy_calls(
        lambda a, b, c, d: c4_factorized(m, a, b, c, d),
        x,
        y,
        z,
        w,
    )

    assert actual.shape == (2, 3, 4, 4)
    torch.testing.assert_close(actual, expected)


def test_third_order_gather_preserves_all_leading_axes():
    coefficients = torch.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
    target_indices = torch.tensor([[0, 2], [6, 1]])

    actual = gather_s3_third_factor(coefficients, target_indices)
    expected = torch.stack(
        [
            torch.stack(
                [coefficients[..., frequency] for frequency in row],
                dim=-1,
            )
            for row in target_indices
        ],
        dim=-2,
    )

    assert actual.shape == (2, 3, 5, 2, 2)
    torch.testing.assert_close(actual, expected)


def test_third_order_cache_retains_only_unique_valid_closing_coefficients():
    frequency_plan = SampledFrequencyPlan(
        full_fft_frequencies=np.arange(8, dtype=np.float64),
        band_frequencies=np.arange(4, 8, dtype=np.float64),
        band_start=4,
        band_stop=8,
    )
    runtime = SimpleNamespace(device=torch.device("cpu"))

    cache = build_third_order_cache(runtime, frequency_plan)

    axis_offsets = torch.arange(4)
    dense_fft_indices = 4 - (
        axis_offsets[:, None] + axis_offsets[None, :]
    )
    expected_valid = (dense_fft_indices >= 0) & (dense_fft_indices < 8)
    expected_unique = torch.unique(dense_fft_indices[expected_valid], sorted=True)

    torch.testing.assert_close(cache.closing_fft_indices, expected_unique)
    torch.testing.assert_close(cache.valid_mask, expected_valid)
    torch.testing.assert_close(
        cache.closing_fft_indices[cache.gather_indices[cache.valid_mask]],
        dense_fft_indices[expected_valid],
    )
    assert cache.closing_fft_indices.numel() < cache.gather_indices.numel()


def test_timestamp_third_order_cache_maps_every_pair_to_compact_closing_frequency():
    grid_indices = np.arange(-2, 3, dtype=np.int64)
    frequency_plan = TimestampFrequencyPlan(
        actual_df=0.1,
        grid_indices=grid_indices,
        band_frequencies=grid_indices.astype(np.float64) * 0.1,
    )
    runtime = SimpleNamespace(device=torch.device("cpu"))

    cache = build_timestamp_third_order_cache(runtime, frequency_plan)

    target_grid_indices = -(
        grid_indices[:, None] + grid_indices[None, :]
    )
    expected_closing_indices = np.arange(-4, 5, dtype=np.int64)
    gathered_frequencies = cache.closing_frequencies[
        cache.gather_indices.numpy()
    ]

    np.testing.assert_array_equal(
        cache.closing_frequencies,
        expected_closing_indices.astype(np.float64) * 0.1,
    )
    np.testing.assert_array_equal(
        gathered_frequencies,
        target_grid_indices.astype(np.float64) * 0.1,
    )
    assert cache.gather_indices.dtype == torch.long
    assert cache.gather_indices.device == torch.device("cpu")
    assert torch.all(cache.valid_mask)
    assert cache.closing_frequencies.size < cache.gather_indices.numel()


def test_empty_compact_third_order_coefficients_produce_placeholder_grid():
    coefficients = ThirdOrderCoefficients(
        values=torch.empty(2, 3, 4, 0),
        gather_indices=torch.zeros(5, 5, dtype=torch.int64),
        valid_mask=torch.zeros(5, 5, dtype=torch.bool),
    )

    gathered = coefficients.gathered_centered_values()

    assert gathered.shape == (2, 3, 4, 5, 5)
    assert torch.count_nonzero(gathered) == 0


def test_third_order_coefficients_center_only_over_window_axis():
    values = torch.arange(2 * 3 * 4 * 2, dtype=torch.float64).reshape(2, 3, 4, 2)
    gather_indices = torch.tensor([[0, 1], [1, 0]])
    coefficients = ThirdOrderCoefficients(
        values=values,
        gather_indices=gather_indices,
        valid_mask=torch.ones(2, 2, dtype=torch.bool),
    )

    actual = coefficients.gathered_centered_values()
    centered = values - values.mean(dim=-2, keepdim=True)
    expected = centered[..., gather_indices]

    assert actual.shape == (2, 3, 4, 2, 2)
    torch.testing.assert_close(actual, expected)


def test_coefficient_batch_is_compact_and_independent_of_full_fft_storage():
    full_fft = torch.arange(2 * 3 * 4 * 8, dtype=torch.float64).reshape(2, 3, 4, 8)
    frequency_plan = SampledFrequencyPlan(
        full_fft_frequencies=np.arange(8, dtype=np.float64),
        band_frequencies=np.arange(3, 6, dtype=np.float64),
        band_start=3,
        band_stop=6,
    )
    runtime = SimpleNamespace(device=torch.device("cpu"))
    third_order_cache = build_third_order_cache(runtime, frequency_plan)

    batch = build_coefficient_batch(
        frequency_plan=frequency_plan,
        coeffs_by_channel={7: full_fft},
        third_order_cache=third_order_cache,
    )
    coefficients = batch.by_channel[7]
    expected_dc = full_fft[..., 4].clone()
    expected_output = full_fft[..., 3:6].clone()
    expected_closing = full_fft[..., third_order_cache.closing_fft_indices].clone()

    full_fft.fill_(float("nan"))

    assert coefficients.dc.shape == (2, 3, 4)
    assert coefficients.output.shape == (2, 3, 4, 3)
    assert coefficients.third_order is not None
    assert coefficients.third_order.values.shape[-1] < 3 * 3
    torch.testing.assert_close(coefficients.dc, expected_dc)
    torch.testing.assert_close(coefficients.output, expected_output)
    torch.testing.assert_close(coefficients.third_order.values, expected_closing)


def test_channel_coefficients_cache_output_centered_only_over_windows():
    output = torch.arange(2 * 3 * 4 * 5, dtype=torch.float64).reshape(2, 3, 4, 5)
    coefficients = ChannelCoefficients(
        dc=torch.zeros(2, 3, 4),
        output=output,
    )

    centered = coefficients.centered_output()

    assert coefficients.centered_output() is centered
    centered_mean = centered.mean(dim=-2)
    torch.testing.assert_close(centered_mean, torch.zeros_like(centered_mean))
    torch.testing.assert_close(
        coefficients.centered_output(conjugated=True),
        torch.conj(centered),
    )


def test_repetition_plan_yields_bounded_stable_realization_ids():
    plan = RepetitionPlan(count=8, batch_size=3, resolved_seed=42)

    batches = list(plan.iter_batches())

    assert [tuple(batch) for batch in batches] == [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7),
    ]
    assert all(len(batch) <= plan.batch_size for batch in batches)


def test_average_is_applied_to_spectra_instead_of_coefficients():
    output = torch.tensor(
        [
            [[[-1.0], [1.0]]],
            [[[-3.0], [3.0]]],
        ],
        dtype=torch.float64,
    )
    window_buffer = WindowBuffer(
        window=torch.ones(1, dtype=torch.float64),
        norm_all_orders=tuple(torch.tensor(1.0) for _ in range(4)),
    )
    runtime = SimpleNamespace(
        window_plan=SimpleNamespace(windows_per_estimate=2),
    )
    coefficient_batch = CoefficientBatch(
        by_channel={
            0: ChannelCoefficients(
                dc=torch.zeros(2, 1, 2, dtype=torch.float64),
                output=output,
            )
        }
    )

    estimates = compute_spectral_estimates(
        channels=(0, 0),
        coefficient_batch=coefficient_batch,
        window_buffer=window_buffer,
        runtime=runtime,
    )
    spectrum_average = estimates.mean(dim=0)

    averaged_coefficient_batch = CoefficientBatch(
        by_channel={
            0: ChannelCoefficients(
                dc=torch.zeros(1, 1, 2, dtype=torch.float64),
                output=output.mean(dim=0, keepdim=True),
            )
        }
    )
    coefficient_average = compute_spectral_estimates(
        channels=(0, 0),
        coefficient_batch=averaged_coefficient_batch,
        window_buffer=window_buffer,
        runtime=runtime,
    ).squeeze(0)

    torch.testing.assert_close(
        spectrum_average,
        torch.tensor([[10.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        coefficient_average,
        torch.tensor([[8.0]], dtype=torch.float64),
    )
