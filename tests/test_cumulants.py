from types import SimpleNamespace

import numpy as np
import pytest
import torch

from signalsnap_pytorch._core.cumulants import (
    c2_factorized,
    c3_factorized,
    c4_factorized,
)
from signalsnap_pytorch._core.plans import (
    DirectFrequencyPlan,
    FFTFrequencyPlan,
    RepetitionPlan,
    SampledChannelPlan,
    WindowBatch,
)
from signalsnap_pytorch._core.spectra import (
    ChannelCoefficients,
    ThirdOrderCoefficients,
    _build_coefficient_batch,
    build_third_order_cache,
    build_timestamp_third_order_cache,
    compute_spectral_estimates,
    expand_deterministic_coefficients,
    prepare_sampled_channel_coefficients,
)
from signalsnap_pytorch._core.window import SampledWindow


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


def test_third_order_cache_retains_only_unique_valid_closing_coefficients():
    frequency_plan = FFTFrequencyPlan(
        shifted_full_fft_frequencies=np.arange(8, dtype=np.float64),
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
    frequency_plan = DirectFrequencyPlan(
        actual_df=0.1,
        grid_indices=grid_indices,
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


def test_timestamp_closing_cache_accepts_sampled_view_beyond_fft_support():
    full_frequencies = np.arange(-3.0, 3.0)
    frequency_plan = FFTFrequencyPlan(
        shifted_full_fft_frequencies=full_frequencies,
        band_start=2,
        band_stop=6,
    )
    runtime = SimpleNamespace(
        device=torch.device("cpu"),
        window_plan=SimpleNamespace(duration=1.0),
    )

    cache = build_timestamp_third_order_cache(runtime, frequency_plan)

    output_grid_indices = np.array([-1, 0, 1, 2], dtype=np.int64)
    target_grid_indices = -(
        output_grid_indices[:, None] + output_grid_indices[None, :]
    )
    gathered_frequencies = cache.closing_frequencies[
        cache.gather_indices.numpy()
    ]

    np.testing.assert_array_equal(
        cache.closing_frequencies,
        np.arange(-4.0, 3.0),
    )
    np.testing.assert_array_equal(
        gathered_frequencies,
        target_grid_indices.astype(np.float64),
    )
    assert cache.closing_frequencies[0] < full_frequencies[0]
    assert cache.closing_frequencies.size == 2 * len(output_grid_indices) - 1
    assert torch.all(cache.valid_mask)


def test_timestamp_closing_cache_supports_one_point_fft_grid():
    frequency_plan = FFTFrequencyPlan(
        shifted_full_fft_frequencies=np.array([0.0]),
        band_start=0,
        band_stop=1,
    )
    runtime = SimpleNamespace(
        device=torch.device("cpu"),
        window_plan=SimpleNamespace(duration=2.0),
    )

    cache = build_timestamp_third_order_cache(runtime, frequency_plan)

    np.testing.assert_array_equal(cache.closing_frequencies, np.array([0.0]))
    torch.testing.assert_close(cache.gather_indices, torch.zeros((1, 1), dtype=torch.long))
    assert torch.all(cache.valid_mask)


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
    frequency_plan = FFTFrequencyPlan(
        shifted_full_fft_frequencies=np.arange(8, dtype=np.float64),
        band_start=3,
        band_stop=6,
    )
    runtime = SimpleNamespace(device=torch.device("cpu"))
    third_order_cache = build_third_order_cache(runtime, frequency_plan)

    batch = _build_coefficient_batch(
        frequency_plan=frequency_plan,
        coeffs_by_channel={7: full_fft},
        third_order_cache=third_order_cache,
    )
    coefficients = batch[7]
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


def test_sampled_channel_producer_matches_independent_fft_reference():
    data = np.arange(16, dtype=np.float64)
    dt = 0.5
    window_points = 4
    full_frequencies = np.fft.fftshift(np.fft.fftfreq(window_points, d=dt))
    frequency_plan = FFTFrequencyPlan(
        shifted_full_fft_frequencies=full_frequencies,
        band_start=1,
        band_stop=4,
    )
    runtime = SimpleNamespace(
        device=torch.device("cpu"),
        real_dtype=torch.float64,
        window_plan=SimpleNamespace(windows_per_estimate=2),
    )
    sampled_window = SampledWindow(
        window=torch.ones(window_points, dtype=torch.float64),
        norm_all_orders=tuple(torch.tensor(1.0) for _ in range(4)),
    )
    batch = WindowBatch(
        relative_starts=np.array([[0.0, 2.0], [4.0, 6.0]]),
        duration=2.0,
        estimate_count=2,
        shifted=False,
    )
    third_order_cache = build_third_order_cache(runtime, frequency_plan)

    actual = prepare_sampled_channel_coefficients(
        channel_index=3,
        source=data,
        channel_plan=SampledChannelPlan(sample_count=data.size, dt=dt),
        batch=batch,
        frequency_plan=frequency_plan,
        sampled_window=sampled_window,
        runtime=runtime,
        third_order_cache=third_order_cache,
    )

    chunks = data.reshape(2, 2, window_points)
    full_coefficients = np.fft.fftshift(
        np.fft.ifft(chunks, axis=-1) * window_points * dt,
        axes=-1,
    )[None, ...]
    np.testing.assert_allclose(actual.dc.numpy(), full_coefficients[..., 2], atol=1e-14)
    np.testing.assert_allclose(
        actual.output.numpy(),
        full_coefficients[..., 1:4],
        atol=1e-14,
    )
    assert actual.third_order is not None
    np.testing.assert_allclose(
        actual.third_order.values.numpy(),
        full_coefficients[..., third_order_cache.closing_fft_indices.numpy()],
        atol=1e-14,
    )


def test_deterministic_coefficient_expansion_uses_zero_stride_views():
    gather_indices = torch.tensor([[0, 1], [1, 0]])
    valid_mask = torch.ones((2, 2), dtype=torch.bool)
    coefficients = ChannelCoefficients(
        dc=torch.arange(6, dtype=torch.float64).reshape(1, 2, 3),
        output=torch.arange(24, dtype=torch.float64).reshape(1, 2, 3, 4),
        third_order=ThirdOrderCoefficients(
            values=torch.arange(30, dtype=torch.float64).reshape(1, 2, 3, 5),
            gather_indices=gather_indices,
            valid_mask=valid_mask,
        ),
    )
    coefficients_by_channel = {4: coefficients}

    expanded = expand_deterministic_coefficients(
        coefficients_by_channel,
        realization_count=7,
    )
    actual = expanded[4]

    for original_values, expanded_values in (
        (coefficients.dc, actual.dc),
        (coefficients.output, actual.output),
        (coefficients.third_order.values, actual.third_order.values),
    ):
        assert expanded_values.shape[0] == 7
        assert expanded_values.stride(0) == 0
        assert (
            expanded_values.untyped_storage().data_ptr()
            == original_values.untyped_storage().data_ptr()
        )
        torch.testing.assert_close(expanded_values[6], original_values[0])

    assert actual.third_order is not None
    assert actual.third_order.gather_indices is gather_indices
    assert actual.third_order.valid_mask is valid_mask


def test_deterministic_coefficient_expansion_returns_original_for_one_realization():
    coefficients_by_channel = {
        0: ChannelCoefficients(
            dc=torch.ones((1, 2, 3)),
            output=torch.ones((1, 2, 3, 4)),
        )
    }

    assert (
        expand_deterministic_coefficients(coefficients_by_channel, 1)
        is coefficients_by_channel
    )


def test_deterministic_coefficient_expansion_rejects_invalid_axes():
    coefficients_by_channel = {
        0: ChannelCoefficients(
            dc=torch.ones((2, 2, 3)),
            output=torch.ones((2, 2, 3, 4)),
        )
    }

    with pytest.raises(ValueError, match="At least one realization"):
        expand_deterministic_coefficients(coefficients_by_channel, 0)

    with pytest.raises(RuntimeError, match="exactly one realization"):
        expand_deterministic_coefficients(coefficients_by_channel, 3)


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
    sampled_window = SampledWindow(
        window=torch.ones(1, dtype=torch.float64),
        norm_all_orders=tuple(torch.tensor(1.0) for _ in range(4)),
    )
    runtime = SimpleNamespace(
        window_plan=SimpleNamespace(windows_per_estimate=2),
    )
    coefficients_by_channel = {
        0: ChannelCoefficients(
            dc=torch.zeros(2, 1, 2, dtype=torch.float64),
            output=output,
        )
    }

    estimates = compute_spectral_estimates(
        channels=(0, 0),
        coefficients_by_channel=coefficients_by_channel,
        normalization=sampled_window.norm(2),
        runtime=runtime,
    )
    spectrum_average = estimates.mean(dim=0)

    averaged_coefficients_by_channel = {
        0: ChannelCoefficients(
            dc=torch.zeros(1, 1, 2, dtype=torch.float64),
            output=output.mean(dim=0, keepdim=True),
        )
    }
    coefficient_average = compute_spectral_estimates(
        channels=(0, 0),
        coefficients_by_channel=averaged_coefficients_by_channel,
        normalization=sampled_window.norm(2),
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
