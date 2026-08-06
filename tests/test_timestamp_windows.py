from types import SimpleNamespace

import numpy as np
import pytest
import torch

from signalsnap_pytorch._core.fft import (
    DefaultTimestampWindow,
    LegacyTimestampWindow,
    prepare_default_timestamp_window,
    prepare_legacy_timestamp_window,
    prepare_timestamp_window,
)


def _reference_default_window(normalized_times):
    sigma = 0.14

    def gaussian(values):
        return np.exp(-((values - 0.5) / (2.0 * sigma)) ** 2)

    edge = gaussian(0.0)
    denominator = gaussian(1.0) + gaussian(-1.0)
    raw = gaussian(normalized_times) - edge * (
        gaussian(normalized_times + 1.0)
        + gaussian(normalized_times - 1.0)
    ) / denominator
    midpoint_raw = gaussian(0.5) - edge * (
        gaussian(1.5) + gaussian(-0.5)
    ) / denominator
    return raw / midpoint_raw


def _runtime(duration, dtype=torch.float64):
    return SimpleNamespace(
        window_plan=SimpleNamespace(duration=duration),
        real_dtype=dtype,
        device=torch.device("cpu"),
    )


def _reference_legacy_window(relative_times, duration):
    reference_points = 70
    length = reference_points + 1
    sigma = 0.14
    reference_dt = duration / reference_points

    def gaussian(values):
        return np.exp(
            -(
                (values - reference_points / 2.0)
                / (2.0 * length * sigma)
            )
            ** 2
        )

    def raw(values):
        edge = gaussian(-0.5)
        denominator = gaussian(-0.5 + length) + gaussian(-0.5 - length)
        return gaussian(values) - edge * (
            gaussian(values + length) + gaussian(values - length)
        ) / denominator

    reference_grid = np.linspace(0.0, reference_points, reference_points)
    reference_raw = raw(reference_grid)
    norm2 = reference_dt * np.sum(reference_raw**2)
    scale = 1.0 / np.sqrt(norm2)
    values = raw(np.asarray(relative_times) / reference_dt) * scale
    normalizations = tuple(
        reference_dt * np.sum((reference_raw * scale) ** order)
        for order in range(1, 5)
    )
    return values, normalizations


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_default_timestamp_window_matches_independent_formula(dtype):
    duration = 2.5
    normalized_times = np.array([0.0, 0.125, 0.5, 0.875, 1.0])
    relative_times = torch.tensor(normalized_times * duration, dtype=dtype)
    prepared = prepare_default_timestamp_window(_runtime(duration, dtype))

    actual = prepared.evaluate(relative_times)
    expected = torch.tensor(_reference_default_window(normalized_times), dtype=dtype)

    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[[0, -1]], torch.zeros(2, dtype=dtype), atol=1e-6, rtol=0)
    assert actual[2].item() == 1.0


def test_default_timestamp_normalizations_match_higher_order_quadrature():
    duration = 2.5
    prepared = prepare_default_timestamp_window(_runtime(duration))
    nodes, weights = np.polynomial.legendre.leggauss(512)
    window = _reference_default_window((nodes + 1.0) / 2.0)

    expected = tuple(
        duration * 0.5 * np.dot(weights, window**order)
        for order in range(1, 5)
    )

    for order, expected_norm in enumerate(expected, start=1):
        assert prepared.norm(order).item() == pytest.approx(expected_norm, rel=1e-13)


def test_default_timestamp_normalizations_scale_with_duration():
    first = prepare_default_timestamp_window(_runtime(2.5))
    second = prepare_default_timestamp_window(_runtime(5.0))

    for order in range(1, 5):
        torch.testing.assert_close(second.norm(order), 2.0 * first.norm(order))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_legacy_timestamp_window_matches_independent_v1_formula(dtype):
    duration = 3.5
    relative_times = np.array([0.0, 0.25, 1.75, 3.0])
    expected_values, expected_norms = _reference_legacy_window(
        relative_times,
        duration,
    )
    prepared = prepare_legacy_timestamp_window(_runtime(duration, dtype))

    actual = prepared.evaluate(torch.tensor(relative_times, dtype=dtype))

    assert actual.dtype == dtype
    torch.testing.assert_close(actual, torch.tensor(expected_values, dtype=dtype))
    for order, expected_norm in enumerate(expected_norms, start=1):
        assert prepared.norm(order).item() == pytest.approx(
            expected_norm,
            rel=2e-6 if dtype == torch.float32 else 1e-13,
        )


def test_legacy_second_order_reference_normalization_is_one():
    prepared = prepare_legacy_timestamp_window(_runtime(3.5))

    assert prepared.norm(2).item() == pytest.approx(1.0, rel=1e-14)


@pytest.mark.parametrize(
    ("old_window", "expected_type"),
    [
        pytest.param(False, DefaultTimestampWindow, id="default"),
        pytest.param(True, LegacyTimestampWindow, id="legacy"),
    ],
)
def test_prepare_timestamp_window_dispatches_selected_convention(
    old_window,
    expected_type,
):
    runtime = _runtime(2.5)
    runtime.old_window = old_window

    assert isinstance(prepare_timestamp_window(runtime), expected_type)
