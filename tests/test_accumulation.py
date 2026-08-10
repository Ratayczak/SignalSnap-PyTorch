from typing import Literal

import numpy as np
import pytest
import torch

from signalsnap_pytorch._core.accumulation import (
    SpectrumAccumulator,
    _batch_mean_m2,
    accumulate_spectral_estimates,
    finalize_result,
)


def make_accumulator(
    channels: tuple[int, ...] = (0, 0),
    frequency_points: int = 2,
    uncertainty_estimation: Literal["global", "short_term"] = "global",
    m_var: int = 10,
) -> SpectrumAccumulator:
    freq = np.asarray([0.0]) if len(channels) == 1 else np.arange(frequency_points, dtype=float)
    return SpectrumAccumulator(
        channels=channels,
        freq=freq,
        freq_unit="Hz",
        uncertainty_estimation=uncertainty_estimation,
        m_var=m_var,
    )


def accumulate_spectrum(
    accumulator: SpectrumAccumulator,
    spectral_estimate: torch.Tensor,
    shifted: bool = False,
) -> None:
    """Test helper that submits one estimate through the batched accumulator API."""
    accumulate_spectral_estimates(
        accumulator,
        spectral_estimate.unsqueeze(0),
        shifted=shifted,
    )


def test_global_accumulation_keeps_shifted_and_unshifted_state_separate():
    accumulator = make_accumulator()
    unshifted = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex128)
    shifted = torch.tensor([5 + 6j, 7 + 8j], dtype=torch.complex128)

    accumulate_spectrum(accumulator, unshifted)
    accumulate_spectrum(accumulator, shifted, shifted=True)

    torch.testing.assert_close(accumulator.unshifted.spectrum_sum, unshifted)
    torch.testing.assert_close(accumulator.shifted.spectrum_sum, shifted)
    assert accumulator.unshifted.count == 1
    assert accumulator.shifted.count == 1

    for group, expected in (
        (accumulator.unshifted, unshifted),
        (accumulator.shifted, shifted),
    ):
        global_state = group.global_state
        torch.testing.assert_close(global_state.mean_re, expected.real)
        torch.testing.assert_close(global_state.mean_im, expected.imag)
        torch.testing.assert_close(global_state.m2_re, torch.zeros_like(expected.real))
        torch.testing.assert_close(global_state.m2_im, torch.zeros_like(expected.imag))

        state = group.short_term_state
        assert state.current_count == 0
        assert state.completed_batches == 0
        assert state.current_mean_re is None
        assert state.current_mean_im is None
        assert state.current_m2_re is None
        assert state.current_m2_im is None
        assert state.variance_sum_re is None
        assert state.variance_sum_im is None


def test_global_finalization_calculates_mean_and_componentwise_sem():
    accumulator = make_accumulator()
    accumulate_spectrum(
        accumulator,
        torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex128),
    )
    accumulate_spectrum(
        accumulator,
        torch.tensor([3 + 4j, 5 + 8j], dtype=torch.complex128),
    )

    result = finalize_result(accumulator)

    np.testing.assert_allclose(result.spectrum, np.asarray([2 + 3j, 4 + 6j]))
    assert result.spectrum_uncertainty is not None
    np.testing.assert_allclose(result.spectrum_uncertainty, np.asarray([1 + 1j, 1 + 2j]))
    assert result.channels == accumulator.channels
    np.testing.assert_array_equal(result.freq, accumulator.freq)
    assert result.freq_unit == accumulator.freq_unit


@pytest.mark.parametrize(
    ("uncertainty_estimation", "m_var"),
    [
        pytest.param("global", 10, id="global"),
        pytest.param("short_term", 3, id="short-term-crosses-multiple-boundaries"),
    ],
)
def test_batched_accumulation_matches_single_estimate_accumulation(
    uncertainty_estimation,
    m_var,
):
    estimates = torch.tensor(
        [
            [1 + 2j, 3 + 4j],
            [2 + 1j, 5 + 8j],
            [4 + 6j, 7 + 2j],
            [8 + 3j, 9 + 5j],
            [3 + 9j, 2 + 6j],
            [6 + 4j, 1 + 7j],
            [5 + 8j, 4 + 3j],
        ],
        dtype=torch.complex128,
    )
    batched = make_accumulator(
        uncertainty_estimation=uncertainty_estimation,
        m_var=m_var,
    )
    singles = make_accumulator(
        uncertainty_estimation=uncertainty_estimation,
        m_var=m_var,
    )

    # Deliberately use calculation-batch boundaries that do not align with m_var=3.
    for start, stop in ((0, 2), (2, 6), (6, 7)):
        accumulate_spectral_estimates(batched, estimates[start:stop])

    for estimate in estimates:
        accumulate_spectrum(singles, estimate)

    batched_result = finalize_result(batched)
    singles_result = finalize_result(singles)

    np.testing.assert_allclose(batched_result.spectrum, singles_result.spectrum)
    assert batched_result.spectrum_uncertainty is not None
    assert singles_result.spectrum_uncertainty is not None
    np.testing.assert_allclose(
        batched_result.spectrum_uncertainty,
        singles_result.spectrum_uncertainty,
    )
    assert batched.unshifted.count == estimates.shape[0]

    if uncertainty_estimation == "short_term":
        state = batched.unshifted.short_term_state
        assert state.completed_batches == 2
        assert state.current_count == 1


def test_global_batch_accumulates_componentwise_welford_state():
    accumulator = make_accumulator()
    estimates = torch.tensor(
        [
            [1 + 2j, 3 + 4j],
            [5 + 6j, 7 + 8j],
        ],
        dtype=torch.complex128,
    )

    accumulate_spectral_estimates(accumulator, estimates)

    torch.testing.assert_close(
        accumulator.unshifted.spectrum_sum,
        estimates.sum(dim=0),
    )
    group = accumulator.unshifted
    state = group.global_state

    assert state.mean_re is not None
    assert state.mean_im is not None
    assert state.m2_re is not None
    assert state.m2_im is not None

    expected_mean_re = estimates.real.mean(dim=0)
    expected_mean_im = estimates.imag.mean(dim=0)
    expected_m2_re = torch.square(estimates.real - expected_mean_re).sum(dim=0)
    expected_m2_im = torch.square(estimates.imag - expected_mean_im).sum(dim=0)

    torch.testing.assert_close(state.mean_re, expected_mean_re)
    torch.testing.assert_close(state.mean_im, expected_mean_im)
    torch.testing.assert_close(state.m2_re, expected_m2_re)
    torch.testing.assert_close(state.m2_im, expected_m2_im)
    assert group.count == 2


def test_batch_mean_m2_cpu_uses_two_pass_reduction(monkeypatch):
    values = torch.tensor(
        [
            [[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]],
            [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
        ],
        dtype=torch.float64,
    )

    def reject_var_mean(*args, **kwargs):
        raise AssertionError("The CPU path must not call torch.var_mean.")

    monkeypatch.setattr(torch, "var_mean", reject_var_mean)

    mean, m2 = _batch_mean_m2(values, dim=1)

    expected_mean = torch.tensor([[5.0, 7.0], [6.0, 8.0]], dtype=torch.float64)
    expected_m2 = torch.full((2, 2), 32.0, dtype=torch.float64)
    torch.testing.assert_close(mean, expected_mean)
    torch.testing.assert_close(m2, expected_m2)


def test_batch_mean_m2_non_cpu_path_honors_reduction_dimension():
    values = torch.empty((2, 3, 4), device="meta")

    mean, m2 = _batch_mean_m2(values, dim=1)

    assert mean.shape == (2, 4)
    assert m2.shape == (2, 4)


def test_batched_accumulation_rejects_an_empty_batch():
    accumulator = make_accumulator()

    with pytest.raises(ValueError, match="empty batch"):
        accumulate_spectral_estimates(
            accumulator,
            torch.empty((0, 2), dtype=torch.complex128),
        )


@pytest.mark.parametrize(
    "spectral_estimates",
    [
        pytest.param(torch.tensor(1 + 2j), id="scalar"),
        pytest.param(torch.ones(2, dtype=torch.complex128), id="unbatched-spectrum"),
    ],
)
def test_batched_accumulation_requires_a_leading_batch_dimension(spectral_estimates):
    accumulator = make_accumulator()

    with pytest.raises(ValueError, match="leading batch dimension"):
        accumulate_spectral_estimates(accumulator, spectral_estimates)


def test_batched_accumulation_rejects_incorrect_per_estimate_shape():
    accumulator = make_accumulator()

    with pytest.raises(ValueError, match=r"per-estimate shape.*expected"):
        accumulate_spectral_estimates(
            accumulator,
            torch.ones((2, 3), dtype=torch.complex128),
        )


def test_global_finalization_combines_groups_with_componentwise_maximum():
    accumulator = make_accumulator(channels=(0,), frequency_points=1)

    for value in (1 + 1j, 3 + 3j):
        accumulate_spectrum(accumulator, torch.tensor([value], dtype=torch.complex128))

    for value in (10 + 2j, 14 + 6j):
        accumulate_spectrum(
            accumulator,
            torch.tensor([value], dtype=torch.complex128),
            shifted=True,
        )

    result = finalize_result(accumulator)

    np.testing.assert_allclose(result.spectrum, np.asarray([7 + 3j]))
    assert result.spectrum_uncertainty is not None
    np.testing.assert_allclose(result.spectrum_uncertainty, np.asarray([2 + 2j]))


def test_global_finalization_with_one_estimate_warns_and_returns_no_uncertainty():
    accumulator = make_accumulator()
    estimate = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex128)
    accumulate_spectrum(accumulator, estimate)
    assert accumulator.unshifted.spectrum_sum is not None
    sum_before = accumulator.unshifted.spectrum_sum.clone()

    with pytest.warns(RuntimeWarning, match="at least two spectral estimates"):
        result = finalize_result(accumulator)

    np.testing.assert_allclose(result.spectrum, estimate.numpy())
    assert result.spectrum_uncertainty is None
    torch.testing.assert_close(accumulator.unshifted.spectrum_sum, sum_before)
    assert accumulator.unshifted.count == 1


def test_short_term_finalization_calculates_one_complete_batch():
    accumulator = make_accumulator(
        channels=(0,),
        frequency_points=1,
        uncertainty_estimation="short_term",
        m_var=2,
    )

    for value in (1 + 2j, 3 + 6j):
        accumulate_spectrum(
            accumulator,
            torch.tensor([value], dtype=torch.complex128),
        )

    result = finalize_result(accumulator)

    np.testing.assert_allclose(result.spectrum, np.asarray([2 + 4j]))
    assert result.spectrum_uncertainty is not None
    np.testing.assert_allclose(result.spectrum_uncertainty, np.asarray([1 + 2j]))

    group = accumulator.unshifted
    state = group.short_term_state
    assert all(
        value is None
        for value in (
            group.global_state.mean_re,
            group.global_state.mean_im,
            group.global_state.m2_re,
            group.global_state.m2_im,
        )
    )
    assert group.count == 2
    assert state.current_count == 0
    assert state.completed_batches == 1
    torch.testing.assert_close(
        state.variance_sum_re,
        torch.tensor([1.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        state.variance_sum_im,
        torch.tensor([4.0], dtype=torch.float64),
    )


def test_short_term_finalization_averages_batch_variances_before_square_root():
    accumulator = make_accumulator(
        channels=(0,),
        frequency_points=1,
        uncertainty_estimation="short_term",
        m_var=2,
    )

    for value in (1 + 2j, 3 + 6j, 10 + 1j, 14 + 7j):
        accumulate_spectrum(
            accumulator,
            torch.tensor([value], dtype=torch.complex128),
        )

    result = finalize_result(accumulator)

    np.testing.assert_allclose(result.spectrum, np.asarray([7 + 4j]))
    assert result.spectrum_uncertainty is not None
    expected_uncertainty = np.sqrt(2.5) + 1j * np.sqrt(6.5)
    np.testing.assert_allclose(result.spectrum_uncertainty, np.asarray([expected_uncertainty]))

    state = accumulator.unshifted.short_term_state
    assert state.current_count == 0
    assert state.completed_batches == 2


def test_incomplete_short_term_batch_affects_spectrum_but_not_uncertainty():
    accumulator = make_accumulator(
        channels=(0,),
        frequency_points=1,
        uncertainty_estimation="short_term",
        m_var=2,
    )

    for value in (1 + 0j, 3 + 0j, 100 + 0j):
        accumulate_spectrum(
            accumulator,
            torch.tensor([value], dtype=torch.complex128),
        )

    result = finalize_result(accumulator)

    np.testing.assert_allclose(result.spectrum, np.asarray([104 / 3]))
    assert result.spectrum_uncertainty is not None
    np.testing.assert_allclose(result.spectrum_uncertainty, np.asarray([1 + 0j]))

    state = accumulator.unshifted.short_term_state
    assert state.completed_batches == 1
    assert state.current_count == 1


def test_short_term_groups_are_batched_separately_and_combined_componentwise():
    accumulator = make_accumulator(
        channels=(0,),
        frequency_points=1,
        uncertainty_estimation="short_term",
        m_var=2,
    )

    for value in (1 + 1j, 3 + 3j):
        accumulate_spectrum(
            accumulator,
            torch.tensor([value], dtype=torch.complex128),
        )

    for value in (10 + 2j, 14 + 8j):
        accumulate_spectrum(
            accumulator,
            torch.tensor([value], dtype=torch.complex128),
            shifted=True,
        )

    result = finalize_result(accumulator)

    np.testing.assert_allclose(result.spectrum, np.asarray([7 + 3.5j]))
    assert result.spectrum_uncertainty is not None
    np.testing.assert_allclose(result.spectrum_uncertainty, np.asarray([2 + 3j]))
    assert accumulator.unshifted.short_term_state.completed_batches == 1
    assert accumulator.shifted.short_term_state.completed_batches == 1


def test_short_term_finalization_uses_qualified_group_and_count_weighted_spectrum():
    accumulator = make_accumulator(
        channels=(0,),
        frequency_points=1,
        uncertainty_estimation="short_term",
        m_var=2,
    )

    for value in (1 + 0j, 3 + 0j):
        accumulate_spectrum(
            accumulator,
            torch.tensor([value], dtype=torch.complex128),
        )

    accumulate_spectrum(
        accumulator,
        torch.tensor([10 + 0j], dtype=torch.complex128),
        shifted=True,
    )

    result = finalize_result(accumulator)

    np.testing.assert_allclose(result.spectrum, np.asarray([14 / 3]))
    assert result.spectrum_uncertainty is not None
    np.testing.assert_allclose(result.spectrum_uncertainty, np.asarray([1 + 0j]))
    assert accumulator.unshifted.short_term_state.completed_batches == 1
    assert accumulator.shifted.short_term_state.completed_batches == 0
    assert accumulator.shifted.short_term_state.current_count == 1


def test_short_term_finalization_without_complete_batch_warns_and_returns_no_uncertainty():
    accumulator = make_accumulator(
        channels=(0,),
        frequency_points=1,
        uncertainty_estimation="short_term",
        m_var=3,
    )

    for value in (1 + 2j, 3 + 4j):
        accumulate_spectrum(
            accumulator,
            torch.tensor([value], dtype=torch.complex128),
        )

    with pytest.warns(RuntimeWarning, match="complete batch"):
        result = finalize_result(accumulator)

    np.testing.assert_allclose(result.spectrum, np.asarray([2 + 3j]))
    assert result.spectrum_uncertainty is None
    assert accumulator.unshifted.short_term_state.current_count == 2
    assert accumulator.unshifted.short_term_state.completed_batches == 0


def test_short_term_batch_handles_prefix_complete_groups_and_suffix():
    accumulator = make_accumulator(
        channels=(0,),
        frequency_points=1,
        uncertainty_estimation="short_term",
        m_var=3,
    )
    estimates = torch.tensor(
        [
            1 + 2j,
            2 + 4j,
            4 + 8j,
            10 + 1j,
            14 + 3j,
            20 + 7j,
            30 + 5j,
            35 + 9j,
            41 + 15j,
            50 + 6j,
            58 + 12j,
            68 + 20j,
            100 + 30j,
        ],
        dtype=torch.complex128,
    ).unsqueeze(1)

    # The second call first completes the existing group, then processes
    # three complete groups and finally retains one estimate as a suffix.
    accumulate_spectral_estimates(accumulator, estimates[:2])
    accumulate_spectral_estimates(accumulator, estimates[2:])

    state = accumulator.unshifted.short_term_state
    complete_groups = estimates[:12].reshape(4, 3, 1)
    expected_variance_sum_re = (
        complete_groups.real.var(dim=1, correction=1) / 3
    ).sum(dim=0)
    expected_variance_sum_im = (
        complete_groups.imag.var(dim=1, correction=1) / 3
    ).sum(dim=0)

    assert accumulator.unshifted.count == 13
    assert state.completed_batches == 4
    assert state.current_count == 1
    torch.testing.assert_close(state.variance_sum_re, expected_variance_sum_re)
    torch.testing.assert_close(state.variance_sum_im, expected_variance_sum_im)
    torch.testing.assert_close(state.current_mean_re, estimates[-1].real)
    torch.testing.assert_close(state.current_mean_im, estimates[-1].imag)
    torch.testing.assert_close(state.current_m2_re, torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(state.current_m2_im, torch.zeros(1, dtype=torch.float64))


@pytest.mark.parametrize(
    ("batch_slices", "rtol"),
    [
        pytest.param(((0, 4),), 1e-12, id="vectorized-complete-group"),
        pytest.param(((0, 3), (3, 4)), 2e-5, id="parallel-welford-merge"),
        pytest.param(
            ((0, 1), (1, 2), (2, 3), (3, 4)),
            2e-5,
            id="singleton-segments",
        ),
    ],
)
def test_short_term_welford_accumulation_is_stable_for_large_offsets(
    batch_slices,
    rtol,
):
    accumulator = make_accumulator(
        channels=(0,),
        frequency_points=1,
        uncertainty_estimation="short_term",
        m_var=4,
    )
    base = 1e12
    estimates = torch.tensor(
        [
            [(base - 1) + 1j * (2 * base - 2)],
            [(base + 1) + 1j * (2 * base + 2)],
            [(base - 1) + 1j * (2 * base - 2)],
            [(base + 1) + 1j * (2 * base + 2)],
        ],
        dtype=torch.complex128,
    )

    for start, stop in batch_slices:
        accumulate_spectral_estimates(accumulator, estimates[start:stop])

    result = finalize_result(accumulator)

    assert result.spectrum_uncertainty is not None
    expected = np.sqrt(1 / 3) + 1j * np.sqrt(4 / 3)
    np.testing.assert_allclose(
        result.spectrum_uncertainty,
        np.asarray([expected]),
        rtol=rtol,
    )


@pytest.mark.parametrize(
    ("batch_slices", "rtol"),
    [
        pytest.param(((0, 4),), 1e-12, id="single-batch"),
        pytest.param(((0, 2), (2, 4)), 2e-5, id="two-batches"),
        pytest.param(
            ((0, 1), (1, 2), (2, 3), (3, 4)),
            2e-5,
            id="singleton-batches",
        ),
    ],
)
def test_global_welford_accumulation_is_stable_for_large_offsets(batch_slices, rtol):
    accumulator = make_accumulator(
        channels=(0,),
        frequency_points=1,
        uncertainty_estimation="global",
    )
    base = 1e12
    estimates = torch.tensor(
        [
            [(base - 1) + 1j * (2 * base - 2)],
            [(base + 1) + 1j * (2 * base + 2)],
            [(base - 1) + 1j * (2 * base - 2)],
            [(base + 1) + 1j * (2 * base + 2)],
        ],
        dtype=torch.complex128,
    )

    for start, stop in batch_slices:
        accumulate_spectral_estimates(accumulator, estimates[start:stop])

    result = finalize_result(accumulator)

    assert result.spectrum_uncertainty is not None
    expected = np.sqrt(1 / 3) + 1j * np.sqrt(4 / 3)
    np.testing.assert_allclose(
        result.spectrum_uncertainty,
        np.asarray([expected]),
        rtol=rtol,
    )


def test_finalize_result_rejects_empty_accumulator():
    accumulator = make_accumulator()

    with pytest.raises(RuntimeError, match="no spectra were accumulated"):
        finalize_result(accumulator)


@pytest.mark.parametrize(
    ("spectrum_sum", "state_values", "count", "message"),
    [
        pytest.param(
            None,
            (torch.ones(2), torch.ones(2), torch.ones(2), torch.ones(2)),
            0,
            "inconsistent",
            id="welford-state-without-sum",
        ),
        pytest.param(
            None,
            (None, None, None, None),
            1,
            "inconsistent",
            id="count-without-sum",
        ),
        pytest.param(
            torch.ones(2, dtype=torch.complex128),
            (None, None, None, None),
            1,
            "missing required state",
            id="sum-without-welford-state",
        ),
        pytest.param(
            torch.ones(2, dtype=torch.complex128),
            (torch.ones(2), torch.ones(2), None, None),
            1,
            "missing required state",
            id="incomplete-welford-state",
        ),
        pytest.param(
            torch.ones(2, dtype=torch.complex128),
            (torch.ones(2), torch.ones(2), torch.zeros(2), torch.zeros(2)),
            0,
            "inconsistent",
            id="sum-without-count",
        ),
    ],
)
def test_global_finalization_rejects_inconsistent_group_state(
    spectrum_sum,
    state_values,
    count,
    message,
):
    accumulator = make_accumulator()
    group = accumulator.unshifted
    group.spectrum_sum = spectrum_sum
    group.count = count
    (
        group.global_state.mean_re,
        group.global_state.mean_im,
        group.global_state.m2_re,
        group.global_state.m2_im,
    ) = state_values

    with pytest.raises(RuntimeError, match=message):
        finalize_result(accumulator)


def test_global_finalization_rejects_populated_short_term_state():
    accumulator = make_accumulator()
    accumulate_spectrum(
        accumulator,
        torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex128),
    )
    accumulator.unshifted.short_term_state.current_count = 1

    with pytest.raises(RuntimeError, match="must remain empty"):
        finalize_result(accumulator)


def test_short_term_finalization_rejects_global_welford_state():
    accumulator = make_accumulator(uncertainty_estimation="short_term", m_var=2)
    accumulate_spectrum(
        accumulator,
        torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex128),
    )
    accumulator.unshifted.global_state.mean_re = torch.ones(2)

    with pytest.raises(RuntimeError, match="must remain empty"):
        finalize_result(accumulator)


def test_short_term_finalization_rejects_inconsistent_batch_count():
    accumulator = make_accumulator(uncertainty_estimation="short_term", m_var=2)
    accumulate_spectrum(
        accumulator,
        torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex128),
    )
    accumulator.unshifted.short_term_state.current_count = 0

    with pytest.raises(RuntimeError, match="batch count is inconsistent"):
        finalize_result(accumulator)
