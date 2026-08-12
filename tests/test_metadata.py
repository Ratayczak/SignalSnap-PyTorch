import numpy as np
import pytest

from signalsnap_pytorch import (
    DataConfig,
    TimestampOptions,
    SampledChannel,
    SpectrumConfig,
    TimestampedChannel,
    calculate_spectra,
)
from signalsnap_pytorch._core import accumulation as _accumulation
from signalsnap_pytorch._core import planning as _planning
from signalsnap_pytorch._core.data_access import open_channels
from signalsnap_pytorch._core.planning import build_runtime_config, resolve_requested_spectra
from signalsnap_pytorch.metadata import build_result_metadata


def _build_metadata(data_config, spectrum_config, requested_spectra):
    spectra, active_channels = resolve_requested_spectra(
        requested_spectra,
        channel_count=len(data_config.channels),
    )
    with open_channels(data_config, active_channels) as opened_channels:
        runtime = build_runtime_config(
            data_config,
            opened_channels,
            spectrum_config,
            spectra,
        )

    return build_result_metadata(data_config, spectrum_config, runtime)


def test_sampled_metadata_records_requested_and_resolved_values():
    data_config = DataConfig(
        channels=(
            SampledChannel(np.ones(40), dt=0.5),
            TimestampedChannel(np.array([], dtype=float)),
        ),
        t_unit="ms",
    )
    spectrum_config = SpectrumConfig(
        df=0.45,
        f_min=-0.5,
        f_max=0.75,
        m=3,
        m_var=4,
        spectral_estimates_max=2,
        interlacing=True,
        precision="double",
    )

    calculation, spectra = _build_metadata(
        data_config,
        spectrum_config,
        [(0,), (0, 0, 0)],
    )

    assert calculation.channel_kinds == ("sampled", "timestamped")
    assert calculation.active_channels == (0,)
    assert calculation.requested_spectra == ((0,), (0, 0, 0))
    assert calculation.observation_start == 0.0
    assert calculation.observation_stop == 20.0
    assert calculation.time_unit == "ms"
    assert calculation.frequency_unit == "kHz"
    assert calculation.requested_df == 0.45
    assert calculation.actual_df == 0.5
    assert calculation.requested_f_min == -0.5
    assert calculation.requested_f_max == 0.75
    assert calculation.window_duration == 2.0
    assert calculation.unshifted_offset == 0.0
    assert calculation.shifted_offset == 1.0
    assert calculation.window_convention == "confined_gaussian"
    assert calculation.timestamp_weighting is None
    assert calculation.exponential_scale is None
    assert calculation.repetition_count == 1
    assert calculation.requested_repetition_batch_size is None
    assert calculation.resolved_repetition_batch_size == 1
    assert calculation.user_seed is None
    assert calculation.resolved_seed is None
    assert calculation.requested_m == 3
    assert calculation.effective_m == 3
    assert calculation.requested_m_var == 4
    assert calculation.effective_m_var == 4
    assert calculation.uncertainty_estimation == "global"
    assert calculation.unshifted_physical_estimate_count == 2
    assert calculation.shifted_physical_estimate_count == 2
    assert calculation.unshifted_coefficient_window_count == 6
    assert calculation.shifted_coefficient_window_count == 6
    assert calculation.real_dtype == "torch.float64"
    assert calculation.complex_dtype == "torch.complex128"
    assert calculation.requested_device == "cpu"
    assert calculation.resolved_device == "cpu"

    first_order = spectra[(0,)]
    third_order = spectra[(0, 0, 0)]
    assert first_order.channels == (0,)
    assert first_order.order == 1
    assert first_order.frequency_view == "sampled_fft"
    assert (first_order.effective_f_min, first_order.effective_f_max) == (0.0, 0.0)
    assert first_order.normalization_convention == "sampled_discrete_default"
    assert first_order.closing_frequency_support == "not_applicable"
    assert third_order.frequency_view == "sampled_fft"
    assert (third_order.effective_f_min, third_order.effective_f_max) == (-0.5, 0.5)
    assert third_order.closing_frequency_support == "sampled_fft"


def test_mixed_metadata_distinguishes_frequency_and_closing_views():
    data_config = DataConfig(
        channels=(
            SampledChannel(np.ones(64), dt=0.25),
            TimestampedChannel(np.array([0.25, 1.5, 4.0, 10.0])),
        ),
        observation_start=0.0,
        observation_stop=16.0,
    )
    spectrum_config = SpectrumConfig(
        df=0.9,
        f_min=-2.0,
        f_max=3.0,
        timestamp_options=TimestampOptions(weighting="unit"),
        m=3,
        interlacing=False,
    )
    requested = [(1, 1), (0, 1), (0, 0, 1), (1, 1, 0)]

    calculation, spectra = _build_metadata(data_config, spectrum_config, requested)
    assert calculation.channel_kinds == ("sampled", "timestamped")
    assert calculation.active_channels == (1, 0)
    assert calculation.actual_df == 1.0
    assert calculation.shifted_offset is None
    assert calculation.timestamp_weighting == "unit"
    assert calculation.repetition_count == 1
    assert calculation.resolved_seed is None

    timestamp_only = spectra[(1, 1)]
    assert timestamp_only.frequency_view == "direct_transform"
    assert (timestamp_only.effective_f_min, timestamp_only.effective_f_max) == (-2.0, 3.0)
    assert timestamp_only.normalization_convention == "timestamp_continuous_default"

    mixed = spectra[(0, 1)]
    assert mixed.frequency_view == "sampled_fft"
    assert (mixed.effective_f_min, mixed.effective_f_max) == (-2.0, 1.0)
    assert mixed.normalization_convention == "mixed_discrete_overlap_default"
    assert spectra[(0, 0, 1)].closing_frequency_support == "direct_transform"
    assert spectra[(1, 1, 0)].closing_frequency_support == "sampled_fft"


def test_legacy_and_exponential_metadata_records_requested_options():
    data_config = DataConfig(
        channels=(TimestampedChannel(np.array([0.25, 1.25, 2.25, 3.25])),),
        observation_start=0.0,
        observation_stop=4.0,
    )
    spectrum_config = SpectrumConfig(
        df=1.0,
        f_min=0.0,
        f_max=1.0,
        timestamp_options=TimestampOptions(
            weighting="exponential",
            scale=2.5,
            repetitions=7,
            repetitions_per_batch=3,
            seed=19,
        ),
        m=2,
        old_window=True,
    )

    calculation, spectra = _build_metadata(
        data_config,
        spectrum_config,
        [(0, 0)],
    )

    assert calculation.window_convention == "legacy_confined_gaussian"
    assert calculation.timestamp_weighting == "exponential"
    assert calculation.exponential_scale == 2.5
    assert calculation.repetition_count == 7
    assert calculation.requested_repetition_batch_size == 3
    assert calculation.resolved_repetition_batch_size == 3
    assert calculation.user_seed == 19
    assert calculation.resolved_seed == 19
    assert spectra[(0, 0)].normalization_convention == "timestamp_fixed_grid_legacy"


def test_pipeline_results_share_store_metadata_by_identity():
    results = calculate_spectra(
        DataConfig(channels=(SampledChannel(np.arange(32), dt=1.0),)),
        SpectrumConfig(
            df=0.25,
            f_min=0.0,
            f_max=0.5,
            m=2,
            spectral_estimates_max=3,
        ),
        requested_spectra=[(0,), (0, 0)],
        show_progress=False,
    )

    assert results.calculation_metadata is not None
    assert tuple(results.spectra_metadata) == (
        (0,),
        (0, 0),
    )
    for result in results:
        assert result.calculation_metadata is results.calculation_metadata
        assert result.spectrum_metadata is results.spectra_metadata[result.channels]


def test_pipeline_metadata_retains_generated_seed(monkeypatch):
    monkeypatch.setattr(_planning.secrets, "randbits", lambda bit_count: 123456)
    results = calculate_spectra(
        DataConfig(
            channels=(TimestampedChannel(np.arange(8) + 0.5),),
            observation_start=0.0,
            observation_stop=8.0,
        ),
        SpectrumConfig(
            df=1.0,
            f_min=0.0,
            f_max=1.0,
            timestamp_options=TimestampOptions(
                weighting="exponential",
                scale=1.5,
                repetitions=2,
            ),
            m=2,
        ),
        requested_spectra=[(0, 0)],
        show_progress=False,
    )

    assert results.calculation_metadata is not None
    assert results.calculation_metadata.user_seed is None
    assert results.calculation_metadata.resolved_seed == 123456


def test_pipeline_metadata_keeps_request_when_finalization_fails(monkeypatch):
    original_finalize = _accumulation.finalize_result

    def fail_one_result(accumulator, **kwargs):
        if accumulator.channels == (0,):
            raise RuntimeError("intentional metadata test failure")
        return original_finalize(accumulator, **kwargs)

    monkeypatch.setattr(_accumulation, "finalize_result", fail_one_result)

    with pytest.warns(RuntimeWarning, match="intentional metadata test failure"):
        results = calculate_spectra(
            DataConfig(channels=(SampledChannel(np.arange(32), dt=1.0),)),
            SpectrumConfig(
                df=0.25,
                f_min=0.0,
                f_max=0.5,
                m=2,
                spectral_estimates_max=3,
            ),
            requested_spectra=[(0,), (0, 0)],
            show_progress=False,
        )

    assert (0,) not in results
    assert (0, 0) in results
    assert results.calculation_metadata is not None
    assert results.calculation_metadata.requested_spectra == ((0,), (0, 0))
    assert tuple(results.spectra_metadata) == (
        (0,),
        (0, 0),
    )
