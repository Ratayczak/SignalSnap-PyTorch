from __future__ import annotations

import numpy as np

from signalsnap_pytorch import DataConfig, SampledChannel

# The largest order-4 regression case peaks near 42 MiB on CUDA at this size while avoiding most
# per-estimate loop overhead.
TEST_SPECTRAL_ESTIMATES_PER_BATCH = 32


def sampled_data_config(
    *,
    channels: tuple[object, ...] | list[object],
    dt: float,
    t_unit: str = "s",
) -> DataConfig:
    """Build a sampled-only configuration for tests unrelated to public API validation."""

    return DataConfig(
        channels=tuple(
            channel
            if isinstance(channel, SampledChannel)
            else SampledChannel(data=channel, dt=dt)
            for channel in channels
        ),
        t_unit=t_unit,
    )


def indices_for_freqs(actual_freq: np.ndarray, expected_freq: np.ndarray) -> np.ndarray:
    """Return indices locating every expected frequency exactly once on an actual axis."""
    indices = []
    for freq in expected_freq:
        matches = np.flatnonzero(np.isclose(actual_freq, freq, rtol=0.0, atol=1e-12))
        if matches.size != 1:
            raise AssertionError(f"Frequency {freq} is not represented exactly once.")
        indices.append(matches[0])
    return np.asarray(indices)


def align_legacy_spectrum_region(
    actual_spectrum: np.ndarray,
    actual_freq: np.ndarray,
    legacy_spectrum: np.ndarray,
    legacy_freq: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Align legacy and current spectra by frequency rather than by array position."""
    if order == 1:
        return actual_spectrum, legacy_spectrum

    if order == 3:
        if legacy_spectrum.shape == (legacy_freq.size, legacy_freq.size):
            row_freq = legacy_freq
            col_freq = legacy_freq
        elif legacy_spectrum.ndim == 2 and legacy_spectrum.shape[1] == legacy_freq.size:
            row_freq = legacy_freq[legacy_freq.size // 2 :]
            col_freq = legacy_freq
        else:
            raise AssertionError(
                f"Unsupported legacy third-order shape {legacy_spectrum.shape} "
                f"for frequency axis length {legacy_freq.size}."
            )

        row_indices = indices_for_freqs(actual_freq, row_freq)
        col_indices = indices_for_freqs(actual_freq, col_freq)
        return actual_spectrum[np.ix_(row_indices, col_indices)], legacy_spectrum

    if order not in (2, 4):
        raise AssertionError(f"Unsupported spectrum order: {order}")

    legacy_indices = indices_for_freqs(legacy_freq, actual_freq)
    if order == 2:
        return actual_spectrum, legacy_spectrum[legacy_indices]
    return actual_spectrum, legacy_spectrum[np.ix_(legacy_indices, legacy_indices)]
