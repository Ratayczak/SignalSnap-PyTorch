import pytest
from pydantic import ValidationError

from multichss.configurators import CrossConfig, SpectrumConfig


def test_spectrum_config_rejects_negative_frequencies_for_order_3():
    with pytest.raises(ValidationError, match="Third-order spectra cannot be requested"):
        SpectrumConfig(f_min=-1, f_max=1, orders=[3])


def test_spectrum_config_rejects_negative_frequencies_when_orders_all():
    with pytest.raises(ValidationError, match="Third-order spectra cannot be requested"):
        SpectrumConfig(f_min=-1, f_max=1, orders="all")


def test_cross_config_allows_repeated_channels_in_cross_correlations():
    CrossConfig(cross_corr_3=[(1, 1, 0)])


def test_cross_config_rejects_auto_correlations_in_cross_correlations():
    with pytest.raises(ValidationError, match="cannot include auto-correlations"):
        CrossConfig(cross_corr_3=[(1, 1, 1)])
