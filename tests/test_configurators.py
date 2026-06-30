import pytest
from pydantic import ValidationError

from multichss.configurators import SpectrumConfig


def test_spectrum_config_rejects_negative_frequencies_for_order_3():
    with pytest.raises(ValidationError, match="Third-order spectra cannot be requested"):
        SpectrumConfig(f_min=-1, f_max=1, orders=[3])


def test_spectrum_config_rejects_negative_frequencies_when_orders_all():
    with pytest.raises(ValidationError, match="Third-order spectra cannot be requested"):
        SpectrumConfig(f_min=-1, f_max=1, orders="all")
