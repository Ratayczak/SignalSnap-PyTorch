# This test doesn't showcase what higher order spectra are actually meant for
# since I am using them on trigonometrical functions with no noise. 
# However i know the results so they are good for testing.

import numpy as np
from multichss.MultiChSS_SpectrumConfig import SpectrumConfig, DataImportConfig
from multichss.MultiChSS_SpectrumCalculator import SpectrumCalculator
from multichss.MultiChSS_CrossConfig import CrossConfig

def test_c1_returns_correct_mean():
    """
    Test that the first-order cumulant correctly returns the mean of the signal.
    Since the signal is sin(2πt) + 2, the mean should be 2.0.
    """

    # Generate test signal
    t = np.linspace(0, 10000, 1000000)
    y = np.sin(2 * np.pi * t) + 2  # known mean = 2.0

    # Wrap into config objects
    config1 = DataImportConfig(data=y)
    selected_data = [0]

    sconfig = SpectrumConfig(
        dt=0.01, f_min=0, f_max=2, s3_calc='1/4',
        f_unit='Hz', backend='cpu', order_in=[1],
        spectrum_size=100, show_first_frame=False
    )

    cconfig = CrossConfig(auto_corr=True)

    # Run the spectrum calculator
    scalc = SpectrumCalculator(sconfig, cconfig, [config1], selected=selected_data)
    scalc.calc_spec()

    # Grab the component from the first-order spectrum
    real_part = scalc.s[0][1][0].real
    imag_part = scalc.s[0][1][0].imag

    # Assert
    assert abs(real_part - 2.0) < 1e-6, f"Expected real=2.0, got {real_part}"
    assert abs(imag_part - 0.0) < 1e-6, f"Expected imag=0.0, got {imag_part}"
