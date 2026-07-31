# SignalSnap (PyTorch)

Higher-order spectral analysis for real, multi-channel time series.

SignalSnap estimates first- through fourth-order auto- and cross-spectra (also called polyspectra)
from real, finite measurement traces using unbiased estimators for multivariate cumulants. Its
PyTorch backend supports calculations on CPUs, NVIDIA GPUs via CUDA, and Apple Silicon through MPS,
with additional experimental support for AMD GPUs through ROCm and Intel GPUs through XPU. Input
channels can be in-memory arrays, lazily read HDF5 selections, or a mixture of both.

This repository contains the PyTorch rewrite of the original
[SignalSnap](https://github.com/MarkusSifft/SignalSnap). The current 2.0 API intentionally differs
from the original object-oriented API.

## Features

- First- through fourth-order auto- and cross-polyspectra; order four is returned as a diagonal
  two-dimensional slice
- Unbiased multivariate cumulant estimators
- Global standard errors and short-term uncertainty estimates from repeated spectral estimates
- Optional interlaced estimates to reduce window-edge effects
- Accelerated computing on a variety of GPUs via PyTorch
- Lazy reading from HDF5 datasets larger than system memory
- Plotting of spectra, uncertainties, and statistical significance

## Installation

SignalSnap requires Python 3.12 or newer. Install the package by first cloning this repository and
installing it via pip:

```bash
git clone https://github.com/Ratayczak/SignalSnap-PyTorch.git
cd SignalSnap-PyTorch
python -m pip install .
```

Install optional HDF5 and plotting support with:

```bash
python -m pip install ".[hdf5,plotting]"
```

For development, including the test dependencies, use an editable installation:

```bash
python -m pip install -e ".[test]"
```

For CUDA or experimental ROCm support, install the appropriate PyTorch build first using the
[PyTorch installation selector](https://pytorch.org/get-started/locally/). SignalSnap will reuse a
compatible installation. For experimental Intel GPU support, install a compatible XPU-enabled
PyTorch build and Intel GPU driver by following the
[PyTorch Intel GPU instructions](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)
before installing SignalSnap.

The distribution name used by `pip` is `signalsnap-pytorch`; the Python import name is
`signalsnap_pytorch`.

## Quick Start

The following example calculates a second-order auto-spectrum and cross-spectrum for two related
signals:

```python
import numpy as np

from signalsnap_pytorch import DataConfig, SampledChannel, SpectrumConfig, calculate_spectra

rng = np.random.default_rng(42)

dt = 1e-3
time = np.arange(20_000) * dt

channel_0 = np.sin(2 * np.pi * 20 * time) + 0.1 * rng.normal(size=time.size)
channel_1 = np.sin(2 * np.pi * 20 * time + 0.5) + 0.1 * rng.normal(size=time.size)

data_config = DataConfig(
    channels=(
        SampledChannel(data=channel_0, dt=dt),
        SampledChannel(data=channel_1, dt=dt),
    ),
    t_unit="s",
)

spectrum_config = SpectrumConfig(
    df=1,
    f_min=0,
    f_max=100,
    m=10,
    device="cpu",
)

results = calculate_spectra(
    data_config,
    spectrum_config,
    requested_spectra=[
        (0, 0),  # power spectrum of channel 0
        (0, 1),  # cross-spectrum of channels 0 and 1
    ],
)

cross_spectrum = results[(0, 1)]
```

Calculation progress is displayed by default. Pass `show_progress=False` to disable it.

### Requesting spectra

Each entry in `requested_spectra` is a tuple of channel indices. Its length determines the order of
the spectrum; its entries select and order the channels:

| Request | Meaning |
| --- | --- |
| `(0,)` | First-order spectrum of channel 0 |
| `(0, 0)` | Second-order auto-spectrum of channel 0 |
| `(0, 1)` | Second-order cross-spectrum of channels 0 and 1 |
| `(0, 1, 0)` | Third-order cross-spectrum using channels 0, 1, and 0 |
| `(0, 1, 0, 1)` | Fourth-order diagonal cross-spectrum using channels 0, 1, 0, and 1 |
| `(1, 1, 1, 1)` | Fourth-order diagonal auto-spectrum of channel 1 |

The indices refer to positions in `DataConfig.channels`. Their order matters, so `(0, 1)` and
`(1, 0)` are distinct requests. Duplicate requests are rejected.

If `requested_spectra` is omitted, SignalSnap calculates order-one through order-four
auto-spectra for each input channel. Explicit requests avoid unnecessary work when only a subset is
needed, particularly for large datasets and higher orders.

### TODO: Expand Quick Start example to include pictures.

See [Calculation configuration](docs/configuration.md) to learn how to use `DataConfig` and
`SpectrumConfig`, and [Working with results](docs/results.md) to learn how results are stored.

## Documentation

The documentation can be found in [`docs/`](docs/README.md):

| Guide | Contents |
| --- | --- |
| [Calculation configuration](docs/configuration.md) | How to use `DataConfig` and `SpectrumConfig` |
| [Working with results](docs/results.md) | Understand result shapes, uncertainties, units, selection, and partial failures |
| [Plotting](docs/plotting.md) | Line plots, color maps, significance, saving, and window inspection |
| [HDF5 input](docs/hdf5.md) | Working with HDF5 files that may exceed memory limits |
| [Scientific background](docs/scientific-background.md) | Definitions, estimators, assumptions, and current scope |
| [Migration guide](docs/migration.md) | Mapping the original API to version 2.0 |

## Plotting

Plotting is provided by `signalsnap_pytorch.plotting`. Assuming that `results` is the
`SpectrumResultStore` from a `calculate_spectra` call:

```python
from signalsnap_pytorch.plotting import PlotStyle, build_order_1_table, create_spectrum_figures

print(build_order_1_table(results))

plot_style = PlotStyle(
    f_min=0,
    f_max=100,
    sigma=3,
    plot_format=["re", "im"],
)

figures = create_spectrum_figures(results, plot_style)
```

Display the figures with:

```python
import matplotlib.pyplot as plt

plt.show()
```

or save them with:

```python
from signalsnap_pytorch.plotting import save_figures

saved_paths = save_figures(
    figures,
    "./figures",
    dpi=200,
)
```

Order-one results are represented as a text table, order-two results as line plots, and order-three
and order-four results as two-dimensional color maps. The [plotting guide](docs/plotting.md) covers
uncertainty bands, significance overlays, arcsinh scaling, saving figures, and inspecting the first
FFT window.

## HDF5 input

Large measurements can be processed without loading the entire dataset into memory:

```python
from pathlib import Path

from signalsnap_pytorch import DataConfig, HDF5Source, SampledChannel

data_config = DataConfig(
    channels=(
        SampledChannel(
            data=HDF5Source(
                file=Path("measurement.h5"),
                dataset="/signals",
                selection=(slice(None), slice(None), 0),
            ),
            dt=2.0,
        ),
    ),
    t_unit="ns",
)
```

Only requested channels and required chunks are read. Selection rules and mixed in-memory/HDF5
workflows are described in [HDF5 input](docs/hdf5.md).

## Scientific background

The scientific background on what polyspectra are, how they can be interpreted, and how they
can be estimated from finite data is provided by
[Sifft et al., *Digital Signal Processing* 173 (2026), 105893](https://doi.org/10.1016/j.dsp.2026.105893).

A summary of the most important formulas used in this library can be found in the
[scientific background](docs/scientific-background.md).

## Benchmark

The following benchmark compares the PyTorch rewrite with the original ArrayFire implementation
for a second-order spectrum calculated from 5 GB of data on an NVIDIA GeForce RTX 4090:

![Runtime comparison between the PyTorch and ArrayFire implementations](Images/cuda_comparison.png)

"Resolution" refers to the number of frequency points between `f_min` and `f_max`.

## Migrating from the original API

Version 2.0 replaces per-signal data configurations, `CrossConfig`, and `SpectrumCalculator` with a
single `DataConfig`, channel tuples, and the `calculate_spectra` pipeline. See the
[migration guide](docs/migration.md) for a concept mapping and examples.

The original [SignalSnap](https://github.com/MarkusSifft/SignalSnap) remains available for
established ArrayFire workflows and currently provides features that are not part of the PyTorch
rewrite:

- downsampling;
- single-photon-regime measurements;
- stationarity testing;
- adding random phase to data; and
- potential support for GPUs not available in PyTorch.

## Development

Run the test suite from an editable development installation:

```bash
python -m pytest
```

## Authors and support

SignalSnap is developed by Armin Ghorbanietemad, Markus Sifft, and Daniel Hägele. The PyTorch
rewrite is currently maintained by David Ratayczak.

Development is supported by the Spectroscopy of Condensed Matter group at the Faculty of Physics and
Astronomy, Ruhr University Bochum.

## License

SignalSnap is distributed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.
