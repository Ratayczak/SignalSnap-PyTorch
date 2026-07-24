# Migrating from the original API

[Documentation index](README.md) · [Scientific background](scientific-background.md)

SignalSnap 2.0 replaces the original calculator and cross-configuration objects with a functional
pipeline.

## Concept mapping

| Original API | Current API |
| --- | --- |
| One data configuration per signal | One `DataConfig` containing all channels |
| `SpectrumConfig.spectrum_size` | `SpectrumConfig.df` |
| `CrossConfig` | Channel tuples in `requested_spectra` |
| `order_in` | Tuple lengths in `requested_spectra` |
| `SpectrumCalculator` | `calculate_spectra` |
| Spectrum dictionaries | `SpectrumResultStore` and `SpectrumResult` |
| `PlotConfig` and `SpectrumPlotter` | `PlotStyle` and plotting functions |

`SpectrumConfig.spectrum_size` has been replaced by `SpectrumConfig.df`. To preserve the
requested spacing of an existing configuration, convert it with:

$$
df = \frac{f_\mathrm{max}-f_\mathrm{min}}
           {\mathrm{spectrum\_size}-1}.
$$

For example, `f_min=0`, `f_max=0.5`, and `spectrum_size=9` become `df=0.0625`.

## Current workflow

Instead of building a separate data object for each signal, place all channels in one configuration:

```python
data_config = DataConfig(
    channels=(signal_trace_0, signal_trace_1, signal_trace_2),
    dt=0.001,
    t_unit="s",
)
```

Express auto- and cross-spectra directly as tuples:

```python
requested_spectra = [
    (0,),
    (0, 0),
    (0, 1),
    (0, 1, 1),
    (1, 2, 0, 2),
]

results = calculate_spectra(
    data_config,
    spectrum_config,
    requested_spectra=requested_spectra,
)
```

There is no separate `auto_corr` switch. Request auto-spectra with repeated indices and omit them
when only cross-spectra are required.

Retrieve results by the same tuple:

```python
cross_spectrum = results[(0, 1)]
values = cross_spectrum.spectrum
uncertainties = cross_spectrum.spectrum_uncertainty
frequencies = cross_spectrum.freq
```

Plotting is separate from calculation and consumes result objects. See the
[plotting guide](plotting.md).
