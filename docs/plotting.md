# Plotting your results

[Documentation index](README.md) · [Working with results](results.md)

Install Matplotlib support with:

```bash
python -m pip install ".[plotting]"
```

The helpers in `signalsnap_pytorch.plotting` display order-one results as a text table, order-two
results as line plots, and order-three and order-four results as two-dimensional color maps.

## Typical workflow

Assuming `results` is the `SpectrumResultStore` returned by `calculate_spectra`:

```python
from signalsnap_pytorch.plotting import (
    PlotStyle,
    build_order_1_table,
    create_spectrum_figures,
    save_figures,
)

plot_style = PlotStyle(
    f_min=0,
    f_max=100,
    sigma=3,
    uncertainty_levels=[1, 2, 3],
    plot_format=["re", "im"],
    arcsinh_ratio=0.02,
    insignificance_alpha=0.8,
)

print(build_order_1_table(results))

figures = create_spectrum_figures(results, plot_style)
saved_paths = save_figures(figures, "figures", dpi=200)
```

`create_spectrum_figures` skips order-one results and returns one `SpectrumFigure` for each result
of orders two through four, preserving result-store order. A `SpectrumFigure` contains the
Matplotlib figure and the corresponding order and channel metadata:

```python
spectrum_figure = figures[0]
print(spectrum_figure.order, spectrum_figure.channels)
spectrum_figure.figure.suptitle("My measurement")
```

Create a figure for one result with:

```python
from signalsnap_pytorch.plotting import create_spectrum_figure

spectrum_figure = create_spectrum_figure(results[(0, 1)], plot_style)
```

## Plot style

`PlotStyle` changes presentation, not calculated data:

| Setting | Description |
| --- | --- |
| `f_min`, `f_max` | Displayed limits. Crop the displayed axes without recalculating or resampling the spectrum. Applied to both axes for orders three and four. |
| `plot_format` | Components to draw: `["re"]`, `["im"]`, or `["re", "im"]`. |
| `sigma` | Uncertainty multiplier; defaults to `1`. |
| `uncertainty_levels` | Positive band multipliers for order-two plots, such as `[1, 2, 3]`. |
| `arcsinh_ratio` | Positive fraction controlling the approximately linear region of the arcsinh scaling; `None` uses linear scaling. |
| `insignificance_alpha` | Opacity from `0` to `1` of the white insignificance overlay. |

### Order-two uncertainty bands

When `spectrum_uncertainty` is available, order-two plots show shaded multiples of the component-wise
uncertainty estimate. If `uncertainty_levels` is set, it takes precedence over `sigma` for these
bands.

### Higher-order significance

Orders three and four use one color-map subplot per selected component and a symmetric color range
centered on zero. When uncertainties are available, points satisfying

```text
abs(real(spectrum)) < sigma * abs(real(spectrum_uncertainty))
abs(imag(spectrum)) < sigma * abs(imag(spectrum_uncertainty))
```

receive a white overlay with opacity `insignificance_alpha`. Invalid values, including unsupported
frequency combinations in a third-order result, are masked.

### Arcsinh scaling

Providing `arcsinh_ratio` enables arcsinh scaling in the plots. The width of the approximately
linear region is determined as the provided fraction (`arcsinh_ratio`) of the largest finite
absolute value in the uncropped real or imaginary spectrum. Axis limits do not affect this scale.

Arcsinh scaling can reveal smaller structures when a few large values dominate a plot while
retaining the sign and a nearly linear neighborhood around zero. Set `arcsinh_ratio` to a positive
fraction. This affects only the display.

## Saving figures

`save_figures` creates the output directory and returns the saved paths. Filenames are derived from
the order and channels, for example `s2_channels_0_1.png`:

```python
saved_paths = save_figures(
    figures,
    "./figures",
    extension="pdf",
    dpi=300,
    close=False,
)
```

The default output is PNG at 150 dpi. Figures are closed after saving by default to release
Matplotlib resources. Pass `close=False` to display or modify them afterwards. Supply `extension`
without a leading period.

## Inspecting the first FFT window

Inspect the first window before starting an expensive calculation:

```python
from signalsnap_pytorch.plotting import create_first_window_figure

figure = create_first_window_figure(
    data_config,
    spectrum_config,
    channels=[0, 1],
)
figure.show()
```

The window length uses the same frequency-resolution logic as the calculation. Omit `channels` to
plot every channel. Indices refer to `DataConfig.channels` and retain the requested order.

Array- and HDF5-backed channels are supported. Only the first window is read. A `ValueError` is
raised when a selected channel is shorter than the resolved window.

Next: [HDF5 input](hdf5.md).
