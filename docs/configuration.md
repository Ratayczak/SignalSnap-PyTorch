# Calculation configuration

[Documentation index](README.md)

## Input data

`DataConfig` describes the channels and their common sampling interval:

```python
data_config = DataConfig(
    channels=(channel_0, channel_1),
    dt=2.0,
    t_unit="ns",
)
```

`dt` is the time interval between consecutive samples in units of `t_unit`. Supported time units are
`"s"`, `"ms"`, `"us"`, `"ns"`, and `"ps"`; SignalSnap selects the corresponding frequency unit. For
example, `t_unit = "us"` will select the frequency unit `"MHz"`.

Array-backed channels must be one-dimensional, nonempty, real-valued numeric or Boolen arrays. All
channels used in one calculation must contain the same number of samples, e.g., through slicing.
HDF5-backed channels are described in the [HDF5 guide](hdf5.md).

## Spectrum settings

```python
spectrum_config = SpectrumConfig(
    df=0.0005,
    f_min=-0.10,
    f_max=0.25,
    m=10,
    uncertainty_estimation="global",
    m_var=10,
    device="cuda",
    precision="auto",
    spectral_estimates_max=1000,
    interlacing=True,
)
```

| Setting | Description |
| --- | --- |
| `df` | Requested frequency spacing. If omitted, each FFT window uses 1000 samples. |
| `f_min`, `f_max` | Requested frequency interval. `f_max=None` uses the Nyquist frequency as an upper bound. `f_min` may be negative. |
| `m` | Number of FFT windows contributing to each cumulant estimate. |
| `uncertainty_estimation` | `"global"` for the global standard error, or `"short_term"` for a typical local uncertainty. |
| `m_var` | Number of consecutive estimates in each short-term uncertainty batch. |
| `device` | `"cpu"`, `"cuda"`, `"cuda:N"`, `"mps"`, `"xpu"`, or `"xpu:N"`. |
| `precision` | `"single"`, `"double"`, or device-dependent `"auto"`. |
| `spectral_estimates_max` | Maximum unshifted estimates, or `None` to use all available data. |
| `interlacing` | Also calculate estimates shifted by half a window. |
| `old_window` | Compatibility option: uses the approximate confined Gaussian window from the original API. Intended only for reproducing results from the original API. |

Configuration objects are immutable and reject unknown fields.

### Frequency resolution and FFT windows

`df` specifies the requested frequency spacing. Together with the sampling interval `dt`, it
determines the FFT window length:

$$
\mathrm{window\_points} = \mathrm{round}\left(\frac{1}{\mathrm{dt} \cdot \mathrm{df}}\right).
$$

Because the window length must be an integer, the actual frequency spacing can differ slightly from
the requested value:

$$
\mathrm{df}_\mathrm{actual}
= \frac{1}{\mathrm{dt} \cdot \mathrm{window\_points}}.
$$

If `df` is omitted, `window_points` defaults to 1000 samples. Use `result.freq` as the authoritative
frequency axis.

The requested interval must remain within the Nyquist bounds `[-1/(2*dt), 1/(2*dt)]`. When `f_max`
is omitted, the positive Nyquist frequency is used as an upper bound.

Each channel is divided into FFT windows of `window_points` samples. One spectral estimate consumes
`m * window_points` samples: the `m` Fourier-coefficient vectors form the sample used by the
multivariate k-statistic. Consecutive groups produce repeated spectral estimates, which are averaged
to obtain the final result and its uncertainty estimate. See the
[Scientific background](scientific-background.md) for a detailed description of the calculation.

### Choosing `m`

For a spectrum of order `n`, the effective `m` must be at least `n`. If the trace is too short for
the configured `m`, SignalSnap warns and reduces it to the largest usable value. The calculation
fails if the reduced value is smaller than the highest requested order. `m=10` is the default value.

Lowering `m` below the default results in noisier cumulant estimates and is generally not
recommended if the data trace is long enough. Larger values of `m` provide more Fourier-coefficient
vectors per cumulant estimate but produce fewer spectral estimates for a fixed trace length. This
can improve computational efficiency since more data is processed in parallel. The
[statistical assumptions](scientific-background.md#statistical-assumptions) remain important.

### Devices and precision

`device` accepts `cpu`, `mps`, `cuda`, `xpu`, or a numbered accelerator such as `cuda:1` or
`xpu:1`. ROCm devices are also chosen using `cuda`. XPU and ROCm support is experimental because the
complete spectrum calculation has not yet been verified on Intel and AMD GPU hardware.

`precision="single"` uses `float32` and `complex64`; `"double"` uses `float64` and `complex128`.
`"auto"` selects single precision on MPS and XPU and double precision elsewhere.

On a machine with an available Intel GPU, users can run the numerical CPU/XPU comparison with
`python -m pytest tests/test_xpu.py` to verify that its outputs match the CPU results. The test is
skipped automatically on other systems.

### Limiting repeated estimates

`spectral_estimates_max` limits the number of estimates computed from the signal. The actual number
may still be smaller when the trace contains insufficient data. Set it to `None` to calculate as
many estimates as possible.

At least two estimates are needed for a global standard-error result. Short-term estimation needs
at least one complete batch of `m_var` estimates. Longer traces usually provide more useful
uncertainty estimates.

### Uncertainty estimation

`uncertainty_estimation="global"` reports the component-wise standard error of the mean calculated
from all estimates in each placement group.

`uncertainty_estimation="short_term"` divides consecutive estimates into complete batches of `m_var`.
For each batch it calculates the component-wise variance of the batch mean, then averages these
variances and takes their square root. This is a typical short-term uncertainty and does not shrink
with the number of completed batches. Incomplete trailing batches contribute to the spectrum but
not its uncertainty.

### Interlacing

With `interlacing=True`, SignalSnap also calculates estimates shifted by half an FFT window. This
reduces the low weight assigned to samples near the edges of the original window placement. The
trace must be long enough to contain at least one shifted estimate. `spectral_estimates_max` applies
only to unshifted estimates.

The final spectrum is averaged over the available unshifted and shifted estimates. Uncertainties are
calculated separately for the two groups. If both provide an uncertainty, their component-wise
maximum is reported as a conservative bound rather than the exact standard error of the combined
spectrum. If only one group has enough estimates for the configured method, its uncertainty is used.

Next: [Working with results](results.md).
