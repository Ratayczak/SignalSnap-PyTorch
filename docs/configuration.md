# Calculation configuration

[Documentation index](README.md)

## Input data

`DataConfig` accepts only explicit `SampledChannel` and `TimestampedChannel` objects. A sampled
channel owns its sampling interval:

```python
from signalsnap_pytorch import DataConfig, SampledChannel

data_config = DataConfig(
    channels=(
        SampledChannel(data=channel_0, dt=2.0),
        SampledChannel(data=channel_1, dt=2.0),
    ),
    t_unit="ns",
)
```

`dt` is the time interval between consecutive samples in units of `t_unit`. All sampled channels
active in one calculation currently require equal dt values and equal logical lengths. Supported
time units are `"s"`, `"ms"`, `"us"`, `"ns"`, and `"ps"`; SignalSnap selects the corresponding
frequency unit. For example, `t_unit = "us"` will select the frequency unit `"MHz"`.

Sampled in-memory data must be a one-dimensional, nonempty NumPy array or CPU PyTorch tensor
containing real numeric or Boolean values. HDF5-backed channels are described in the
[HDF5 guide](hdf5.md).

A `TimestampedChannel` instead stores the occurrence time of each discrete event:

```python
from signalsnap_pytorch import TimestampedChannel

data_config = DataConfig(
    channels=(TimestampedChannel(timestamps=event_times),),
    observation_start=0.0,
    observation_stop=10.0,
    t_unit="s",
)
```

In-memory timestamps must be a one-dimensional NumPy array or CPU PyTorch tensor of finite,
nondecreasing real numbers. Duplicate timestamps represent distinct events, and an empty event
stream is valid. Use timestamps relative to a nearby origin when large absolute times would lose
the required floating-point resolution.

Configuration objects retain their potentially large arrays and tensors without copying them.
Although the configuration models are frozen, the referenced data remains mutable and must not be
changed during a calculation.

`DataConfig.observation_start` and `observation_stop` describe a common half-open physical
interval. Sampled-only planning may default the start to zero and infer the stop from the active
channel length and `dt`. Both bounds are required when an active channel is timestamped. Events at
`observation_start` are included; events at `observation_stop` are outside the interval. In a mixed
calculation, the interval duration must equal the active sampled channels' length times `dt`.

Only channels used by `requested_spectra` are active. An unused timestamped channel therefore does
not require observation bounds or timestamp-specific spectrum options.

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
    spectral_estimates_per_batch=8,
    interlacing=True,
)
```

| Setting | Description |
| --- | --- |
| `df` | Requested frequency spacing. If omitted in a calculation with sampled channels, each FFT window uses 1000 samples. Timestamp-only calculations require it. |
| `f_min`, `f_max` | Inclusive requested frequency interval. `f_min` and `f_max` may be negative. `f_max=None` uses the sampled Nyquist frequency; timestamp-only calculations require an explicit `f_max`. |
| `photon_options` | Required event-amplitude treatment for active timestamped channels; rejected for sampled-only calculations. |
| `m` | Number of physical windows contributing to each cumulant estimate. |
| `uncertainty_estimation` | `"global"` for the global standard error, or `"short_term"` for a typical local uncertainty. |
| `m_var` | Number of consecutive estimates in each short-term uncertainty batch. |
| `device` | `"cpu"`, `"cuda"`, `"cuda:N"`, `"mps"`, `"xpu"`, or `"xpu:N"`. |
| `precision` | `"single"`, `"double"`, or device-dependent `"auto"`. |
| `spectral_estimates_max` | Maximum unshifted estimates, or `None` to use all available data. |
| `spectral_estimates_per_batch` | Number of independent spectral estimates calculated in parallel. |
| `interlacing` | Also calculate estimates shifted by half a window. |
| `old_window` | Compatibility option: uses the original API's sampled or timestamped window convention. Intended only for reproducing results from the original API. |

Configuration objects are immutable and reject unknown fields.

### Timestamped event weighting

`PhotonOptions` applies to every active timestamped channel. Unit weighting assigns amplitude one
to each event and performs one deterministic calculation:

```python
from signalsnap_pytorch import PhotonOptions

photon_options = PhotonOptions(weighting="unit")
```

Exponential weighting draws independent positive event amplitudes for each realization and
averages the resulting spectra:

```python
photon_options = PhotonOptions(
    weighting="exponential",
    scale=1.0,
    repetitions=100,
    repetitions_per_batch=10,
    seed=1234,
)
```

A positive `scale` and a positive integer `repetitions` are required for exponential weighting.
The positive integer `repetitions_per_batch` limits how many realizations are processed together
and defaults to at most 10. An explicit nonnegative `seed` makes the generated amplitudes
reproducible independently of batching; omitting it chooses a new seed for each calculation. These
exponential-only fields are invalid with unit weighting.

Unit weighting treats the timestamps as a counting measure. To reproduce the exponentially
distributed detector-pulse amplitudes described by Sifft et al. in
[*Physical Review A* 109, 062210 (2024)](https://doi.org/10.1103/PhysRevA.109.062210), use
exponential weighting with `scale=1.0` and typically 100 repetitions. Add `old_window=True` when
reproducing the historical SignalSnap window convention.

### Frequency resolution and physical windows

`df` specifies the requested frequency spacing. Together with the active sampled channels' common
sampling interval `dt`, it determines the FFT window length:

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

Timestamp-only calculations require `df` and `f_max`. Their window duration is exactly `1 / df`,
and timestamps are transformed directly at the zero-anchored frequency grid within the inclusive
`f_min` and `f_max` bounds. Empty physical windows and a complete event-free tail still contribute
to the available estimate count.

In mixed calculations, sampled data determines the physical window duration and therefore the
actual frequency spacing. A result containing any sampled channel uses the frequencies supported
by the sampled FFT. A timestamp-only result may extend beyond the sampled Nyquist range, so results
in the same `SpectrumResultStore` can have different frequency axes. Always use `result.freq` as
the authoritative axis.

Each channel is divided into the same physical windows. Sampled values are transformed by FFT;
timestamped events retain their arrival times and are transformed directly. One spectral estimate
uses `m` Fourier-coefficient vectors as the sample for the multivariate k-statistic. Consecutive
groups produce repeated spectral estimates, which are averaged to obtain the final result and its
uncertainty estimate. See the
[Scientific background](scientific-background.md) for a detailed description of the calculation.

### Choosing `m`

For a spectrum of order `n`, the effective `m` must be at least `n`. If the observation is too short
for the configured `m`, SignalSnap warns and reduces it to the largest usable value. The
calculation fails if the reduced value is smaller than the highest requested order. `m=10` is the
default value.

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

### Batching spectral estimates

`spectral_estimates_per_batch` controls how many independent spectral estimates are calculated in
parallel. Increasing it can reduce calculation time, especially on accelerators, while increasing
device-memory use. The final calculation batch may contain fewer estimates when the available
estimate count is not divisible by the configured batch size.

This setting changes only computational batching. Each spectral estimate still contains `m`
physical windows, and short-term uncertainty batches are still defined independently by `m_var`.
The default value is `1`.

### Uncertainty estimation

`uncertainty_estimation="global"` reports the component-wise standard error of the mean calculated
from all estimates in each placement group.

`uncertainty_estimation="short_term"` divides consecutive estimates into complete batches of `m_var`.
For each batch it calculates the component-wise variance of the batch mean, then averages these
variances and takes their square root. This is a typical short-term uncertainty and does not shrink
with the number of completed batches. Incomplete trailing batches contribute to the spectrum but
not its uncertainty.

### Interlacing

With `interlacing=True`, SignalSnap also calculates estimates shifted by half a physical window (or
the corresponding whole-sample offset for sampled input). This reduces the low weight assigned to
measurements near the edges of the original window placement. The observation must be long enough
to contain at least one shifted estimate. `spectral_estimates_max` applies only to unshifted
estimates.

The final spectrum is averaged over the available unshifted and shifted estimates. Uncertainties are
calculated separately for the two groups. If both provide an uncertainty, their component-wise
maximum is reported as a conservative bound rather than the exact standard error of the combined
spectrum. If only one group has enough estimates for the configured method, its uncertainty is used.

Next: [Working with results](results.md).
