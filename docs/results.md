# Working with results

[Documentation index](README.md) · [Calculation configuration](configuration.md)

`calculate_spectra` returns a `SpectrumResultStore`. Retrieve a result using the same channel tuple
used in the request:

```python
results.spectra_metadata  # immutable mapping for all planned spectra

result = results[(0, 1)]

result.channels         # (0, 1)
result.order            # 2
result.freq             # one-dimensional NumPy frequency axis
result.freq_unit        # for example, "Hz"
result.spectrum         # complex NumPy array
result.spectrum_uncertainty   # complex NumPy array, or None
result.calculation_metadata   # CalculationMetadata object
result.spectrum_metadata      # SpectrumMetadata object
```

## Shapes and frequency coordinates

`result.freq` is the authoritative frequency axis. `f_min`, `f_max`, and `df` from `SpectrumConfig`
should not be used to reconstruct the frequency axis.

In a mixed calculation, results containing a sampled channel use the sampled FFT frequency view,
while results containing only timestamped channels use the direct-transform view and may extend
beyond the sampled Nyquist range. Consequently, different results in one store can have different
frequency axes.

For `N = len(result.freq)`, the returned shapes are:

| Order | `spectrum` shape | Interpretation |
| --- | --- | --- |
| 1 | `(1,)` | Zero-frequency component corresponding to the signal mean |
| 2 | `(N,)` | Spectrum evaluated along one frequency axis |
| 3 | `(N, N)` | Bispectrum evaluated on a two-dimensional grid |
| 4 | `(N, N)` | Diagonal two-dimensional trispectrum slice |

For order one, `result.freq` is just `[0]`.

A third-order value `S^(3)(frequency[i], frequency[j])` is stored in
`result.spectrum[i, j]`. It is `NaN` when the required third frequency,
`-(frequency[i] + frequency[j])`, is unavailable. In particular, a sampled closing channel is
limited by its FFT support, while a timestamped closing channel is transformed directly at the
required frequency.

An order-four value at `[i, j]` belongs to the diagonal slice
`(frequency[i], -frequency[i], frequency[j], -frequency[j])`. SignalSnap does not currently return
the full three-dimensional trispectrum.

## Uncertainty estimates

`spectrum_uncertainty` has the same shape as `spectrum`.
Its real and imaginary components independently store uncertainty estimates for the corresponding
spectrum components; the complex values themselves have no statistical interpretation. With global
uncertainty estimation these are standard errors of the mean. With short-term estimation they are
typical local uncertainties calculated from complete batches of `m_var` estimates.

```python
real_uncertainty = result.spectrum_uncertainty.real
imaginary_uncertainty = result.spectrum_uncertainty.imag
```

It is `None` unless at least two unshifted estimates are available.

See [Calculation configuration](configuration.md#uncertainty-estimation) for details. With
interlacing,
the component-wise maximum of the available placement-group uncertainties is a conservative bound,
not the exact standard error of the combined spectrum.

## Physical units

SignalSnap does not attach amplitude units to channels. If channel $k$ has amplitude unit $X_k$,
an order-$n$ spectrum has units

$$
\left(\prod_{k=1}^{n} X_k \right)\mathrm{t\_unit}^{n-1}
$$

or equivalently

$$
\left(\prod_{k=1}^{n} X_k \right)\mathrm{freq\_unit}^{1-n}.
$$

`freq_unit` is the inverse-time unit corresponding to the `t_unit` supplied in `DataConfig`.

For a unit-weighted `TimestampedChannel`, the signal is a counting measure and the order-one result
is a window-normalized average event rate rather than a raw event count. Exponential weighting
retains the configured amplitude scale and moments in the result.

## Calculation metadata

Pipeline results contain immutable metadata without retaining input arrays, tensors, open files, or
other runtime calculation state.

Calculation-wide metadata is available from the store and every returned result:

```python
calculation = results.calculation_metadata
result = results[(0, 1)]

assert result.calculation_metadata is calculation
```

`calculation_metadata` is a frozen `CalculationMetadata` object with these fields:

| Field | Meaning |
| --- | --- |
| `channel_kinds` | Kind of every configured channel, including inactive channels |
| `active_channels` | Channel indices used by the calculation, in resolved first-use order |
| `requested_spectra` | Every planned channel tuple in resolved request order |
| `observation_start`, `observation_stop` | Resolved half-open observation interval |
| `time_unit`, `frequency_unit` | Physical time unit and its reciprocal frequency unit |
| `requested_df`, `actual_df` | Requested and resolved Fourier spacing |
| `requested_f_min`, `requested_f_max` | Requested bounds; an omitted upper bound remains `None` |
| `window_duration` | Duration of one physical coefficient window |
| `unshifted_offset`, `shifted_offset` | Placement offsets; the latter is `None` without interlacing |
| `window_convention` | Default or legacy confined-Gaussian window convention |
| `photon_weighting` | Timestamp amplitude model, or `None` for sampled-only calculations |
| `exponential_scale` | Exponential amplitude scale when `photon_weighting = "exponential"`, is `None` for unit amplitudes |
| `repetition_count` | Number of amplitude realizations, is `1` for unit amplitudes  |
| `requested_repetition_batch_size` | Configured repetition batch size for exponential amplitude scaling, is `None` for unit amplitudes |
| `resolved_repetition_batch_size` | Repetition batch size actually used, is `1` for unit amplitudes |
| `user_seed` | Configured seed for exponential amplitude scaling, is `None` for unit amplitudes |
| `resolved_seed` | Seed actually used for exponential amplitude scaling, is `None` for unit amplitudes |
| `requested_m`, `effective_m` | Requested and resolved coefficient windows per estimate |
| `requested_m_var`, `effective_m_var` | Requested and resolved short-term uncertainty group size |
| `uncertainty_estimation` | `"global"` or `"short_term"` |
| `unshifted_physical_estimate_count` | Number of unshifted spectral estimates |
| `shifted_physical_estimate_count` | Number of interlaced spectral estimates |
| `unshifted_coefficient_window_count` | Coefficient windows used by unshifted estimates |
| `shifted_coefficient_window_count` | Coefficient windows used by shifted estimates |
| `real_dtype`, `complex_dtype` | Resolved calculation dtypes |
| `requested_device`, `resolved_device` | Requested and resolved PyTorch device |

For exponential weighting with no configured seed, `user_seed` is `None` and `resolved_seed`
contains the generated seed needed to reproduce the calculation:

```python
generated_seed = results.calculation_metadata.resolved_seed
```

Per-spectrum metadata is available both as the complete planned mapping on the store and as the
specific object attached to each returned result:

```python
all_spectrum_metadata = results.spectra_metadata
spectrum_metadata = result.spectrum_metadata

assert spectrum_metadata is all_spectrum_metadata[result.channels]
```

Each frozen `SpectrumMetadata` contains:

| Field | Meaning |
| --- | --- |
| `channels` | Requested channel tuple |
| `order` | Spectrum order derived from `channels` |
| `frequency_view` | `"sampled_fft"` or `"direct_transform"` |
| `effective_f_min`, `effective_f_max` | Minimum and maximum returned frequencies |
| `normalization_convention` | Applied sampled, timestamp, or mixed window normalization |
| `closing_frequency_support` | Third-order closing-frequency behavior, or `"not_applicable"` |

For first-order results, the effective bounds are both zero because the returned frequency axis is
`[0]`. The calculation still records its resolved Fourier spacing and requested bounds.

`requested_spectra` and `spectra_metadata` describe the complete planned calculation. The `results`
mapping contains only successfully returned spectra and may therefore contain fewer entries.
Filtered stores also retain the complete metadata mapping from the original calculation.

Selections preserve the original metadata objects:

```python
selected = results.select_by_order(2)

assert selected.calculation_metadata is results.calculation_metadata
assert selected.spectra_metadata is results.spectra_metadata
```

Manually constructed `SpectrumResult` and `SpectrumResultStore` objects may omit metadata. In that
case their metadata attributes are `None` and the store's `spectra_metadata` mapping is empty.


## Using the result store

The store preserves insertion order. Iteration yields `SpectrumResult` objects rather than channel
tuples:

```python
for result in results:
    print(result.channels, result.spectrum.shape)

print(f"Received {len(results)} results")

if (0, 1) in results:
    cross_spectrum = results[(0, 1)]
```

Accessing an unavailable tuple raises `KeyError`, distinguishing a missing result from one
containing zeros or `NaN` values.

Select an explicit set of results:

```python
selected = results.select([(0, 0), (0, 1)])
```

Filter by order or by the presence of a channel anywhere in the tuple:

```python
second_order = results.select_by_order(2)
involving_channel_zero = results.select_by_channel(0)
```

Filtered selections return an empty `SpectrumResultStore` when nothing matches. Every selection
returns a new store that shares the same `SpectrumResult` objects and underlying arrays with the
original store rather than copying them.

## Partial failures and warnings

Failures while computing, accumulating, or finalizing an individual requested spectrum are isolated
to the affected channel tuple. SignalSnap emits a `RuntimeWarning`, omits that result, and
continues with the other requests. Treat these warnings as significant and verify that the returned
store contains every requested tuple.

Next: [Plotting your results](plotting.md).
