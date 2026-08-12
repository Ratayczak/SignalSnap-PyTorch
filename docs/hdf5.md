# HDF5 input

[Documentation index](README.md) · [Plotting your results](plotting.md)

SignalSnap can read only the required slices of HDF5 datasets instead of loading an entire
measurement into memory. This allows datasets much larger than system memory to be processed
incrementally.

Install the optional dependency:

```bash
python -m pip install ".[hdf5]"
```

## Defining channels

An `HDF5Source` identifies a file, dataset, and selection. The selected values are flattened in C
order into one logical signal channel:

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
        SampledChannel(
            data=HDF5Source(
                file=Path("measurement.h5"),
                dataset="/signals",
                selection=(slice(None), slice(None), 1),
            ),
            dt=2.0,
        ),
    ),
    t_unit="ns",
)
```

This example turns the last-axis entries `0` and `1` into separate SignalSnap channels.

The same storage descriptor can provide timestamped events:

```python
from signalsnap_pytorch import TimestampedChannel

timestamped_channel = TimestampedChannel(
    timestamps=HDF5Source(
        file=Path("measurement.h5"),
        dataset="/click_times",
        selection=(slice(None), slice(None)),
    )
)
```

## Selection rules

- Selection entries must be integers or `slice(...)`.
- Slice steps other than `1` are not supported.
- A selection must leave one or two dataset dimensions unfixed.
- Remaining dimensions are flattened in C/row-major order.
- Sampled selections must be nonempty and contain real numeric or Boolean data.
- Timestamp selections may be empty and must contain finite real numeric values that are
  nondecreasing after C-order flattening. Duplicate timestamps remain distinct events.
- All active sampled channels must contain the same number of selected samples. Timestamped
  channels may contain different numbers of events.

## Mixing storage types

Both channel types may reference either in-memory data or an `HDF5Source`. Storage types can be
mixed freely, for example between sampled channels:

```python
data_config = DataConfig(
    channels=(
        SampledChannel(data=in_memory_reference, dt=2.0),
        SampledChannel(data=hdf5_measurement, dt=2.0),
    ),
    t_unit="ns",
)
```

Only channels used by `requested_spectra` are opened and read. During calculation, SignalSnap reads
one required chunk at a time, so a selected dataset can exceed system memory. Fourier coefficients
and accumulators still require memory on the selected compute device; the exact amount depends on
the frequency grid, requested orders, and channel tuples.

Next: [Scientific background](scientific-background.md).
