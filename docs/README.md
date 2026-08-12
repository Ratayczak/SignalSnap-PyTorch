# SignalSnap documentation

SignalSnap calculates first- through fourth-order auto- and cross-polyspectra for real,
multi-channel sampled signals and timestamped event streams.

## Guides

- [Calculation configuration](configuration.md): how to configure sampled, timestamped, and mixed
  calculations with `DataConfig` and `SpectrumConfig`.
- [Working with results](results.md): understand result and attached metadata.
- [Plotting your results](plotting.md): display spectra and uncertainties and inspect the first FFT
  window.
- [HDF5 input](hdf5.md): sampled and timestamped input from HDF5 files that may exceed memory
  limits.
- [Scientific background](scientific-background.md): review the definition, assumptions,
  estimators, and current scientific scope.
- [Migration from the original API](migration.md): translate ArrayFire-era workflows to the 2.0
  API.

Return to the [project README](../README.md).
