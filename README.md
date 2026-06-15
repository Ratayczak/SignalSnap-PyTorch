# SignalSnap (PyTorch): Signal Analysis In Python Made Easy
by A. Ghorbanietemad, M. Sifft, and D. Hägele

After the initial release of SignalSnap, [here](https://github.com/MarkusSifft/SignalSnap), we have decided to rewrite and optimize it using PyTorch. This new version of SignalSnap aims to be more user friendly, faster and easier to maintain. The decision to change the backend library from ArrayFire to PyTorch was based on the following factors:
* Apple has been switching to its own GPU designs as part of its transition to Apple Silicon, meaning that supporting the MPS backend is now essential.
* PyTorch has rapidly grown in popularity, becoming one of the most widely-used libraries.
* More reliable and no occurrance of Heisenbugs (!) leading to an easier debugging process. 
* This active community provides extensive documentation, tutorials, and community-driven support, making it easier for developers to adopt and contribute to SignalSnap.
* Transitioning to PyTorch positions SignalSnap favorably for future development. This ensures long-term compatibility and easier maintenance compared to less mainstream frameworks.

<br> 

## Installation
Right now, you can clone this repository

```bash
git clone https://github.com/ArminGEtemad/SignalSnap-PyTorch.git
```
and install the package in your python environment via PIP. To include PyTorch with CUDA support adjust the extra index URL below to match your specific CUDA version (e.g., `cu126` for CUDA 12.6):
```bash
cd SignalSnap-PyTorch
pip install . --extra-index-url https://download.pytorch.org/whl/cu126
```

In the future, we plan to add PyPI support for a seamless installation of the package.

<br> 

## What are higher-order spectra?
Higher-order spectra contain additional information that is not contained within the second-order spectrum. SignalSnap is not only capable of calculating the first- and second-order but also third- and four-order spectrum. These have the following properties:
* Spectra beyond second-order are intrinsically insensitive to Gaussian noise.
* Third-order spectra show contributions whenever the phase of two frequencies and their sum are phase correlated.
* Fourth-order spectra are to be interpreted as intensity correlation between two frequencies or more simply, the correlation of two second-order spectra.

SignalSnap uses the definition of higher-order spectra $S^{(n)}_z$ introduced by [Brillinger in 1965](https://www.researchgate.net/publication/38365655_An_Introduction_to_Polyspectra): 
$$
2\pi\,\delta(\omega_1 + \cdots + \omega_n) \,S^{(n)}_z(\omega_1, \cdots , \omega_{n-1})= C_n\big(z(\omega_1), \cdots, z(\omega_n)\big),
$$
where $C_n$ are multivariate cumulants. The numerical calculation of these spectra is backed by __unbiased estimators for multivariate cumulants__ introduced by [Fabian Schefczik and Daniel Hägele](https://arxiv.org/pdf/1904.12154), leading to highly rigorous and reliable results which makes SignalSnap unique. 

In SignalSnap, you can calculate these spectra by first converting your data into a specific SignalSnap data object, putting these objects into a list, and select them (if necessary) by their index:
```python
data_object_1 = DataImportConfig(data=signal_trace_1)
data_object_2 = DataImportConfig(data=signal_trace_2)
data_object_list = [data_object_1, data_object_2]
selected_data = [0, 1]
```
Then you need to give SignalSnap your settings and configuration for further calculation:
```python
sconfig = SpectrumConfig(dt=0.001, f_min=0, f_max=5, s3_calc='1/2',f_unit='MHz',
                         backend='mps', order_in=[1, 2, 3, 4], spectrum_size=500,
                         show_first_frame=True)

# Configure for auto-correlation (cross-correlation options discussed below)
cconfig = CrossConfig(auto_corr=True) 
```
and in the end, SignalSnap needs your command to execute the calculation:
```python
scalc = SpectrumCalculator(sconfig, cconfig, data_object_list, selected=selected_data)
scalc.calc_spec();
```
and voilà the spectra are now accessible as python dictionaries:
```python
scalc.s # this is the spectrum value
scalc.s_err # this is the error value
scalc.freq # this is the frequency value
```
Of course, SignalSnap has its own built-in plotting functions with extensive configuration as well:
```python
pconfig = PlotConfig(f_min=0, f_max=5, display_orders=[1, 2, 3, 4],
                     significance=3, arcsinh_scale=(False, 0.02),
                     plot_format=['re', 'im'], insignif_transparency=0.5)
plotter = SpectrumPlotter(sconfig, cconfig, scalc, pconfig)
plotter.display()
```

<br> 

## Why higher-order multi-channel spectra?
Multiple stationary real signals may exhibit correlations. Such correlations can be characterized by cross-correlation spectra. Therefore, we generalized Brillinger's definition of higher-order spectra for such correlations:
$$
2\pi\,\delta(\omega_1 + \cdots + \omega_n) \,S^{(n)}_{z_1, \cdots, z_n}(\omega_1, \cdots , \omega_{n-1})= C_n\big(z_1(\omega_1), \cdots, z_n(\omega_n)\big)
$$
The cross-correlation spectra can be calculated simply by giving SignalSnap the instruction:
```python
cconfig = CrossConfig(auto_corr=True, 
                      cross_corr_2=[(0, 1), (1, 0)], # or any other permutation
                      cross_corr_3=[(0, 1, 1), (1, 0, 0), (0, 0, 1)], # or any other permutation
                      cross_corr_4=[(1, 0, 0, 1), (1, 1, 0, 0)]) # or any other permutation
```
Of course, if you are only interested in cross-correlation spectra you could set `auto_corr=False`. If you have more than two signal traces, e.g., `data_object_list = [data_object_1, data_object_2, data_object_3]` and `selected_data = [0, 1, 2]` the cross-correlation spectra could have more indices:
```python
cconfig = CrossConfig(auto_corr=True, cross_corr_2=[(0, 1), (2, 0)], # or any other permutation
                                      cross_corr_3=[(0, 1, 1), (2, 0, 0), (0, 2, 1)], # or any other permutation
                                      cross_corr_4=[(1, 0, 0, 1), (1, 1, 0, 0), (1, 2, 0, 2), (1, 2, 2, 1)]) # or any other permutation
```

<br> 

## Some benchmarking
The optimized new version of SignalSnap is much faster for higher resolutions. This calculation was done on 5GB of data on our PC with a GeForce RTX 4090 and for now only for the second-order spectrum.
![Runtime Comparison](Images/cuda_comparison.png)

<br> 

## Why is SignalSnap (ArrayFire) still important?
The version of SignalSnap introduced [here](https://github.com/MarkusSifft/SignalSnap) is still important and of interest, when you are looking for more functions, such as 
* downsampling
* single photon regime measurements
* test of stationarity
* adding random phase to data
* if you have an AMD graphic card

<br> 

## Dependencies
For the package multiple libraries are used for the numerics and displaying the results:
* NumPy
* SciPy
* MatPlotLib
* tqdm
* Numba
* h5py
* PyTorch (for CUDA)
* Pandas
* tabulate

<br> 

## Support
The development of the SignalSnap package is supported by the working group Spectroscopy of Condensed Matter of the Faculty of Physics and Astronomy at the Ruhr University Bochum.
