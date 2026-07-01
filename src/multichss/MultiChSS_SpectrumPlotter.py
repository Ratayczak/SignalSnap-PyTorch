"""Legacy plotting helpers for the deprecated SpectrumCalculator API."""

# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from multichss.MultiChSS_SpectrumCalculator import SpectrumCalculator
from multichss.configurators import CrossConfig, PlotConfig, SpectrumConfig
 
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

def custom_colormap():
    # Define the first list of colors (normalized to [0, 1] already)
    colors_list_1 = [
        (0, 0, 0.8),         # Dark blue
        (0.2, 0.4, 1),       # Lighter blue
        (0.6, 0.7, 1),       # Pale blue
        (0.92, 0.92, 0.92),   # Neutral gray at zero
        (1, 0.6, 0.6),       # Pale red
        (1, 0.2, 0.2),       # Medium red
        (0.8, 0, 0)          # Dark red
    ]
    
    # Define the second list of colors in RGB (0-255 scale)
    colors_list = [
        (23, 51, 107),      # Dark blue
        (82, 137, 190),     # Lighter blue
        (165, 203, 230),    # Pale blue
        (0.92*255, 0.92*255, 0.92*255),  # Neutral gray at zero
        (235, 164, 120),    # Pale red
        (188, 84, 68),
        (107, 22, 38)       # Medium red
    ]
    # Normalize to [0, 1]
    colors_list = np.array(colors_list) / 255.0

    # Average the two lists
    colors_list_2 = (np.array(colors_list_1) + colors_list) / 2.0

    # Create the colormap
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors_list_2)
    return cmap

def custom_error_colormap(insignif_transparency):

    color_array = [
        (0.0, 0.0, 0.0, 0.0),  # transparent
        (1.0, 1.0, 1.0, insignif_transparency),  # semi-transparent green
    ]
    return LinearSegmentedColormap.from_list('white_alpha', color_array)

class SpectrumPlotter:
    def __init__(self, sconfig: SpectrumConfig, cconfig: CrossConfig, scalc: SpectrumCalculator,  pconfig: PlotConfig):
        self.sconfig = sconfig
        self.scalc = scalc
        self.cconfig = cconfig
        self.pconfig = pconfig

    def import_spec_data(self, order, keys):
        """
        Import spectrum data, handling cases where the order or keys might not exist.
        """
        try:
            s_data = self.scalc.s[keys][order].copy() if self.scalc.s[keys][order] is not None else None
            s_err_data = self.scalc.s_err[keys][order].copy() if self.scalc.s_err[keys][order] is not None else None
            freq_data = self.scalc.freq[keys][order].copy() if self.scalc.freq[keys][order] is not None else None
        except (KeyError, AttributeError):
            s_data, s_err_data, freq_data = None, None, None
        return s_data, s_err_data, freq_data

    def signif_bound_calculate(self, s_data, s_err_data):
        """
        Calculates the significance bounds for the given spectrum data.
        """
        return [
            [s_data + (i + 1) * s_err_data for i in range(self.pconfig.significance)],
            [s_data - (i + 1) * s_err_data for i in range(self.pconfig.significance)]
        ]

    def arcsinh_scale(self, s_data, s_err_data):
        s_max = np.max(np.abs(s_data))
        scale = 1 / (s_max * self.pconfig.arcsinh_scale[1])
        s_data = np.arcsinh(scale * s_data) / scale

        if s_err_data is not None:
            s_err_data = np.arcsinh(scale * s_err_data) / scale

        return s_data, s_err_data

    def plot_first_frames(self, diconfig_list, selected, window_size):
        """
        visualizes the first frame of the selected data.
        """
        n_plots = len(selected)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots))
        if n_plots == 1:
            axes = [axes]
        for i, idx in enumerate(selected):
            data_config = diconfig_list[idx]
            first_frame = data_config.data[:window_size]
            t = np.arange(len(first_frame)) * self.sconfig.dt
            axes[i].plot(t, first_frame)
            axes[i].set_xlim([0, t[-1]])
            axes[i].set_title(f'First frame for data {idx}')
            axes[i].set_xlabel(f't / ({self.scalc.t_unit})')
            axes[i].set_ylabel('Amplitude')
        plt.tight_layout()
        if self.pconfig.output == "show":
            plt.show()
        elif self.pconfig.output == "save":
            plt.savefig(self.pconfig.output_folder / "first_frame.png")
        else:
            # Runtime guard clause in case they ignore type hints
            raise ValueError(f"Invalid action '{self.pconfig.output}'. Expected 'show' or 'save'.")

    def display_s1(self, order, keys, source):
        """
        Function to handle the processing and display for order 1.
        """
        all_results = []
        s_data, s_err_data, _ = self.import_spec_data(order, keys)

        if s_data is not None and s_err_data is not None:
            # Create a list of dictionaries for each row
            spectrum = s_data
            error_estimate = s_err_data

            for i in range(len(spectrum)):
                all_results.append({
                    'Dataset Index': keys,
                    'S1': spectrum[i].real,
                    'Error S1': error_estimate[i].real
                })

        return all_results

    def display_s2(self, order, datasets):
        """
        Function to handle plotting for order 2. Displays real and/or imaginary parts based on plot_format,
        with significance bounds shaded in gray and the area between filled.
        Additionally, displays a table for scaling information (once, as it's shared across all plots).
        """
        def plot_data(ax, freq_data, s_data, signif_bounds, component, label_prefix):
            """
            Helper to plot the data for the specified component (real/imag).
            """
            # Plot main data
            data = getattr(s_data, component)
            ax.plot(freq_data, data, label=f'{label_prefix} ({component.capitalize()})')

            # Plot significance bounds
            num_bounds = len(signif_bounds[0])
            grays = np.linspace(0.8, 0.3, num_bounds)  # Gradual shades of gray
            for i, (upper, lower) in enumerate(zip(signif_bounds[0], signif_bounds[1])):
                upper_data = getattr(upper, component)
                lower_data = getattr(lower, component)
                ax.plot(freq_data, upper_data, linestyle='--', color=str(grays[i]), alpha=0.5)
                ax.plot(freq_data, lower_data, linestyle='--', color=str(grays[i]), alpha=0.5)
                ax.fill_between(
                    freq_data, lower_data, upper_data,
                    color=str(grays[i]), alpha=0.2,
                    label='Significance Bounds' if i == 0 else None
                )

        def configure_axes_s2(ax, title, ylabel):
            """
            Helper to configure axis labels and title.
            """
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlim(self.pconfig.f_min, self.pconfig.f_max)
            ax.legend()

        # Initialize variables for scaling information
        scaled = False
        scale_factor = None

        num_columns = len(self.pconfig.plot_format)
        num_datasets = len(datasets)

        # Create subplots with the required number of columns
        fig, axes = plt.subplots(
            num_datasets, num_columns, figsize=(8 * num_columns, 4 * num_datasets), sharex=True
        )

        # Normalize axes for consistent handling
        if num_datasets == 1:
            axes = [axes] if num_columns == 1 else [axes]
        elif num_columns == 1:
            axes = [[ax] for ax in axes]

        for (keys, source), ax_row in zip(datasets, axes):
            s_data, s_err_data, freq_data = self.import_spec_data(order, keys)

            if s_data is not None and freq_data is not None and s_err_data is not None:
                # Check and apply arcsinh scaling (once)
                if scale_factor is None and self.pconfig.arcsinh_scale[0]:
                    scaled = True
                    s_max = np.max(np.abs(s_data))
                    scale_factor = self.pconfig.arcsinh_scale[1]

                # Apply scaling (if enabled)
                if self.pconfig.arcsinh_scale[0]:
                    s_data, s_err_data = self.arcsinh_scale(s_data, s_err_data)

                # Plot the data
                signif_bounds = self.signif_bound_calculate(s_data, s_err_data)
                for col, ax in enumerate(ax_row):
                    component = 'real' if self.pconfig.plot_format[col] == 're' else 'imag'
                    plot_data(ax, freq_data, s_data, signif_bounds, component, f'{source}: Dataset {keys}')
                    ylabel = f'{component.capitalize()} S Data'
                    title = f'{source}: {component.capitalize()} Part - Dataset {keys}'
                    configure_axes_s2(ax, title, ylabel)
            else:
                for ax in ax_row:
                    ax.text(0.5, 0.5, f"No data for {source}: key {keys}", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{source}: Dataset {keys}')
                    ax.set_ylabel('S Data')
                    ax.set_xlim(self.pconfig.f_min, self.pconfig.f_max)

        # Set shared x-axis label
        if num_datasets == 1 and num_columns == 1:
            axes[0][0].set_xlabel('Frequency')
        else:
            for ax in axes[-1]:
                ax.set_xlabel('Frequency')

        plt.tight_layout()
        if self.pconfig.output == "show":
            plt.show()
        elif self.pconfig.output == "save":
            plt.savefig(self.pconfig.output_folder / "s2.png")
        else:
            # Runtime guard clause in case they ignore type hints
            raise ValueError(f"Invalid action '{self.pconfig.output}'. Expected 'show' or 'save'.")

        # Display the scaling information (once)
        s2_table_data = [{
            'Arcsinh Scaled': scaled,
            'Scale Factor': scale_factor if scaled else "N/A"
        }]
        s2_table = pd.DataFrame(s2_table_data)
        print("\nS2 Scaling Information:")
        print(tabulate(s2_table, headers='keys', tablefmt='pretty', showindex=False))

    def display_sN(self, order):
        """
        Function to handle plotting for a given 'order' (e.g., 3 or 4).
        Displays real and/or imaginary parts based on plot_format,
        optionally with arcsinh scaling if configured.
        Plots a 2D color map (pcolormesh) for each dataset.
        This function plots both normal (selected) order-N spectra and,
        additionally, cross correlations for order N.
        """

        def plot_data(ax, X, Y, Z, title, freq_label, cmap):
            """
            Helper to plot the 2D data (Z) on ax with pcolormesh.
            X and Y are meshgrids of the frequency axis.
            """
            # Symmetric limits around zero
            data_max = Z.max()
            data_min = Z.min()
            limit = max(abs(data_max), abs(data_min))
            # pcolormesh ensures zero is centered if we set vmin=-limit, vmax=limit
            cmesh = ax.pcolormesh(
                X, Y, Z,
                cmap=cmap,
                vmin=-limit, vmax=limit
            )
            ax.set_title(title)
            ax.set_xlabel(freq_label)
            ax.set_ylabel(freq_label)
            if order == 3:
                if self.pconfig.f_min < 0:
                    ax.set_ylim(0, self.pconfig.f_max/2)
                else:
                    ax.set_ylim(self.pconfig.f_min/2, self.pconfig.f_max/2)
            else:
                ax.set_ylim(self.pconfig.f_min, self.pconfig.f_max)
            if order == 3:
                ax.set_xlim(self.pconfig.f_min/2, self.pconfig.f_max/2)
            else:
                ax.set_xlim(self.pconfig.f_min, self.pconfig.f_max)
            return cmesh

        def configure_axes(fig, ax, cmesh):
            """
            Helper to configure axes (labels, colorbars, etc.) for sN plots.
            """
            # Attach a colorbar specific to this Axes
            fig.colorbar(cmesh, ax=ax)

        cmap = custom_colormap()
        cmap_err  = custom_error_colormap(self.pconfig.insignif_transparency)

        # -------------------------------------------------------------------------
        # 1. Plot normal (selected) order-N spectra
        # -------------------------------------------------------------------------
        datasets_normal = []
        for source, selected_keys in [("selected", self.scalc.selected)]:
            for keys in selected_keys:
                s_data, s_err_data, freq_data = self.import_spec_data(order, keys)

                if s_data is not None and freq_data is not None:
                    datasets_normal.append((keys, source, s_data, s_err_data, freq_data))

        if datasets_normal:
            scaled = False
    
            scale_factor = None
            num_datasets = len(datasets_normal)
            num_columns = len(self.pconfig.plot_format)
            fig, axes = plt.subplots(
                num_datasets, num_columns,
                figsize=(8 * num_columns, 5 * num_datasets),
                squeeze=False
            )
            for (keys, source, s_data, s_err_data, freq_data), ax_row in zip(datasets_normal, axes):
                # Apply arcsinh scaling if enabled
                if self.pconfig.arcsinh_scale[0]:
                    if scale_factor is None:
                        s_max = np.max(np.abs(s_data))
                        scale_factor = self.pconfig.arcsinh_scale[1]
                        scaled = True
                    s_data, s_err_data = self.arcsinh_scale(s_data, s_err_data)

                # Create a 2D grid from 1D frequency data
                if order == 3:
                    if self.sconfig.s3_calc == '1/2':
                        X, Y = np.meshgrid(freq_data, freq_data[freq_data.size//2:])
                    elif self.sconfig.s3_calc == '1/4':
                        X, Y = np.meshgrid(freq_data, freq_data)
                else:
                    X, Y = np.meshgrid(freq_data, freq_data)

                for col, ax in enumerate(ax_row):
                    component = self.pconfig.plot_format[col]
                    if component == 're':
                        Z = np.real(s_data)
                        comp_label = "Real"
                    elif component == 'im':
                        Z = np.imag(s_data)
                        comp_label = "Imag"
                    else:
                        # Default to Real part if something else is specified
                        Z = np.real(s_data)
                        comp_label = "Real"

                    plot_title = f"{source}: Order {order} {comp_label} - Dataset {keys}"
                    cmesh = plot_data(
                        ax=ax,
                        X=X,
                        Y=Y,
                        Z=Z,
                        title=plot_title,
                        freq_label="Frequency",
                        cmap=cmap
                    )
                    configure_axes(fig, ax, cmesh)

                    if s_err_data is not None:
                        s_err_data *= self.pconfig.significance 
                        err_matrix = np.zeros_like(Z, dtype=float)
                        # Mark 1 where the error is larger than the signal
                        err_matrix[np.abs(Z) < s_err_data] = 1
                        # Overlay with semi-transparent green
                        ax.pcolormesh(X, Y, err_matrix,cmap=cmap_err,vmin=0,vmax=1,shading='auto')
            plt.tight_layout()
            if self.pconfig.output == "show":
                plt.show()
            elif self.pconfig.output == "save":
                plt.savefig(self.pconfig.output_folder / "sN1.png")
            else:
                # Runtime guard clause in case they ignore type hints
                raise ValueError(f"Invalid action '{self.pconfig.output}'. Expected 'show' or 'save'.")

            sN_table_data = [{
                'Arcsinh Scaled': scaled,
                'Scale Factor': scale_factor if scaled else "N/A"
            }]
            sN_table = pd.DataFrame(sN_table_data)
            print(f"\nS{order} Scaling Information (Normal):")
            print(tabulate(sN_table, headers='keys', tablefmt='pretty', showindex=False))
        else:
            print(f"No normal order {order} data available.")

        # -------------------------------------------------------------------------
        # 2. Plot cross correlation order-N spectra
        # -------------------------------------------------------------------------
        cross_list = getattr(self.scalc, f"cross{order}_selected", [])

        datasets_cross = []
        for source, cross_keys in [("cross", cross_list)]:
            for keys in cross_keys:
                s_data, s_err_data, freq_data = self.import_spec_data(order, keys)
                if s_data is not None and freq_data is not None:
                    datasets_cross.append((keys, source, s_data, s_err_data, freq_data))

        if datasets_cross:
            scaled = False
            scale_factor = None
            num_datasets = len(datasets_cross)
            num_columns = len(self.pconfig.plot_format)
            fig, axes = plt.subplots(
                num_datasets, num_columns,
                figsize=(8 * num_columns, 5 * num_datasets),
                squeeze=False
            )
            for (keys, source, s_data, s_err_data, freq_data), ax_row in zip(datasets_cross, axes):
                # Apply arcsinh scaling if enabled
                if self.pconfig.arcsinh_scale[0]:
                    if scale_factor is None:
                        s_max = np.max(np.abs(s_data))
                        scale_factor = self.pconfig.arcsinh_scale[1]
                        scaled = True
                    s_data, s_err_data = self.arcsinh_scale(s_data, s_err_data)

                if order == 3:
                    if self.sconfig.s3_calc == '1/2':
                        X, Y = np.meshgrid(freq_data, freq_data[freq_data.size//2:])
                    elif self.sconfig.s3_calc == '1/4':
                        X, Y = np.meshgrid(freq_data, freq_data)
                else:
                    X, Y = np.meshgrid(freq_data, freq_data)

                for col, ax in enumerate(ax_row):
                    component = self.pconfig.plot_format[col]
                    if component == 're':
                        Z = np.real(s_data)
                        comp_label = "Real"
                    elif component == 'im':
                        Z = np.imag(s_data)
                        comp_label = "Imag"
                    else:
                        Z = np.real(s_data)
                        comp_label = "Real"

                    plot_title = f"{source}: Order {order} {comp_label} - Datasets {keys}"
                    if order == 3:
                        cmesh = plot_data(
                            ax=ax,
                            X=X,
                            Y=Y,
                            Z=Z,
                            title=plot_title,
                            freq_label="Frequency",
                            cmap=cmap)
                    else:
                        cmesh = plot_data(
                        ax=ax,
                        X=Y,
                        Y=X,
                        Z=Z,
                        title=plot_title,
                        freq_label="Frequency",
                        cmap=cmap)

                    if s_err_data is not None:
                        s_err_data *= self.pconfig.significance 
                        err_matrix = np.zeros_like(Z, dtype=float)
                        # Mark 1 where the error is larger than the signal
                        err_matrix[np.abs(Z) < s_err_data] = 1
                        # Overlay with semi-transparent green
                        if order ==3:
                            ax.pcolormesh(X, Y, err_matrix,cmap=cmap_err,vmin=0,vmax=1,shading='auto')
                        else:
                            ax.pcolormesh(Y, X, err_matrix,cmap=cmap_err,vmin=0,vmax=1,shading='auto')


                    configure_axes(fig, ax, cmesh)
            plt.tight_layout()
            if self.pconfig.output == "show":
                plt.show()
            elif self.pconfig.output == "save":
                plt.savefig(self.pconfig.output_folder / "sN2.png")
            else:
                # Runtime guard clause in case they ignore type hints
                raise ValueError(f"Invalid action '{self.pconfig.output}'. Expected 'show' or 'save'.")

            sN_table_data = [{
                'Arcsinh Scaled': scaled,
                'Scale Factor': scale_factor if scaled else "N/A"
            }]
            sN_table = pd.DataFrame(sN_table_data)
            print(f"\nS{order} Scaling Information (Cross):")
            print(tabulate(sN_table, headers='keys', tablefmt='pretty', showindex=False))
        else:
            print(f"No cross order {order} data available.")

    def display(self):
        all_results = []
        datasets = []
        generate_s2_plots = False
        generate_s3_plots = False
        generate_s4_plots = False

        if self.sconfig.show_first_frame:
            self.plot_first_frames(self.scalc.diconfig_list, self.scalc.selected, self.scalc.window_points)

        # For the 'selected' datasets, only process orders that are in display_orders.
        for source, selected_keys in [
            ("selected", self.scalc.selected),
            ("cross2_selected", self.scalc.cross2_selected)
        ]:
            for keys in selected_keys:
                # For cross2_selected, we assume only order 2 is valid.
                if source == "cross2_selected":
                    orders_to_process = [order for order in self.pconfig.display_orders if order == 2]
                else:
                    orders_to_process = self.pconfig.display_orders

                for order in orders_to_process:
                    if order == 1:
                        results = self.display_s1(order, keys, source)
                        all_results.extend(results)
                    elif order == 2:
                        datasets.append((keys, source))
                        generate_s2_plots = True
                    elif order == 3:
                        generate_s3_plots = True
                    elif order == 4:
                        generate_s4_plots = True

        # Display the S1 results as a table.
        df = pd.DataFrame(all_results)
        if not df.empty:
            print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))
        else:
            print("No results available for order 1.")

        # Then handle the S2 and S4 plots.
        if generate_s2_plots:
            self.display_s2(order=2, datasets=datasets)
        if generate_s3_plots:
            self.display_sN(order=3)
        if generate_s4_plots:
            self.display_sN(order=4)