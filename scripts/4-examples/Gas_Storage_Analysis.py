""" Shows possible correlation of surface motion with gas storage activity """


import os

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import u4py.addons.gas_storage as u4gas
import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4plotfmt
import u4py.utils.files as u4files


def main():
    # Paths
    base_path = u4files.get_folder_paths(title="Select base folder")
    psivert_path = u4files.get_folder_paths(title="Select PSI data folder")
    inversion_path = u4files.get_file_paths(
        filetypes=((".pkl", ".pkl"),), title="Select inversion results"
    )
    ext_path = os.path.join(base_path, "ExternalData")
    # places_path = os.path.join(base_path, "Places")
    # output_path = os.path.join(base_path, "INSAR_plots")
    # os.makedirs(output_path, exist_ok=True)
    # fig_path = os.path.join(output_path, "Riedstadt_GasField")

    points = u4files.get_point_points(
        (464350, 5516100), 300, "OilWell", psivert_path
    )

    # Load Data
    data_well = u4files.load_data_from_points(psivert_path, points)
    inversion_data, _ = u4files.get_pickled_inversion_results(
        inversion_path, points=points
    )
    date_gas, _, level_gas = u4gas.load_gas_data(
        os.path.join(ext_path, "Inventory Turnover Data_23.txt")
    )

    fig, axes = plt.subplots(nrows=3, sharex=True)
    u4ax.plot_timeseries(
        data_well["time"], data_well["timeseries"], ax=axes[0], color="C0"
    )
    u4ax.plot_inversion_fit(
        data_well["time"], inversion_data[0][2], ax=axes[0]
    )

    u4ax.plot_fit_residuals(data_well, inversion_data[0][2], ax=axes[1])

    axes[2].plot(date_gas, level_gas)
    u4plotfmt.add_copyright("Source: MND Energies, online", ax=axes[2])

    # Formatting
    axes[0].legend(loc="upper left")
    for ax in axes[:-1]:
        ax.set_ylabel("Vertical Displacement (mm)")
    axes[-1].set_ylabel("Fill level of gas storage (%)")
    axes[-1].set_xlabel("Year")

    plt.show()


def add_gas_geology(places_path: os.PathLike, ax: Axes):
    u4ax.add_shapefile(
        os.path.join(places_path, "Tiefenlinie_Top_Sand_7.shp"),
        ax=ax,
        column="Z",
        facecolor="none",
        zorder=1,
    )
    u4ax.add_shapefile(
        os.path.join(places_path, "Gas_Störungen.shp"),
        ax=ax,
        color="k",
        label="Faults",
        zorder=1,
    )


if __name__ == "__main__":
    main()
