import os

import geopandas as gp
import matplotlib.pyplot as plt

import u4py.analysis.processing as u4process
import u4py.plotting.axes as u4ax
import u4py.plotting.plots as u4plots
import u4py.plotting.preparation as u4plotprep
import u4py.utils.files as u4files


def main():
    # chunked_analysis(interactive=True)
    single_h5_analysis(interactive=True)


def chunked_analysis(interactive=False):
    """
    Processes a full suite of datasets
    """
    # file_list = u4files.get_file_paths(filetypes=(("*.h5", "*.h5"),))
    file_list = u4files.get_file_list()
    base_folder = u4files.multi_split(file_list[0], nsplits=3)
    places_path = os.path.join(base_folder, "Places")
    plot_folder = os.path.join(base_folder, "INSAR_plots")
    suptitle = get_suptitle(file_list[0])
    piloten = gp.read_file(
        os.path.join(base_folder, "Places", "Pilotregionen.shp")
    )

    slp_2d, sea_2d, extent = load_data(file_list)
    base_map_path = os.path.join(places_path, "hessen_map.tif")
    tektonik_path = os.path.join(places_path, "tektonik_cropped.shp")
    if interactive:
        u4plots.plot_gridded(
            slp_2d,
            sea_2d,
            extent,
            suptitle=suptitle,
            base_map_path=base_map_path,
            tektonik_path=tektonik_path,
            dpi=100,
        )
    else:
        u4plots.plot_gridded(
            slp_2d,
            sea_2d,
            extent,
            suptitle=suptitle,
            base_map_path=base_map_path,
            tektonik_path=tektonik_path,
            save_path=os.path.join(plot_folder, f"Hessen_{suptitle}"),
        )
        names = ["Kassel", "Hoher_Meissner", "Werra_Kali", "Rhein-Main"]
        for num, pilot in piloten.values:
            name = names[num - 1]
            roi = pilot.bounds
            u4plots.plot_gridded(
                slp_2d,
                sea_2d,
                extent,
                suptitle=suptitle,
                base_map_path=base_map_path,
                tektonik_path=tektonik_path,
                roi=roi,
                save_path=os.path.join(plot_folder, f"roi_{name}_{suptitle}"),
            )


def load_data(file_list: list) -> tuple:
    """
    Gets processing results or recalculates them in case they don't exist.
    """
    results, chunk_size = u4process.get_processing_results(file_list)
    prepped_data = u4plotprep.convert_results_for_grid(results, chunk_size)
    gridded_data = u4plotprep.make_gridded_data(*prepped_data)
    return gridded_data


def get_suptitle(fname: str) -> str:
    """Generates the figure title depending on the filename"""
    if "ASCE" in fname:
        suptitle = "LOS Ascending (curve fitting)"
    elif "DESC" in fname:
        suptitle = "LOS Descending (curve fitting)"
    elif "BBD_Vert" in fname:
        suptitle = "Vertical (curve fitting)"
    elif "BBD_EW" in fname:
        suptitle = "East-West (curve fitting)"
    elif "merged" in fname:
        suptitle = "Vertical (full inversion)"
    return suptitle


def single_h5_analysis(interactive=True):
    file_list = u4files.get_file_paths(filetypes=(("*.h5", "*.h5"),))
    for file_path in file_list:
        data = u4files.load_hdf5(file_path)
        fig, ax = u4ax.plot_stat_func(data)
        # fig, ax = plot_statistics(data, timeslot=-1)
        # fig, ax = plot_cwt(data)
        # fig, ax = plot_timeseries(data)
        # plot_mean(data)
        if interactive:
            plt.show()
        else:
            fig.savefig(file_path.replace(".h5", "_simple"))


if __name__ == "__main__":
    main()
