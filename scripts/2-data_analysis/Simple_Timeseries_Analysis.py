import os

import geopandas
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.plot as rioplot

import u4py.analysis.processing as u4process
import u4py.plotting.axes as u4ax
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
    piloten = geopandas.read_file(
        os.path.join(base_folder, "Places", "Pilotregionen.dbf")
    )

    slp_2d, sea_2d, extent = load_data(file_list)

    tektonik_hessen = get_tektonik_hessen(
        os.path.join(places_path, "tektonik.dbf"),
        os.path.join(places_path, "vg2500_bld.dbf"),
    )
    if interactive:
        plot_gridded(
            slp_2d,
            sea_2d,
            extent,
            suptitle,
            places_path,
            tektonik_hessen,
            dpi=100,
        )
    else:
        plot_gridded(
            slp_2d,
            sea_2d,
            extent,
            suptitle,
            places_path,
            tektonik_hessen,
            save_path=os.path.join(plot_folder, f"Hessen_{suptitle}"),
        )
        names = ["Kassel", "Hoher_Meissner", "Werra_Kali", "Rhein-Main"]
        for num, pilot in piloten.values:
            name = names[num - 1]
            roi = pilot.bounds
            plot_gridded(
                slp_2d,
                sea_2d,
                extent,
                suptitle,
                places_path,
                tektonik_hessen,
                roi=roi,
                save_path=os.path.join(plot_folder, f"roi_{name}_{suptitle}"),
            )


def get_tektonik_hessen(tektonik_path, bld_path):
    tektonik = geopandas.read_file(tektonik_path).to_crs("EPSG:32632")
    bld = geopandas.read_file(bld_path).to_crs("EPSG:32632")
    hessen = bld[bld["GEN"] == "Hessen"]
    tektonik_hessen = geopandas.clip(tektonik, hessen)
    return tektonik_hessen


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


def plot_gridded(
    slp_2d: np.ndarray,
    sea_2d: np.ndarray,
    extent: tuple,
    suptitle: str,
    places_path: os.PathLike,
    tektonik_hessen: geopandas.GeoDataFrame,
    roi: geopandas.GeoDataFrame = None,
    save_path: os.PathLike = None,
    perc: int = 95,
    dpi: int = 300,
):
    """Creates a plot for gridded data

    Arguments:
        slp_2d -- A 2D Array containing the linear trend data.
        sea_2d -- A 2D Array containing the seasonal variation data.
        extent -- The extend of the 2D grid as (minx, maxx, miny, maxy) tuple.
        suptitle -- The title for the plot.
        places_path -- Path to the file containing the additional shape files to be plotted.
        tektonik_hessen -- GeoDataFrame with tectonic information of hessen.

    Keyword Arguments:
        roi -- GeoDataFrame containing the regions of interest for detailed plots. (default: {None})
        save_path -- Path where to save the plot. (default: {None})
        perc -- Percentile for the visualization. (default: {95})
        dpi -- Resolution of the plot for saving to png. (default: {300})
    """
    figwidth = 11.7
    figheight = 8.27

    if roi is not None:
        width = roi[2] - roi[0]
        height = roi[3] - roi[1]
        ratio = width / height
        figwidth = ratio * 1.25 * figwidth

    fig, axes = plt.subplots(
        ncols=2,
        sharex=True,
        sharey=True,
        dpi=dpi,
        figsize=(figwidth, figheight),
        layout="constrained",
    )
    rng = np.nanpercentile(np.abs(slp_2d), perc)
    slp = axes[0].imshow(
        slp_2d,
        vmin=-rng,
        vmax=rng,
        origin="lower",
        extent=extent,
        cmap="RdYlBu",
        zorder=1,
        alpha=0.8,
    )
    plt.colorbar(
        slp,
        ax=axes[0],
        label="Displacement (mm/a)",
        orientation="horizontal",
        extend="both",
        shrink=0.5,
    )
    rng = np.nanpercentile(sea_2d, perc)
    seas = axes[1].imshow(
        sea_2d,
        vmin=0,
        vmax=rng,
        origin="lower",
        extent=extent,
        zorder=1,
        alpha=0.8,
    )

    for ax in axes:
        with rasterio.open(
            os.path.join(places_path, "hessen_map.tif")
        ) as hessen_map:
            rioplot.show(hessen_map, ax=ax, zorder=0)
        tektonik_hessen.plot(ax=ax, color="k", zorder=2, linewidth=1.5)
        ax.grid("True", color="r", alpha=0.3)
    plt.colorbar(
        seas,
        ax=axes[1],
        label="Amplitude (mm)",
        orientation="horizontal",
        extend="both",
        shrink=0.5,
    )
    axes[0].set_title("Linear Component", fontweight="bold")
    axes[1].set_title("Seasonal Component", fontweight="bold")
    fig.suptitle(suptitle, fontsize="large", fontweight="bold")
    if roi is not None:
        axes[0].set_xlim(roi[0], roi[2])
        axes[0].set_ylim(roi[1], roi[3])
    # fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


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
