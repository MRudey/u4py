import os
from ast import Interactive
from datetime import datetime, timedelta

import contextily as cx
import geopandas
import matplotlib.gridspec as mplgrid
import matplotlib.pyplot as plt
import matplotlib.widgets as mplwid
import numpy as np
import pycwt
import rasterio
import rasterio.plot as rioplot
import scipy.ndimage as spimg
import scipy.optimize as spopt
import scipy.signal as spsignal
import scipy.stats as spstats
import u4py.analysis.processing as u4process
import u4py.utils.files as u4files
import uncertainties as unc


def main():
    chunked_analysis(interactive=True)
    # original_file(interactive=False)


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

    data = load_data(file_list)
    slp_2d, sea_2d, extent = make_gridded_data(*data)
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


def load_data(file_list):
    results, chunk_size = u4process.get_processing_results(file_list)
    if not chunk_size:
        chunk_size = 250
    xmids = []
    ymids = []
    slope = []
    season = []
    for (xmid, ymid, components) in results:
        xmids.append(xmid)
        ymids.append(ymid)
        if components is not None:
            if len(components) < 5:
                slope.append(components[0] * 365.25)
                season.append(np.abs(components[1]))
            else:
                slope.append(components[13])
                season.append(np.abs(components[14]))
        else:
            season.append(np.nan)
    return xmids, ymids, slope, season, chunk_size


def get_suptitle(fname):
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


def make_gridded_data(xmids, ymids, slope, season, chunk_size):
    minx = np.min(xmids)
    maxx = np.max(xmids) + chunk_size
    miny = np.min(ymids)
    maxy = np.max(ymids) + chunk_size
    extent = (minx, maxx, miny, maxy)
    x = np.arange(minx, maxx, chunk_size)
    y = np.arange(miny, maxy, chunk_size)

    XX, YY = np.meshgrid(x, y)
    slp_2d = np.ones_like(XX) * np.nan
    sea_2d = np.ones_like(XX) * np.nan

    for xi, yi, sl, se in zip(xmids, ymids, slope, season):
        xn = int((xi - minx) / chunk_size)
        yn = int((yi - miny) / chunk_size)
        slp_2d[yn, xn] = sl
        sea_2d[yn, xn] = se

    slp_2d = spimg.median_filter(slp_2d, 3)
    sea_2d = spimg.median_filter(sea_2d, 3)

    return slp_2d, sea_2d, extent


def plot_gridded(
    slp_2d,
    sea_2d,
    extent,
    suptitle,
    places_path,
    tektonik_hessen,
    roi=None,
    save_path=None,
    perc=95,
    dpi=300,
):
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


def original_file(interactive=True):
    file_list = u4files.get_file_paths(filetypes=(("*.h5", "*.h5"),))
    for file_path in file_list:
        data = u4files.load_hdf5(file_path)
        # plot_statistics(data, timeslot=-1)
        # plot_cwt(data)
        fig, ax = plot_timeseries(data)
        # plot_mean(data)
        if interactive:
            plt.show()
        else:
            fig.savefig(file_path.replace(".h5", "_simple"))


def plot_mean(data):
    fig, ax = plt.subplots()
    ax.plot(
        np.mean(data["timeseries"], axis=0)
        / np.median(data["timeseries"], axis=0)
    )
    # ax.plot(run_stat_mean(data["timeseries"], fnc=spstats.t))
    plt.show()


def plot_cwt(data):
    time = data["time"]
    y = np.median(data["timeseries"], axis=0)
    freqs, coi, power = cwt(y, 6)

    fig, ax = plt.subplots()
    xx, yy = np.meshgrid(time, freqs)
    ax.contourf(xx, yy, power, levels=20)
    ax.fill_between(time, coi, np.min(freqs), color="w", alpha=0.25)
    plt.show()


def plot_timeseries(data):
    time = data["time"]
    y = np.nanmedian(data["timeseries"], axis=0)
    slc = np.nonzero(np.isfinite(y))
    time = time[slc]
    y = y[slc]
    time_days = np.linspace(0, len(time) * 6, len(time))

    lin_popt, lin_pcov = spopt.curve_fit(
        poly1,
        time_days,
        y,
    )
    linear_component = poly1(time_days, *lin_popt)
    y_detrend = y - linear_component

    popt, pcov = spopt.curve_fit(
        sinefunc,
        time_days,
        y_detrend,
        p0=[2, 6 / 365.25, 0],
    )
    sinus_component = sinefunc(time_days, *popt)
    y_residual = y_detrend - sinus_component

    fig, axes = plt.subplots(
        ncols=3, figsize=(15, 5), sharex=True, sharey=True
    )
    axes[0].set_title("Linear Trend")
    # axes[0].plot(time, y, ".")
    axes[0].plot(time, y, ".-", linewidth=0.5)
    axes[0].plot(time, linear_component)
    axes[0].annotate(
        "Jährliche Hebung/Senkung: %.1f mm/a" % ((lin_popt[0]) * 365.25),
        (0.95, 0.05),
        xycoords="axes fraction",
        horizontalalignment="right",
    )

    axes[1].set_title("Signal - Linear = Sinusoidal Trend")
    # axes[1].plot(time, y_detrend, ".")
    axes[1].plot(time, y_detrend, ".-", linewidth=0.5)
    axes[1].plot(time, sinus_component)
    shift = datetime(2015, 1, 1) - timedelta(days=np.abs(popt[2]))

    if shift.day > 10:
        prefix = "Mitte"
    elif shift.day > 20:
        prefix = "Ende"
    else:
        prefix = "Anfang"

    axes[1].annotate(
        "Jährliche Schwankung: $\\pm$%.1f mm\nPeriodizität: %i Tage\nMaximum: %s %s"
        % (np.abs(popt[0]), (1 / popt[1]) * 6, prefix, shift.strftime("%B")),
        (0.95, 0.95),
        xycoords="axes fraction",
        horizontalalignment="right",
        verticalalignment="top",
    )

    axes[2].set_title("Residuals")
    # axes[2].plot(time, y_residual, ".")
    axes[2].plot(time, y_residual, ".-", linewidth=0.5)

    return fig, axes


def plot_statistics(data, timeslot):
    yerr_min = np.percentile(data["timeseries"], 5, axis=0)
    yerr_max = np.percentile(data["timeseries"], 95, axis=0)
    rng = np.max(np.abs([np.min(yerr_min), np.max(yerr_max)]))
    points = data["timeseries"][:, timeslot]
    slc = clean_points(points, fnc=spstats.t)

    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    axes[0].set_title("PSI - Map")
    axes[0].scatter(
        data["x"][slc],
        data["y"][slc],
        c=points[slc],
        marker=".",
        cmap="RdBu",
        vmin=-rng,
        vmax=rng,
    )

    axes[1].set_title("PSI Density")
    axes[1].hexbin(data["x"], data["y"], gridsize=10)
    axes[1].sharex(axes[0])
    axes[1].sharey(axes[0])

    axes[2].set_title("All PSI - Single Interval")
    axes[2].hist(data["timeseries"][:, timeslot], "auto", density=True)
    xlims = axes[2].get_xlim()
    stat_dist = add_pdf(
        xlims, data["timeseries"][:, timeslot], axes[2], fnc=spstats.t
    )
    add_pdf(xlims, data["timeseries"][:, timeslot], axes[2], fnc=spstats.norm)
    mean_val = unc.ufloat(stat_dist.median(), 2 * stat_dist.std())
    axes[2].annotate(
        "N = %i\nMedian = %s mm"
        % (len(data["timeseries"][:, timeslot]), mean_val),
        (0.05, 0.95),
        xycoords="axes fraction",
        verticalalignment="top",
    )

    plt.show()


def add_pdf(xlims, ypoints, ax, fnc=spstats.norm):
    stat_vals = fnc.fit(ypoints)
    t_x = np.linspace(xlims[0], xlims[1], 100)
    ax.plot(t_x, fnc.pdf(t_x, *stat_vals))
    return fnc(*stat_vals)


def clean_points(points, sigma=2, fnc=spstats.norm):
    limit = sigma * fnc(*fnc.fit(points)).std()
    slc = np.nonzero(points > -limit) and np.nonzero(points < limit)
    return slc


def run_stat_mean(timeseries, fnc=spstats.norm):
    """
    Computes the running mean/median with the given statistical distribution
    """
    _, c = timeseries.shape
    ts_out = np.zeros(c)
    for ii in range(c):
        ts_out[ii] = get_stat_val(timeseries[:, ii], fnc=spstats.t)
    return ts_out


def get_stat_val(values, fnc=spstats.norm):
    return fnc(*fnc.fit(values)).mean()


def sinefunc(x, amplitude, width, shift):
    return amplitude * np.cos(width * (x + shift))


def poly1(x, slope, offset):
    return slope * x + offset


def cwt(y, dt):
    """
    Does a continuous wavelet transformation of the input data with a Morlet
    """
    # Detrend and normalize data for better cwt analysis
    y_detrend = spsignal.detrend(y)
    std = np.std(y_detrend)  # Standard deviation
    dat_norm = y_detrend / std  # Normalized dataset

    # Wavelet parameters
    mother = pycwt.wavelet.Morlet(6.0)
    s0 = 8 * dt  # Starting scale
    dj = 1 / 12  # sub-octaves
    J = 7 / dj  # Seven powers of two with dj sub-octaves

    # Do continous transform
    wave, scales, freqs, coi, _, _ = pycwt.wavelet.cwt(
        dat_norm, dt, dj, s0, J, mother
    )

    # Convert cone of influence from periods to frequency and set values above
    # threshold to maximum for better plotting
    coi = 1 / coi
    coi[coi >= np.max(freqs)] = np.max(freqs)

    # Calculate power spectrum
    power = (np.abs(wave)) ** 2
    power /= scales[:, None]

    return freqs, coi, power


if __name__ == "__main__":
    main()
