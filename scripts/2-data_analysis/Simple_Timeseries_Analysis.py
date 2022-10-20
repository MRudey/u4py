import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pycwt
import scipy.ndimage as spimg
import scipy.optimize as spopt
import scipy.signal as spsignal
import scipy.stats as spstats
import u4py.analysis.processing as u4process
import u4py.analysis.spatial as u4spatial
import u4py.utils.convert as u4convert
import u4py.utils.files as u4files
import uncertainties as unc


def main():
    chunked_analysis()
    # original_file()


def chunked_analysis():
    """
    Processes a full suite of datasets
    """
    # file_list = u4files.get_file_paths(filetypes=(("*.h5", "*.h5"),))
    file_list = u4files.get_file_list()
    base_folder = u4files.multi_split(file_list[0], nsplits=3)

    places = u4convert.dbf_to_dict(
        os.path.join(base_folder, "Places", "gis_osm_places_free_1.dbf")
    )

    city_names, city_coords = u4spatial.get_features(places, ["city", "town"])


def load_data(file_list):
    results, chunk_size = u4process.get_processing_results(file_list)
    if not chunk_size:
        chunk_size = 500
    xmids = []
    ymids = []
    slope = []
    season = []
    for (xmid, ymid, time_components) in results:
        xmids.append(xmid)
        ymids.append(ymid)
        slope.append(time_components[0] * 365.25)
        season.append(np.abs(time_components[1]))
    return xmids, ymids, slope, season


def plot_gridded(xmids, ymids, chunk_size, slope, season):
    minx = np.min(xmids)
    maxx = np.max(xmids) + chunk_size
    miny = np.min(ymids)
    maxy = np.max(ymids) + chunk_size
    x = np.arange(minx, maxx, chunk_size)
    y = np.arange(miny, maxy, chunk_size)

    XX, YY = np.meshgrid(x, y)
    SLP = np.ones_like(XX) * np.nan
    SEA = np.ones_like(XX) * np.nan

    for xi, yi, sl, se in zip(xmids, ymids, slope, season):
        xn = int((xi - minx) / chunk_size)
        yn = int((yi - miny) / chunk_size)
        SLP[yn, xn] = sl
        SEA[yn, xn] = se

    SLP = spimg.median_filter(SLP, 3)
    SEA = spimg.median_filter(SEA, 3)


def plot_gridded(
    SLP,
    SEA,
):
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    rng = np.percentile(np.abs(slope), 95)
    slp = axes[0].imshow(
        SLP,
        vmin=-rng,
        vmax=rng,
        origin="lower",
        extent=(minx, maxx, miny, maxy),
        cmap="RdYlBu_r",
    )
    plt.colorbar(slp, ax=axes[0])
    rng = np.nanpercentile(season, 95)
    seas = axes[1].imshow(
        SEA, vmin=0, vmax=rng, origin="lower", extent=(minx, maxx, miny, maxy)
    )

    for ax in axes:
        for citname, citcoords in zip(city_names, city_coords):
            ax.annotate(
                citname,
                xy=citcoords,
                horizontalalignment="center",
                fontweight="bold",
                fontsize="small",
                fontfamily="Verdana",
            )
    plt.colorbar(seas, ax=axes[1])
    plt.tight_layout()
    plt.show()


def original_file():
    file_list = u4files.get_file_paths(filetypes=(("*.h5", "*.h5"),))
    for file_path in file_list:
        data = u4files.load_hdf5(file_path)
        # plot_statistics(data, timeslot=-1)
        # plot_cwt(data)
        plot_timeseries(data)
        # plot_mean(data)


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

    plt.show()


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
