import matplotlib.pyplot as plt
import numpy as np


def plot_inversion_results(
    time, time2, data, inversion_results, save_path=None
):
    ew_mov = inversion_results["matrix_ori"][1]
    ud_mov = inversion_results["matrix_ori"][13]
    ux = np.unique(time[0])

    fig, axes = plt.subplots(ncols=2, figsize=(10, 5), sharex=True)
    y = [np.nanmedian(data["dataE"][np.argwhere(time[0] == uu)]) for uu in ux]
    yep = [
        np.nanpercentile(data["dataE"][np.argwhere(time[0] == uu)], 95)
        for uu in ux
    ]
    yed = [
        np.nanpercentile(data["dataE"][np.argwhere(time[0] == uu)], 5)
        for uu in ux
    ]
    yep2 = [
        np.nanpercentile(data["dataE"][np.argwhere(time[0] == uu)], 68)
        for uu in ux
    ]
    yed2 = [
        np.nanpercentile(data["dataE"][np.argwhere(time[0] == uu)], 32)
        for uu in ux
    ]
    axes[0].set_title("East-West Component")
    axes[0].plot(ux, y, ".", label="Median")
    axes[0].fill_between(
        ux, yed, yep, color="C0", alpha=0.5, edgecolor=None, label="95% range"
    )
    axes[0].fill_between(
        ux,
        yed2,
        yep2,
        color="C0",
        alpha=0.5,
        edgecolor=None,
        label="68% range",
    )
    axes[0].plot(
        time[0], data["ori_dhat_data"]["dhatE"], color="C1", label="Fit"
    )
    axes[0].plot(
        time2[0],
        data["dhat_data"]["dhatE"],
        color="C2",
        label="Fit (w/o outliers)",
    )
    axes[0].annotate(
        f"{ew_mov:.2} mm/yr", (0.05, 0.05), xycoords="axes fraction"
    )
    axes[0].legend(loc="best", fontsize="small", markerscale=0.5)
    # axes[1][0].plot(time[0], data["ori_dhat_data"]["dhatE"], "C1")
    y = [np.nanmedian(data["dataU"][np.argwhere(time[0] == uu)]) for uu in ux]
    yep = [
        np.nanpercentile(data["dataU"][np.argwhere(time[0] == uu)], 95)
        for uu in ux
    ]
    yed = [
        np.nanpercentile(data["dataU"][np.argwhere(time[0] == uu)], 5)
        for uu in ux
    ]
    yep2 = [
        np.nanpercentile(data["dataU"][np.argwhere(time[0] == uu)], 68)
        for uu in ux
    ]
    yed2 = [
        np.nanpercentile(data["dataU"][np.argwhere(time[0] == uu)], 32)
        for uu in ux
    ]
    axes[1].set_title("Vertical Component")
    axes[1].plot(ux, y, ".")
    axes[1].fill_between(ux, yed, yep, color="C0", alpha=0.5, edgecolor=None)
    axes[1].fill_between(ux, yed2, yep2, color="C0", alpha=0.5, edgecolor=None)
    axes[1].plot(time[0], data["ori_dhat_data"]["dhatU"])
    axes[1].plot(time2[0], data["dhat_data"]["dhatU"])
    axes[1].annotate(
        f"Hebung/Senkung = {ud_mov:.2} mm/yr",
        (0.05, 0.05),
        xycoords="axes fraction",
    )
    # axes[1][1].plot(time[0], data["ori_dhat_data"]["dhatU"], "C1")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()
