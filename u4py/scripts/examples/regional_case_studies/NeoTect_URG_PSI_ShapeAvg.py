import os
import pickle as pkl
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt


def main():
    path = Path(
        r"~\Documents\ArcGIS\INSAR_results\NeoTect_URG_PSI\Faults"
    ).expanduser()
    out_path = Path(
        r"~\nextHessenbox\Entwurf_Neotectonics_URG (Johannes Mair)\Python Plots"
    ).expanduser()

    fault_list = ["1A", "1B", "1C", "1D", "1E", "2A", "2B"]
    surveys = ["BBD", "EGMS_1", "EGMS_2"]

    fig, axes = plt.subplots(
        ncols=len(fault_list),
        sharey=True,
        # sharex=True,
        figsize=(12, 3),
    )
    for ii, fault in enumerate(fault_list):
        if "1" in fault:
            dirs = ["West", "Ost"]
        else:
            dirs = ["Nord", "Süd"]

        for jj, surv in enumerate(surveys):
            for nn, dr in enumerate(dirs):
                fault_name = f"{surv}_{fault} {dr}.pkl"
                fpath = os.path.join(path, fault_name)
                with open(fpath, "rb") as pkl_file:
                    data = pkl.load(pkl_file)
                    axes[ii].plot(
                        nn,
                        data["inversion_results"][7],
                        marker="s",
                        color=f"C{jj}",
                    )

        axes[ii].set_title(fault)
        axes[ii].set_xlim(-0.5, 1.5)
        axes[ii].set_xticks([0, 1])
        axes[ii].set_xticklabels(dirs)
        axes[ii].axhline(0, color="k")
    axes[0].set_ylabel("linear trend (mm/yr)")
    axes[-1].legend(
        handles=[
            mlines.Line2D(
                [],
                [],
                color=f"C{ii}",
                marker="s",
                linestyle="None",
                label=surv.replace("_", " "),
            )
            for ii, surv in enumerate(surveys)
        ],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_path, "FaultPolygon_AverageLinearComponents.png")
    )
    fig.savefig(
        os.path.join(out_path, "FaultPolygon_AverageLinearComponents.pdf")
    )


if __name__ == "__main__":
    main()
