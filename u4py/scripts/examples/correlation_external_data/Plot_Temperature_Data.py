"""
Simple plotting script for temperature data
"""

import os
from pathlib import Path

import u4py.addons.climate as u4climate
import u4py.analysis.inversion as u4invert
import u4py.plotting.plots as u4plots
import u4py.utils.projects as u4proj


def main():
    project = u4proj.get_project(
        proj_path=Path(
            "~/Documents/ArcGIS/U4_projects/Timeseries_External.u4project"
        ).expanduser(),
        required=["base_path", "ext_path", "output_path"],
        interactive=False,
    )

    data = u4climate.get_temperature_data(
        os.path.join(
            project["paths"]["ext_path"],
            "Lufttemperatur",
            "TU_temp_TMW2015_2022.csv",
        )
    )

    selected_stations = ["Darmstadt", "Frankfurt-Höchst", "Frankfurt Ost"]
    title = ""
    data_inv = dict()
    for station in selected_stations:
        data_inv[station] = u4invert.reformat_simple_timeseries(
            data["time"], data[station], station=station
        )
        if title:
            title += ", "
        title += station
    stacked_data = u4invert.stack_data(data_inv)
    # stacked_data = u4invert.smooth_stacked_data(stacked_data)
    (
        matrix,
        data_inv_out,
        time_vector,
        parameters_list,
    ) = u4invert.invert_time_series(stacked_data)
    inversion_results = {"matrix_ori": matrix}
    # inversion_string = u4invert.print_inversion_results(
    #     matrix, parameters_list
    # )
    fig, ax = u4plots.plot_inversion_results(
        time=time_vector,
        data=data_inv_out,
        inversion_results=inversion_results,
        unit="°C",
        single_dim=True,
    )
    ax.legend(loc="upper right")
    ax.set_title(title.title())
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature daily avg. (°C)")
    ax.set_ylim(-20, 40)
    ax.annotate(
        (
            f"linear trend: {matrix[1]:.2f} °C/yr\n"
            + f"annual sine: {matrix[2]:.2f}\n"
            + f"annual cosine: {matrix[3]:.2f}\n"
            + f"semi-annual sine: {matrix[4]:.2f}\n"
            + f"semi-annual cosine: {matrix[5]:.2f}"
        ),
        (0.01, 0.01),
        xycoords="axes fraction",
        verticalalignment="bottom",
    )
    fig.tight_layout()
    fig_path = os.path.join(
        project["paths"]["output_path"], "TemperatureTSA.pdf"
    )
    fig.savefig(fig_path)
    fig.savefig(fig_path.replace("pdf", "png"))


if __name__ == "__main__":
    main()
