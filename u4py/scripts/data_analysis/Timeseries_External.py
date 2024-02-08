"""
Time series analysis of external data, such as groundwater, weather or other data.
"""

# import logging
import os
from pathlib import Path

import u4py.addons.groundwater as u4gw
import u4py.analysis.inversion as u4invert
import u4py.plotting.plots as u4plots
import u4py.utils.projects as u4proj

# import u4py.utils.config as u4config
# u4config.start_logger()


def main():
    project = u4proj.get_project(
        proj_path=Path(
            "~/Documents/ArcGIS/U4_projects/Timeseries_External.u4project"
        ).expanduser(),
        required=["base_path", "ext_path", "output_path"],
        interactive=False,
    )

    data = u4gw.get_groundwater_data(
        os.path.join(
            project["paths"]["ext_path"], "GWStände_2015", "GWStände_2015.pkl"
        )
    )

    # Enter Keys of selected stations:
    selected_stations = ["CRUMSTADT", "HAHN flach", "ALLMENDFELD (alt)"]
    title = ""
    data_inv = dict()
    for station in selected_stations:
        data_inv[station] = u4invert.reformat_simple_timeseries(
            data[station]["time"],
            data[station]["height"],
            station=station,
            xmid=data[station]["easting"],
            ymid=data[station]["northing"],
        )
        if title:
            title += ", "
        title += data[station]["name"]
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
        unit="m",
        single_dim=True,
    )
    ax.set_ylabel("Ground water level (m.a.s.l.)")
    ax.set_xlabel("Time")
    ax.legend()
    ax.set_title(title.title())
    fig.tight_layout()
    fig_path = os.path.join(
        project["paths"]["output_path"], "GroundwaterTSA.pdf"
    )
    fig.savefig(fig_path)
    fig.savefig(fig_path.replace("pdf", "png"))


if __name__ == "__main__":
    main()
