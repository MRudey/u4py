"""
Functions for creating a TeX-based report of the classified anomalies.
"""

import os
import subprocess
from typing import Tuple

import geopandas as gp
import humanize
import numpy as np
import uncertainties as unc
from uncertainties import unumpy as unp


def main_report(output_path: os.PathLike | str):
    html_folder = os.path.join(output_path, "html_includes")
    include_list = [
        os.path.join(html_folder, fp)
        for fp in os.listdir(html_folder)
        if fp.endswith("html")
    ]
    html = (
        "<!DOCTYPE html>\n"
        + "<html>\n"
        + "  <head>\n"
        + '    <meta charset="utf-8">\n'
        + '    <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        + "    <title>U4Py Overview</title>\n"
        + '    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@1.0.2/css/bulma.min.css">\n'
        + "  </head>\n"
        + '  <nav class="navbar" role="navigation", aria-label="main navigation">\n'
        + '    <div class="navbar-brand">\n'
        + '      <a class="navbar-item" href="main.html">\n'
        + '        <svg width="160" height="160" viewBox="0 0 25 25" fill="" xmlns="http://www.w3.org/2000/svg">\n'
        + '          <path d="M20 17.0002V11.4522C20 10.9179 19.9995 10.6506 19.9346 10.4019C19.877 10.1816 19.7825 9.97307 19.6546 9.78464C19.5102 9.57201 19.3096 9.39569 18.9074 9.04383L14.1074 4.84383C13.3608 4.19054 12.9875 3.86406 12.5674 3.73982C12.1972 3.63035 11.8026 3.63035 11.4324 3.73982C11.0126 3.86397 10.6398 4.19014 9.89436 4.84244L5.09277 9.04383C4.69064 9.39569 4.49004 9.57201 4.3457 9.78464C4.21779 9.97307 4.12255 10.1816 4.06497 10.4019C4 10.6506 4 10.9179 4 11.4522V17.0002C4 17.932 4 18.3978 4.15224 18.7654C4.35523 19.2554 4.74432 19.6452 5.23438 19.8482C5.60192 20.0005 6.06786 20.0005 6.99974 20.0005C7.93163 20.0005 8.39808 20.0005 8.76562 19.8482C9.25568 19.6452 9.64467 19.2555 9.84766 18.7654C9.9999 18.3979 10 17.932 10 17.0001V16.0001C10 14.8955 10.8954 14.0001 12 14.0001C13.1046 14.0001 14 14.8955 14 16.0001V17.0001C14 17.932 14 18.3979 14.1522 18.7654C14.3552 19.2555 14.7443 19.6452 15.2344 19.8482C15.6019 20.0005 16.0679 20.0005 16.9997 20.0005C17.9316 20.0005 18.3981 20.0005 18.7656 19.8482C19.2557 19.6452 19.6447 19.2554 19.8477 18.7654C19.9999 18.3978 20 17.932 20 17.0002Z" stroke-linecap="round" stroke-linejoin="round" fill="white"/>\n'
        + "        </svg> Home\n"
        + "      </a>\n"
        + '      <a role="button" class="navbar-burger" aria-label="menu" aria-expanded="false" data-target="navbarBasicExample">\n'
        + '        <span aria-hidden="true"></span>\n'
        + '        <span aria-hidden="true"></span>\n'
        + '        <span aria-hidden="true"></span>\n'
        + '        <span aria-hidden="true"></span>\n'
        + "      </a>\n"
        + "    </div>\n"
        + "\n"
        + '    <div id="navbarBasicExample" class="navbar-menu">\n'
        + '      <div class="navbar-start">\n'
        + '        <a class="navbar-item" href="database.html">\n'
        + "          Datenbank\n"
        + "        </a>\n"
        + '        <a class="navbar-item" href="https://rudolf.pages.git-ce.rwth-aachen.de/u4py/">\n'
        + "          U4Py\n"
        + "        </a>\n"
        + "      </div>\n"
        + "    </div>\n"
        + "\n"
        + "  </nav>\n"
        + "  <body>\n"
        + '  <section class="section">'
        + '<div class="container">\n'
        + '<h1 class="title">\n'
        + "  Datenbank\n"
        + "</h1>\n"
        + '<p class="subtitle">\n'
        + "  Liste aller Anomalien\n"
        + "</p>\n"
        + "<p>\n"
        + f"  Insgesamt {len(include_list)} Einträge\n"
        + "</p>\n"
        + '<table class="table is-striped is-hoverable">\n'
        + "  <thead>\n"
        + "      <tr>\n"
        + "          <th>Nummer</th>\n"
        + "          <th>Eintrag</th>\n"
        + "      </tr>\n"
        + "  </thead>\n"
        + "  <tfoot>\n"
        + "      <tr>\n"
        + "          <th>Nummer</th>\n"
        + "          <th>Eintrag</th>\n"
        + "      </tr>\n"
        + "  </tfoot>\n"
        + "  <tbody>\n"
    )
    include_list.sort()
    for incl in include_list:
        fname = os.path.split(incl)[-1]
        group = int(fname.split("_")[0])
        html += (
            f"            <th>{group}</th>\n"
            + f'            <td><a href="html_includes/{fname}" title="Gruppe {group}">\n'
            + f"                Gruppe Nummer {group}\n"
            + "            </a></td><tr></tr>\n"
            + "\n"
        )
    html += (
        "        </tbody>\n"
        + "      </table>\n"
        + "    </div>\n"
        + "  </section>\n"
        + "  </body>\n"
        + "</html>\n"
    )

    report_path = os.path.join(output_path, "database.html")
    with open(report_path, "wt", encoding="utf-8", newline="\n") as html_file:
        html_file.write(html)


def site_report(
    row: tuple,
    output_path: os.PathLike | str,
    suffix: str,
    hlnug_data: gp.GeoDataFrame,
    prev_grp: str,
    next_grp: str,
):
    """Creates a report for each area of interest using LaTeX. This is later merged together into a larger main document by the `main` function.

    :param row: The index and data for the area of interest.
    :type row: tuple
    :param output_path: The path where to store the outputs.
    :type output_path: os.PathLike | str
    :param suffix: The subfolder to use for the LaTeX files.
    :type suffix: str
    :param hlnug_data: More info loaded from HLNUG Dataset
    :type hlnug_data: gp.GeoDataFrame
    """

    # Setting Paths
    group = row[1].group
    output_path_html = os.path.join(output_path, suffix)
    os.makedirs(output_path_html, exist_ok=True)
    img_path = os.path.join(
        output_path, "Detailed_Maps", "known_features", f"{group:05}"
    )

    # Create TeX code

    html = (
        "<!DOCTYPE html>\n"
        + '<html class="has-navbar-fixed-top">\n'
        + "\n"
        + "<head>\n"
        + '  <meta charset="utf-8">\n'
        + '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        + "  <title>U4Py Overview</title>\n"
        + '  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@1.0.2/css/bulma.min.css">\n'
        + "  <style>\n"
        + "    #tab-content p {\n"
        + "      display: none;\n"
        + "    }\n"
        + "\n"
        + "    #tab-content p.is-active {\n"
        + "      display: block;\n"
        + "    }\n"
        + "\n"
        + "    #tab-content2 p {\n"
        + "      display: none;\n"
        + "    }\n"
        + "\n"
        + "    #tab-content2 p.is-active {\n"
        + "      display: block;\n"
        + "    }\n"
        + "\n"
        + "    .box {\n"
        + "      padding-top: 50px;\n"
        + "    }\n"
        + "  </style>\n"
        + '  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>\n'
        + "  <script>\n"
        + "    $(document).ready(function () {\n"
        + "      $('#tabs li').on('click', function () {\n"
        + "        var tab = $(this).data('tab');\n"
        + "\n"
        + "        $('#tabs li').removeClass('is-active');\n"
        + "        $(this).addClass('is-active');\n"
        + "\n"
        + "        $('#tab-content p').removeClass('is-active');\n"
        + "        $('p[data-content="
        " + tab + "
        "]').addClass('is-active');\n"
        + "      });\n"
        + "      $('#tabs2 li').on('click', function () {\n"
        + "        var tab = $(this).data('tab');\n"
        + "\n"
        + "        $('#tabs2 li').removeClass('is-active');\n"
        + "        $(this).addClass('is-active');\n"
        + "\n"
        + "        $('#tab-content2 p').removeClass('is-active');\n"
        + "        $('p[data-content="
        " + tab + "
        "]').addClass('is-active');\n"
        + "      });\n"
        + "    });\n"
        + "  </script>\n"
        + "</head>\n"
        + '<nav class="navbar is-fixed-top" role="navigation" , aria-label="main navigation">\n'
        + '  <div class="navbar-brand">\n'
        + '    <a class="navbar-item" href="../main.html">\n'
        + '      <svg width="160" height="160" viewBox="0 0 25 25" xmlns="http://www.w3.org/2000/svg">\n'
        + "        <path\n"
        + '          d="M20 17.0002V11.4522C20 10.9179 19.9995 10.6506 19.9346 10.4019C19.877 10.1816 19.7825 9.97307 19.6546 9.78464C19.5102 9.57201 19.3096 9.39569 18.9074 9.04383L14.1074 4.84383C13.3608 4.19054 12.9875 3.86406 12.5674 3.73982C12.1972 3.63035 11.8026 3.63035 11.4324 3.73982C11.0126 3.86397 10.6398 4.19014 9.89436 4.84244L5.09277 9.04383C4.69064 9.39569 4.49004 9.57201 4.3457 9.78464C4.21779 9.97307 4.12255 10.1816 4.06497 10.4019C4 10.6506 4 10.9179 4 11.4522V17.0002C4 17.932 4 18.3978 4.15224 18.7654C4.35523 19.2554 4.74432 19.6452 5.23438 19.8482C5.60192 20.0005 6.06786 20.0005 6.99974 20.0005C7.93163 20.0005 8.39808 20.0005 8.76562 19.8482C9.25568 19.6452 9.64467 19.2555 9.84766 18.7654C9.9999 18.3979 10 17.932 10 17.0001V16.0001C10 14.8955 10.8954 14.0001 12 14.0001C13.1046 14.0001 14 14.8955 14 16.0001V17.0001C14 17.932 14 18.3979 14.1522 18.7654C14.3552 19.2555 14.7443 19.6452 15.2344 19.8482C15.6019 20.0005 16.0679 20.0005 16.9997 20.0005C17.9316 20.0005 18.3981 20.0005 18.7656 19.8482C19.2557 19.6452 19.6447 19.2554 19.8477 18.7654C19.9999 18.3978 20 17.932 20 17.0002Z"\n'
        + '          stroke-linecap="round" stroke-linejoin="round" fill="white" />\n'
        + "      </svg> Home\n"
        + "    </a>\n"
        + "  </div>\n"
        + "\n"
        + '  <div class="navbar-menu">\n'
        + '    <div class="navbar-start">\n'
        + '      <a class="navbar-item" href="../database.html">Datenbank</a>\n'
        + '      <a class="navbar-item" href="https://rudolf.pages.git-ce.rwth-aachen.de/u4py/">U4Py</a>\n'
        + "    </div>\n"
        + '    <div class="navbar-end">\n'
        + f'      <a class="navbar-item is-active" href="../{prev_grp}.html"><svg fill="white" width="128px" height="128px"\n'
        + '          viewBox="0 0 32 32" version="1.1" xmlns="http://www.w3.org/2000/svg">\n'
        + "          <path\n"
        + '            d="M16 0c-8.837 0-16 7.163-16 16s7.163 16 16 16c8.837 0 16-7.163 16-16s-7.163-16-16-16zM16 30.032c-7.72 0-14-6.312-14-14.032s6.28-14 14-14 14 6.28 14 14-6.28 14.032-14 14.032zM23.010 14.989h-11.264l3.617-3.617c0.39-0.39 0.39-1.024 0-1.414s-1.024-0.39-1.414 0l-5.907 6.062 5.907 6.063c0.195 0.195 0.451 0.293 0.707 0.293s0.511-0.098 0.707-0.293c0.39-0.39 0.39-1.023 0-1.414l-3.68-3.68h11.327c0.552 0 1-0.448 1-1s-0.448-1-1-1z">\n'
        + "          </path>\n"
        + "        </svg></a>\n"
        + '      <a class="navbar-item" href="#home">Top</a>\n'
        + '      <a class="navbar-item" href="#loc">Lokalität</a>\n'
        + '      <a class="navbar-item" href="#desc">Beschreibung</a>\n'
        + '      <a class="navbar-item" href="#diff">Differenzenplan</a>\n'
        + '      <a class="navbar-item" href="#topo">Topographie</a>\n'
        + '      <a class="navbar-item" href="#haz">Geogefahren</a>\n'
        + '      <a class="navbar-item" href="#geol">Geodaten</a>\n'
        + f'      <a class="navbar-item is-active" href="{next_grp}.html"><svg fill="white" width="128px" height="128px"\n'
        + '          viewBox="0 0 32 32" version="1.1" xmlns="http://www.w3.org/2000/svg">\n'
        + "          <path\n"
        + '            d="M16 0c-8.836 0-16 7.163-16 16s7.163 16 16 16c8.837 0 16-7.163 16-16s-7.163-16-16-16zM16 30.032c-7.72 0-14-6.312-14-14.032s6.28-14 14-14 14 6.28 14 14-6.28 14.032-14 14.032zM16.637 9.957c-0.39 0.39-0.39 1.024 0 1.414l3.617 3.617h-11.264c-0.553 0-1 0.448-1 1s0.447 1 1 1h11.327l-3.68 3.68c-0.39 0.39-0.39 1.023 0 1.414 0.195 0.195 0.451 0.293 0.707 0.293s0.512-0.098 0.707-0.293l5.907-6.063-5.907-6.063c-0.391-0.39-1.023-0.39-1.415 0z">\n'
        + "          </path>\n"
        + "        </svg></i></a>\n"
        + "    </div>\n"
        + "  </div>\n"
        + "\n"
        + "</nav>\n"
        + "\n"
        + '<body id="home">\n'
        + "\n"
        + '  <section class="section" style="max-width: 900px;">\n'
        + '    <div class="content">\n'
        + f'      <h1 class="title is-1">Gruppe {group}</h1>\n'
    )

    # Overview plot and satellite image
    html += location(row[1])
    if os.path.exists(img_path + "_map.png") and os.path.exists(
        img_path + "_satimg.png"
    ):
        html += details_and_satellite(img_path)
    # Manual Classification or HLNUG data
    if len(hlnug_data) > 0:
        html += hlnug_description(hlnug_data[hlnug_data.AMT_NR_ == group])
    else:
        html += manual_description(row[1])
    # html += shape(row[1])
    # html += landuse(row[1])

    # # Volumina
    # html += moved_volumes(row[1])

    # # Difference maps
    # if os.path.exists(img_path + "_diffplan.pdf"):
    #     html += difference(img_path)

    # # Topographie
    # if os.path.exists(img_path + "_slope.pdf") or os.path.exists(
    #     img_path + "_aspect_slope.pdf"
    # ):
    #     html += topography(row[1], img_path)

    # # PSI Data
    # if os.path.exists(img_path + "_psi.png"):
    #     html += psi_map(img_path)

    # # Geohazard
    # html += geohazard(row[1])

    # # Geologie etc...
    # if os.path.exists(img_path + "_GK25.pdf"):
    #     html += geology(img_path)
    # if os.path.exists(img_path + "_HUEK200.pdf"):
    #     html += hydrogeology(img_path)
    # if os.path.exists(img_path + "_BFD50.pdf"):
    #     html += soils(img_path)

    html += (
        "    </div>\n"
        + "\n"
        + "  </section>\n"
        + "</body>\n"
        + "\n"
        + "</html>\n"
    )

    # Save to html file
    with open(
        os.path.join(output_path_html, f"{group:05}_info.html"),
        "wt",
        encoding="utf-8",
        newline="\n",
    ) as html_file:
        html_file.write(html)


def location(series: gp.GeoSeries) -> str:
    """Adds location information to the document

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The html code.
    :rtype: str
    """

    wgs_point = gp.GeoDataFrame(
        geometry=[series.geometry.centroid], crs="EPSG:32632"
    ).to_crs("EPSG:4326")
    lat = np.round(float(wgs_point.geometry.y.iloc[0]), 6)
    lng = np.round(float(wgs_point.geometry.x.iloc[0]), 6)

    html = (
        '      <div class="box" id="loc">\n'
        + '        <h2 class="title is-2">Lokalität:</h2>\n'
        + "\n"
        + '        <table class="table">\n'
        + "          <tbody>\n"
        + "            <tr>\n"
        + f'              <td colspan="3"><strong>Adresse:</strong> {series.locations}</td>\n'
        + "            </tr>\n"
        + "            <tr>\n"
        + f'              <td colspan="3"><strong>Koordinaten (UTM 32N):</strong> {int(series.geometry.centroid.y)}&nbsp;N, {int(series.geometry.centroid.x)}&nbsp;E</td>\n'
        + "            </tr>\n"
        + f'            <td><a href="https://www.google.com/maps/place/{lat},{lng}/@{lat},{lng}/data=!3m1!1e3"\n'
        + '                target="_blank">\n'
        + f'                <img src="https://icons.duckduckgo.com/ip3/google.com.ico" width="5%"> {np.round(lat, 3)}&nbsp;N, {np.round(lng, 3)}&nbsp;E\n'
        + "              </a>\n"
        + "            </td>\n"
        + f'            <td><a href="https://bing.com/maps/default.aspx?cp={lat}~{lng}&style=h&lvl=15" target="_blank"><img\n'
        + f'                  src="https://icons.duckduckgo.com/ip3/bing.com.ico" width="5%"> {np.round(lat, 3)}&nbsp;N, {np.round(lng, 3)}&nbsp;E</a>\n'
        + "            </td>\n"
        + f'            <td><a href="http://www.openstreetmap.org/?lat={lat}&lon={lng}&zoom=17&layers=M" target="_blank"><img\n'
        + f'                  src="https://icons.duckduckgo.com/ip3/openstreetmap.com.ico" width="5%"> {np.round(lat, 3)}&nbsp;N, {np.round(lng, 3)}&nbsp;E</a>\n'
        + "            </td>\n"
        + "            <tr></tr>\n"
        + "          </tbody>\n"
        + "        </table>\n"
        + "\n"
    )
    return html


def shape(series: gp.GeoSeries) -> str:
    """Adds shape information to the document

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The html code.
    :rtype: str
    """
    humanize.activate("de")
    html = "\\paragraph{Größe und Form}\n"

    long_ax = eval(series["shape_ellipse_a"])
    short_ax = eval(series["shape_ellipse_b"])
    if isinstance(long_ax, list):
        areas = [np.pi * a * b for a, b in zip(long_ax, short_ax)]
        if len(areas) > 0:
            imax = np.argmax(areas)
            imin = np.argmin(areas)
            if len(long_ax) > 2:
                lax = str(
                    unc.ufloat(np.mean(long_ax), 2 * np.std(long_ax))
                ).replace("+/-", "$\\pm$")
                sax = str(
                    unc.ufloat(np.mean(short_ax), 2 * np.std(short_ax))
                ).replace("+/-", "$\\pm$")
            else:
                lax = f"{round(long_ax[0])} und {round(long_ax[1])}"
                sax = f"{round(short_ax[0])} und {round(short_ax[1])}"

            html += (
                f"Es handelt sich um {humanize.apnumber(len(long_ax))} Anomalien. "
                + f"Die Anomalien sind ca. {lax}&nbsp;m lang und ca. {sax}&nbsp;m breit. "
            )
            if len(long_ax) > 2:
                html += (
                    "Die flächenmäßig kleinste Anomalie ist hierbei ca. "
                    + f"{round(long_ax[imin])}&nbsp;m lang und "
                    + f"{round(short_ax[imin])}&nbsp;m breit, die größte ca. "
                    + f"{round(long_ax[imax])}&nbsp;m lang und "
                    + f"{round(short_ax[imax])}&nbsp;m breit. "
                )
    if isinstance(long_ax, float):
        lax = f"{round(long_ax)}"
        sax = f"{round(short_ax)}"
        html += f"Die Anomalie ist ca. {lax}&nbsp;m lang und ca. {sax}&nbsp;m breit. "

    return html


def manual_description(series: gp.GeoSeries) -> str:
    html = "\\subsection*{Manuelle Klassifizierung}\n\n"
    if int(series.manual_known):
        html += (
            "Die Anomalie ist bereits in den Datenbanken des HLNUG vorhanden. "
        )
    else:
        html += "Die Anomalie ist noch nicht in den Datenbanken des HLNUG vorhanden. "
    ncls = len(
        [
            series[f"manual_class_{ii}"]
            for ii in range(1, 4)
            if series[f"manual_class_{ii}"]
        ]
    )
    prob_txt = ["wahrscheinlich", "möglicherweise"]
    if ncls == 1:
        html += (
            f"Es handelt sich {prob_txt[int(series['manual_unclear_1'])]} "
            + f"um ein/e {series['manual_class_1']}: \n"
            + "\\begin{itemize}\n"
            + f"\\item[$\\rightarrow$] {series['manual_comment']}\n"
            + "\\end{itemize}\n"
        )
    elif ncls == 2:
        html += (
            "Mehrere Ursachen kommen in Frage. "
            + f"Es handelt sich {prob_txt[int(series['manual_unclear_1'])]} "
            + f"um ein/e {series['manual_class_1']} oder "
            + f"{prob_txt[int(series['manual_unclear_2'])]} "
            + f"um ein/e {series['manual_class_2']}: \n"
            + "\\begin{itemize}\n"
            + f"\\item[$\\rightarrow$] {series['manual_comment']}\n"
            + "\\end{itemize}\n"
        )
    elif ncls == 3:
        html += (
            "Mehrere Ursachen kommen in Frage. "
            + f"Es handelt sich {prob_txt[int(series['manual_unclear_1'])]} "
            + f"um ein/e {series['manual_class_1']}, "
            + f"{prob_txt[int(series['manual_unclear_2'])]} "
            + f"um ein/e {series['manual_class_2']} oder "
            + f"{prob_txt[int(series['manual_unclear_3'])]} "
            + f"um ein/e {series['manual_class_3']}: \n"
            + "\\begin{itemize}\n"
            + f"\\item[$\\rightarrow$] {series['manual_comment']}\n"
            + "\\end{itemize}\n"
        )
    else:
        html += "Eine manuelle Klassifikation ist noch nicht erfolgt. "
    if int(series["manual_research"]):
        html += (
            "Aufgrund der Nähe zu Infrastruktur oder der unklaren Lage "
            + "sollte die Anomalie einer genaueren Untersuchung unterzogen "
            + "werden. "
        )
    html += "\n\n"
    return html


def details_and_satellite(img_path: os.PathLike | str) -> str:
    """Adds the detailed map and the satellite image map.

    :param img_path: The path to the image folder including group name.
    :type img_path: os.PathLike | str
    :return: The html code.
    :rtype: str
    """
    fname = os.path.split(img_path)[-1]
    html = (
        "        <figure>\n"
        + f'          <img width="49.5%" src="../Detailed_Maps/known_features/{fname}_map.png" alt="Kartenausschnitt der Anomalie" />\n'
        + f'          <img width="49.5%" src="../Detailed_Maps/known_features/{fname}_satimg.png" alt="Satellitenbild der Anomalie" />\n'
        + "          <figcaption>\n"
        + "            Abbildung 1: Übersicht über das Gebiet der Gruppe inklusive verschiedener Geogefahren und der detektierten\n"
        + "            Anomalien (Kartengrundlage: OpenStreetMap, Satellitenbild: ESRI Imagery).\n"
        + "          </figcaption>\n"
        + "        </figure>\n"
        + "      </div>\n"
    )
    return html


def moved_volumes(series: gp.GeoSeries) -> str:
    """Adds description of moved volumes to the document.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The html code.
    :rtype: str
    """
    html = (
        "\\clearpage\n\\subsection*{Höhenveränderungen}\n"
        + "Im Gebiet um die detektierte Anomalie wurde insgesamt "
        + f"{series.volumes_moved}&nbsp;m$^3$ Material bewegt, "
        + f"wovon {series.volumes_added}&nbsp;m$^3$ hinzugefügt und "
        + f"{abs(series.volumes_removed)}&nbsp;m$^3$ abgetragen wurde. "
        + f"Dies ergibt eine Gesamtbilanz von {series.volumes_total}&nbsp;m$^3$,"
        + f" in Summe wurde also {vol_str(series.volumes_total)}.\n\n"
    )
    return html


def vol_str(val: float) -> str:
    """Converts a volume estimate into a descriptive htmlt.

    :param val: The value.
    :type val: float
    :return: The descriptive htmlt.
    :rtype: str
    """
    if val > 100:
        return "Material hinzugefügt"
    elif val < 100:
        return "Material abgetragen"
    else:
        return "das Gesamtvolumen nur wenig verändert"


def difference(img_path: os.PathLike | str) -> str:
    """Adds the difference and slope maps.

    :param img_path: The path to the image folder including group name.
    :type img_path: os.PathLike | str
    :return: The html code.
    :rtype: str
    """
    html = ""
    if os.path.exists(img_path + "_diffplan.pdf"):
        html += (
            "\\begin{figure}[!ht]\n"
            + "  \\centering"
            + f"  \\includegraphics[width=.9\\htmltwidth]{{{img_path + '_diffplan.pdf'}}}\n"
            + "  \\caption{Differenzenplan im Gebiet.}\n"
            + "\\end{figure}\n"
        )

    if os.path.exists(img_path + "_dem.pdf"):
        html += (
            "\\begin{figure}[!ht]\n"
            + "  \\centering"
            + f"  \\includegraphics[width=.9\\htmltwidth]{{{img_path + '_dem.pdf'}}}\n"
            + "  \\caption{Digitales Höhenmodell (Schummerung).}\n"
            + "\\end{figure}\n\n"
        )
    return html


def topography(series: gp.GeoSeries, img_path: os.PathLike | str) -> str:
    """Converts the slope into a descriptive htmlt.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The html code.
    :rtype: str
    """
    html = "\n\\subsection*{Topographie}\n\n"

    for yy in ["14", "19", "21"]:
        year = f"20{yy}"
        html += f"\\paragraph*{{{year}}}\n"

        if not series[f"slope_polygons_mean_{yy}"] == "[]":
            # Values for the individual polygons (can be empty)
            sl_m = eval(series[f"slope_polygons_mean_{yy}"])
            sl_s = eval(series[f"slope_polygons_std_{yy}"])
            as_m = eval(series[f"aspect_polygons_mean_{yy}"])
            as_s = eval(series[f"aspect_polygons_std_{yy}"])
            if isinstance(sl_m, list) or isinstance(sl_m, float):
                usl, usl_str = topo_htmlt(sl_m, sl_s, "\\%")
                html += (
                    f"Im Bereich der Anomalie {slope_std_str(usl.s)} und "
                    + f"{slope_str(usl.n)} ({usl_str}). "
                )
                if as_m:
                    uas, uas_str = topo_htmlt(as_m, as_s, "°")
                    html += (
                        "Die Anomalien fallen nach "
                        + f"{direction_to_htmlt(uas.n)} ({uas_str}) ein. "
                    )
            else:
                html += "Es liegen für den inneren Bereich der Anomalie keine Daten vor (außerhalb DEM). "

            # Values for the hull around all anomalies
            usl, usl_str = topo_htmlt(
                eval(series[f"slope_hull_mean_{yy}"]),
                eval(series[f"slope_hull_std_{yy}"]),
                "\\%",
            )
            if isinstance(usl, unc.UFloat):
                html += (
                    f"Im näheren Umfeld ist das Gelände {slope_std_str(usl.s)}"
                    + f" und {slope_str(usl.n)} ({usl_str}). "
                )
                uas, uas_str = topo_htmlt(
                    eval(series[f"aspect_hull_mean_{yy}"]),
                    eval(series[f"aspect_hull_std_{yy}"]),
                    "°",
                )
                if isinstance(uas, unc.UFloat):
                    html += (
                        "Der Bereich fällt im Mittel nach "
                        + f"{direction_to_htmlt(uas.n)} ({uas_str}) ein. "
                    )
            else:
                html += "Es liegen für das nähere Umfeld der Anomalien keine Werte für die Steigung vor. "
            html += "\n\n"
    if os.path.exists(img_path + "_slope.pdf") and os.path.exists(
        img_path + "_aspect.pdf"
    ):
        html += (
            "\n\\begin{figure}[!ht]\n"
            + "  \\begin{subfigure}[][][t]{.49\\htmltwidth}\n"
            + f"  \\includegraphics[width=\\htmltwidth]{{{img_path + '_slope.pdf'}}}\n"
            + "  \\caption{Steigung}\n"
            + "  \\end{subfigure}\n\hfill\n"
            + "  \\begin{subfigure}[][][t]{.49\\htmltwidth}\n"
            + "\\centering\n"
            + f"  \\includegraphics[width=\\htmltwidth]{{{img_path + '_aspect.pdf'}}}\n"
            + "  \\caption{Exposition.}\n"
            + "  \\end{subfigure}\n\hfill\n"
            + "  \\caption{Topographie im Gebiet.}"
            + "\\end{figure}\n\n"
        )

    if os.path.exists(img_path + "_aspect_slope.pdf"):
        html += (
            "\\begin{figure}[!ht]\n"
            + "  \\centering\n"
            + f"  \\includegraphics[width=.95\\htmltwidth]{{{img_path + '_aspect_slope.pdf'}}}\n"
            + "  \\caption{Steigung und Exposition}\n"
            + "\\end{figure}\n"
        )
    html += "\n\n"
    return html


def slope_str(val: float) -> str:
    """Converts a slope estimate into a descriptive htmlt.

    :param val: The value.
    :type val: float
    :return: The descriptive htmlt.
    :rtype: str
    """
    if val < 1.1:
        return "nahezu eben"
    if val < 3.0:
        return "sehr leicht fallend"
    if val < 5.0:
        return "sanft geneigt"
    if val < 8.5:
        return "mäßig geneigt"
    if val < 16.5:
        return "stark ansteigend"
    if val < 24.0:
        return "sehr stark ansteigend"
    if val < 35.0:
        return "extrem ansteigend"
    if val < 45.0:
        return "steil"
    else:
        return "sehr steil"


def slope_std_str(val: float) -> str:
    """Converts a standard deviation of data into a descriptive htmlt.

    :param val: The value.
    :type val: float
    :return: The descriptive htmlt.
    :rtype: str
    """
    if val < 5:
        return "gleichmäßig"
    elif val < 10:
        return "etwas unregelmäßig"
    elif val < 15:
        return "unregelmäßig"
    else:
        return "sehr variabel"


def landuse(series: gp.GeoSeries) -> str:
    """Converts the landuse into a descriptive htmlt.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The html code.
    :rtype: str
    """
    landuse = ""
    html = "\\paragraph{Landnutzung}\n\n"
    try:
        landuse = eval(series.landuse_names)
    except NameError:
        landuse = series.landuse_names
    landuse_perc = eval(series.landuse_percent)
    if landuse:
        html += (
            "Der überwiegende Teil wird durch "
            + f"{landuse_str(series.landuse_major)} bedeckt. "
            + f"Die Anteile der Landnutzung sind: \n\n"
        )
        if isinstance(landuse, list):
            for ii in range(1, len(landuse) + 1):
                html += f" {landuse_perc[-ii]:.1f}\\% {landuse_str(landuse[-ii])}, "
            html = html[:-2] + ".\n"
        else:
            html += f"{landuse_perc:.1f}\\% {landuse_str(landuse)}"
    return html


def landuse_str(in_str: str) -> str:
    """Translates the landuse strings from OSM into German.

    :param in_str: The landuse htmlt from `fclass`.
    :type in_str: str
    :return: The German translation.
    :rtype: str
    """
    convert = {
        "allotments": "Kleingärten",
        "buildings": "Gebäude",
        "cemetery": "Friedhof",
        "commercial": "Gewerbe",
        "farmland": "Ackerland",
        "farmyard": "Hof",
        "forest": "Wald",
        "grass": "Gras",
        "heath": "Heide",
        "industrial": "Industrie",
        "meadow": "Wiese",
        "military": "Sperrgebiet",
        "nature_reserve": "Naturschutzgebiet",
        "orchard": "Obstgarten",
        "park": "Park",
        "quarry": "Steinbruch",
        "recreation_ground": "Erholungsgebiet",
        "residential": "Wohngebiet",
        "retail": "Einzelhandel",
        "roads": "Straßen",
        "scrub": "Gestrüpp",
        "unclassified": "nicht klassifiziert",
        "water": "Gewässer",
        "vineyard": "Weinberg",
    }
    return convert[in_str]


def part_str(val: float) -> str:
    """Gets a qualitative descriptor for the area.

    :param val: The area coverage
    :type val: float
    :return: A string describing the proportion of an area.
    :rtype: str
    """
    if val < 10:
        return "zu einem geringen Teil"
    elif val < 33:
        return "teilweise"
    elif val < 66:
        return "zu einem großen Teil"
    elif val < 90:
        return "zu einem überwiegenden Teil"
    else:
        return "quasi vollständig"


def psi_map(img_path: os.PathLike | str) -> str:
    """Adds the psi map with timeseries.

    :param img_path: The path to the image folder including group name.
    :type img_path: os.PathLike | str
    :return: The html code.
    :rtype: str
    """
    html = (
        "\\subsection*{InSAR Daten}\n\n"
        + "\\begin{figure}[h!]\n"
        + "  \\centering\n"
        + f"  \\includegraphics[width=\\htmltwidth]{{{img_path + '_psi.png'}}}\n"
        + "  \\caption{Persistent scatterer und Zeitreihe der Deformation "
        + "im Gebiet der Gruppe.}\n"
        + "\\end{figure}\n\n"
    )
    return html


def geohazard(series: gp.GeoSeries) -> str:
    """Converts the known geohazards into a descriptive htmlt.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The html code.
    :rtype: str
    """
    html = (
        "\\subsection*{Geogefahren}\n\n"
        + "\\begin{table}[H]\n"
        + "  \\centering"
        + "  \\caption{Bekannte Geogefahren}\n"
        + "  \\begin{tabular}{r|ll}\n"
        + "    Typ & Innerhalb des Areals & Im Umkreis von 1 km\\\\\\hline\n"
        + f"    Hangrutschungen & {series.landslides_num_inside} & "
        + f"{series.landslides_num_1km}\\\\\n"
        + f"    Karsterscheinungen & {series.karst_num_inside} & "
        + f"{series.karst_num_1km}\\\\\n"
        + f"    Steinschläge & {series.rockfall_num_inside} & "
        + f"{series.rockfall_num_1km}\n"
        + "  \\end{tabular}\n"
        + "\\end{table}\n\n"
    )
    html += landslide_risk(series)
    html += karst_risk(series)
    html += subsidence_risk(series)
    return html


def landslide_risk(series: gp.GeoSeries) -> str:
    """Converts the known landslides into a descriptive htmlt.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The html code.
    :rtype: str
    """
    lsar = series.landslide_total
    html = "\\paragraph*{Rutschungsgefährdung}\n\n"
    if lsar > 0:
        html += f"Das Gebiet liegt {part_str(lsar)} ({lsar:.0f}\%) in einem gefährdeten Bereich mit rutschungsanfälligen Schichten. "
        try:
            landslide_units = eval(series.landslide_units)
        except (NameError, SyntaxError):
            landslide_units = series.landslide_units
        if isinstance(landslide_units, list):
            html += f"Die Einheiten sind: "
            for unit in landslide_units:
                html += f"{unit}, "
            html = html[:-2] + ".\n\n"
        else:
            html += f"Wichtigste Einheiten sind {landslide_units}.\n\n"
    else:
        html += (
            "Es liegen keine Informationen zur Rutschungsgefährdung vor.\n\n"
        )
    return html


def karst_risk(series: gp.GeoSeries) -> str:
    """Converts the known karst phenomena into a descriptive htmlt.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The html code.
    :rtype: str
    """
    ksar = series.karst_total
    html = "\\paragraph*{Karstgefährdung}\n\n"
    if ksar > 0:
        html += f"Das Gebiet liegt {part_str(ksar)} ({ksar:.0f}\%) in einem Bereich bekannter verkarsteter Schichten. "
        try:
            karst_units = eval(series.karst_units)
        except (NameError, SyntaxError):
            karst_units = series.karst_units
        if isinstance(karst_units, list):
            html += f"Die Einheiten sind: "
            for unit in karst_units:
                html += f"{unit}, "
            html = html[:-2] + ".\n\n"
        else:
            html += f"Wichtigste Einheiten sind {karst_units}.\n\n"
    else:
        html += "Es liegen keine Informationen zur Karstgefährdung vor.\n\n"
    return html


def subsidence_risk(series: gp.GeoSeries) -> str:
    """Converts the area of known subsidence into a descriptive htmlt.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The html code.
    :rtype: str
    """
    subsar = series.subsidence_total
    html = "\\paragraph*{Setzungsgefährdung}\n\n"
    if subsar > 0:
        html += f"Das Gebiet liegt {part_str(subsar)} ({subsar:.0f}\%) in einem Bereich bekannter setzungsgefährdeter Schichten. "
        try:
            subsidence_units = eval(series.subsidence_units)
        except (NameError, SyntaxError):
            subsidence_units = series.subsidence_units
        if isinstance(subsidence_units, list):
            html += f"Die Einheiten sind: "
            for unit in subsidence_units:
                html += f"{unit}, "
            html = html[:-2] + ".\n\n"
        else:
            html += f"Wichtigste Einheiten sind {subsidence_units}.\n\n"
    else:
        html += "Es liegen keine Informationen zur Setzungsgefährdung vor.\n\n"
    return html


def geology(img_path) -> str:
    html = (
        "\n\\subsection*{Geologie}\n\n"
        + "\\begin{figure}[H]\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=\\htmltwidth]{{{img_path + '_GK25.pdf'}}}\n"
        + "\\end{figure}\n"
        + "\\vspace{-2ex}\n"
        + "\\begin{figure}[H]\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=.75\\htmltwidth]{{{img_path + '_GK25_leg.pdf'}}}\n"
        + "  \\caption{Geologie im Gebiet basierend auf GK25 (Quelle: HLNUG).}\n"
        + "\\end{figure}\n\n"
    )
    return html


def hydrogeology(img_path: os.PathLike | str) -> str:
    html = (
        "\n\\subsection*{Hydrogeologie}\n\n"
        + "\\begin{figure}[H]\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=\\htmltwidth]{{{img_path + '_HUEK200.pdf'}}}\n"
        + "\\end{figure}\n"
        + "\\vspace{-2ex}\n"
        + "\\begin{figure}[H]\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=.75\\htmltwidth]{{{img_path + '_HUEK200_leg.pdf'}}}\n"
        + "  \\caption{Hydrogeologische Einheiten im Gebiet basierend auf HÜK200 (Quelle: HLNUG).}\n"
        + "\\end{figure}\n\n"
    )
    return html


def soils(img_path: os.PathLike | str) -> str:
    html = (
        "\n\\subsection*{Bodengruppen}\n\n"
        + "\\begin{figure}[H]\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=\\htmltwidth]{{{img_path + '_BFD50.pdf'}}}\n"
        + "\\end{figure}\n"
        + "\\vspace{-2ex}\n"
        + "\\begin{figure}[H]\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=.75\\htmltwidth]{{{img_path + '_BFD50_leg.pdf'}}}\n"
        + "  \\caption{Bodenhauptgruppen im Gebiet basierend auf der BFD50 (Quelle: HLNUG).}\n"
        + "\\end{figure}\n\n"
    )
    return html


def direction_to_htmlt(direction: float, lang: str = "de") -> str:
    """Converts an azimut between 0 and 360 to ordinal directions.

    :param direction: The direction with 0 = North and 180 = South
    :type direction: float
    :param lang: The language of the htmlt (supported values: "en", "de", "abbrev"), defaults to "de"
    :type lang: str, optional
    :return: The ordinal direction as a string
    :rtype: str
    """

    limits = np.arange(11.25, 360 + 22.25, 22.5)
    abbrevs = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
        "N",
    ]
    translate_abbrev = {
        "de": {
            "N": "Norden",
            "NNE": "Nordnordosten",
            "NE": "Nordosten",
            "ENE": "Ostnordosten",
            "E": "Osten",
            "ESE": "Ostsüdosten",
            "SE": "Südosten",
            "SSE": "Südsüdosten",
            "S": "Süden",
            "SSW": "Südsüdwesten",
            "SW": "Südwesten",
            "WSW": "Westsüdwesten",
            "W": "Westen",
            "WNW": "Westnordwesten",
            "NW": "Nordwesten",
            "NNW": "Nordnordwesten",
        },
        "en": {
            "N": "North",
            "NNE": "North-northeast",
            "NE": "Northeast",
            "ENE": "East-northeast",
            "E": "East",
            "ESE": "East-southeast",
            "SE": "Southeast",
            "SSE": "South-southeast",
            "S": "South",
            "SSW": "South-southwest",
            "SW": "Southwest",
            "WSW": "West-southwest",
            "W": "West",
            "WNW": "West-northwest",
            "NW": "Northwest",
            "NNW": "North-northwest",
        },
    }
    for ii in range(len(limits)):
        desc = abbrevs[ii]
        if direction <= limits[ii]:
            break
    if lang != "abbrev":
        desc = translate_abbrev[lang][desc]
    return desc


def topo_htmlt(
    means: float | list, std: float | list, unit: str
) -> Tuple[unc.ufloat, str]:
    """Converts the measured topography index, such as slope or aspect to a descriptive htmlt.

    :param means: The mean of the value
    :type means: float | list
    :param std: The standard deviation of the value
    :type std: float | list
    """
    if isinstance(means, list):
        slope_unp = unp.uarray(
            means,
            std,
        )
        usl = np.mean(slope_unp)
    elif isinstance(means, float):
        usl = unc.ufloat(means, std)

    ustr = str(usl).replace("+/-", "$\\pm$") + f"&nbsp;{unit}"
    if "e" in ustr:
        ustr = f"{usl:.0f}&nbsp;{unit}".replace("+/-", "$\\pm$")
    return usl, ustr


def hlnug_description(hld: gp.GeoDataFrame) -> str:
    """Adds a description based on HLNUG data

    :param hld: The dataset
    :type hld: gp.GeoDataFrame
    :return: The description
    :rtype: str
    """

    def kart_str(in_str):
        if "ja" in in_str:
            spl = in_str.split(" ")
            if len(spl) > 2:
                return f"von {spl[1]} am {spl[2]}"
            else:
                return f"von {spl[1]}"
        else:
            return "aus dem DGM"

    html = (
        '      <div class="box" id="desc">\n'
        + '        <h2 class="title is-2">Beschreibung</h2>\n'
        + '<div class="block">\n'
        + f"Es handelt sich hierbei um eine {hld.OBJEKT.values[0]} "
    )
    if hld.HERKUNFT.values[0]:
        html += f"welche durch {hld.HERKUNFT.values[0]} "
        if hld.KARTIERT.values[0]:
            html += f"{kart_str(hld.KARTIERT.values[0])} "
        html += "kartiert wurde"
    html += ". "
    if hld.KLASSI_DGM.values[0]:
        html += f"Der Befund im DGM ist {hld.KLASSI_DGM.values[0]}. "
    if hld.RU_SCHICHT.values[0]:
        html += f"Die betroffenen Einheiten sind {hld.RU_SCHICHT.values[0]} "
        if hld.RU_SCHIC_2.values[0]:
            html += f"und {hld.RU_SCHIC_2.values[0]} "
        if hld.GEOLOGIE.values[0]:
            html += f"auf {hld.GEOLOGIE.values[0]} "
            if hld.STR_SYSTEM.values[0]:
                html += f"({hld.STR_SYSTEM.values[0]})"
        html += ". "
    else:
        if hld.GEOLOGIE.values[0]:
            html += f"Die Geologie besteht aus {hld.GEOLOGIE.values[0]} "
            if hld.STR_SYSTEM.values[0]:
                html += f"({hld.STR_SYSTEM.values[0]})"
        html += ". "

    if hld.FLAECHE_M2.values[0]:
        html += f"Die betroffene Fläche beträgt ca. {np.round(hld.FLAECHE_M2.values[0], -2)}&nbsp;m<sup>2</sup>. "

    if hld.LAENGE_M.values[0] and hld.BREITE_M.values[0]:
        html += f"Sie ist ca. {hld.LAENGE_M.values[0]}&nbsp;m lang und {hld.BREITE_M.values[0]}&nbsp;m breit"
        if (
            hld.H_MAX_MNN.values[0]
            and hld.H_MIN_MNN.values[0]
            and hld.H_DIFF_M.values[0]
        ):
            html += f" und erstreckt sich von {hld.H_MAX_MNN.values[0]}&nbsp;m&nbsp;NN bis {hld.H_MIN_MNN.values[0]}&nbsp;m&nbsp;NN über ca. {hld.H_DIFF_M.values[0]}&nbsp;m Höhendifferenz"
        html += ". "
    exp2txt = {
        "N": "Norden",
        "NNO": "Nordnordosten",
        "NNW": "Nordnordwesten",
        "NO": "Nordosten",
        "NW": "Nordwesten",
        "O": "Osten",
        "ONO": "Ostnordosten",
        "OSO": "Ostsüdosten",
        "S": "Süden",
        "SO": "Südosten",
        "SSO": "Südsüdosten",
        "SSW": "Südsüdwesten",
        "SW": "Südwesten",
        "W": "Westen",
        "WNW": "Westnordwesten",
        "WSW": "Westsüdwesten",
    }
    if hld.EXPOSITION.values[0]:
        if hld.EXPOSITION.values[0] != "n.b.":
            html += f"Das Gelände fällt nach {exp2txt[hld.EXPOSITION.values[0]]} ein. "

    if hld.LANDNUTZUN.values[0]:
        html += f"Im wesentlichen ist das Gebiet von {hld.LANDNUTZUN.values[0]} bedeckt. "

    if hld.URSACHE.values[0]:
        html += f"Eine mögliche Ursache ist {hld.URSACHE.values[0]}. "

    if hld.SCHUTZ_OBJ.values[0]:
        if hld.SCHUTZ_OBJ.values[0] == "nicht bekannt":
            html += "Eine potentielle Gefährdung ist nicht bekannt. "
        else:
            html += f"Eine potentielle Gefährdung für {hld.SCHUTZ_OBJ.values[0]} könnte vorliegen. "

    if hld.AKTIVITAET.values[0]:
        if hld.AKTIVITAET.values[0] == "nicht bekannt":
            html += "Eine mögliche Aktivität ist nicht bekannt. "
        if hld.AKTIVITAET.values[0] == "aktiv":
            html += f"Die {hld.OBJEKT.values[0]} ist aktiv. "

    if hld.MASSNAHME.values[0]:
        if hld.MASSNAHME.values[0] == "nicht bekannt":
            html += "Über unternommene Maßnahmen ist nichts bekannt. \n</div>"
        else:
            html += "\n</div>"

    if hld.BEMERKUNG.values[0]:
        html += (
            '\n\n<div class="notification">Kommentar: '
            + hld.BEMERKUNG.values[0]
            + ".</div>\n\n"
        )

    return html
