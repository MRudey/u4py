"""
Functions for creating a TeX-based report of the classified anomalies.
"""

import os
import shutil
import subprocess

import geopandas as gp
import humanize
import numpy as np
import uncertainties as unc
from tqdm import tqdm

import u4py.io.human_text as u4human


def multi_report(output_path: os.PathLike):
    tex_folder = os.path.join(output_path, "tex_includes")
    include_list = [
        os.path.join(tex_folder, fp)
        for fp in os.listdir(tex_folder)
        if fp.endswith("tex")
    ]
    base_tex = (
        "\\documentclass[\n"
        + "  ngerman,\n"
        + "  logofile=/home/rudolf/Documents/umwelt4/tuda_logo.pdf,\n"
        + "  accentcolor=8c,\n"
        + "]{tudapub}\n"
        + "\\usepackage{graphicx}\n"
        + "\\usepackage{subcaption}\n"
        + "\\usepackage{enumitem}\n"
        + "\\usepackage{wrapfig}\n"
        + "\\usepackage{float}\n"
        + "\\usepackage{fontawesome}\n"
        + "\\usepackage[ngerman]{babel}\n"
        + "\\usepackage{tocloft}"
        + "\\addtolength{\\cftsecnumwidth}{10pt}"
        + "\\begin{document}\n"
    )
    include_list.sort()
    for incl in tqdm(
        include_list, desc="Generating individual reports", leave=False
    ):
        num = os.path.splitext(os.path.split(incl)[-1])[0].replace("_info", "")
        tex = f"{base_tex}\\input{{{incl}}}\n\\clearpage\n" + "\\end{document}"
        report_folder = os.path.join(output_path, "single_reports_tex")
        report_path = os.path.join(report_folder, f"Site_Report_{num}.tex")
        os.makedirs(report_folder, exist_ok=True)
        with open(
            report_path, "wt", encoding="utf-8", newline="\n"
        ) as tex_file:
            tex_file.write(tex)

        report_out_path = os.path.join(output_path, "single_reports_pdf")
        os.makedirs(report_out_path, exist_ok=True)
        run_tool_chain(report_path, report_out_path)


def main_report(
    output_path: os.PathLike,
    title: str,
    subtitle: str,
    suffix: str = "",
):
    """Generates a single report from all files in the `tex_includes` folder.

    :param output_path: The path to the output folder, where also the `tex_includes` are located.
    :type output_path: os.PathLike
    :param suffix: The suffix added to the report to distinguish different datasets, defaults to ""
    :type suffix: str, optional
    """
    tex_folder = os.path.join(output_path, "tex_includes")
    include_list = [
        os.path.join(tex_folder, fp)
        for fp in os.listdir(tex_folder)
        if fp.endswith("tex")
    ]
    tex = (
        "\\documentclass[\n"
        + "  ngerman,\n"
        + "  logofile=/home/rudolf/Documents/umwelt4/tuda_logo.pdf,\n"
        + "  accentcolor=8c,\n"
        + "]{tudapub}\n"
        + "\\usepackage{graphicx}\n"
        + "\\usepackage{subcaption}\n"
        + "\\usepackage{enumitem}\n"
        + "\\usepackage{wrapfig}\n"
        + "\\usepackage{float}\n"
        + "\\usepackage{fontawesome}\n"
        + "\\usepackage[ngerman]{babel}\n"
        + "\\usepackage{tocloft}"
        + "\\addtolength{\\cftsecnumwidth}{10pt}"
        # + "\\usepackage[margin=1in]{geometry}\n"
        + "\\begin{document}\n"
        + f"\\title{{{title}}}\n"
        + f"\\subtitle{{{subtitle}}}\n"
        + "\\author{automatisch generierter Report aus U4Py}\n"
        + "\\date{\\today}\n"
        + "\\addTitleBox{Institut für Angewandte Geowissenschaften}\n\n"
        + "\\maketitle\n\n"
        + "\\tableofcontents\n\n"
        + "\\clearpage\n"
    )
    include_list.sort()
    for incl in include_list:
        tex += f"\\input{{{incl}}}\n\\clearpage\n"
    tex += "\\end{document}"

    report_path = os.path.join(output_path, f"Site_Report{suffix}.tex")
    with open(report_path, "wt", encoding="utf-8", newline="\n") as tex_file:
        tex_file.write(tex)

    report_out_path = os.path.split(output_path)[0]
    run_tool_chain(report_path, report_out_path, suffix)


def run_tool_chain(
    report_path: os.PathLike, report_out_path: os.PathLike, suffix: str = ""
):
    """Runs a 3x Latex toolchain, first two in draft mode and then full compilation.

    Requires Latex.

    :param report_path: The path to the main file
    :type report_path: os.PathLike
    :param report_out_path: The output path of the final file
    :type report_out_path: os.PathLike
    :param suffix: A suffix that was added to the report_path, defaults to ""
    :type suffix: str, optional
    """
    if shutil.which("pdflatex") is not None:
        if not suffix:
            suffix = os.path.splitext(report_path)[0][-6:]
            clean_aux_files(report_out_path, suffix)
        else:
            clean_aux_files(report_out_path, suffix)
        subprocess.run(
            ["pdflatex", "-draftmode", f"{report_path}"],
            cwd=report_out_path,
        )
        subprocess.run(
            ["pdflatex", "-draftmode", f"{report_path}"],
            cwd=report_out_path,
        )
        subprocess.run(
            ["pdflatex", f"{report_path}"],
            cwd=report_out_path,
        )
        clean_aux_files(report_out_path, suffix)
    else:
        raise FileNotFoundError(
            "pdflatex is not installed or unavailable to Python.\n"
            + "Consider installing or check the path variable."
        )


def clean_aux_files(report_out_path: os.PathLike, suffix: str):
    """Cleans Latex auxiliary files

    :param report_out_path: The path to the latex file
    :type report_out_path: os.PathLike
    :param suffix: The suffix of the file
    :type suffix: str
    """
    tex_temps = [
        "pdfa.xmpi",
        f"Site_Report{suffix}.aux",
        f"Site_Report{suffix}.log",
        f"Site_Report{suffix}.out",
        f"Site_Report{suffix}.xmpdata",
        f"Site_Report{suffix}.toc",
    ]
    for tp in tex_temps:
        fp = os.path.join(report_out_path, tp)
        if os.path.exists(fp):
            os.remove(fp)


def site_report(
    row: tuple,
    output_path: os.PathLike,
    suffix: str,
    img_fmt: str = "pdf",
):
    """Creates a report for each area of interest using LaTeX. This is later merged together into a larger main document by the `main` function.

    :param row: The index and data for the area of interest.
    :type row: tuple
    :param output_path: The path where to store the outputs.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for the LaTeX files.
    :type suffix: str
    :param out_format: The output format (pdf or docx), determines which kind of images are included (pdf or png).
    :type out_format: str
    """

    # Setting Paths
    group = row[1].group
    output_path_tex = os.path.join(output_path, suffix)
    os.makedirs(output_path_tex, exist_ok=True)
    img_path = os.path.join(output_path, "known_features", f"{group:05}")

    # Create TeX code
    heading = sanitize_text(",".join(row[1].locations.split(",")[:-4]))
    tex = "\\section{" + heading + f" ({group})" + "}\n\n"

    # Overview plot and satellite image
    tex += location(row[1])
    if os.path.exists(img_path + f"_map.{img_fmt}") and os.path.exists(
        img_path + f"_satimg.{img_fmt}"
    ):
        tex += details_and_satellite(img_path, img_fmt)
    # Manual Classification or HLNUG data
    tex += manual_description(row[1])
    tex += shape(row[1])
    tex += landuse(row[1])

    # Volumina
    tex += moved_volumes(row[1])

    # Difference maps
    if os.path.exists(img_path + f"_diffplan.{img_fmt}"):
        tex += difference(img_path, img_fmt)

    # Topographie
    if os.path.exists(img_path + f"_slope.{img_fmt}") or os.path.exists(
        img_path + f"_aspect_slope.{img_fmt}"
    ):
        tex += topography(row[1], img_path, img_fmt)

    # PSI Data
    if os.path.exists(img_path + f"_psi.{img_fmt}"):
        tex += psi_map(img_path, img_fmt)

    # Geohazard
    tex += geohazard(row[1])

    # Geologie etc...
    if os.path.exists(img_path + f"_GK25.{img_fmt}"):
        tex += geology(img_path, img_fmt)
    if os.path.exists(img_path + f"_HUEK200.{img_fmt}"):
        tex += hydrogeology(img_path, img_fmt)
    if os.path.exists(img_path + f"_BFD50.{img_fmt}"):
        tex += soils(img_path, img_fmt)

    # Save to tex file
    with open(
        os.path.join(output_path_tex, f"{group:05}_info.tex"),
        "wt",
        encoding="utf-8",
        newline="\n",
    ) as tex_file:
        tex_file.write(tex)


def location(series: gp.GeoSeries) -> str:
    """Adds location information to the document

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The tex code.
    :rtype: str
    """

    wgs_point = gp.GeoDataFrame(
        geometry=[series.geometry.centroid], crs="EPSG:32632"
    ).to_crs("EPSG:4326")
    lat = np.round(float(wgs_point.geometry.y.iloc[0]), 6)
    lng = np.round(float(wgs_point.geometry.x.iloc[0]), 6)
    address = sanitize_text(series.locations)
    tex = (
        "\\subsection*{{Lokalität:}}\n"
        + "\\textbf{Adresse:} "
        + f"{address}\n\n"
        + "\\textbf{{Koordinaten (UTM 32N):}} "
        + f"{int(series.geometry.centroid.y)}\\,N "
        + f"{int(series.geometry.centroid.x)}\\,E\n\n"
        + "\\textbf{{Google Maps:}} "
        + f"\\href{{https://www.google.com/maps/place/{lat},{lng}/@{lat},{lng}/data=!3m1!1e3}}"
        + f"{{\\faExternalLink {np.round(lat, 3)}\\,N, {np.round(lng, 3)}\\,E}}\n\n"
        + "\\textbf{{Bing Maps:}} "
        + f"\\href{{https://bing.com/maps/default.aspx?cp={lat}~{lng}&style=h&lvl=15}}"
        + f"{{\\faExternalLink {np.round(lat, 3)}\\,N, {np.round(lng, 3)}\\,E}}\n\n"
        + "\\textbf{{OpenStreetMap:}} "
        + f"\\href{{http://www.openstreetmap.org/?lat={lat}&lon={lng}&zoom=17&layers=M}}"
        + f"{{\\faExternalLink {np.round(lat, 3)}\\,N, {np.round(lng, 3)}\\,E}}\n\n"
    )
    return tex


def shape(series: gp.GeoSeries) -> str:
    """Adds shape information to the document

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The tex code.
    :rtype: str
    """
    humanize.activate("de")
    tex = "\\paragraph{Größe und Form}\n"

    long_ax = eval(series["shape_ellipse_a"])
    short_ax = eval(series["shape_ellipse_b"])
    if isinstance(long_ax, list):
        areas = [np.pi * a * b for a, b in zip(long_ax, short_ax)]
        if len(areas) > 0:
            imax = np.argmax(areas)
            imin = np.argmin(areas)
            if len(long_ax) > 2:
                lax = (
                    f"zwischen {round(np.min(long_ax))} und "
                    + f"{round(np.max(long_ax))}"
                )
                sax = (
                    f"zwischen {round(np.min(short_ax))} und "
                    + f"{round(np.max(short_ax))}"
                )
            else:
                lax = f"{round(long_ax[0])} und {round(long_ax[1])}"
                sax = f"{round(short_ax[0])} und {round(short_ax[1])}"

            tex += (
                f"Es handelt sich um {humanize.apnumber(len(long_ax))} Anomalien. "
                + f"Die Anomalien sind {lax}\\,m lang und {sax}\\,m breit. "
            )
            if len(long_ax) > 2:
                tex += (
                    "Die flächenmäßig kleinste Anomalie ist hierbei "
                    + f"{round(long_ax[imin])}\\,m lang und "
                    + f"{round(short_ax[imin])}\\,m breit, die größte "
                    + f"{round(long_ax[imax])}\\,m lang und "
                    + f"{round(short_ax[imax])}\\,m breit. "
                )
    if isinstance(long_ax, float):
        lax = f"{round(long_ax)}"
        sax = f"{round(short_ax)}"
        tex += f"Die Anomalie ist {lax}\\,m lang und {sax}\\,m breit. "

    return tex


def manual_description(series: gp.GeoSeries) -> str:
    tex = "\\subsection*{Manuelle Klassifizierung}\n\n"
    if int(series.manual_known):
        tex += (
            "Die Anomalie ist bereits in den Datenbanken des HLNUG vorhanden. "
        )
    else:
        tex += "Die Anomalie ist noch nicht in den Datenbanken des HLNUG vorhanden. "
    ncls = len(
        [
            series[f"manual_class_{ii}"]
            for ii in range(1, 4)
            if series[f"manual_class_{ii}"]
            and series[f"manual_class_{ii}"] != "(empty)"
        ]
    )
    prob_txt = ["wahrscheinlich", "möglicherweise"]
    if ncls == 1:
        tex += (
            f"Es handelt sich {prob_txt[int(series['manual_unclear_1'])]} "
            + f"um ein/e {series['manual_class_1']}: \n"
            + "\\begin{itemize}\n"
            + f"\\item[$\\rightarrow$] {series['manual_comment']}\n"
            + "\\end{itemize}\n"
        )
    elif ncls == 2:
        tex += (
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
        tex += (
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
        tex += "Eine manuelle Klassifikation ist noch nicht erfolgt. "
    if int(series["manual_research"]):
        tex += (
            "Aufgrund der Nähe zu Infrastruktur oder der unklaren Lage "
            + "sollte die Anomalie einer genaueren Untersuchung unterzogen "
            + "werden. "
        )
    tex += "\n\n"
    return tex


def details_and_satellite(img_path: os.PathLike, img_fmt: str) -> str:
    """Adds the detailed map and the satellite image map.

    :param img_path: The path to the image folder including group name.
    :type img_path: os.PathLike
    :return: The tex code.
    :rtype: str
    """
    tex = (
        "\\begin{figure}[h!]\n"
        + "  \\centering\n"
        + "  \\begin{subfigure}[][][t]{.49\\textwidth}\n"
        + f"    \\includegraphics[width=\\textwidth]{{{img_path + f'_map.{img_fmt}'}}}\n"
        + "    \\caption{Übersicht über das Gebiet der Gruppe inklusive verschiedener Geogefahren und der detektierten Anomalien (Kartengrundlage OpenStreetMap).}\n"
        + "  \\end{subfigure}\n\hfill\n"
        + "  \\begin{subfigure}[][][t]{.49\\textwidth}\n"
        + f"    \\includegraphics[width=\\textwidth]{{{img_path + f'_satimg.{img_fmt}'}}}\n"
        + "    \\caption{Luftbild basierend auf ESRI Imagery.}\n"
        + "  \\end{subfigure}\n"
        + "  \\caption{Lokalität der Anomalie.}"
        + "\\end{figure}\n\n"
    )
    return tex


def moved_volumes(series: gp.GeoSeries) -> str:
    """Adds description of moved volumes to the document.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The tex code.
    :rtype: str
    """
    tex = (
        "\\clearpage\n\\subsection*{Höhenveränderungen}\n"
        + "Im Gebiet um die detektierte Anomalie wurde insgesamt "
        + f"{series.volumes_moved}\\,m$^3$ Material bewegt, "
        + f"wovon {series.volumes_added}\\,m$^3$ hinzugefügt und "
        + f"{abs(series.volumes_removed)}\\,m$^3$ abgetragen wurde. "
        + f"Dies ergibt eine Gesamtbilanz von {series.volumes_total}\\,m$^3$,"
        + f" in Summe wurde also {u4human.vol_str(series.volumes_total)}.\n\n"
    )
    return tex


def difference(img_path: os.PathLike, img_fmt: str) -> str:
    """Adds the difference and slope maps.

    :param img_path: The path to the image folder including group name.
    :type img_path: os.PathLike
    :return: The tex code.
    :rtype: str
    """
    tex = ""
    if os.path.exists(img_path + f"_diffplan.{img_fmt}"):
        tex += (
            "\\begin{figure}[!ht]\n"
            + "  \\centering"
            + f"  \\includegraphics[width=.9\\textwidth]{{{img_path + f'_diffplan.{img_fmt}'}}}\n"
            + "  \\caption{Differenzenplan im Gebiet.}\n"
            + "\\end{figure}\n"
        )

    if os.path.exists(img_path + f"_dem.{img_fmt}"):
        tex += (
            "\\begin{figure}[!ht]\n"
            + "  \\centering"
            + f"  \\includegraphics[width=.9\\textwidth]{{{img_path + f'_dem.{img_fmt}'}}}\n"
            + "  \\caption{Digitales Höhenmodell (Schummerung).}\n"
            + "\\end{figure}\n\n"
        )
    return tex


def topography(
    series: gp.GeoSeries, img_path: os.PathLike, img_fmt: str
) -> str:
    """Converts the slope into a descriptive text.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The tex code.
    :rtype: str
    """
    tex = "\n\\subsection*{Topographie}\n\n"

    for yy in ["14", "19", "21"]:
        year = f"20{yy}"
        tex += f"\\paragraph*{{{year}}}\n"

        if not series[f"slope_polygons_mean_{yy}"] == "[]":
            # Values for the individual polygons (can be empty)
            sl_m = eval(series[f"slope_polygons_mean_{yy}"])
            sl_s = eval(series[f"slope_polygons_std_{yy}"])
            as_m = eval(series[f"aspect_polygons_mean_{yy}"])
            as_s = eval(series[f"aspect_polygons_std_{yy}"])
            if isinstance(sl_m, list) or isinstance(sl_m, float):
                usl, usl_str = u4human.topo_text(
                    sl_m, sl_s, "\\%", is_tex=True
                )
                tex += (
                    f"Im Bereich der Anomalie {u4human.slope_std_str(usl.s)} und "
                    + f"{u4human.slope_str(usl.n)} ({usl_str}). "
                )
                if as_m:
                    uas, uas_str = u4human.topo_text(
                        as_m, as_s, "°", is_tex=True
                    )
                    tex += (
                        "Die Anomalien fallen nach "
                        + f"{u4human.direction_to_text(uas.n)} ({uas_str}) ein. "
                    )
            else:
                tex += "Es liegen für den inneren Bereich der Anomalie keine Daten vor (außerhalb DEM). "

            # Values for the hull around all anomalies
            usl, usl_str = u4human.topo_text(
                eval(series[f"slope_hull_mean_{yy}"]),
                eval(series[f"slope_hull_std_{yy}"]),
                "\\%",
                is_tex=True,
            )
            if isinstance(usl, unc.UFloat):
                tex += (
                    f"Im näheren Umfeld ist das Gelände {u4human.slope_std_str(usl.s)}"
                    + f" und {u4human.slope_str(usl.n)} ({usl_str}). "
                )
                uas, uas_str = u4human.topo_text(
                    eval(series[f"aspect_hull_mean_{yy}"]),
                    eval(series[f"aspect_hull_std_{yy}"]),
                    "°",
                    is_tex=True,
                )
                if isinstance(uas, unc.UFloat):
                    tex += (
                        "Der Bereich fällt im Mittel nach "
                        + f"{u4human.direction_to_text(uas.n)} ({uas_str}) ein. "
                    )
            else:
                tex += "Es liegen für das nähere Umfeld der Anomalien keine Werte für die Steigung vor. "
            tex += "\n\n"
    if os.path.exists(img_path + f"_slope.{img_fmt}") and os.path.exists(
        img_path + f"_aspect.{img_fmt}"
    ):
        tex += (
            "\n\\begin{figure}[!ht]\n"
            + "  \\begin{subfigure}[][][t]{.49\\textwidth}\n"
            + f"  \\includegraphics[width=\\textwidth]{{{img_path + f'_slope.{img_fmt}'}}}\n"
            + "  \\caption{Steigung}\n"
            + "  \\end{subfigure}\n\hfill\n"
            + "  \\begin{subfigure}[][][t]{.49\\textwidth}\n"
            + "\\centering\n"
            + f"  \\includegraphics[width=\\textwidth]{{{img_path + f'_aspect.{img_fmt}'}}}\n"
            + "  \\caption{Exposition.}\n"
            + "  \\end{subfigure}\n\hfill\n"
            + "  \\caption{Topographie im Gebiet.}"
            + "\\end{figure}\n\n"
        )

    if os.path.exists(img_path + f"_aspect_slope.{img_fmt}"):
        tex += (
            "\\begin{figure}[!ht]\n"
            + "  \\centering\n"
            + f"  \\includegraphics[width=.95\\textwidth]{{{img_path + f'_aspect_slope.{img_fmt}'}}}\n"
            + "  \\caption{Steigung und Exposition}\n"
            + "\\end{figure}\n"
        )
    tex += "\n\n"
    return tex


def landuse(series: gp.GeoSeries) -> str:
    """Converts the landuse into a descriptive text.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The tex code.
    :rtype: str
    """
    landuse = ""
    tex = "\\paragraph{Landnutzung}\n\n"
    try:
        landuse = eval(series.landuse_names)
    except NameError:
        landuse = series.landuse_names
    landuse_perc = eval(series.landuse_percent)
    if landuse:
        tex += (
            "Der überwiegende Teil wird durch "
            + f"{u4human.landuse_str(series.landuse_major)} bedeckt. "
            + "Die Anteile der Landnutzung sind: \n\n"
        )
        if isinstance(landuse, list):
            for ii in range(1, len(landuse) + 1):
                tex += f" {landuse_perc[-ii]:.1f}\\% {u4human.landuse_str(landuse[-ii])}, "
            tex = tex[:-2] + ".\n"
        else:
            tex += f"{landuse_perc:.1f}\\% {u4human.landuse_str(landuse)}"
    return tex


def psi_map(img_path: os.PathLike, img_fmt: str) -> str:
    """Adds the psi map with timeseries.

    :param img_path: The path to the image folder including group name.
    :type img_path: os.PathLike
    :return: The tex code.
    :rtype: str
    """
    tex = (
        "\\subsection*{InSAR Daten}\n\n"
        + "\\begin{figure}[h!]\n"
        + "  \\centering\n"
        + f"  \\includegraphics[width=\\textwidth]{{{img_path + f'_psi.{img_fmt}'}}}\n"
        + "  \\caption{Persistent scatterer und Zeitreihe der Deformation "
        + "im Gebiet der Gruppe.}\n"
        + "\\end{figure}\n\n"
    )
    return tex


def geohazard(series: gp.GeoSeries) -> str:
    """Converts the known geohazards into a descriptive text.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The tex code.
    :rtype: str
    """
    tex = (
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
    tex += landslide_risk(series)
    tex += karst_risk(series)
    tex += subsidence_risk(series)
    return tex


def landslide_risk(series: gp.GeoSeries) -> str:
    """Converts the known landslides into a descriptive text.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The tex code.
    :rtype: str
    """
    lsar = series.landslide_total
    tex = "\\paragraph*{Rutschungsgefährdung}\n\n"
    if lsar > 0:
        tex += f"Das Gebiet liegt {u4human.part_str(lsar)} ({lsar:.0f}\%) in einem gefährdeten Bereich mit rutschungsanfälligen Schichten. "
        try:
            landslide_units = eval(series.landslide_units)
        except (NameError, SyntaxError):
            landslide_units = series.landslide_units
        if isinstance(landslide_units, list):
            tex += "Die Einheiten sind: "
            for unit in landslide_units:
                tex += f"{unit}, "
            tex = tex[:-2] + ".\n\n"
        else:
            tex += f"Wichtigste Einheiten sind {landslide_units}.\n\n"
    else:
        tex += (
            "Es liegen keine Informationen zur Rutschungsgefährdung vor.\n\n"
        )
    return tex


def karst_risk(series: gp.GeoSeries) -> str:
    """Converts the known karst phenomena into a descriptive text.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The tex code.
    :rtype: str
    """
    ksar = series.karst_total
    tex = "\\paragraph*{Karstgefährdung}\n\n"
    if ksar > 0:
        tex += f"Das Gebiet liegt {u4human.part_str(ksar)} ({ksar:.0f}\%) in einem Bereich bekannter verkarsteter Schichten. "
        try:
            karst_units = eval(series.karst_units)
        except (NameError, SyntaxError):
            karst_units = series.karst_units
        if isinstance(karst_units, list):
            tex += "Die Einheiten sind: "
            for unit in karst_units:
                tex += f"{unit}, "
            tex = tex[:-2] + ".\n\n"
        else:
            tex += f"Wichtigste Einheiten sind {karst_units}.\n\n"
    else:
        tex += "Es liegen keine Informationen zur Karstgefährdung vor.\n\n"
    return tex


def subsidence_risk(series: gp.GeoSeries) -> str:
    """Converts the area of known subsidence into a descriptive text.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The tex code.
    :rtype: str
    """
    subsar = series.subsidence_total
    tex = "\\paragraph*{Setzungsgefährdung}\n\n"
    if subsar > 0:
        tex += f"Das Gebiet liegt {u4human.part_str(subsar)} ({subsar:.0f}\%) in einem Bereich bekannter setzungsgefährdeter Schichten. "
        try:
            subsidence_units = eval(series.subsidence_units)
        except (NameError, SyntaxError):
            subsidence_units = series.subsidence_units
        if isinstance(subsidence_units, list):
            tex += "Die Einheiten sind: "
            for unit in subsidence_units:
                tex += f"{unit}, "
            tex = tex[:-2] + ".\n\n"
        else:
            tex += f"Wichtigste Einheiten sind {subsidence_units}.\n\n"
    else:
        tex += "Es liegen keine Informationen zur Setzungsgefährdung vor.\n\n"
    return tex


def geology(img_path: os.PathLike, img_fmt: str) -> str:
    """Adds information for geology to the docum

    :param img_path: The path to the geology image file.
    :type img_path: os.PathLike
    :param img_fmt: The image file format.
    :type img_fmt: str
    :return: The tex code
    :rtype: str
    """
    tex = (
        "\n\\subsection*{Geologie}\n\n"
        + "\\begin{figure}[H]\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=\\textwidth]{{{img_path + f'_GK25.{img_fmt}'}}}\n"
        + "\\end{figure}\n"
        + "\\vspace{-2ex}\n"
        + "\\begin{figure}[H]\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=.75\\textwidth]{{{img_path + f'_GK25_leg.{img_fmt}'}}}\n"
        + "  \\caption{Geologie im Gebiet basierend auf GK25 (Quelle: HLNUG).}\n"
        + "\\end{figure}\n\n"
    )
    return tex


def hydrogeology(img_path: os.PathLike, img_fmt: str) -> str:
    """Adds information for hydrogeology to the document

    :param img_path: The path to the hydrogeology image file.
    :type img_path: os.PathLike
    :param img_fmt: The image file format.
    :type img_fmt: str
    :return: The tex code
    :rtype: str
    """
    tex = (
        "\n\\subsection*{Hydrogeologie}\n\n"
        + "\\begin{figure}[H]\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=\\textwidth]{{{img_path + f'_HUEK200.{img_fmt}'}}}\n"
        + "\\end{figure}\n"
        + "\\vspace{-2ex}\n"
        + "\\begin{figure}[H]\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=.75\\textwidth]{{{img_path + f'_HUEK200_leg.{img_fmt}'}}}\n"
        + "  \\caption{Hydrogeologische Einheiten im Gebiet basierend auf HÜK200 (Quelle: HLNUG).}\n"
        + "\\end{figure}\n\n"
    )
    return tex


def soils(img_path: os.PathLike, img_fmt: str) -> str:
    """Adds information for soils to the documen

    :param img_path: The path to the soils image file.
    :type img_path: os.PathLike
    :param img_fmt: The image file format.
    :type img_fmt: str
    :return: The tex code
    :rtype: str
    """
    tex = (
        "\n\\subsection*{Bodengruppen}\n\n"
        + "\\begin{figure}[H]\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=\\textwidth]{{{img_path + f'_BFD50.{img_fmt}'}}}\n"
        + "\\end{figure}\n"
        + "\\vspace{-2ex}\n"
        + "\\begin{figure}[H]\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=.75\\textwidth]{{{img_path + f'_BFD50_leg.{img_fmt}'}}}\n"
        + "  \\caption{Bodenhauptgruppen im Gebiet basierend auf der BFD50 (Quelle: HLNUG).}\n"
        + "\\end{figure}\n\n"
    )
    return tex


def sanitize_text(in_str: str) -> str:
    """Escapes all special characters in the input string for LaTeX.

    :param in_str: The input string, possibly containing symbols with special meaning in LaTeX.
    :type in_str: str
    :return: The sanitized string.
    :rtype: str
    """
    # Somehow regex did not work properly...
    characters = ["&", "%", "$", "#", "_", "{", "}"]
    for char in characters:
        in_str = in_str.replace(char, f"\\{char}")
    return in_str
