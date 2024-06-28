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


def main_report(output_path: os.PathLike):
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
        + "\\title{Detektierte Anomalien}\n"
        + "\\subtitle{Atlas anomaler Bodenbewegungen in Hessen}\n"
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

    report_path = os.path.join(output_path, "Site_Report.tex")
    with open(report_path, "wt", encoding="utf-8", newline="\n") as tex_file:
        tex_file.write(tex)

    report_out_path = os.path.split(output_path)[0]
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
    clean_aux_files(report_out_path)


def clean_aux_files(report_out_path: os.PathLike):
    tex_temps = [
        "pdfa.xmpi",
        "Site_Report.aux",
        "Site_Report.log",
        "Site_Report.out",
        "Site_Report.xmpdata",
    ]
    for tp in tex_temps:
        fp = os.path.join(report_out_path, tp)
        if os.path.exists(fp):
            os.remove(fp)


def site_report(row: tuple, output_path: os.PathLike, suffix: str):
    """Creates a report for each area of interest using LaTeX. This is later merged together into a larger main document by the `main` function.

    :param row: The index and data for the area of interest.
    :type row: tuple
    :param output_path: The path where to store the outputs.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for the LaTeX files.
    :type suffix: str
    """

    # Setting Paths
    group = row[1].group
    output_path_tex = os.path.join(output_path, suffix)
    os.makedirs(output_path_tex, exist_ok=True)
    img_path = os.path.join(output_path, "known_features", f"{group:05}")

    # Create TeX code

    tex = (
        "\\section{"
        + ",".join(row[1].locations.split(",")[:-4])
        + f" ({group})"
        + "}\n\n"
    )

    # Overview plot and satellite image
    tex += location(row[1])
    if os.path.exists(img_path + "_map.pdf") and os.path.exists(
        img_path + "_satimg.pdf"
    ):
        tex += details_and_satellite(img_path)
    tex += shape(row[1])
    tex += landuse(row[1])

    # Manual Classification
    tex += manual_description(row[1])

    # Volumina
    tex += moved_volumes(row[1])

    # Difference maps
    if os.path.exists(img_path + "_diffplan.pdf"):
        tex += difference(img_path)

    # Topographie
    if os.path.exists(img_path + "_slope.pdf"):
        tex += topography(row[1], img_path)

    # PSI Data
    if os.path.exists(img_path + "_psi.png"):
        tex += psi_map(img_path)

    # Geohazard
    tex += geohazard(row[1])

    # Geologie etc...
    if os.path.exists(img_path + "_GK25.pdf"):
        tex += geology(img_path)
    if os.path.exists(img_path + "_HUEK200.pdf"):
        tex += hydrogeology(img_path)
    if os.path.exists(img_path + "_BFD50.pdf"):
        tex += soils(img_path)

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
    lat = np.round(float(wgs_point.geometry.y), 6)
    lng = np.round(float(wgs_point.geometry.x), 6)

    tex = (
        "\\subsection*{{Lokalität:}}\n"
        + "\\textbf{Adresse:} "
        + f"{series.locations}\n\n"
        + f"\\textbf{{Koordinaten (UTM 32N):}} "
        + f"{int(series.geometry.centroid.y)}\\,N "
        + f"{int(series.geometry.centroid.x)}\\,E\n\n"
        + f"\\textbf{{Google Maps:}} "
        + f"\\href{{https://www.google.com/maps/place/{lat},{lng}/@{lat},{lng}/data=!3m1!1e3}}"
        + f"{{\\faExternalLink {np.round(lat,3)}\\,N, {np.round(lng,3)}\\,E}}\n\n"
        + f"\\textbf{{Bing Maps:}} "
        + f"\\href{{https://bing.com/maps/default.aspx?cp={lat}~{lng}&style=h&lvl=15}}"
        + f"{{\\faExternalLink {np.round(lat,3)}\\,N, {np.round(lng,3)}\\,E}}\n\n"
        + f"\\textbf{{OpenStreetMap:}} "
        + f"\\href{{http://www.openstreetmap.org/?lat={lat}&lon={lng}&zoom=17&layers=M}}"
        + f"{{\\faExternalLink {np.round(lat,3)}\\,N, {np.round(lng,3)}\\,E}}\n\n"
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

        tex += (
            f"Es handelt sich um {humanize.apnumber(len(long_ax))} Anomalien. "
            + f"Die Anomalien sind ca. {lax}\\,m lang und ca. {sax}\\,m breit. "
        )
        if len(long_ax) > 2:
            tex += (
                "Die flächenmäßig kleinste Anomalie ist hierbei ca. "
                + f"{round(long_ax[imin])}\\,m lang und "
                + f"{round(short_ax[imin])}\\,m breit, die größte ca. "
                + f"{round(long_ax[imax])}\\,m lang und "
                + f"{round(short_ax[imax])}\\,m breit. "
            )
    if isinstance(long_ax, float):
        lax = f"{round(long_ax)}"
        sax = f"{round(short_ax)}"
        tex += f"Die Anomalie ist ca. {lax}\\,m lang und ca. {sax}\\,m breit. "

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


def details_and_satellite(img_path: os.PathLike) -> str:
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
        + f"    \\includegraphics[width=\\textwidth]{{{img_path+'_map.pdf'}}}\n"
        + "    \\caption{Übersicht über das Gebiet der Gruppe inklusive verschiedener Geogefahren und der detektierten Anomalien (Kartengrundlage OpenStreetMap).}\n"
        + "  \\end{subfigure}\n\hfill\n"
        + "  \\begin{subfigure}[][][t]{.49\\textwidth}\n"
        + f"    \\includegraphics[width=\\textwidth]{{{img_path+'_satimg.pdf'}}}\n"
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
        + f" in Summe wurde also {vol_str(series.volumes_total)}.\n\n"
    )
    return tex


def vol_str(val: float) -> str:
    """Converts a volume estimate into a descriptive text.

    :param val: The value.
    :type val: float
    :return: The descriptive text.
    :rtype: str
    """
    if val > 100:
        return "Material hinzugefügt"
    elif val < 100:
        return "Material abgetragen"
    else:
        return "das Gesamtvolumen nur wenig verändert"


def difference(img_path: os.PathLike) -> str:
    """Adds the difference and slope maps.

    :param img_path: The path to the image folder including group name.
    :type img_path: os.PathLike
    :return: The tex code.
    :rtype: str
    """
    tex = (
        "\\begin{figure}[!ht]\n"
        + "  \\centering"
        + f"  \\includegraphics[width=.9\\textwidth]{{{img_path+'_diffplan.pdf'}}}\n"
        + "  \\caption{Differenzenplan im Gebiet.}\n"
        + "\\end{figure}\n"
        + "\\begin{figure}[!ht]\n"
        + "  \\centering"
        + f"  \\includegraphics[width=.9\\textwidth]{{{img_path+'_dem.pdf'}}}\n"
        + "  \\caption{Digitales Höhenmodell (Schummerung).}\n"
        + "\\end{figure}\n\n"
    )
    return tex


def topography(series: gp.GeoSeries, img_path: os.PathLike) -> str:
    """Converts the slope into a descriptive text.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The tex code.
    :rtype: str
    """
    tex = "\\clearpage\n\\subsection*{Topographie}\n\n"

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
                usl, usl_str = topo_text(sl_m, sl_s, "\\%")
                tex += (
                    f"Im Bereich der Anomalie {slope_std_str(usl.s)} und "
                    + f"{slope_str(usl.n)} ({usl_str}). "
                )
                if as_m:
                    uas, uas_str = topo_text(as_m, as_s, "°")
                    tex += (
                        "Die Anomalien fallen nach "
                        + f"{direction_to_text(uas.n)} ({uas_str}) ein. "
                    )
            else:
                tex += "Es liegen für den inneren Bereich der Anomalie keine Daten vor (außerhalb DEM). "

            # Values for the hull around all anomalies
            usl, usl_str = topo_text(
                eval(series[f"slope_hull_mean_{yy}"]),
                eval(series[f"slope_hull_std_{yy}"]),
                "\\%",
            )
            if isinstance(usl, unc.UFloat):
                tex += (
                    f"Im näheren Umfeld ist das Gelände {slope_std_str(usl.s)}"
                    + f" und {slope_str(usl.n)} ({usl_str}). "
                )
                uas, uas_str = topo_text(
                    eval(series[f"aspect_hull_mean_{yy}"]),
                    eval(series[f"aspect_hull_std_{yy}"]),
                    "°",
                )
                if isinstance(uas, unc.UFloat):
                    tex += (
                        "Der Bereich fällt im Mittel nach "
                        + f"{direction_to_text(uas.n)} ({uas_str}) ein. "
                    )
            else:
                tex += "Es liegen für das nähere Umfeld der Anomalien keine Werte für die Steigung vor. "
            tex += "\n\n"

    tex += (
        "\n\\begin{figure}[!ht]\n"
        + "  \\begin{subfigure}[][][t]{.49\\textwidth}\n"
        + f"  \\includegraphics[width=\\textwidth]{{{img_path+'_slope.pdf'}}}\n"
        + "  \\caption{Steigung}\n"
        + "  \\end{subfigure}\n\hfill\n"
        + "  \\begin{subfigure}[][][t]{.49\\textwidth}\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=\\textwidth]{{{img_path+'_aspect.pdf'}}}\n"
        + "  \\caption{Exposition.}\n"
        + "  \\end{subfigure}\n\hfill\n"
        + "  \\caption{Topographie im Gebiet.}"
        + "\\end{figure}\n\n"
        + "\\begin{figure}[!ht]\n"
        + "  \\centering\n"
        + f"  \\includegraphics[width=.95\\textwidth]{{{img_path+'_aspect_slope.pdf'}}}\n"
        + "  \\caption{Steigung und Exposition}\n"
        + "\\end{figure}\n"
    )
    tex += "\n\n"
    return tex


def slope_str(val: float) -> str:
    """Converts a slope estimate into a descriptive text.

    :param val: The value.
    :type val: float
    :return: The descriptive text.
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
    """Converts a standard deviation of data into a descriptive text.

    :param val: The value.
    :type val: float
    :return: The descriptive text.
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
            + f"{landuse_str(series.landuse_major)} bedeckt. "
            + f"Die Anteile der Landnutzung sind: \n\n"
        )
        if isinstance(landuse, list):
            for ii in range(1, len(landuse) + 1):
                tex += f" {landuse_perc[-ii]:.1f}\\% {landuse_str(landuse[-ii])}, "
            tex = tex[:-2] + ".\n"
        else:
            tex += f"{landuse_perc:.1f}\\% {landuse_str(landuse)}"
    return tex


def landuse_str(in_str: str) -> str:
    """Translates the landuse strings from OSM into German.

    :param in_str: The landuse text from `fclass`.
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


def psi_map(img_path: os.PathLike) -> str:
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
        + f"  \\includegraphics[width=\\textwidth]{{{img_path+'_psi.png'}}}\n"
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
        tex += f"Das Gebiet liegt {part_str(lsar)} ({lsar:.0f}\%) in einem gefährdeten Bereich mit rutschungsanfälligen Schichten. "
        try:
            landslide_units = eval(series.landslide_units)
        except (NameError, SyntaxError):
            landslide_units = series.landslide_units
        if isinstance(landslide_units, list):
            tex += f"Die Einheiten sind: "
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
        tex += f"Das Gebiet liegt {part_str(ksar)} ({ksar:.0f}\%) in einem Bereich bekannter verkarsteter Schichten. "
        try:
            karst_units = eval(series.karst_units)
        except (NameError, SyntaxError):
            karst_units = series.karst_units
        if isinstance(karst_units, list):
            tex += f"Die Einheiten sind: "
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
        tex += f"Das Gebiet liegt {part_str(subsar)} ({subsar:.0f}\%) in einem Bereich bekannter setzungsgefährdeter Schichten. "
        try:
            subsidence_units = eval(series.subsidence_units)
        except (NameError, SyntaxError):
            subsidence_units = series.subsidence_units
        if isinstance(subsidence_units, list):
            tex += f"Die Einheiten sind: "
            for unit in subsidence_units:
                tex += f"{unit}, "
            tex = tex[:-2] + ".\n\n"
        else:
            tex += f"Wichtigste Einheiten sind {subsidence_units}.\n\n"
    else:
        tex += "Es liegen keine Informationen zur Setzungsgefährdung vor.\n\n"
    return tex


def geology(img_path) -> str:
    tex = (
        "\\clearpage\n\\subsection*{Geologie}\n\n"
        + "\\begin{figure}[H]\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=\\textwidth]{{{img_path+'_GK25.pdf'}}}\n"
        + "\\end{figure}\n"
        + "\\vspace{-2ex}\n"
        + "\\begin{figure}[H]\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=.75\\textwidth]{{{img_path+'_GK25_leg.pdf'}}}\n"
        + "  \\caption{Geologie im Gebiet basierend auf GK25 (Quelle: HLNUG).}\n"
        + "\\end{figure}\n\n"
    )
    return tex


def hydrogeology(img_path: os.PathLike) -> str:
    tex = (
        "\\clearpage\n\\subsection*{Hydrogeologie}\n\n"
        + "\\begin{figure}[H]\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=\\textwidth]{{{img_path+'_HUEK200.pdf'}}}\n"
        + "\\end{figure}\n"
        + "\\vspace{-2ex}\n"
        + "\\begin{figure}[H]\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=.75\\textwidth]{{{img_path+'_HUEK200_leg.pdf'}}}\n"
        + "  \\caption{Hydrogeologische Einheiten im Gebiet basierend auf HÜK200 (Quelle: HLNUG).}\n"
        + "\\end{figure}\n\n"
    )
    return tex


def soils(img_path: os.PathLike) -> str:
    tex = (
        "\\clearpage\n\\subsection*{Bodengruppen}\n\n"
        + "\\begin{figure}[H]\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=\\textwidth]{{{img_path+'_BFD50.pdf'}}}\n"
        + "\\end{figure}\n"
        + "\\vspace{-2ex}\n"
        + "\\begin{figure}[H]\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=.75\\textwidth]{{{img_path+'_BFD50_leg.pdf'}}}\n"
        + "  \\caption{Bodenhauptgruppen im Gebiet basierend auf der BFD50 (Quelle: HLNUG).}\n"
        + "\\end{figure}\n\n"
    )
    return tex


def direction_to_text(direction: float, lang: str = "de") -> str:
    """Converts an azimut between 0 and 360 to ordinal directions.

    :param direction: The direction with 0 = North and 180 = South
    :type direction: float
    :param lang: The language of the text (supported values: "en", "de", "abbrev"), defaults to "de"
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


def topo_text(
    means: float | list, std: float | list, unit: str
) -> Tuple[unc.ufloat, str]:
    """Converts the measured topography index, such as slope or aspect to a descriptive text.

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

    ustr = str(usl).replace("+/-", "$\\pm$") + f"\\,{unit}"
    if "e" in ustr:
        ustr = f"{usl:.0f}\\,{unit}".replace("+/-", "$\\pm$")
    return usl, ustr
