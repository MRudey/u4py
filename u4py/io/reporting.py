"""
Functions for creating a TeX-based report of the classified anomalies.
"""

import os
import subprocess

import geopandas as gp
import numpy as np
import uncertainties as unc


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
        + "\\setlength{\\cftsubsecnumwidth}{4em}"
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
    subprocess.run(["pdflatex", f"{report_path}"], cwd=report_out_path)
    subprocess.run(["pdflatex", f"{report_path}"], cwd=report_out_path)
    subprocess.run(["pdflatex", f"{report_path}"], cwd=report_out_path)
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
    tex = f"\\section{{Gruppe {group}}}\n\n"

    tex += location(row[1])

    # Overview plot and satellite image
    if os.path.exists(img_path + "_map.pdf") and os.path.exists(
        img_path + "_satimg.pdf"
    ):
        tex += details_and_satellite(img_path)

    # Volumina
    tex += moved_volumes(row[1])

    # Difference maps
    if os.path.exists(img_path + "_diffplan.pdf") and os.path.exists(
        img_path + "_slope.pdf"
    ):
        tex += difference(img_path)

    # Topographie
    tex += topography(row[1])

    # Landuse
    tex += landuse(row[1])

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
        f"\\textbf{{Lokalität:}} {series.locations}\n\n"
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
        + "  \\begin{subfigure}[][][t]{.45\\textwidth}\n"
        + f"    \\includegraphics[width=\\textwidth]{{{img_path+'_map.pdf'}}}\n"
        + "    \\caption{Übersicht über das Gebiet der Gruppe inklusive verschiedener Geogefahren und der detektierten Anomalien (Kartengrundlage OpenStreetMap).}\n"
        + "  \\end{subfigure}\n\hfill\n"
        + "  \\begin{subfigure}[][][t]{.45\\textwidth}\n"
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
        "Im Gebiet um die detektierte Anomalie wurde insgesamt "
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
        + "  \\begin{subfigure}[][][t]{.45\\textwidth}\n"
        + f"  \\includegraphics[width=\\textwidth]{{{img_path+'_diffplan.pdf'}}}\n"
        + "  \\caption{Differenzenplan im Gebiet.}\n"
        + "  \\end{subfigure}\n\hfill\n"
        + "  \\begin{subfigure}[][][t]{.45\\textwidth}\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=\\textwidth]{{{img_path+'_dem.pdf'}}}\n"
        + "  \\caption{Digitales Höhenmodell (Schummerung).}\n"
        + "  \\end{subfigure}\n\hfill\n"
        + "  \\caption{Höhenmodell und dessen Veränderung im Gebiet.}"
        + "\\end{figure}\n\n"
        + "\\begin{figure}[!ht]\n"
        + "  \\begin{subfigure}[][][t]{.3\\textwidth}\n"
        + f"  \\includegraphics[width=\\textwidth]{{{img_path+'_slope.pdf'}}}\n"
        + "  \\caption{Steigung}\n"
        + "  \\end{subfigure}\n\hfill\n"
        + "  \\begin{subfigure}[][][t]{.3\\textwidth}\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=\\textwidth]{{{img_path+'_aspect.pdf'}}}\n"
        + "  \\caption{Exposition.}\n"
        + "  \\end{subfigure}\n\hfill\n"
        + "  \\begin{subfigure}[][][t]{.3\\textwidth}\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=\\textwidth]{{{img_path+'_aspect_slope.pdf'}}}\n"
        + "  \\caption{Steigung und Exposition}\n"
        + "  \\end{subfigure}\n\hfill\n"
        + "  \\caption{Topographie im Gebiet.}"
        + "\\end{figure}\n\n"
    )
    return tex


def topography(series: gp.GeoSeries) -> str:
    """Converts the slope into a descriptive text.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The tex code.
    :rtype: str
    """
    # Post event data
    slope_new = float(series.slope_hull_median_new)
    slope_std_new = float(series.slope_hull_std_new)
    usl_new = unc.ufloat(slope_new, slope_std_new)
    ustr_new = str(usl_new).replace("+/-", "$\\pm$") + "\\,\\%"
    # Pre event data
    slope_old = float(series.slope_hull_median_old)
    slope_std_old = float(series.slope_hull_std_old)
    usl_old = unc.ufloat(slope_old, slope_std_old)
    ustr_old = str(usl_old).replace("+/-", "$\\pm$") + "\\,\\%"

    tex = (
        f"Die Steigung im Gebiet ist {slope_std_str(slope_std_new)} und "
        + f"{slope_str(slope_new)} ({ustr_new}). "
        + f"Vor dem Ereignis war die Steigung {slope_std_str(slope_std_old)} "
        + f"und {slope_str(slope_old)} ({ustr_old}).\n"
    )
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
    tex = ""
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
        "\\subsection*{Geologie}\n\n"
        + "\\begin{figure}[H]\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=.9\\textwidth]{{{img_path+'_GK25.pdf'}}}\n"
        + "  \\caption{Geologie im Gebiet basierend auf GK25 (Quelle: HLNUG).}\n"
        + "\\end{figure}\n\n"
    )
    return tex


def hydrogeology(img_path: os.PathLike) -> str:
    tex = (
        "\\subsection*{Hydrogeologie}\n\n"
        + "\\begin{figure}[H]\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=.9\\textwidth]{{{img_path+'_HUEK200.pdf'}}}\n"
        + "  \\caption{Hydrogeologische Einheiten im Gebiet basierend auf HÜK200 (Quelle: HLNUG).}\n"
        + "\\end{figure}\n\n"
    )
    return tex


def soils(img_path: os.PathLike) -> str:
    tex = (
        "\\subsection*{Bodengruppen}\n\n"
        + "\\begin{figure}[H]\n"
        + "\\centering\n"
        + f"  \\includegraphics[width=.9\\textwidth]{{{img_path+'_BFD50.pdf'}}}\n"
        + "  \\caption{Bodenhauptgruppen im Gebiet basierend auf der BFD50 (Quelle: HLNUG).}\n"
        + "\\end{figure}\n\n"
    )
    return tex
