"""
Functions for creating a Word-Based report of the anomalies, taylored to the
requirements of the HLNUG.
"""

import os

import docx
import docx.parts
import docx.shared
import geopandas as gp
import humanize
import numpy as np
import uncertainties as unc
from docx.document import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH

import u4py.io.human_text as u4human

FIGURENUM = 1
TABLENUM = 1


def site_report(
    row: tuple,
    output_path: os.PathLike,
    suffix: str,
    hlnug_data: gp.GeoDataFrame,
    img_fmt: str = "png",
):
    """Creates a report for each area of interest using python-docx.

    :param row: The index and data for the area of interest.
    :type row: tuple
    :param output_path: The path where to store the outputs.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for the docx files.
    :type suffix: str
    """
    global FIGURENUM
    # Setting Paths
    group = row[1].group
    output_path_docx = os.path.join(output_path, suffix)
    os.makedirs(output_path_docx, exist_ok=True)
    img_path = os.path.join(output_path, "known_features", f"{group:05}")

    # Create Document and apply style
    document = docx.Document()
    styles = document.styles
    styles["Normal"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    section = document.sections[0]
    section.page_height = docx.shared.Mm(297)
    section.page_width = docx.shared.Mm(210)
    section.left_margin = docx.shared.Mm(25.4)
    section.right_margin = docx.shared.Mm(25.4)
    section.top_margin = docx.shared.Mm(25.4)
    section.bottom_margin = docx.shared.Mm(25.4)
    section.header_distance = docx.shared.Mm(12.7)
    section.footer_distance = docx.shared.Mm(12.7)

    # Start filling the Document
    document.add_heading(f"Beschreibung für Nr. {group}", level=0)

    # Add header
    heading = ",".join(row[1].locations.split(",")[:-4])
    document.add_heading(heading, level=1)

    # Overview plot and satellite image
    document = location(row[1], document)
    if os.path.exists(img_path + f"_map.{img_fmt}") and os.path.exists(
        img_path + f"_satimg.{img_fmt}"
    ):
        document = details_and_satellite(img_path, img_fmt, document)
    # Manual Classification or HLNUG data
    document = hlnug_description(
        hlnug_data[hlnug_data.AMT_NR_ == group], document
    )
    document = shape(row[1], document)
    document = landuse(row[1], document)

    # Volumina
    document = moved_volumes(row[1], document)

    # Difference maps
    if os.path.exists(img_path + f"_diffplan.{img_fmt}"):
        document = difference(img_path, img_fmt, document)

    # Topographie
    if os.path.exists(img_path + f"_slope.{img_fmt}") or os.path.exists(
        img_path + f"_aspect_slope.{img_fmt}"
    ):
        document = topography(row[1], img_path, img_fmt, document)

    # PSI Data
    if os.path.exists(img_path + f"_psi.{img_fmt}"):
        document = psi_map(img_path, img_fmt, document)

    # Geohazard
    document = geohazard(row[1], document)

    # Geologie etc...
    if os.path.exists(img_path + f"_GK25.{img_fmt}"):
        document = geology(img_path, img_fmt, document)
    if os.path.exists(img_path + f"_HUEK200.{img_fmt}"):
        document = hydrogeology(img_path, img_fmt, document)
    if os.path.exists(img_path + f"_BFD50.{img_fmt}"):
        document = soils(img_path, img_fmt, document)
    FIGURENUM = 1
    # Save to docx file
    document.save(os.path.join(output_path_docx, f"{group:05}_info.docx"))


def location(series: gp.GeoSeries, document: Document) -> Document:
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
    document.add_heading("Lokalität:", level=1)

    prgph = document.add_paragraph()
    prgph.add_run("Adresse: ").bold = True
    prgph.add_run(f"{series.locations}".replace("\n", ""))

    prgph = document.add_paragraph()
    prgph.add_run("Koordinaten (UTM 32N): ").bold = True
    prgph.add_run(f"{int(series.geometry.centroid.y)} N ")
    prgph.add_run(f"{int(series.geometry.centroid.x)} E")

    prgph = document.add_paragraph()
    prgph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    prgph.add_run("Google Maps: ").bold = True
    prgph.add_run(
        f"https://www.google.com/maps/place/{lat},{lng}/@{lat},{lng}/data=!3m1!1e3"
    )

    prgph = document.add_paragraph()
    prgph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    prgph.add_run("Bing Maps: ").bold = True
    prgph.add_run(
        f"https://bing.com/maps/default.aspx?cp={lat}~{lng}&style=h&lvl=15"
    )

    prgph = document.add_paragraph()
    prgph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    prgph.add_run("OpenStreetMap: ").bold = True
    prgph.add_run(
        f"http://www.openstreetmap.org/?lat={lat}&lon={lng}&zoom=17&layers=M"
    )
    return document


def details_and_satellite(
    img_path: os.PathLike, img_fmt: str, document: Document
) -> Document:
    """Adds the detailed map and the satellite image map.

    :param img_path: The path to the image folder including group name.
    :type img_path: os.PathLike
    :return: The tex code.
    :rtype: str
    """
    global FIGURENUM
    prgph = document.add_paragraph()
    prgph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = prgph.add_run()
    run.add_picture(img_path + f"_map.{img_fmt}", width=docx.shared.Mm(70))
    run.add_picture(img_path + f"_satimg.{img_fmt}", width=docx.shared.Mm(70))

    prgph = document.add_paragraph()
    prgph.add_run(f"Abbildung {FIGURENUM}: ").bold = True
    FIGURENUM += 1
    prgph.add_run("Lokalität der Anomalie. ")
    prgph.add_run("Links: ").italic = True
    prgph.add_run(
        "Übersicht über das Gebiet der Gruppe inklusive verschiedener "
        + "Geogefahren und der detektierten Anomalien "
        + "(Kartengrundlage OpenStreetMap). "
    )
    prgph.add_run("Rechts: ").italic = True
    prgph.add_run("Luftbild basierend auf ESRI Imagery.")

    return document


def hlnug_description(hld: gp.GeoDataFrame, document: Document) -> Document:
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

    document.add_heading("Beschreibung", level=1)
    prgph = document.add_paragraph()
    prgph.add_run(f"Es handelt sich hierbei um eine {hld.OBJEKT.values[0]} ")
    if hld.HERKUNFT.values[0]:
        prgph.add_run(f"welche durch {hld.HERKUNFT.values[0]} ")
        if hld.KARTIERT.values[0]:
            prgph.add_run(f"{kart_str(hld.KARTIERT.values[0])} ")
        prgph.add_run("kartiert wurde")
    prgph.add_run(". ")
    if hld.KLASSI_DGM.values[0]:
        prgph.add_run(f"Der Befund im DGM ist {hld.KLASSI_DGM.values[0]}. ")
    if hld.RU_SCHICHT.values[0]:
        prgph.add_run(
            f"Die betroffenen Einheiten sind {hld.RU_SCHICHT.values[0]} "
        )
        if hld.RU_SCHIC_2.values[0]:
            prgph.add_run(f"und {hld.RU_SCHIC_2.values[0]} ")
        if hld.GEOLOGIE.values[0]:
            prgph.add_run(f"auf {hld.GEOLOGIE.values[0]} ")
            if hld.STR_SYSTEM.values[0]:
                prgph.add_run(f"({hld.STR_SYSTEM.values[0]})")
        prgph.add_run(". ")
    else:
        if hld.GEOLOGIE.values[0]:
            prgph.add_run(
                f"Die Geologie besteht aus {hld.GEOLOGIE.values[0]} "
            )
            if hld.STR_SYSTEM.values[0]:
                prgph.add_run(f"({hld.STR_SYSTEM.values[0]})")
        prgph.add_run(". ")

    if hld.FLAECHE_M2.values[0]:
        prgph.add_run(
            f"Die betroffene Fläche beträgt {np.round(hld.FLAECHE_M2.values[0], -2)} m². "
        )

    if hld.LAENGE_M.values[0] and hld.BREITE_M.values[0]:
        prgph.add_run(
            f"Sie ist {hld.LAENGE_M.values[0]}\u00a0m lang und {hld.BREITE_M.values[0]}\u00a0m breit"
        )
        if (
            hld.H_MAX_MNN.values[0]
            and hld.H_MIN_MNN.values[0]
            and hld.H_DIFF_M.values[0]
        ):
            prgph.add_run(
                f" und erstreckt sich von {hld.H_MAX_MNN.values[0]}\u00a0m NN bis {hld.H_MIN_MNN.values[0]}\u00a0m NN über {hld.H_DIFF_M.values[0]}\u00a0m Höhendifferenz"
            )
        prgph.add_run(". ")

    if hld.EXPOSITION.values[0]:
        if hld.EXPOSITION.values[0] != "n.b.":
            prgph.add_run(
                f"Das Gelände fällt nach {u4human.direction_to_text(hld.EXPOSITION.values[0], in_lang='de')} ein. "
            )

    if hld.LANDNUTZUN.values[0]:
        prgph.add_run(
            f"Im wesentlichen ist das Gebiet von {hld.LANDNUTZUN.values[0]} bedeckt. "
        )

    if hld.URSACHE.values[0]:
        prgph.add_run(f"Eine mögliche Ursache ist {hld.URSACHE.values[0]}. ")

    if hld.SCHUTZ_OBJ.values[0]:
        if hld.SCHUTZ_OBJ.values[0] == "nicht bekannt":
            prgph.add_run("Eine potentielle Gefährdung ist nicht bekannt. ")
        else:
            prgph.add_run(
                f"Eine potentielle Gefährdung für {hld.SCHUTZ_OBJ.values[0]} könnte vorliegen. "
            )

    if hld.AKTIVITAET.values[0]:
        if hld.AKTIVITAET.values[0] == "nicht bekannt":
            prgph.add_run("Eine mögliche Aktivität ist nicht bekannt. ")
        if hld.AKTIVITAET.values[0] == "aktiv":
            prgph.add_run(f"Die {hld.OBJEKT.values[0]} ist aktiv. ")

    if hld.MASSNAHME.values[0]:
        if hld.MASSNAHME.values[0] == "nicht bekannt":
            prgph.add_run("Über unternommene Maßnahmen ist nichts bekannt. ")
        else:
            prgph.add_run("")

    if hld.BEMERKUNG.values[0]:
        prgph = document.add_paragraph()
        prgph.add_run(
            "Kommentar: " + hld.BEMERKUNG.values[0] + "."
        ).italic = True

    return document


def shape(series: gp.GeoSeries, document: Document) -> Document:
    """Adds shape information to the document

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The tex code.
    :rtype: str
    """
    humanize.activate("de")
    prgph = document.add_paragraph()
    prgph.add_run("Größe und Form: ").bold = True

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

            prgph.add_run(
                f"Es handelt sich um {humanize.apnumber(len(long_ax))} "
                + f"Anomalien. Die Anomalien sind {lax}\u00a0m lang und {sax}"
                + "\u00a0m breit. "
            )
            if len(long_ax) > 2:
                prgph.add_run(
                    "Die flächenmäßig kleinste Anomalie ist hierbei "
                    + f"{round(long_ax[imin])}\u00a0m lang und "
                    + f"{round(short_ax[imin])}\u00a0m breit, die größte "
                    + f"{round(long_ax[imax])}\u00a0m lang und "
                    + f"{round(short_ax[imax])}\u00a0m breit. "
                )
    if isinstance(long_ax, float):
        lax = f"{round(long_ax)}"
        sax = f"{round(short_ax)}"
        prgph.add_run(f"Die Anomalie ist {lax} m lang und {sax} m breit. ")

    return document


def landuse(series: gp.GeoSeries, document: Document) -> Document:
    """Converts the landuse into a descriptive text.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The tex code.
    :rtype: str
    """
    prgph = document.add_paragraph()
    prgph.add_run("Landnutzung: ").bold = True
    landuse = ""
    try:
        landuse = eval(series.landuse_names)
    except NameError:
        landuse = series.landuse_names
    landuse_perc = eval(series.landuse_percent)
    if landuse:
        prgph.add_run(
            "Der überwiegende Teil wird durch "
            + f"{u4human.landuse_str(series.landuse_major)} bedeckt. "
            + "Die Anteile der Landnutzung sind: "
        )
        if isinstance(landuse, list):
            lnd_str = ""
            for ii in range(1, len(landuse) + 1):
                lnd_str += f" {landuse_perc[-ii]:.1f}% {u4human.landuse_str(landuse[-ii])}, "

            lnd_str = lnd_str[:-2] + "."
            prgph.add_run(lnd_str)
        else:
            prgph.add_run(
                f"{landuse_perc:.1f}% {u4human.landuse_str(landuse)}"
            )
    return document


def moved_volumes(series: gp.GeoSeries, document: Document) -> Document:
    """Adds description of moved volumes to the document.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The tex code.
    :rtype: str
    """
    document.add_heading("Höhenveränderungen", level=1)
    document.add_paragraph(
        "Im Gebiet um die detektierte Anomalie wurde insgesamt "
        + f"{series.volumes_moved}\u00a0m³ Material bewegt, "
        + f"wovon {series.volumes_added}\u00a0m³ hinzugefügt und "
        + f"{abs(series.volumes_removed)}\u00a0m³ abgetragen wurde. "
        + f"Dies ergibt eine Gesamtbilanz von {series.volumes_total}\u00a0m³,"
        + f" in Summe wurde also {u4human.vol_str(series.volumes_total)}."
    )
    return document


def difference(
    img_path: os.PathLike, img_fmt: str, document: Document
) -> Document:
    """Adds the difference and slope maps.

    :param img_path: The path to the image folder including group name.
    :type img_path: os.PathLike
    :return: The tex code.
    :rtype: str
    """
    global FIGURENUM
    if os.path.exists(img_path + f"_diffplan.{img_fmt}") and os.path.exists(
        img_path + f"_dem.{img_fmt}"
    ):
        prgph = document.add_paragraph()
        prgph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = prgph.add_run()
        run.add_picture(
            img_path + f"_diffplan.{img_fmt}", width=docx.shared.Mm(70)
        )
        run.add_picture(img_path + f"_dem.{img_fmt}", width=docx.shared.Mm(70))
    prgph = document.add_paragraph()
    prgph.add_run(f"Abbildung {FIGURENUM}: ").bold = True
    FIGURENUM += 1
    prgph.add_run("Höhendaten im Bereich der Anomalie. ")
    prgph.add_run("Links: ").italic = True
    prgph.add_run("Differenzenplan im Gebiet.")
    prgph.add_run("Rechts: ").italic = True
    prgph.add_run("Digitales Höhenmodell (Schummerung).")
    return document


def topography(
    series: gp.GeoSeries,
    img_path: os.PathLike,
    img_fmt: str,
    document: Document,
) -> Document:
    """Converts the slope into a descriptive text.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The tex code.
    :rtype: str
    """
    global FIGURENUM
    document.add_heading("Topographie", level=1)

    document.add_paragraph(
        "In den folgenden Jahren war die Topographie im Bereich der "
        + "Anomalien wie folgt:"
    )
    for yy in ["14", "19", "21"]:
        year = f"20{yy}"
        prgph = document.add_paragraph()
        prgph.add_run(f"{year}: ").bold = True

        if not series[f"slope_polygons_mean_{yy}"] == "[]":
            # Values for the individual polygons (can be empty)
            sl_m = eval(series[f"slope_polygons_mean_{yy}"])
            sl_s = eval(series[f"slope_polygons_std_{yy}"])
            as_m = eval(series[f"aspect_polygons_mean_{yy}"])
            as_s = eval(series[f"aspect_polygons_std_{yy}"])
            if isinstance(sl_m, list) or isinstance(sl_m, float):
                usl, usl_str = u4human.topo_text(sl_m, sl_s, "%", is_tex=False)
                prgph.add_run(
                    f"Im Bereich der Anomalie {u4human.slope_std_str(usl.s)} "
                    + f"und {u4human.slope_str(usl.n)} ({usl_str}). "
                )
                if as_m:
                    uas, uas_str = u4human.topo_text(
                        as_m, as_s, "°", is_tex=False
                    )
                    prgph.add_run(
                        "Die Anomalien fallen nach "
                        + f"{u4human.direction_to_text(uas.n)} ({uas_str}) ein. "
                    )
            else:
                prgph.add_run(
                    "Es liegen für den inneren Bereich der Anomalie keine Daten vor (außerhalb DEM). "
                )

            # Values for the hull around all anomalies
            usl, usl_str = u4human.topo_text(
                eval(series[f"slope_hull_mean_{yy}"]),
                eval(series[f"slope_hull_std_{yy}"]),
                "%",
                is_tex=False,
            )
            if isinstance(usl, unc.UFloat):
                prgph.add_run(
                    f"Im näheren Umfeld ist das Gelände {u4human.slope_std_str(usl.s)}"
                    + f" und {u4human.slope_str(usl.n)} ({usl_str}). "
                )
                uas, uas_str = u4human.topo_text(
                    eval(series[f"aspect_hull_mean_{yy}"]),
                    eval(series[f"aspect_hull_std_{yy}"]),
                    "°",
                    is_tex=False,
                )
                if isinstance(uas, unc.UFloat):
                    prgph.add_run(
                        "Der Bereich fällt im Mittel nach "
                        + f"{u4human.direction_to_text(uas.n)} ({uas_str}) ein. "
                    )
            else:
                prgph.add_run(
                    "Es liegen für das nähere Umfeld der Anomalien keine Werte für die Steigung vor. "
                )

    if os.path.exists(img_path + f"_slope.{img_fmt}") and os.path.exists(
        img_path + f"_aspect.{img_fmt}"
    ):
        prgph = document.add_paragraph()
        prgph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = prgph.add_run()
        run.add_picture(
            img_path + f"_slope.{img_fmt}", width=docx.shared.Mm(70)
        )
        run.add_picture(
            img_path + f"_aspect.{img_fmt}", width=docx.shared.Mm(70)
        )

        prgph = document.add_paragraph()
        prgph.add_run(f"Abbildung {FIGURENUM}: ").bold = True
        FIGURENUM += 1
        prgph.add_run("Topographie im Gebiet. ")
        prgph.add_run("Links: ").italic = True
        prgph.add_run("Steigung. ")
        prgph.add_run("Rechts: ").italic = True
        prgph.add_run("Exposition.")

    if os.path.exists(img_path + f"_aspect_slope.{img_fmt}"):
        prgph = document.add_paragraph()
        prgph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = prgph.add_run()
        run.add_picture(
            img_path + f"_aspect_slope.{img_fmt}",
            width=docx.shared.Mm(150),
        )
        prgph = document.add_paragraph()
        prgph.add_run(f"Abbildung {FIGURENUM}: ").bold = True
        FIGURENUM += 1
        prgph.add_run(
            "Gemischte Darstellung von Steigung (Sättigung) und Exposition (Farbe)."
        )

    return document


def psi_map(
    img_path: os.PathLike, img_fmt: str, document: Document
) -> Document:
    """Adds the psi map with timeseries.

    :param img_path: The path to the image folder including group name.
    :type img_path: os.PathLike
    :return: The tex code.
    :rtype: str
    """
    global FIGURENUM
    document.add_heading("InSAR Daten", level=1)
    prgph = document.add_paragraph()
    prgph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = prgph.add_run()
    run.add_picture(img_path + f"_psi.{img_fmt}", width=docx.shared.Mm(150))
    prgph.add_run(f"Abbildung {FIGURENUM}: ").bold = True
    FIGURENUM += 1
    prgph.add_run(
        "Persistent scatterer und Zeitreihe der Deformation "
        + "im Gebiet der Gruppe."
    )
    print(img_path)
    return document


def geohazard(series: gp.GeoSeries, document: Document) -> Document:
    """Converts the known geohazards into a descriptive text.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The tex code.
    :rtype: str
    """
    global TABLENUM
    document.add_heading("Geogefahren", level=1)
    prgph = document.add_paragraph()
    prgph.add_run(f"Tabelle {TABLENUM}: ").bold = True
    prgph.add_run("Bekannte Geogefahren im Bereich der Rutschung.")
    table = document.add_table(rows=4, cols=3, style="Light Shading Accent 1")
    table.cell(0, 0).text = "Typ"
    table.cell(0, 1).text = "Innerhalb des Areals"
    table.cell(0, 2).text = "Im Umkreis von 1 km"
    table.cell(1, 0).text = "Hangrutschungen"
    table.cell(1, 1).text = f"{series.landslides_num_inside}"
    table.cell(1, 2).text = f"{series.landslides_num_1km}"
    table.cell(2, 0).text = "Karsterscheinungen"
    table.cell(2, 1).text = f"{series.karst_num_inside}"
    table.cell(2, 2).text = f"{series.karst_num_1km}"
    table.cell(3, 0).text = "Steinschläge"
    table.cell(3, 1).text = f"{series.rockfall_num_inside}"
    table.cell(3, 2).text = f"{series.rockfall_num_1km}"

    document = landslide_risk(series, document)
    document = karst_risk(series, document)
    document = subsidence_risk(series, document)
    return document


def landslide_risk(series: gp.GeoSeries, document: Document) -> Document:
    """Converts the known landslides into a descriptive text.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The tex code.
    :rtype: str
    """
    lsar = series.landslide_total
    document.add_heading("Rutschungsgefährdung", level=2)
    prgph = document.add_paragraph()
    if lsar > 0:
        prgph.add_run(
            f"Das Gebiet liegt {u4human.part_str(lsar)} ({lsar:.0f}%) in "
            + "einem gefährdeten Bereich mit rutschungsanfälligen Schichten. "
        )
        try:
            landslide_units = eval(series.landslide_units)
        except (NameError, SyntaxError):
            landslide_units = series.landslide_units
        if isinstance(landslide_units, list):
            text = "Die Einheiten sind: "
            for unit in landslide_units:
                text += f"{unit}, "
            prgph.add_run(text[:-2] + ".")
        else:
            prgph.add_run(f"Wichtigste Einheiten sind {landslide_units}.")
    else:
        prgph.add_run(
            "Es liegen keine Informationen zur Rutschungsgefährdung vor."
        )
    return document


def karst_risk(series: gp.GeoSeries, document: Document) -> Document:
    """Converts the known karst phenomena into a descriptive text.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The tex code.
    :rtype: str
    """
    ksar = series.karst_total
    document.add_heading("Karstgefährdung", level=2)
    prgph = document.add_paragraph()
    if ksar > 0:
        prgph.add_run(
            f"Das Gebiet liegt {u4human.part_str(ksar)} ({ksar:.0f}\%) in "
            + "einem Bereich bekannter verkarsteter Schichten. "
        )
        try:
            karst_units = eval(series.karst_units)
        except (NameError, SyntaxError):
            karst_units = series.karst_units
        if isinstance(karst_units, list):
            text = "Die Einheiten sind: "
            for unit in karst_units:
                text += f"{unit}, "
            prgph.add_run(text[:-2] + ".")
        else:
            prgph.add_run(f"Wichtigste Einheiten sind {karst_units}.")
    else:
        prgph.add_run("Es liegen keine Informationen zur Karstgefährdung vor.")
    return document


def subsidence_risk(series: gp.GeoSeries, document: Document) -> Document:
    """Converts the area of known subsidence into a descriptive text.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The tex code.
    :rtype: str
    """
    subsar = series.subsidence_total
    document.add_heading("Setzungsgefährdung", level=2)
    prgph = document.add_paragraph()
    if subsar > 0:
        prgph.add_run(
            f"Das Gebiet liegt {u4human.part_str(subsar)} ({subsar:.0f}\%) in "
            + "einem Bereich bekannter setzungsgefährdeter Schichten. "
        )
        try:
            subsidence_units = eval(series.subsidence_units)
        except (NameError, SyntaxError):
            subsidence_units = series.subsidence_units
        if isinstance(subsidence_units, list):
            text = "Die Einheiten sind: "
            for unit in subsidence_units:
                text += f"{unit}, "
            prgph.add_run(text[:-2] + ".")
        else:
            prgph.add_run(f"Wichtigste Einheiten sind {subsidence_units}.")
    else:
        prgph.add_run(
            "Es liegen keine Informationen zur Setzungsgefährdung vor."
        )
    return document


def geology(
    img_path: os.PathLike, img_fmt: str, document: Document
) -> Document:
    """Adds information for geology to the docum

    :param img_path: The path to the geology image file.
    :type img_path: os.PathLike
    :param img_fmt: The image file format.
    :type img_fmt: str
    :return: The tex code
    :rtype: str
    """
    global FIGURENUM

    document.add_heading("Geologie", level=1)

    prgph = document.add_paragraph()
    prgph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = prgph.add_run()
    run.add_picture(img_path + f"_GK25.{img_fmt}", width=docx.shared.Mm(150))

    prgph = document.add_paragraph()
    prgph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = prgph.add_run()
    run.add_picture(
        img_path + f"_GK25_leg.{img_fmt}", width=docx.shared.Mm(150)
    )

    prgph = document.add_paragraph()
    prgph.add_run(f"Abbildung {FIGURENUM}: ").bold = True
    FIGURENUM += 1
    prgph.add_run("Geologie im Gebiet basierend auf GK25 (Quelle: HLNUG).")

    return document


def hydrogeology(
    img_path: os.PathLike, img_fmt: str, document: Document
) -> Document:
    """Adds information for hydrogeology to the document

    :param img_path: The path to the hydrogeology image file.
    :type img_path: os.PathLike
    :param img_fmt: The image file format.
    :type img_fmt: str
    :return: The tex code
    :rtype: str
    """
    global FIGURENUM

    document.add_heading("Hydrogeologie", level=1)

    prgph = document.add_paragraph()
    prgph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = prgph.add_run()
    run.add_picture(
        img_path + f"_HUEK200.{img_fmt}", width=docx.shared.Mm(150)
    )

    prgph = document.add_paragraph()
    prgph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = prgph.add_run()
    run.add_picture(
        img_path + f"_HUEK200_leg.{img_fmt}", width=docx.shared.Mm(150)
    )

    prgph = document.add_paragraph()
    prgph.add_run(f"Abbildung {FIGURENUM}: ").bold = True
    FIGURENUM += 1
    prgph.add_run(
        "Hydrogeologische Einheiten im Gebiet basierend auf HÜK200 (Quelle: HLNUG)."
    )

    return document


def soils(img_path: os.PathLike, img_fmt: str, document: Document) -> Document:
    """Adds information for soils to the documen

    :param img_path: The path to the soils image file.
    :type img_path: os.PathLike
    :param img_fmt: The image file format.
    :type img_fmt: str
    :return: The tex code
    :rtype: str
    """
    global FIGURENUM

    document.add_heading("Bodengruppen", level=1)

    prgph = document.add_paragraph()
    prgph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = prgph.add_run()
    run.add_picture(img_path + f"_BFD50.{img_fmt}", width=docx.shared.Mm(150))

    prgph = document.add_paragraph()
    prgph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = prgph.add_run()
    run.add_picture(
        img_path + f"_BFD50_leg.{img_fmt}", width=docx.shared.Mm(150)
    )

    prgph = document.add_paragraph()
    prgph.add_run(f"Abbildung {FIGURENUM}: ").bold = True
    FIGURENUM += 1
    prgph.add_run(
        "Bodenhauptgruppen im Gebiet basierend auf der BFD50 (Quelle: HLNUG)."
    )

    return document
