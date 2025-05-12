"""
Functions for creating a Word-Based report of the anomalies, taylored to the
requirements of the HLNUG.
"""

import os

import numpy as np
import docx
import docx.parts
import docx.shared
import geopandas as gp
import humanize
import uncertainties as unc
from docx.document import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH

import u4py.io.human_text as u4human
from u4py.utils.types import U4ResDict

FIGURENUM = 1
TABLENUM = 1
humanize.activate("de")


def site_report(
    row: U4ResDict,
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
    site_path = os.path.split(os.path.split(output_path)[0])[0]
    web_query_path = os.path.join(
        site_path,
        "Places",
        "Classifier_shapes",
        "Web_Queries_" + suffix.split("_")[-1],
    )

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
    document.add_heading(f"Beschreibung für Amt-Nr. {group}", level=0)

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
        hlnug_data[hlnug_data.AMT_NR_ == group],
        row[1],
        document,
        web_query_path,
    )
    # document = landuse(row[1], document)

    # Volumina
    document = moved_volumes(row[1], document)

    # Difference maps
    if os.path.exists(img_path + f"_dem.{img_fmt}"):
        document = dem(img_path, img_fmt, document)

    # Topographie
    if os.path.exists(img_path + f"_slope.{img_fmt}") or os.path.exists(
        img_path + f"_aspect_slope.{img_fmt}"
    ):
        document = topography(row[1], img_path, img_fmt, document)

    # PSI Data
    if os.path.exists(img_path + f"_psi.{img_fmt}"):
        document = psi_map(img_path, img_fmt, document)

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

    document.add_heading("Lokalität:", level=1)

    prgph = document.add_paragraph()
    prgph.add_run("Adresse: ").bold = True
    prgph.add_run(f"{series.locations}".replace("\n", ""))

    prgph = document.add_paragraph()
    prgph.add_run("Koordinaten (UTM 32N): ").bold = True
    prgph.add_run(f"{int(series.geometry.centroid.y)} N ")
    prgph.add_run(f"{int(series.geometry.centroid.x)} E")
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
    prgph.add_run("Lokalität der Rutschung. ")
    prgph.add_run("Links: ").italic = True
    prgph.add_run(
        "Übersicht über das Gebiet der Rutschung inklusive weiterer Rutschungen (Basiskarte OpenStreetMap). "
    )
    prgph.add_run("Rechts: ").italic = True
    prgph.add_run("Luftbild basierend auf ESRI Imagery.")

    return document


def hlnug_description(
    hld: gp.GeoDataFrame,
    row: U4ResDict,
    document: Document,
    web_query_path: os.PathLike,
) -> Document:
    """Adds a description based on HLNUG data

    :param hld: The dataset
    :type hld: gp.GeoDataFrame
    :return: The description
    :rtype: str
    """
    document.add_heading("Beschreibung", level=1)
    prgph = document.add_paragraph()

    # Add geological structural area
    structure_string = u4human.listed_strings(row["structural_region"])
    if isinstance(structure_string, list):
        prgph.add_run(
            f"Die Rutschung liegt in den geologischen Strukturräumen {structure_string} "
        )
    else:
        prgph.add_run(
            f'Die Rutschung liegt im geologischen Strukturraum "{structure_string}" '
        )

    # Add number and name of geological map
    mapnum = eval(row["geology_mapnum"])
    if mapnum:
        if isinstance(mapnum, list):
            mapname = eval(row["geology_mapname"])
            map_list = [
                f"{mpnum} {mpname}"
                for mpnum, mpname in zip(
                    mapnum, mapname
                )
            ]
            mapnum_string = u4human.listed_strings(map_list)
            prgph.add_run(f"auf den Kartenblättern {mapnum_string}. ")
        else:
            prgph.add_run(
                f"auf dem Kartenblatt {row['geology_mapnum']} {row['geology_mapname']}. "
            )
    else:
        prgph.add_run(". ")

    # Add dimensions
    prgph.add_run(
        f"Sie hat eine Länge von ca. {hld['LAENGE_M'].values[0]}\u00a0m, eine Breite von ca. {hld['BREITE_M'].values[0]}\u00a0m und verläuft nach {u4human.direction_to_text(hld['EXPOSITION'].values[0], in_lang='de')}. "
    )

    # Add sliding layers
    prgph.add_run(
        f"Die an der Rutschung beteiligten Schichten sind {hld['RU_SCHICHT'].values[0]}"
    )
    if hld["RU_SCHIC_2"].values[0]:
        prgph.add_run(f" und {hld['RU_SCHIC_2'].values[0]}")
    prgph.add_run(". ")

    # Add roads
    if (
        row["roads_has_motorway"]
        or row["roads_has_primary"]
        or row["roads_has_secondary"]
    ):
        road_list = []
        if row["roads_has_motorway"]:
            if row["roads_motorway_names"].startswith("["):
                motorway_names = eval(row["roads_motorway_names"])
            else:
                motorway_names = row["roads_motorway_names"]
            motorway_lengths = eval(row["roads_motorway_length"])
            if isinstance(motorway_lengths, list):
                for rname, rlen in zip(motorway_names, motorway_lengths):
                    road_list.append(
                        f"der {rname} auf einer Länge von {int(rlen)}\u00a0m"
                    )
            else:
                road_list.append(
                    f"der {motorway_names} auf einer Länge von {motorway_lengths:.1f}\u00a0m"
                )
        if row["roads_has_primary"]:
            if row["roads_primary_names"].startswith("["):
                primary_names = eval(row["roads_primary_names"])
            else:
                primary_names = row["roads_primary_names"]
            primary_lengths = eval(row["roads_primary_length"])
            if isinstance(primary_lengths, list):
                for rname, rlen in zip(primary_names, primary_lengths):
                    road_list.append(
                        f"der {rname} auf einer Länge von {int(rlen)}\u00a0m"
                    )
            else:
                road_list.append(
                    f"der {primary_names} auf einer Länge von {primary_lengths:.1f}\u00a0m"
                )
        if row["roads_has_secondary"]:
            if row["roads_secondary_names"].startswith("["):
                secondary_names = eval(row["roads_secondary_names"])
            else:
                secondary_names = row["roads_secondary_names"]
            secondary_lengths = eval(row["roads_secondary_length"])
            if isinstance(secondary_lengths, list):
                for rname, rlen in zip(secondary_names, secondary_lengths):
                    road_list.append(
                        f"der {rname} auf einer Länge von {int(rlen)}\u00a0m"
                    )
            else:
                road_list.append(
                    f"der {secondary_names} auf einer Länge von {secondary_lengths:.1f}\u00a0m"
                )
        prgph.add_run(
            f"Die Rutschung wird von {u4human.listed_strings(road_list)} durchkreuzt. "
        )

    else:
        if row["roads_nearest_motorway_name"]:
            prgph.add_run(
                f"In ca. {int(row['roads_nearest_motorway_dist'])}\u00a0m "
                + "Entfernung vom Mittelpunkt der Rutschung befindet sich die "
                + f"{row['roads_nearest_motorway_name']}. "
            )
        elif row["roads_nearest_primary_name"]:
            prgph.add_run(
                f"In ca. {int(row['roads_nearest_primary_dist'])}\u00a0m "
                + "Entfernung vom Mittelpunkt der Rutschung befindet sich die "
                + f"{row['roads_nearest_primary_name']}. "
            )
        elif row["roads_nearest_secondary_dist"]>0:
            prgph.add_run(
                f"In ca. {int(row['roads_nearest_secondary_dist'])}\u00a0m "
                + "Entfernung vom Mittelpunkt der Rutschung befindet sich "
            )
            if row["roads_nearest_secondary_name"]:
                prgph.add_run(f"die {row['roads_nearest_secondary_name']}. ")
            else:
                prgph.add_run("eine nicht benannte Landstraße. ")
        else:
            prgph.add_run("Im Umkreis von 1\u00a0km sind keine größeren Straßen vorhanden. ")

    if row["railways_has"]:
        prgph.add_run(
            "Eine Bahnlinie durchquert das Gebiet auf "
            +f"{int(row['railways_length'])}\u00a0m Länge. "
        )
    elif row["railways_close"]:
        prgph.add_run(
            f"Eine Bahnlinie liegt ca. {int(row['railways_length'])}\u00a0m " + "entfernt vom Mittelpunkt der Rutschung. "
        )

    # Add drill sites
    drill_path = os.path.join(
        web_query_path,
        f"Archivbohrungen, Endteufe [m]_bohr_{row['group']:05}.gpkg",
    )
    if os.path.exists(drill_path):
        num_wells = len(gp.read_file(drill_path))
        if num_wells > 1:
            prgph.add_run(
                f"Im Umfeld der Rutschung sind {humanize.apnumber(num_wells)} Bohrungen bekannt. "
            )
        elif num_wells == 1:
            prgph.add_run("Im Umfeld der Rutschung ist eine Bohrung bekannt. ")
    return document


def landuse(series: gp.GeoSeries, document: Document) -> Document:
    """Converts the landuse into a descriptive text.

    :param series: The GeoSeries object extracted from the row.
    :type series: gp.GeoSeries
    :return: The tex code.
    :rtype: str
    """
    landuse = ""
    try:
        landuse = eval(series.landuse_names)
    except NameError:
        landuse = series.landuse_names
    landuse_perc = eval(series.landuse_percent)
    if landuse:
        prgph = document.add_paragraph()
        prgph.add_run("Landnutzung: ").bold = True
        prgph.add_run(
            "Der überwiegende Teil wird durch "
            + f"{u4human.landuse_str(series.landuse_major)} bedeckt. "
            + "Die Anteile der Landnutzung nach OpenStreetMap sind: "
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
    if np.all(
        np.abs(
            [
                series.volumes_moved,
                series.volumes_added,
                series.volumes_removed,
                series.volumes_total,
            ]
        )
        <= 200
    ):
        document.add_paragraph(
            "Im Gebiet um die detektierte Anomalie wurde laut vorliegendem "
            + "Differenzenplan (HVBG) keine Bewegung festgestellt."
        )
    else:
        document.add_paragraph(
            "Im Gebiet um die detektierte Anomalie wurde laut vorliegendem "
            + "Differenzenplan (HVBG) insgesamt "
            + f"{series.volumes_moved}\u00a0m³ Material bewegt, "
            + f"wovon {series.volumes_added}\u00a0m³ hinzugefügt und "
            + f"{abs(series.volumes_removed)}\u00a0m³ abgetragen wurde. "
            + f"Dies ergibt eine Gesamtbilanz von {series.volumes_total}"
            + f"\u00b1{series.volumes_error}\u00a0m³,"
            + f" in Summe wurde also {u4human.vol_str(series.volumes_total)}."
        )
    return document


def dem(img_path: os.PathLike, img_fmt: str, document: Document) -> Document:
    """Adds the dem maps.

    :param img_path: The path to the image folder including group name.
    :type img_path: os.PathLike
    :return: The tex code.
    :rtype: str
    """
    global FIGURENUM
    if os.path.exists(img_path + f"_dem.{img_fmt}"):
        prgph = document.add_paragraph()
        prgph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = prgph.add_run()
        run.add_picture(
            img_path + f"_dem.{img_fmt}", width=docx.shared.Mm(140)
        )
    prgph = document.add_paragraph()
    prgph.add_run(f"Abbildung {FIGURENUM}: ").bold = True
    FIGURENUM += 1
    prgph.add_run(
        "Digitales Höhenmodell DGM 1 (HVBG) und vorhandene Bohrungen."
    )
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
        "In den Jahren 2014, 2019 und 2021 war die Topographie anhand des DGM1"
        + " (HVBG) im Bereich der Rutschung wie folgt:"
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
                    f"Im Bereich der Rutschung {u4human.slope_std_str(usl.s)} "
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
                    "Es liegen für den inneren Bereich der Rutschung keine Daten vor (außerhalb DEM). "
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
                    "Es liegen für das nähere Umfeld der Rutschung keine Werte für die Steigung vor. "
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
        + "im Gebiet der Rutschung."
    )
    return document


def geology(
    img_path: os.PathLike, img_fmt: str, document: Document
) -> Document:
    """Adds information for geology to the document.

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
        img_path + f"_GK25_leg.{img_fmt}", width=docx.shared.Mm(100)
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
        img_path + f"_HUEK200_leg.{img_fmt}", width=docx.shared.Mm(100)
    )

    prgph = document.add_paragraph()
    prgph.add_run(f"Abbildung {FIGURENUM}: ").bold = True
    FIGURENUM += 1
    prgph.add_run(
        "Hydrogeologische Einheiten im Gebiet basierend auf HÜK200 (Quelle: HLNUG)."
    )

    return document


def soils(img_path: os.PathLike, img_fmt: str, document: Document) -> Document:
    """Adds information for soils to the document.

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
        img_path + f"_BFD50_leg.{img_fmt}", width=docx.shared.Mm(100)
    )

    prgph = document.add_paragraph()
    prgph.add_run(f"Abbildung {FIGURENUM}: ").bold = True
    FIGURENUM += 1
    prgph.add_run(
        "Bodenhauptgruppen im Gebiet basierend auf der BFD50 (Quelle: HLNUG)."
    )

    return document
