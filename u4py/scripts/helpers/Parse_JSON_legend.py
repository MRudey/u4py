"""
Parses the legend json from an ArcGIS Rest Services Directory to a faster file
format
"""

import json
import os
import pickle
import platform
import subprocess
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def main():
    # Making Paths
    fnames = [
        # "legend_BFD50",
        # "legend_GK25",
        # "legend_HUEK200",
        # "Geologie (Kartiereinheiten)"
        "Hydrogeologische Einheiten"
    ]
    for fname in tqdm(fnames, desc="Parsing legend files"):
        fpath = Path(f"~/Documents/umwelt4/Places/{fname}.json").expanduser()
        if os.path.exists(fpath):
            parse_json_legend(fpath)
        else:
            fpath = Path(
                f"~/Documents/umwelt4/Places/{fname}.lyrx"
            ).expanduser()
            parse_lyrx_legend(fpath)


def load_lyrx_data(fpath: os.PathLike | str):
    with open(fpath, "rt", encoding="utf8") as lyr_file:
        lyr_str = "data = " + lyr_file.read()
        lyr_str = lyr_str.replace("true", "True")
        lyr_str = lyr_str.replace("false", "False")
        scope = dict()
        exec(lyr_str, scope)
    return scope["data"]


def parse_lyrx_legend(fpath: os.PathLike | str):
    """
    Parses the symbology in fpath to a pickle file formatted for plotting
    with matplotlib.

    :param fpath: The file path of the LYRX file.
    :type fpath: os.PathLike | str
    """
    leg_folder = os.path.splitext(fpath)[0]
    os.makedirs(leg_folder, exist_ok=True)

    legend_data = load_lyrx_data(fpath)
    univals = legend_data["layerDefinitions"][0]["renderer"]["groups"][0][
        "classes"
    ]
    num_ent = len(univals)

    # Parsing
    uuids = []
    labels = []
    styles = []
    facecolors = []
    alphas = []
    edgecolors = []
    linewidths = []
    fills = []
    for val in tqdm(univals, desc="Parsing", total=num_ent, leave=False):
        uuid, lbl, sty, fc, fca, ec, eca, lw = parse_lyrx_entry(val)
        if sty:
            if isinstance(sty, str):
                sty *= 3
            elif isinstance(sty, dict):
                sty["hatch"] = [ht * 3 for ht in sty["hatch"]]
        if fc == ec:
            fill = False
        else:
            fill = True
        if not fca:
            fill = False
            fca = eca
        uuids.append(uuid)
        labels.append(lbl)
        styles.append(sty)
        fills.append(fill)
        facecolors.append(fc)
        edgecolors.append(ec)
        alphas.append(fca)
        linewidths.append(lw)

    # Sorting
    sorted = np.argsort(labels)

    # Plotting
    file_list = []
    ii = 100
    jj = 1
    fig, ax = plt.subplots(figsize=(10, 20))
    for srt in tqdm(sorted, desc="Plotting", leave=False):
        lbl = labels[srt]
        sty = styles[srt]
        fill = fills[srt]
        fc = facecolors[srt]
        ec = edgecolors[srt]
        fca = alphas[srt]
        lw = linewidths[srt]

        if isinstance(sty, str):
            ax.add_artist(
                mpatches.Rectangle(
                    (0, ii),
                    width=0.05,
                    height=0.75,
                    facecolor=fc,
                    edgecolor=ec,
                    alpha=fca,
                    fill=fill,
                    hatch=sty,
                    linewidth=lw,
                )
            )
        elif isinstance(sty, dict):
            if not sty["hatch"]:
                ax.add_artist(
                    mpatches.Rectangle(
                        (0, ii),
                        width=0.05,
                        height=0.75,
                        facecolor=fc,
                        edgecolor=ec,
                        alpha=fca,
                        fill=fill,
                        linewidth=lw,
                    )
                )
            else:
                ax.add_artist(
                    mpatches.Rectangle(
                        (0, ii),
                        width=0.05,
                        height=0.75,
                        facecolor=fc,
                        edgecolor=ec,
                        alpha=fca,
                        fill=fill,
                        linewidth=lw,
                    )
                )
                for hii in range(len(sty["hatch"])):
                    ht = sty["hatch"][hii]
                    hc = sty["hatch_color"][0]
                    hca = sty["hatch_alpha"][0]
                    ax.add_artist(
                        mpatches.Rectangle(
                            (0, ii),
                            width=0.05,
                            height=0.75,
                            facecolor="None",
                            edgecolor=hc,
                            alpha=hca,
                            hatch=ht,
                            linewidth=lw,
                        )
                    )
        ax.annotate(lbl, (0.06, ii + 0.75 / 2), verticalalignment="center")
        if ii == 0:
            # Exports after every 100 entries
            ax.set_ylim(-1, 101)
            ax.set_position([0, 0, 1, 1])
            plt.draw()
            ax.axis("off")
            out_path = os.path.join(leg_folder, f"{jj:02}.png")
            fig.savefig(out_path)
            out_path = os.path.join(leg_folder, f"{jj:02}.pdf")
            fig.savefig(out_path)
            file_list.append(f'"{out_path}"\n')
            jj += 1
            plt.cla()
            ii = 100
        ii -= 1

    # Final export
    ax.set_ylim(-1, 101)
    ax.set_position([0, 0, 1, 1])
    plt.draw()
    ax.axis("off")
    out_path = os.path.join(leg_folder, f"{jj:02}.png")
    fig.savefig(out_path)
    out_path = os.path.join(leg_folder, f"{jj:02}.pdf")
    fig.savefig(out_path)
    file_list.append(f'"{out_path}"\n')
    with open(
        os.path.join(leg_folder, "files.txt"),
        "wt",
        encoding="utf8",
        newline="\n",
    ) as ftxt:
        ftxt.writelines(file_list)

    gs_path = {"Linux": "gs", "Darwin": "gs", "Windows": "gswin64.exe"}
    try:
        subprocess.run(
            [
                gs_path[platform.system()],
                "-q",
                "-dNOPAUSE",
                "-dBATCH",
                "-sDEVICE=pdfwrite",
                "-sOutputFile=00_all_entries.pdf",
                "@files.txt",
            ],
            cwd=leg_folder,
        )
    except FileNotFoundError:
        print(
            "Ghostscript not found. Not able to generate merged legend file."
        )

    with open(leg_folder + ".pkl", "wb") as pkl_file:
        pickle.dump(
            [
                uuids,
                labels,
                styles,
                fills,
                facecolors,
                edgecolors,
                alphas,
                linewidths,
            ],
            pkl_file,
        )


def parse_json_legend(fpath: os.PathLike | str):
    """
    Parses the JSON legend in fpath to a pickle file formatted for plotting
    with matplotlib.

    :param fpath: The file path of the JSON file.
    :type fpath: os.PathLike | str
    """
    leg_folder = os.path.splitext(fpath)[0]
    os.makedirs(leg_folder, exist_ok=True)
    with open(fpath, "rt", encoding="utf8") as json_file:
        legend_data = json.load(json_file)

    univals = legend_data["drawingInfo"]["renderer"]["uniqueValueInfos"]
    num_ent = len(univals)

    # Parsing
    uuids = []
    labels = []
    styles = []
    facecolors = []
    alphas = []
    edgecolors = []
    linewidths = []
    fills = []
    for val in tqdm(univals, desc="Parsing", total=num_ent, leave=False):
        uuid, lbl, sty, fc, fca, ec, eca, lw = parse_legend_entry(val)
        if sty:
            sty *= 3
            if fc == ec:
                fill = False
        else:
            fill = True
        if not fca:
            fill = False
            fca = eca
        uuids.append(uuid)
        labels.append(lbl)
        styles.append(sty)
        fills.append(fill)
        facecolors.append(fc)
        edgecolors.append(ec)
        alphas.append(fca)
        linewidths.append(lw)

    # Sorting
    sorted = np.argsort(labels)

    # Plotting
    file_list = []
    ii = 100
    jj = 1
    fig, ax = plt.subplots(figsize=(10, 20))
    for srt in tqdm(sorted, desc="Plotting", leave=False):
        lbl = labels[srt]
        sty = styles[srt]
        fill = fills[srt]
        fc = facecolors[srt]
        ec = edgecolors[srt]
        fca = alphas[srt]
        lw = linewidths[srt]

        ax.add_artist(
            mpatches.Rectangle(
                (0, ii),
                width=0.05,
                height=0.75,
                facecolor=fc,
                edgecolor=ec,
                alpha=fca,
                fill=fill,
                hatch=sty,
                linewidth=lw,
            )
        )
        ax.annotate(lbl, (0.06, ii + 0.75 / 2), verticalalignment="center")
        if ii == 0:
            # Exports after every 100 entries
            ax.set_ylim(-1, 101)
            ax.set_position([0, 0, 1, 1])
            plt.draw()
            ax.axis("off")
            out_path = os.path.join(leg_folder, f"{jj:02}.pdf")
            fig.savefig(out_path)
            file_list.append(f'"{out_path}"\n')
            jj += 1
            plt.cla()
            ii = 100
        ii -= 1

    # Final export
    ax.set_ylim(-1, 101)
    ax.set_position([0, 0, 1, 1])
    plt.draw()
    ax.axis("off")
    out_path = os.path.join(leg_folder, f"{jj:02}.pdf")
    fig.savefig(out_path)
    file_list.append(f'"{out_path}"\n')
    with open(
        os.path.join(leg_folder, "files.txt"),
        "wt",
        encoding="utf8",
        newline="\n",
    ) as ftxt:
        ftxt.writelines(file_list)

    gs_path = {"Linux": "gs", "Darwin": "gs", "Windows": "gswin64.exe"}
    try:
        subprocess.run(
            [
                gs_path[platform.system()],
                "-q",
                "-dNOPAUSE",
                "-dBATCH",
                "-sDEVICE=pdfwrite",
                "-sOutputFile=00_all_entries.pdf",
                "@files.txt",
            ],
            cwd=leg_folder,
        )
    except FileNotFoundError:
        print(
            "Ghostscript not found. Not able to generate merged legend file."
        )

    with open(leg_folder + ".pkl", "wb") as pkl_file:
        pickle.dump(
            [
                uuids,
                labels,
                styles,
                fills,
                facecolors,
                edgecolors,
                alphas,
                linewidths,
            ],
            pkl_file,
        )

    # with open(fpath.replace(".json", ".pkl"), "rb") as pkl_file:
    #     data = pickle.load(pkl_file)
    #     print(data)


def parse_legend_entry(univalinfo: dict) -> tuple:
    """Parses the entry into the correct data values.

    :param univalinfo: The value extracted from the JSON
    :type univalinfo: dict
    :return: The dissected data.
    :rtype: tuple
    """
    style_parser = {
        "esriSFSBackwardDiagonal": "\\",
        "esriSFSForwardDiagonal": "/",
        "esriSFSCross": "+",
        "esriSFSDiagonalCross": "x",
        "esriSFSVertical": "|",
        "esriSFSHorizontal": "-",
        "esriSFSSolid": "",
    }
    try:
        uuid = int(univalinfo["value"])
    except ValueError:
        uuid = univalinfo["value"]
    lbl = univalinfo["label"]
    sty = style_parser[univalinfo["symbol"]["style"]]
    fc, fca = parse_color(univalinfo["symbol"]["color"])
    if univalinfo["symbol"]["outline"]:
        ec, eca = parse_color(univalinfo["symbol"]["outline"]["color"])
        lw = univalinfo["symbol"]["outline"]["width"]
    else:
        ec = "None"
        eca = 1
        lw = 0

    return (uuid, lbl, sty, fc, fca, ec, eca, lw)


def parse_lyrx_entry(univalinfo: dict) -> tuple:
    """Parses the entry into the correct data values.

    :param univalinfo: The value extracted from the JSON
    :type univalinfo: dict
    :return: The dissected data.
    :rtype: tuple
    """
    uuid = None
    lbl = None
    sty = None
    fc = None
    fca = None
    ec = None
    eca = None
    lw = None
    htsty = []
    hc = []
    hca = []

    rotation_parser = {
        -45: "\\",
        15: "/",
        25: "/",
        35: "/",
        45: "/",
        90: "|",
        115: "\\",
        120: "\\",
        135: "\\",
        155: "\\",
        165: "\\",
        180: "-",
        270: "|",
        315: "/",
    }
    try:
        uuid = int(univalinfo["values"][0]["fieldValues"][0])
    except ValueError:
        uuid = univalinfo["values"][0]["fieldValues"][0]
    lbl = univalinfo["label"]
    symlayers = univalinfo["symbol"]["symbol"]["symbolLayers"]
    for symbol in symlayers:
        if symbol["type"] == "CIMSolidFill":
            fc, fca = parse_color(symbol["color"]["values"])
        elif symbol["type"] == "CIMSolidStroke":
            ec, eca = parse_color(symbol["color"]["values"])
            lw = symbol["width"]
        elif symbol["type"] == "CIMVectorMarker":
            if "rings" in symbol["markerGraphics"][0]["geometry"].keys():
                hc_, hca_ = parse_color(
                    symbol["markerGraphics"][0]["symbol"]["symbolLayers"][0][
                        "color"
                    ]["values"]
                )
                hc.append(hc_)
                hca.append(hca_)
                htsty.append(".")
        elif symbol["type"] == "CIMHatchFill":
            if symbol["lineSymbol"]["type"] == "CIMLineSymbol":
                symlyr = symbol["lineSymbol"]["symbolLayers"][0]

                if symlyr["type"] == "CIMVectorMarker":
                    hc_, hca_ = parse_color(
                        symlyr["markerGraphics"][0]["symbol"]["symbolLayers"][
                            0
                        ]["color"]["values"]
                    )
                    hc.append(hc_)
                    hca.append(hca_)
                    if (
                        "rings"
                        in symlyr["markerGraphics"][0]["geometry"].keys()
                    ):
                        htsty.append(".")
                else:
                    hc_, hca_ = parse_color(symlyr["color"]["values"])
                    hc.append(hc_)
                    hca.append(hca_)
                if "rotation" in symbol.keys():
                    htsty.append(rotation_parser[symbol["rotation"]])
                elif "effects" in symlyr.keys():
                    eff = symlyr["effects"][0]["type"]
                    if eff == "CIMGeometricEffectDashes":
                        htsty.append("-")
                    else:
                        raise NotImplementedError(
                            f"Hatch effect {eff} not implemented."
                        )
        else:
            raise NotImplementedError(
                f"Symbol type {symbol['type']} not supported."
            )

    if len(htsty) > 1:
        if "\\" in htsty and "/" in htsty:
            htsty = ["x"]
            hc = [hc[0]]
            hca = [hca[0]]
    if uuid == 5621001:
        pass
    sty = {
        "hatch": htsty,
        "hatch_color": hc,
        "hatch_alpha": hca,
    }

    return (uuid, lbl, sty, fc, fca, ec, eca, lw)


def parse_color(color: list) -> tuple:
    """Converts a 0..255 based RGB into 0..1 based RGB

    :param color: The color as RGBA list in 0..255.
    :type color: list
    :return: The color as RGB in 0..1 color space.
    :rtype: tuple
    """
    return ((color[0] / 255, color[1] / 255, color[2] / 255), color[3] / 255)


if __name__ == "__main__":
    main()
