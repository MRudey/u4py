"""
Parses the legend json from an ArcGIS Rest Services Directory to a faster file
format
"""

import json
import os
import pickle
import subprocess
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def main():
    # Making Paths
    fpath = Path("~\Documents\ArcGIS\Places\legend_GK25.json").expanduser()
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
            ax.set_ylim(-1, 101)
            ax.set_position([0, 0, 1, 1])
            plt.draw()
            ax.axis("off")
            fpath = os.path.join(leg_folder, f"{jj:02}.pdf")
            fig.savefig(fpath)
            file_list.append(f'"{fpath}"\n')
            jj += 1
            plt.cla()
            ii = 100
        ii -= 1
    with open(
        os.path.join(leg_folder, "files.txt"),
        "wt",
        encoding="utf8",
        newline="\n",
    ) as ftxt:
        ftxt.writelines(file_list)

    subprocess.run(
        [
            "gswin64.exe",
            "-q",
            "-dNOPAUSE",
            "-dBATCH",
            "-sDEVICE=pdfwrite",
            "-sOutputFile=00_all_entries.pdf",
            "@files.txt",
        ],
        cwd=leg_folder,
    )

    with open(fpath.replace(".json", ".pkl"), "wb") as pkl_file:
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
