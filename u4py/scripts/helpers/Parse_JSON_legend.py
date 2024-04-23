"""
Parses the legend json from an ArcGIS Rest Services Directory to a faster file
format
"""

import json
import pickle


def main():
    fpath = "/home/rudolf/Documents/umwelt4/Places/legend_HUEK200.json"
    with open(fpath, "rt") as json_file:
        legend_data = json.load(json_file)

    uuids = []
    labels = []
    styles = []
    facecolors = []
    edgecolors = []
    linewidths = []
    for val in legend_data["drawingInfo"]["renderer"]["uniqueValueInfos"]:
        uuid, label, style, facecolor, edgecolor, linewidth = (
            parse_legend_entry(val)
        )
        uuids.append(uuid)
        labels.append(label)
        styles.append(style)
        facecolors.append(facecolor)
        edgecolors.append(edgecolor)
        linewidths.append(linewidth)

    with open(fpath.replace(".json", ".pkl"), "wb") as pkl_file:
        pickle.dump(
            [uuids, labels, styles, facecolors, edgecolors, linewidths],
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
    label = univalinfo["label"]
    style = style_parser[univalinfo["symbol"]["style"]]
    facecolor = parse_color(univalinfo["symbol"]["color"])
    if univalinfo["symbol"]["outline"]:
        edgecolor = parse_color(univalinfo["symbol"]["outline"]["color"])
        linewidth = univalinfo["symbol"]["outline"]["width"]
    else:
        edgecolor = "None"
        linewidth = 0

    return (uuid, label, style, facecolor, edgecolor, linewidth)


def parse_color(color: list) -> tuple:
    """Converts a 0..255 based RGB into 0..1 based RGB

    :param color: The color as RGBA list in 0..255.
    :type color: list
    :return: The color as RGB in 0..1 color space.
    :rtype: tuple
    """
    return (color[0] / 255, color[1] / 255, color[2] / 255)


if __name__ == "__main__":
    main()
