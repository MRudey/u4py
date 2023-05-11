"""
Converts the CSV File by the HLNUG to a pickled dictionary
"""

import csv
import os
import pickle as pkl
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


def main():
    path = Path(
        "~/Documents/ArcGIS/ExternalData/GWStände_2015/GWStände_2015.CSV"
    ).expanduser()
    output_file, _ = os.path.splitext(path)
    with open(path, "rt") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=";")
        next(csv_reader, None)  # skip header

        stations = dict()
        first_run = True
        for row in tqdm(csv_reader):
            if first_run:
                station = new_station(row)
            elif station["gruwahID"] == int(row[1]):
                station["time"].append(datetime.strptime(row[5], "%d.%m.%Y"))
                station["height"].append(dec2float(row[6]))
            else:
                stations[station["name"]] = station
                station = new_station(row)
            first_run = False
    with open(output_file + ".pkl", "wb") as pkl_file:
        pkl.dump(stations, pkl_file)


def new_station(row: list) -> dict:
    """Creates a new station from the entries in the row

    Arguments:
        row -- row read from the csv file

    Returns:
        dictionary with single entry
    """
    try:
        shortID = int(row[0])
    except ValueError:
        shortID = 0
    station = {
        "shortID": shortID,
        "gruwahID": int(row[1]),
        "name": row[2],
        "easting": dec2float(row[3]),
        "northing": dec2float(row[4]),
        "time": [datetime.strptime(row[5], "%d.%m.%Y")],
        "height": [dec2float(row[6])],
    }
    return station


def dec2float(string: str) -> float:
    """Returns a float from a string containing a comma as decimal separator

    Arguments:
        string -- string to be converted

    Returns:
        float of the string
    """
    return float(string.replace(",", "."))


if __name__ == "__main__":
    main()
