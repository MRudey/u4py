"""
Functions for working with gas storage data by MND Energies
"""

from datetime import datetime


def _date2num_gas(y):
    "06. 12. 2020"
    try:
        date = datetime.strptime(y, "%d. %m. %Y")
    except ValueError:
        date = datetime.strptime(y, "%d.%m.%Y")
    return date


def load_gas_data(file_path):
    with open(file_path, "rt") as gasfile:
        first_row = gasfile.readline()
        date = []
        level = []
        for row in gasfile.readlines():
            row_text = row.split("\t")
            date.append(_date2num_gas(row_text[0]))
            level.append(float(row_text[1].replace("\n", "").replace(" ", "")))
    return date, level
