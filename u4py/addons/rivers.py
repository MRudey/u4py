"""
Functions for working with river level data
"""

import locale
from datetime import datetime


def _date2num_rhine(y):
    "Jan 1996"
    locale.setlocale(locale.LC_ALL, "de_DE")
    date = datetime.strptime(y, "%b %Y")
    return date


def load_rhine_date(file_path):
    with open(file_path, "rt") as rhinefile:
        first_row = rhinefile.readline()
        date = []
        level = []
        for row in rhinefile.readlines():
            row_text = row.split(",")
            date.append(_date2num_rhine(row_text[0]))
            level.append(float(row_text[3].replace("\n", "").replace(" ", "")))
    return date, level
