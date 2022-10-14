""" Contains simple file and folder utilities for u4py """

import os
from datetime import datetime
from tkinter import Tk, filedialog

import h5py
import numpy as np


def get_file_paths(**kwargs):
    """Safe wrapper for filedialog by tkinter"""
    file_list = []
    try:
        root = Tk()
        root.withdraw()
        file_list = filedialog.askopenfilenames(**kwargs)
    finally:
        root.destroy()

    return file_list


def get_folder_paths(**kwargs):
    """Safe wrapper for filedialog by tkinter"""
    folder_path = ""
    try:
        root = Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory(**kwargs)
    finally:
        root.destroy()

    return folder_path


def get_file_list(filetype=".h5", **kwargs):
    """Asks for folder and returns all files of given filetype"""
    folder_path = get_folder_paths(**kwargs)
    file_list = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(filetype)
    ]
    return file_list


def load_hdf5(file_path):
    """Loads data from a hdf5 file. Converts timestamps to datetime."""
    data = dict()
    with h5py.File(file_path, "r") as h5file:
        for k in h5file.keys():
            if k == "time":
                v = np.array(
                    [datetime.fromisoformat(val.decode()) for val in h5file[k]]
                )
            else:
                try:
                    v = h5file[k][()]
                except ValueError:
                    v = h5file[k][()]
            data[k] = v

    return data
