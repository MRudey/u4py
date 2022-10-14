""" Converts a shape file containing points with timeseries into hdf5 """


import u4py.utils.convert as u4convert
import u4py.utils.files as u4files
from tqdm import tqdm


def main():
    file_list = u4files.get_file_paths(filetypes=(("*.dbf", "*.dbf"),))
    for fp in tqdm(file_list):
        u4convert.convert_file(fp)


if __name__ == "__main__":
    main()
