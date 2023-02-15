""" Converts a shape file containing points with timeseries into hdf5 """


from tqdm import tqdm

import u4py.utils.convert as u4convert
import u4py.utils.files as u4files


def main():
    file_list = u4files.get_file_list(filetype=".dbf")
    for fp in tqdm(file_list):
        u4convert.convert_shapefile(fp)


if __name__ == "__main__":
    main()
