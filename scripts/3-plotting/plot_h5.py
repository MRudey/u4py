from turtle import ht

import h5py
from matplotlib import pyplot as plt


def main():
    """
    main function
    """
    h5path = r"E:\Projekte\Umwelt4\PythonExports\raster14\raster14_415999_5535999.h5"

    with h5py.File(h5path) as h5file:
        for k in h5file.keys():
            data = h5file['raster'][()]

    fig, ax = plt.subplots()
    ax.imshow(data)
    plt.show()


if __name__ == '__main__':
    main()
