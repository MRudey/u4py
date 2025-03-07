import matplotlib.pyplot as plt

import u4py.plotting.axes as u4ax


def main():
    file_path = (
        r"E:\Projekte\Umwelt4\PSI_Tiffs_EGMS_2015_2021_vertikal\2016-01-05.tif"
    )

    u4ax.add_tile(file_path)
    plt.show()


if __name__ == "__main__":
    main()
