import matplotlib.pyplot as plt
from u4py.analysis import inversion as u4invert


def main():
    ts_inversion()


def ts_inversion(file_path="", do_plot=True, out_path=""):
    """Simultaneous inversion of multiple components of surface motion

    Args:
        station (string, optional): Station name
        do_plot (bool, optional): Create plots. Defaults to True.
        out_path (string, optional): Path where plots are saved
    """
    if not file_path:
        time_series, t_EQ = u4invert.create_synthetic_data()

    matrix_ori, data, time_vector_ori = u4invert.invert_time_series(
        time_series, t_EQ=t_EQ
    )
    # ind = remove_outliers(data["ori_dhat_data"])
    # _, data, time_vector = invert_time_series(time_series, ind=ind)

    fig, ax = plt.subplots()
    ax.plot(time_vector_ori[0], data["dataE"])
    ax.plot(time_vector_ori[0], data["ori_dhat_data"]["dhatE"])
    plt.show()


if __name__ == "__main__":
    main()
