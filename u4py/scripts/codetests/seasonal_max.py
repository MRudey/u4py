"""Tests the superposition of two sine functions"""

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm

import u4py.analysis.other as u4other


def main():
    max_of_added_sines()
    add_two_sines()
    random_add_two_sines(0.0, -0.03)
    # u4other.superpose()


def max_of_added_sines():
    max_k = 1
    res_k = 1000
    x = np.linspace(0, 2 * np.pi, 1000)
    k_sin = np.linspace(-max_k, max_k, res_k)
    k_cos = np.linspace(-max_k, max_k, res_k)
    max_vals = np.zeros((res_k, res_k))
    for ii, ks in tqdm(enumerate(k_sin), total=res_k):
        for jj, kc in enumerate(k_cos):
            max_vals[ii, jj] = np.max(ks * np.sin(x) + kc * np.cos(x))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.imshow(max_vals, extent=(-max_k, max_k, -max_k, max_k))
    # ax.plot(max_vals[:, 250])
    XX, YY = np.meshgrid(k_sin, k_cos)
    ax.plot_surface(XX, YY, max_vals)
    plt.show()


def random_add_two_sines(k_sin=None, k_cos=None):
    x = np.linspace(0, 4 * np.pi, 100)
    if not np.isfinite(k_sin):
        k_sin = np.random.normal(loc=0, scale=0.5)
    if not np.isfinite(k_cos):
        k_cos = np.random.normal(loc=0, scale=0.5)
    fig, ax = plt.subplots()
    ax.plot(x, k_sin * np.sin(x))
    ax.plot(x, k_cos * np.cos(x))
    ax.plot(x, k_sin * np.sin(x) + k_cos * np.cos(x))
    a, phi_0 = u4other.superpose(k_sin, -k_cos)
    ax.axhline(a, linestyle=":", color="C2")
    ax.axvline(phi_0, linestyle=":", color="C2")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(np.pi / 4))
    ax.annotate(
        f"sin: {k_sin:.2f}, cos: {k_cos:.2f}",
        (0.05, 0.05),
        xycoords="axes fraction",
    )
    peak_time = phi_0 / (2 * np.pi) * 365
    peak_date = datetime(2015, 1, 1, 0, 0, 0) + timedelta(days=peak_time)
    print(f"annual peak time: {peak_date.day:02}.{peak_date.month:02}.")
    plt.show()


def add_two_sines():
    x = np.linspace(0, 2 * np.pi, 100)
    k_s = np.random.rand()
    k_c = np.random.rand()
    k_sin = [-k_s, k_s]
    k_cos = [-k_c, k_c]
    fig, axes = plt.subplots(nrows=2, ncols=2)
    for ii, ks in enumerate(k_sin):
        for jj, kc in enumerate(k_cos):
            a, phi = u4other.superpose(ks, -kc)
            axes[ii][jj].plot(x, ks * np.sin(x) + kc * np.cos(x))
            axes[ii][jj].axhline(a, linestyle=":")
            axes[ii][jj].axvline(phi, linestyle=":")
            axes[ii][jj].annotate(
                f"ks: {ks:.2f}, kc: {kc:.2f}",
                (0.05, 0.05),
                xycoords="axes fraction",
            )

    # for ax in axes.flat:
    #     ax.xaxis.set_major_locator(ticker.MultipleLocator(np.pi / 2))
    plt.show()


if __name__ == "__main__":
    main()
