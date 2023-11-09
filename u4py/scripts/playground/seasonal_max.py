from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm


def main():
    # max_of_added_sines()
    # random_add_two_sines()
    add_two_sines()


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


def random_add_two_sines():
    x = np.linspace(0, 2 * np.pi, 100)
    k_sin = np.random.normal(loc=0, scale=0.5)
    k_cos = np.random.normal(loc=0, scale=0.5)
    fig, ax = plt.subplots()
    ax.plot(x, k_sin * np.sin(x))
    ax.plot(x, k_cos * np.cos(x))
    ax.plot(x, k_sin * np.sin(x) + k_cos * np.cos(x))
    a, phi_0 = superpose(k_sin, -k_cos)
    ax.axhline(a, linestyle=":", color="C2")
    ax.axvline(phi_0, linestyle=":", color="C2")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(np.pi / 4))
    ax.annotate(
        f"sin: {k_sin:.2f}, cos: {k_cos:.2f}",
        (0.05, 0.05),
        xycoords="axes fraction",
    )
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
            a, phi = superpose(ks, -kc)
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


def superpose(
    a_1: float, a_2: float, phi_1: float = 0, phi_2: float = np.pi / 2
) -> Tuple[float, float]:
    """Superposes two sine functions and calculates their amplitude and phase.

    The superposition follows this principle:

    ..math::

        y_1 = a_1\,sin (x + \varphi_1) \wedge y_2 = a_2 \, sin (x + \varphi_2)
        y = y_1+y_2 = a sin\,(x+\varphi_0)
        a=\sqrt{a_1^2 + a_2^2+2a_1a_2\,cos(\varphi_2-\varphi_1)}
        \varphi_0=tan^{-1}\left(\frac{a_1\,sin \varphi_1 + a_2\,sin\varphi_2}{a_1\,cos \varphi_1 + a_2\,cos\varphi_2} \right)

    In the case of the inversion results, phase 2 is a cosine which means that :math:`a_2 = -a_2`. The phase change :math:`\varphi_0` is corrected by :math:`\frac{\pi}{2}` and depending on the polarity of :math:`a_1` and :math:`a_2` selected so that the return value is the first maximum in the interval :math:`\[0,2\pi\]`.

    :param a_1: Amplitude of phase 1
    :type a_1: float
    :param a_2: Amplitude of phase 2
    :type a_2: float
    :param phi_1: Phase change of phase 1, defaults to 0
    :type phi_1: float, optional
    :param phi_2: Phase change of phase 2, defaults to np.pi/2
    :type phi_2: float, optional
    :return: Maximum amplitude and phase change of superposed functions
    :rtype: Tuple[float, float]
    """
    a = np.sqrt(a_1**2 + a_2**2 + 2 * a_1 * a_2 * np.cos(phi_2 - phi_1))
    phi_0 = (
        np.arctan(
            (a_1 * np.sin(phi_1) + a_2 * np.sin(phi_2))
            / (a_1 * np.cos(phi_1) + a_2 * np.cos(phi_2))
        )
        + np.pi / 2
    )
    if a_1 < 0:
        phi_0 = phi_0 + np.pi
    return (a, phi_0)


if __name__ == "__main__":
    main()
