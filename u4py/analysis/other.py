"""General functions for data analysis"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pycwt
import scipy.signal as spsignal


def cwt(y: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Does a continuous wavelet transformation of the input data with a Morlet.

    :param y: The input data set as a 1D numpy array.
    :type y: np.ndarray
    :param dt: The time difference between each sample in seconds.
    :type dt: float
    :return: A tuple containing the frequencies, cone of influence and power.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    logging.info("Detrend and normalize data for better cwt analysis")
    y_detrend = spsignal.detrend(y)
    std = np.std(y_detrend)  # Standard deviation
    dat_norm = y_detrend / std  # Normalized dataset type: ignore

    # Wavelet parameters
    s0 = 8 * dt  # Starting scale
    dj = 1 / 12  # sub-octaves
    jj = 7 / dj  # Seven powers of two with dj sub-octaves

    logging.info("Do continous transform")
    wave, scales, freqs, coi, _, _ = pycwt.cwt(
        signal=dat_norm,
        dt=dt,
        dj=dj,
        s0=int(s0),
        J=int(jj),
    )

    # Convert cone of influence from periods to frequency and set values above
    # threshold to maximum for better plotting
    coi = 1 / coi
    coi[coi >= np.max(freqs)] = np.max(freqs)

    # Calculate power spectrum
    power = (np.abs(wave)) ** 2
    power /= scales[:, None]

    return freqs, coi, power


def poly1(x: np.ndarray, slope: float, offset: float) -> np.ndarray:
    """Returns a linear function of the given data.

    :param x: The x axis
    :type x: np.ndarray
    :param slope: The slope of the function
    :type slope: float
    :param offset: The y-axis offset
    :type offset: float
    :return: `slope * x * offset`
    :rtype: np.ndarray
    """
    return slope * x + offset


def find_maximum_sines(
    grid_sin: np.ndarray, grid_cos: np.ndarray, interval: int
) -> Tuple[np.ndarray | float, np.ndarray | float]:
    """Finds the maximum amplitudes of two grids containing coefficients of sine functions. Also removes outliers from the grids which are greater than
    99.99% of the values.

    :param grid_sin: The grid with the sin component.
    :type grid_sin: np.ndarray
    :param grid_cos: The grid with the cos component.
    :type grid_cos: np.ndarray
    :param interval: The interval over which the phase is to be projected (days in a year)
    :type interval: int
    :return: The grids with the maximum amplitude and time of peak.
    :rtype: Tuple[np.ndarray | float, np.ndarray | float]
    """
    # Do superposition
    logging.info("Superposing grids.")
    max_vals, max_time = superpose(grid_sin, grid_cos)
    # Clean outliers
    logging.info("Cleaning and reprojecting.")
    if isinstance(max_vals, np.ndarray):
        max_vals[max_vals > np.nanpercentile(max_vals, 99.99)] = np.nan
    # Project time from radians to halfyear
    max_time = (max_time / (2 * np.pi)) * interval
    return max_vals, max_time


def superpose(
    a_1: float | np.ndarray,
    a_2: float | np.ndarray,
    phi_1: float = 0,
    phi_2: float = np.pi / 2,
) -> Tuple[float | np.ndarray, float | np.ndarray]:
    """Superposes two sine functions and calculates their amplitude and phase.

    The superposition follows this principle:

    ..math::

        y_1 = a_1\,sin (x + \varphi_1) \wedge y_2 = a_2 \, sin (x + \varphi_2)
        y = y_1+y_2 = a sin\,(x+\varphi_0)
        a=\sqrt{a_1^2 + a_2^2+2a_1a_2\,cos(\varphi_2-\varphi_1)}
        \varphi_0=tan^{-1}\left(\frac{a_1\,sin \varphi_1 + a_2\,sin\varphi_2}{a_1\,cos \varphi_1 + a_2\,cos\varphi_2} \right)

    In the case of the inversion results, phase 2 is a cosine which means that :math:`a_2 = -a_2`. The phase change :math:`\varphi_0` is corrected by :math:`\frac{\pi}{2}` and depending on the polarity of :math:`a_1` and :math:`a_2` selected so that the return value is the first maximum in the interval :math:`\[0,2\pi\]`.

    :param a_1: Amplitude of phase 1
    :type a_1: float|np.ndarray
    :param a_2: Amplitude of phase 2
    :type a_2: float|np.ndarray
    :param phi_1: Phase change of phase 1, defaults to 0
    :type phi_1: float, optional
    :param phi_2: Phase change of phase 2, defaults to np.pi/2
    :type phi_2: float, optional
    :return: Maximum amplitude and phase change of superposed functions
    :rtype: Tuple[float | np.ndarray, float | np.ndarray]
    """
    a = np.sqrt(a_1**2 + a_2**2 + 2 * a_1 * a_2 * np.cos(phi_2 - phi_1))
    phi_0 = (
        np.arctan(
            (a_1 * np.sin(phi_1) + a_2 * np.sin(phi_2))
            / (a_1 * np.cos(phi_1) + a_2 * np.cos(phi_2))
        )
        + np.pi / 2
    )
    if isinstance(a_1, (float, np.floating)):
        if a_1 < 0:
            phi_0 = phi_0 + np.pi
    elif isinstance(a_1, np.ndarray):
        phi_0[a_1 < 0] = phi_0[a_1 < 0] + np.pi
    else:
        TypeError("a_1 has wrong type")
    return (a, phi_0)


def adj_R_squared(R_squared: float, p: int, n: int) -> float:
    """Returns the adjusted R² of a given fit

    :param R_squared: The R² value of a fit
    :type R_squared: float
    :param p: The number of variables in the model
    :type p: int
    :param n: The sample size
    :type n: int
    :return: The adjusted R² value
    :rtype: float
    """
    return 1 - (1 - R_squared) * ((n - 1) / (n - p - 1))


def R_squared(data: np.ndarray, residuals: np.ndarray) -> float:
    """Returns the R² of a fit

    :param data: The original data.
    :type data: np.ndarray
    :param residuals: The residuals
    :type residuals: np.ndarray
    :return: The R²
    :rtype: float
    """
    ym = np.mean(data)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((data - ym) ** 2))
    return 1 - (ss_res / ss_tot)
