""" General functions for data analysis """
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
    # Detrend and normalize data for better cwt analysis
    y_detrend = spsignal.detrend(y)
    std = np.std(y_detrend)  # Standard deviation
    dat_norm = y_detrend / std  # Normalized dataset

    # Wavelet parameters
    mother = pycwt.wavelet.Morlet(6.0)
    s0 = 8 * dt  # Starting scale
    dj = 1 / 12  # sub-octaves
    J = 7 / dj  # Seven powers of two with dj sub-octaves

    # Do continous transform
    wave, scales, freqs, coi, _, _ = pycwt.wavelet.cwt(
        dat_norm, dt, dj, s0, J, mother
    )

    # Convert cone of influence from periods to frequency and set values above
    # threshold to maximum for better plotting
    coi = 1 / coi
    coi[coi >= np.max(freqs)] = np.max(freqs)

    # Calculate power spectrum
    power = (np.abs(wave)) ** 2
    power /= scales[:, None]

    return freqs, coi, power


def cosinefunc(
    x: np.ndarray, amplitude: float, width: float, shift: float
) -> np.ndarray:
    """Returns the cosine function of the given data

    :param x: The x axis
    :type x: np.ndarray
    :param amplitude: The amplitude of the cosine
    :type amplitude: float
    :param width: The width/wavelength of the cosine
    :type width: float
    :param shift: The phase shift
    :type shift: float
    :return: `amplitude * cos(width * (x + shift))`
    :rtype: np.ndarray
    """
    return amplitude * np.cos(width * (x + shift))


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
