""" General functions for data analysis """
import numpy as np
import pycwt
import scipy.signal as spsignal
import scipy.stats as spstats


def cwt(y, dt):
    """
    Does a continuous wavelet transformation of the input data with a Morlet
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


def run_stat_mean(timeseries, fnc=spstats.norm):
    """
    Computes the running mean/median with the given statistical distribution
    """
    _, c = timeseries.shape
    ts_out = np.zeros(c)
    for ii in range(c):
        ts_out[ii] = get_stat_val(timeseries[:, ii], fnc=spstats.t)
    return ts_out


def get_stat_val(values, fnc=spstats.norm):
    """
    Fits the data and returns the mean of the given statistical distribution
    """
    return fnc(*fnc.fit(values)).mean()


def sinefunc(
    x: np.ndarray, amplitude: float, width: float, shift: float
) -> np.ndarray:
    """Returns the cosine function of the given data

    Arguments:
        x -- The x axis
        amplitude -- The amplitude of the cosine
        width -- The width/wavelength of the cosine
        shift -- The phase shift

    Returns:
        `amplitude * cos(width * (x + shift))`
    """
    return amplitude * np.cos(width * (x + shift))


def poly1(x: np.ndarray, slope: float, offset: float) -> np.ndarray:
    """Returns a linear function of the given data

    Arguments:
        x -- The x axis
        slope -- The slope of the function
        offset -- The y-axis offset

    Returns:
        `slope * x * offset`
    """
    return slope * x + offset
