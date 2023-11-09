"""
**Inversion of timeseries data**

A timeseries is defined as a three component velocity field and analyzed
through an inversion approach with up to eleven different components. The data
has to be in dictionary format and may include errors for each component:

|    data = {
|        `"t"`: NumPy array of time in float years (e.g. 01.01.2014 = 2014.0),
|        `"dataE"`: NumPy array of E-W component (mm),
|        `"dataN"`: NumPy array of N-S component (mm),
|        `"dataU"`: NumPy array of Up-Down component (mm),
|        `"sigmE"`: NumPy array of Error of E-W component (mm),
|        `"sigmN"`: NumPy array of Error of N-S component (mm),
|        `"sigmU"`: NumPy array of Error of Up-Down component (mm),
|        `"station"`: String of station name,
|    }

**Examples:**
An implementation using test data is found in :func:`invert_test_data`. A best
practice to work with real data is implemented in
:func:`u4py.analysis.processing.invert_file`.

**Source**:
This code has been transcribed from the Matlab source code of S. Metzger, GFZ-Potsdam.

"""
from __future__ import annotations

import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as splinalg
import scipy.sparse as spsparse
import scipy.stats as spstats

import u4py.plotting.plots as u4plots
import u4py.utils.config as u4config
import u4py.utils.convert as u4convert


def create_synthetic_data(
    lin: float = 1.1,
    ann_sin: float = 0.6,
    ann_cos: float = 0.8,
    sem_sin: float = -0.1,
    sem_cos: float = 0.05,
    eq_disp: list = [
        -32,
    ],
    t_EQdummy: list = [
        2010,
    ],
    at_disp: list = [
        10,
    ],
    t_AT: list = [
        2014,
    ],
    ex_disp: list = [
        5,
    ],
    t_EX: list[tuple] = [
        (2020, 2021),
    ],
) -> Tuple[dict, list]:
    """Creates a synthetic dataset for testing the inversion algorithm.

    :param lin: Linear component, defaults to 1.1
    :type lin: float, optional
    :param ann_sin: Annual sine, defaults to 0.6
    :type ann_sin: float, optional
    :param ann_cos: Annual cosine, defaults to 0.8
    :type ann_cos: float, optional
    :param sem_sin: semiannual sine, defaults to -0.1
    :type sem_sin: float, optional
    :param sem_cos: semiannual cosine, defaults to 0.05
    :type sem_cos: float, optional
    :param eq_disp: A list of earthquake displacements, defaults to [32,]
    :type eq_disp: list, optional
    :param t_EQdummy: A list of earthquake times matching `eq_disp`, defaults to [ 2010, ]
    :type t_EQdummy: list, optional
    :param at_disp: A list of displacements due to antenna offsets, defaults to [10,]
    :type at_disp: list, optional
    :param t_AT: A list of Antenna offsets, defaults to [ 2014, ]
    :type t_AT: list, optional
    :param ex_disp: A list of displacements due to water extraction, defaults to [5,]
    :type ex_disp: list, optional
    :param t_EX: A list of water extractions, defaults to [ (2020, 2021),]
    :type t_EX: list[tuple], optional
    :return: A tuple containing (`data`, `eq_list`)
    :rtype: Tuple[dict, list]

    This function creates a hypothetical dataset with a single earthquake which
    can be used to test the inversion algorithm if it works the same as the
    original Matlab source.

    The dataset covers the time from 01.01.2007 to 01.01.2015 in intervals of
    roughly one day. A simulated earthquake is generated on 27.02.2010 at
    00:56h. The synthetic dataset contains:

    - A linear component,
    - two annual (sine/cosine, :math:`\\frac{\\pi}{2}`),
    - two semiannual components (sine/cosine, :math:`\\frac{\\pi}{4}`)
    - a heavyside step function with several mm of displacement and post seismic decay,
    - Gaussian noise.
    """
    t = np.arange(2000, 2023, 0.003)  # sample time
    d_noise = spstats.norm(loc=0, scale=1.5).rvs(len(t))  # noise

    f_Heavi = np.zeros_like(t)  # Heaviside function
    f_eq = np.zeros_like(t)
    for eqd, teq in zip(eq_disp, t_EQdummy):
        f_Heavi[t > teq] = f_Heavi[t > teq] - eqd / 8  # EQ at t_EQdummy
        f_post = np.real(3 * f_Heavi * np.log((1.0 + 0.0j) + (t - teq)))
        f_eq += -8 * f_Heavi + f_post

    f_at = np.zeros_like(t)
    for atd, tat in zip(at_disp, t_AT):
        f_at[t > tat] = f_at[t > tat] + atd

    f_ex = np.zeros_like(t)
    for exd, (t_start, t_end) in zip(ex_disp, t_EX):
        # Factor to fit logistic function into given range
        # duration_factor = 2 * 6.9076907690769
        duration_factor = 2 * 2.92
        duration = t_end - t_start
        k = duration_factor / duration
        t_off = t_start + 0.5 * duration
        f_ex = f_ex + _expit(t, k=k, t_off=t_off) * exd

    # Create synthetic data and uncertainties
    dataE = np.real(
        t * lin
        + ann_sin * np.sin(2 * np.pi * t)
        + ann_cos * np.cos(2 * np.pi * t)
        + sem_sin * np.sin(4 * np.pi * t)
        + sem_cos * np.cos(4 * np.pi * t)
        + f_eq
        + f_at
        + f_ex
        + d_noise
    )

    individual_g_funcs = {
        "lin": t * lin,
        "ann_sin": ann_sin * np.sin(2 * np.pi * t),
        "ann_cos": ann_cos * np.cos(2 * np.pi * t),
        "sem_sin": sem_sin * np.sin(4 * np.pi * t),
        "sem_cos": sem_cos * np.cos(4 * np.pi * t),
        "f_eq": f_eq,
        "f_at": f_at,
        "f_ex": f_ex,
        "d_noise": d_noise,
    }

    sigmE = np.ones_like(t)
    dataE = dataE - dataE[0]
    data = {
        "t": t,
        "dataE": dataE,
        "dataN": dataE,
        "dataU": dataE,
        "sigmE": sigmE,
        "sigmN": sigmE,
        "sigmU": sigmE,
        "station": "Dummy",
    }
    return data, t_EQdummy, t_AT, t_EX, individual_g_funcs


def invert_time_series(
    data: dict,
    t_AT: list = [],
    t_EQ: list = [],
    t_EX: list = [],
    ind: slice = 0,
    num_coeffs: int = 1,
    directions: list = ["dataE", "dataN", "dataU"],
    t_relative: float = 0,
    use_tensorflow: bool = False,
    use_sparse: bool = True,
    use_sigma: bool = False,
) -> Tuple:
    """Inverts a timeseries.

    :param data: The data formatted as a dictionary.
    :type data: dict
    :param t_AT: A list with times of known antenna offsets, defaults to []
    :type t_AT: list, optional
    :param t_EQ: A list with times of known earthquakes, defaults to []
    :type t_EQ: list, optional
    :param ind: Slice to use only a certain time window, defaults to 0
    :type ind: slice, optional
    :param num_coeffs: The number of parameters to use for inversion, defaults to 1
    :type num_coeffs: int, optional
    :param directions: Which directions to use for inversion, defaults to ["dataE","dataN", "dataU"]
    :type directions: list, optional
    :param t_relative: A time offset used mainly for plotting, defaults to 0
    :type t_relative: float, optional
    :param use_tensorflow: Use tensorflow for inversion (WIP), defaults to False
    :type use_tensorflow: bool, optional
    :param use_sparse: Uses sparse matrices for saving memory, defaults to True
    :type use_sparse: bool, optional
    :param use_sigma: Uses weighted fitting with measurement errors, defaults to False
    :type use_sigma: bool, optional
    :return: A tuple with the original data, forward model and time series.
    :rtype: Tuple

    **Inversion Theory**

    *The problem must be linear, otherwise you cannot invert!*

    The measured data :math:`d` is the result of the multiplication of a
    Green's function matrix :math:`G` and the model parameters :math:`m`,
    i.e., the components:

    .. math::
        d = G \\cdot m

    *(d: data, G: solver matrix, m: model parameters)*

    To invert for the model parameters the inverse product of the solver
    matrix with its transposed version :math:`(G'\\cdot G)^{-1}` is multiplied
    by transposed solver matrix :math:`G'` and the measured data :math:`d`:

    .. math::
        m = (G'\\cdot G)^{-1} \\cdot G' \\cdot d


    and, using weights :math:`W`:

    .. math::
      W \\cdot d = W \\cdot G \\cdot m

    .. math::
        m = (G' \\cdot S^{-1} \\cdot G)^{-1} \\cdot G' \\cdot S^{-1} \\cdot d

    with :math:`S^{-1} = W' \\cdot W`, or, simplified: :math:`S = s^2`

    **Setup of the G-Matrix**

    The full G-Matrix contains a y-offset, linear trend, higher polynomial
    trends, sinusoidal oscillations, antenna offsets, earthquake offsets with
    postseismic relaxiation and logistic functions to simulate transients due
    to water extraction:

        | :math:`A_1 + A_2\\cdot(t-t_R) + A_3 \\cdot (t-t_R)^2 + A_n \\cdot (t-t_R)^{n-1} + \\dots`
        | :math:`B_1 \\cdot sin(2 \\pi t) + B_2 \\cdot cos(2 \\pi t) + B_3 \\cdot sin(4 \\pi t) + B_4 \\cdot cos(4 \\pi t) + \\dots`
        | :math:`C_n \\cdot tH(t_{AT}) + \\dots`
        | :math:`D_n \\cdot tH(t_{EQ}) + \\dots`
        | :math:`E_n \\cdot log(1+\\frac{t}{dT_1}) + \\dots`
        | :math:`F_n \\cdot \\frac{1}{1+e^{-kF_n(t-t_0)} (\\frac{F_n}{t_0}-1)}`

    In this equation, the following model parameters are included:

    - :math:`A_1 \\dots A_n`: General offset and steady rate (can be steady, linear , for interseismic deformation, or quadratic, for GIA adjustment)
    - :math:`B_1 \\dots B_4`: Amplitudes of annual and semi-annual oscillations
    - :math:`C_1 \\dots C_n`: Amplitude of Heaviside function for antenna offsets (:math:`d_{AT}`) starting at :math:`t_{AT}`.
    - :math:`D_1 \\dots D_n`: Amplitude of Heaviside function for vertical earthquake displacement (:math:`d_{EQ}`) starting at :math:`t_{EQ}`.
    - :math:`E_1 \\dots E_n`: Amplitude of postseismic signal starting at :math:`t_{EQ}`.
    - :math:`F_1 \\dots F_n`: Amplitude of logistic function, with :math:`k=\\frac{2\\times2.92}{t_{end}-t_{start}}` which sets the increase to fit into the given start and end times, and :math:`t_0=t_{start}+\\frac{1}{2}(t_{end}-t_{start})` which shifts the middle of the logistic function to the middle of the period.

    The G-Matrix needs to be individualized for each station, depending if
    it is affected by these signals. You should create a *GLOBAL* G-Matrix
    including *ALL* stations and not solve this for each single station. You do
    that by concatenating all stations and components into one big G-Matrix.
    """
    _clean_inputs(data)
    if not ind:
        ind = slice(len(data["t"]))
        ori = True
    else:
        ori = False

    g_functions, parameter_list = _prepare_g_functions(
        data=data,
        ind=ind,
        t_AT=t_AT,
        t_EQ=t_EQ,
        t_EX=t_EX,
        num_coeffs=num_coeffs,
        t_relative=t_relative,
    )
    if use_sparse:
        g_matrix = spsparse.csr_matrix(_set_g_matrices(g_functions))
    else:
        g_matrix = _set_g_matrices(*g_functions)

    # The G-matrix has now the following dimension:
    # Length: #samples * #stations * #components
    # Width:  #parameters

    # Set up the DATA vector D, TIME vector T and SIGMA-matrix S
    # Dimension:
    #     (#samples * #components * #stations) x 1
    data_vector = np.hstack([data[kk][ind] for kk in directions])

    # Define SIGMA-Matrix
    # In the best case this should be a full-matrix with correlated errors. In
    # reality, you often only have the diagonal components. Use SPARSE to
    # reduce the size of the matrix.
    # Dimension:
    #    (#samples * #components * #stations) x
    #    (#samples * #components * #stations)
    if use_sigma:
        sigm_primer = np.hstack(
            [data[kk.replace("data", "sigm")][ind] ** 2 for kk in directions]
        )
        if use_sparse:
            sigma_matrix = spsparse.csc_matrix(spsparse.diags(sigm_primer))

        else:
            sigma_matrix = np.diag(sigm_primer)
    else:
        sigma_matrix = np.array([])
    # Define TIME-vector
    # Dimension:
    #     (#samples * #components * #stations) x 1
    time_vector = [data["t"][ind], data["t"][ind], data["t"][ind]]
    if use_sparse:
        matrix = _invert(g_matrix, data_vector, sigma_matrix)
    # elif use_tensorflow:
    #     matrix = _invert_tf(g_matrix, data_vector, sigma_matrix)
    else:
        matrix = _invert_np(g_matrix, data_vector, sigma_matrix)
    data = forward_model(matrix, g_matrix, data, ind, ori=ori)
    return matrix, data, time_vector, parameter_list


def _clean_inputs(data: dict):
    """Cleans all nonfinite data from input

    :param data: The data matrix containing some nonfinite data points
    :type data: dict
    """
    logging.info("Cleaning inputs.")
    for k in data.keys():
        if isinstance(data[k], np.ndarray):
            ind = np.nonzero(np.isfinite(data[k]))
            for k in data.keys():
                if (
                    isinstance(data[k], np.ndarray)
                    and k != "inversion_results"
                ):
                    data[k] = data[k][ind]


def _invert(
    G: np.ndarray, D: np.ndarray, S: np.ndarray = np.array([])
) -> np.ndarray:
    """Invert for your model parameters using scipy's sparse linear algebra.

    :param G: The Green's function matrix
    :type G: np.ndarray
    :param D: The data matrix
    :type D: np.ndarray
    :param S: A matrix containing measurement errors to use for weighting, defaults to np.array([])
    :type S: np.ndarray, optional
    :return: The inverted model parameters (dimension: #parameters x 1)
    :rtype: np.ndarray
    """
    logging.info("Starting inversion.")
    logging.info("Transposing G-Matrix.")
    Gp = G.conj().transpose()
    if S.size > 0:
        logging.info("Inverting S.")
        iS = spsparse.linalg.inv(S)
        logging.info("Summing Matrices.")
        M = iS @ D @ Gp @ spsparse.linalg.inv(spsparse.csc_matrix(G @ iS @ Gp))
    else:
        logging.info("No sigma. Using unweighted inversion.")
        M = D @ Gp @ spsparse.linalg.inv(spsparse.csc_matrix(G @ Gp))
    return M.astype("e")


def _invert_np(
    G: np.ndarray, D: np.ndarray, S: np.ndarray = np.array([])
) -> np.ndarray:
    """Invert for your model parameters using numpy's linear algebra.

    :param G: The Green's function matrix
    :type G: np.ndarray
    :param D: The data matrix
    :type D: np.ndarray
    :param S: A matrix containing measurement errors to use for weighting, defaults to np.array([])
    :type S: np.ndarray, optional
    :return: The inverted model parameters (dimension: #parameters x 1)
    :rtype: np.ndarray
    """
    logging.info("Setting up matrices.")
    G = G.astype(np.double)
    D = D.astype(np.double)
    S = S.astype(np.double)
    logging.info("Transposing G-Matrix.")
    Gp = G.conj().transpose()
    if S.any():
        logging.info("Inverting S.")
        iS = np.linalg.inv(S)
        logging.info("Inverting Matrix Component.")
        iM1 = np.linalg.inv(np.matmul(np.matmul(G, iS), Gp))
        logging.info("Summing Matrices.")
        M = np.matmul(np.matmul(np.matmul(iS, D), Gp), iM1)
    else:
        logging.info("Inverting Matrix Component.")
        iM1 = np.linalg.inv(np.matmul(Gp, G))
        logging.info("Summing Matrices.")
        M = np.matmul(np.matmul(iM1, D), Gp)

    return M.astype("e")


# def _invert_tf(
#     G: np.ndarray, D: np.ndarray, S: np.ndarray = np.array([])
# ) -> np.ndarray:
#     """Invert for your model parameters using tensorflow.

#     :param G: The Green's function matrix
#     :type G: np.ndarray
#     :param D: The data matrix
#     :type D: np.ndarray
#     :param S: A matrix containing measurement errors to use for weighting, defaults to np.array([])
#     :type S: np.ndarray, optional
#     :return: The inverted model parameters (dimension: #parameters x 1)
#     :rtype: np.ndarray
#     """
#     Gp = G.conj().transpose()
#     if S.any():
#         iS = tf.linalg.inv(S)
#         M = iS @ D @ Gp @ tf.linalg.inv(G @ iS @ Gp)
#     else:
#         M = tf.linalg.inv(Gp @ G) @ D @ Gp
#     return M.astype("e")


def forward_model(
    M: np.ndarray, G: np.ndarray, data: dict, ind: slice, ori: bool = True
) -> dict:
    """Forward model the data using the model parameters.

    :param M: The inverted model parameters.
    :type M: np.ndarray
    :param G: The Greens function matrix.
    :type G: np.ndarray
    :param data: The measured data dictionary. The forward model is added as `dhat_data`.
    :type data: dict
    :param ind: The time slice to use for calculating.
    :type ind: slice
    :param ori: Whether the data is the first fit or not. (required to test if the solution improves.), defaults to True
    :type ori: bool, optional
    :return: Dictionary with the forward model attached to it as `dhat_data` or `ori_dhat_data`.
    :rtype: dict

    The big `dhat`-vector is separated into different components. The
    residuals are also added to the returned data dictionary.
    """
    logging.info("Calulating Forward Model.")
    if isinstance(G, spsparse.csr_matrix):
        dhat = np.reshape(M @ G, (3, len(data["t"][ind])))
    else:
        dhat = np.reshape(np.matmul(M, G), (3, len(data["t"][ind])))
    dhat_data = dict()
    dhat_data["dhatE"] = dhat[0]
    dhat_data["dhatN"] = dhat[1]
    dhat_data["dhatU"] = dhat[2]

    # Calculate residuals
    dhat_data["dresE"] = data["dataE"][ind] - dhat_data["dhatE"]
    dhat_data["dresN"] = data["dataN"][ind] - dhat_data["dhatN"]
    dhat_data["dresU"] = data["dataU"][ind] - dhat_data["dhatU"]

    if ori:
        data["ori_dhat_data"] = dhat_data
    else:
        data["dhat_data"] = dhat_data
    return data


def _prepare_g_functions(
    data: dict,
    ind: slice,
    t_AT: list = [],
    t_EQ: list = [],
    t_EX: list = [],
    num_coeffs: int = 2,
    t_relative: float = 0,
) -> Tuple[Tuple[np.ndarray], list]:
    """Prepares the Green's functions.

    :param data: The data formatted as a dictionary.
    :type data: dict
    :param ind: Slice to use only a certain time window, defaults to 0
    :type ind: slice
    :param t_AT: A list with times of known antenna offsets, defaults to []
    :type t_AT: list, optional
    :param t_EQ: A list with times of known earthquakes, defaults to []
    :type t_EQ: list, optional
    :param t_EX: A list with times of known water extractions, defaults to []
    :type t_EX: list, optional
    :param num_coeffs: The number of parameters to use for g_TREND, defaults to 2 (offset and linear)
    :type num_coeffs: int, optional
    :param t_relative: A time offset used mainly for plotting, defaults to 0
    :type t_relative: float, optional
    :return: The Green's functions as a stacked matrix and a list with the names of the components.
    :rtype: Tuple[Tuple[np.ndarray], list]
    """
    g_funcs = []
    parameter_list = []
    g_TREND, parameter_list = _set_g_trend(
        num_coeffs, data["t"][ind], t_relative, parameter_list
    )
    g_funcs.append(np.asarray(g_TREND))
    g_ANNUAL, parameter_list = _set_g_annual(data["t"][ind], parameter_list)
    g_funcs.append(np.asarray(g_ANNUAL))

    if t_AT:
        g_AT, parameter_list = _set_g_antenna(
            data["t"][ind], t_AT, parameter_list
        )
        g_funcs.append(np.asarray(g_AT))

    if t_EQ:
        g_EQ, parameter_list = _set_g_earthquakes(
            data["t"][ind], t_EQ, parameter_list
        )
        g_funcs.append(np.asarray(g_EQ))
        g_POSTSM, parameter_list = _set_g_postseismic(
            data["t"][ind], t_EQ, parameter_list
        )
        g_funcs.append(np.asarray(g_POSTSM))
    if t_EX:
        g_EX, parameter_list = _set_g_extraction(
            data["t"][ind], t_EX, parameter_list
        )
        g_funcs.append(np.asarray(g_EX))

    # RETURN all G-functions.
    return (g_funcs, parameter_list)


def _set_g_trend(
    num_coeffs: int,
    time: np.ndarray,
    t_relative: float,
    parameter_list: list,
) -> Tuple[list, list]:
    """Sets the g-functions for polynomial trends.

    :param num_coeffs: The number of coefficients.
    :type num_coeffs: int
    :param time: The time as array in float year
    :type time: np.ndarray
    :param t_relative: A time offset used mainly for plotting.
    :type t_relative: float
    :param parameter_list: The list of parameters to keep track of the coefficient names
    :type parameter_list: list
    :return: Returns a list of g-functions and the updated parameter list.
    :rtype: Tuple[list, list]
    """
    logging.info("Setting Mini-g-function == TREND")
    g_TREND = [(time - t_relative) ** ii for ii in range(num_coeffs + 1)]
    trend_names = [
        "y-axis offset (year=0!)",
        "linear trend",
        "quadratic trend",
        "order trend",
    ]
    for ii in range(num_coeffs + 1):
        if ii < 3:
            parameter_list.append(trend_names[ii])
        else:
            parameter_list.append(f"{ii}. " + trend_names[-1])
    return g_TREND, parameter_list


def _set_g_annual(time: np.ndarray, parameter_list: list) -> Tuple[list, list]:
    """Sets the four g-functions for the annual trend :math:`B_1` to :math:`B_4`.

    :param time: The time as array in float years
    :type time: np.ndarray
    :param parameter_list: The list of parameters to keep track of the coefficient names.
    :type parameter_list: list
    :return: Returns a list of g-functions and the updated parameter list.
    :rtype: Tuple[list, list]
    """
    logging.info("Setting Mini-g-function: Annual")
    # Mini-g-function == ANNUAL SIGNAL ==== 4 parameters ===== B1 to B4 ===
    g_ANNUAL = [
        np.sin(2 * np.pi * time),
        np.cos(2 * np.pi * time),
        np.sin(4 * np.pi * time),
        np.cos(4 * np.pi * time),
    ]

    parameter_list.extend(
        [
            "semi-annual sine",
            "semi-annual cosine",
            "annual sine",
            "annual cosine",
        ]
    )
    return g_ANNUAL, parameter_list


def _set_g_antenna(
    time: np.ndarray, t_AT: Iterable, parameter_list: list
) -> Tuple[list, list]:
    """Sets a g-function in the form of a heaviside step function for each antenna offset.

    :param time: The time as array in float years.
    :type time: np.ndarray
    :param t_AT: An iterable containing the times of antenna offset.
    :type t_AT: Iterable
    :param parameter_list: The list of parameters to keep track of the coefficient names.
    :type parameter_list: list
    :return: Returns a list of g-functions and the updated parameter list.
    :rtype: Tuple[list, list]
    """
    logging.info("Setting Mini-g-function: Antenna offsets")
    g_AT = []

    # Create a Heaviside vector (or matrix, if t_AT has more than one entry):
    for ii, tat in enumerate(t_AT):
        g_hat = np.zeros_like(time)
        g_hat[time > tat] = 1
        g_AT.append(g_hat)
        parameter_list.append(f"antenna offset no. {ii+1}")
    return g_AT, parameter_list


def _set_g_earthquakes(
    time: np.ndarray, t_EQ: Iterable, parameter_list: list
) -> Tuple[list, list]:
    """Sets a g-function in the form of a heaviside step function for each earthquake.

    :param time: The time as array in float years.
    :type time: np.ndarray
    :param t_EQ: An iterable containing the times of earthquakes
    :type t_EQ: Iterable
    :param parameter_list: The list of parameters to keep track of the coefficient names.
    :type parameter_list: list
    :return: Returns a list of g-functions and the updated parameter list.
    :rtype: Tuple[list, list]
    """
    logging.info("Setting Mini-g-function: Earthquake offsets")
    g_EQ = []

    if (np.max(time) > np.min(t_EQ)) or (np.min(time) < np.max(t_EQ)):
        if t_EQ[0] < time[0]:
            t_EQ[0] = time[0]

        # Here I check if several earthquakes occurred within the gap of a
        # time-series. If yes, I remove the first earthquakes and keep only
        # the last event in this data gap.
        # spaceind = [
        #     len(np.argwhere(data["t"] > t_EQ[k]) and data["t"] < t_EQ[k + 1]))
        #     for k in range(len(t_EQ))
        # ]
        # t_EQ[spaceind == 0] = []

        # Create a Heaviside vector (or matrix, if t_EQ has more than one entry):
        for ii in range(len(t_EQ)):
            g_heq = np.zeros_like(time)
            g_heq[time > t_EQ[ii]] = g_heq[time > t_EQ[ii]] + 1
            g_EQ.append(g_heq)
            parameter_list.append(f"earthquake no. {ii+1}")
    return g_EQ, parameter_list


def _set_g_postseismic(
    time: np.ndarray, t_EQ: Iterable, parameter_list: list
) -> Tuple[list, list]:
    """Sets a g-function in the form of logarithmic increase for each earthquake in `t_EQ`.

    :param time: The time as array in float years.
    :type time: np.ndarray
    :param t_EQ: An iterable containing the times of earthquakes
    :type t_EQ: Iterable
    :param parameter_list: The list of parameters to keep track of the coefficient names.
    :type parameter_list: list
    :return: Returns a list of g-functions and the updated parameter list.
    :rtype: Tuple[list, list]

    __Attention!__

    The postseismic signal is A-PRIORI-LINEARIZED by assuming dT = 1 (Bevis &
    Brown, 2014). If you want to really study the geophysical properties of
    the post-seismic signal, you have to estimate dT using a non-linear
    approximation for each station, or group of stations later on.
    """
    logging.info("Setting Mini-g-function: Postseismic relaxation")

    g_POSTSM = []
    if len(time) > 101:
        g_psm = np.zeros_like(time)
        for ii, te in enumerate(t_EQ):
            iB = time > te
            g_psm[iB] = g_psm[iB] + np.log(1 + (time[iB] - te))
            g_POSTSM.append(g_psm)
            parameter_list.append(f"postseismic no. {ii+1}")
    return g_POSTSM, parameter_list


def _set_g_extraction(
    time: np.ndarray, t_EX: Iterable[Tuple], parameter_list: list
) -> Tuple[list, list]:
    """Sets a g-function in the form of a logistic growth for each extraction timerange in `t_EX`.

    :param time: The time as array in float years.
    :type time: np.ndarray
    :param t_EX: An Iterable with Tuples of extraction start and extraction ends.
    :type t_EX: Iterable[Tuple]
    :param parameter_list: The list of parameters to keep track of the coefficient names.
    :type parameter_list: list
    :return: Returns a list of g-functions and the updated parameter list.
    :rtype: Tuple[list, list]
    """
    logging.info("Setting Mini-g-function: Extraction")
    g_EX = []

    gex = np.zeros_like(time)
    for ii, (t_start, t_end) in enumerate(t_EX):
        # Factor to fit logistic function into given range
        # duration_factor = 2 * 6.9076907690769
        duration_factor = 2 * 2.92
        duration = t_end - t_start
        k = duration_factor / duration
        t_off = t_start + 0.5 * duration
        gex = gex + _expit(time, k=k, t_off=t_off)
        g_EX.append(gex)
        parameter_list.append(f"water extraction no. {ii+1}")
    return g_EX, parameter_list


def _expit(
    t: np.ndarray,
    k: float = 1,
    G: float = 1,
    t0: float = 0.5,
    t_off: float = 0,
) -> np.ndarray:
    """Returns a logistic function.

    :param t: The time in floatyears.
    :type t: np.ndarray
    :param k: The logistic growth rate, defaults to 1
    :type k: float, optional
    :param G: The supremum of the values of the function, defaults to 1
    :type G: float, optional
    :param t0: The time of the functions midpoint, defaults to 0.5
    :type t0: float, optional
    :param t_off: The time offset of the midpoint, defaults to 0
    :type t_off: float, optional
    :return: The logistic curve.
    :rtype: np.ndarray
    """
    result = G * (1 / (1 + np.exp(-1 * k * G * (t - t_off)) * ((G / t0) - 1)))
    return result


def _set_g_matrices(g_funcs) -> np.ndarray:
    """Set up a G-matrices for each component and then the full, global G-matrix.

    :return: Block diagonal matrix of stacked Green's function for each component.
    :rtype: np.ndarray
    """
    logging.info("Setting up G-Matrices")
    g_matrix = np.array([])
    for g_func in g_funcs:
        if g_func.any():
            r, _ = g_func.shape
            for n in range(r):
                if g_matrix.any():
                    g_matrix = np.vstack((g_matrix, g_func[n, :]))
                else:
                    g_matrix = g_func[n, :]
    gE = g_matrix
    gN = g_matrix
    gU = g_matrix

    return splinalg.block_diag(gE, gN, gU)


def remove_outliers(data: dict, threshold: float = 2.5) -> np.ndarray:
    """Creates a numpy slicing array that removes all data which is more than
    `threshold` standard deviations away from median.

    :param data: The data dictionary containing the timeseries.
    :type data: dict
    :param threshold: The number of standard deviations to take, defaults to 2.5
    :type threshold: float, optional
    :return: The indices to remove the outliers.
    :rtype: np.ndarray
    """
    logging.info("Removing Outliers.")
    thrE = threshold * np.std(data["dresE"])
    thrN = threshold * np.std(data["dresN"])
    thrU = threshold * np.std(data["dresU"])
    ind = (
        np.nonzero(np.abs(data["dresE"]) < thrE)
        and np.nonzero(np.abs(data["dresN"]) < thrN)
        and np.nonzero(np.abs(data["dresU"]) < thrU)
    )
    return ind


def medianize_station(
    dataset: dict, data_keys: list, include_sigma=True
) -> dict:
    """Takes the median of the dataset and returns it as a timeseries.

    :param dataset: The input dataset as dictionary.
    :type dataset: dict
    :param data_keys: Keys where the data is found in `dataset`.
    :type data_keys: list
    :param include_sigma: Whether to include the errors or not, defaults to True
    :type include_sigma: bool, optional
    :return: The medianized dataset.
    :rtype: dict
    """
    logging.info("Medianizing Station.")
    time_series = dict()
    for k in ["dataE", "dataN", "dataU"]:
        time_series[k] = np.nanmedian(
            [dataset[kk][k] for kk in data_keys], axis=0
        )
        if include_sigma:
            time_series[k.replace("data", "sigm")] = np.nanstd(
                [dataset[kk][k] for kk in data_keys], axis=0
            )
    time_series["t"] = dataset[data_keys[0]]["t"]
    time_series["station"] = [dataset[kk]["station"] for kk in data_keys]
    return time_series


def stack_data(dataset: dict) -> dict:
    """Stacks all datapoints for inversion

    :param dataset: The dataset containing the data to stack.
    :type dataset: dict
    :return: A dataset with the all data stacked together.
    :rtype: dict
    """
    logging.info("Stacking Data.")
    data_keys = [kk for kk in dataset.keys() if kk != "inversion_results"]
    time_series = dict()

    # Used when dataset is a dictionary of individual stations
    if isinstance(dataset[data_keys[0]], dict):
        time_series["t"] = np.hstack([dataset[kk]["t"] for kk in data_keys])
        asorted = np.argsort(time_series["t"])
        time_series["t"] = time_series["t"][asorted]
        for k in ["dataE", "dataN", "dataU"]:
            time_series[k] = np.hstack([dataset[kk][k] for kk in data_keys])
            time_series[k] = time_series[k][asorted]
        time_series["station"] = [dataset[kk]["station"] for kk in data_keys]
        time_series["xmid"] = dataset[data_keys[0]]["xmid"]
        time_series["ymid"] = dataset[data_keys[0]]["ymid"]

    # Used when dataset is already a merged dataset with numpy arrays
    else:
        time_series["t"] = np.tile(
            u4convert.get_floatyear(dataset["time"]),
            (dataset["num_points"], 1),
        )
        for k in ["dataE", "dataN", "dataU"]:
            time_series[k] = dataset["timeseries"]
        time_series["station"] = [str(nn) for nn in dataset["ps_id"]]
        time_series["xmid"] = dataset["xmid"]
        time_series["ymid"] = dataset["ymid"]
    return time_series


def downsample_timeseries(time_series: dict, maxn: int) -> dict:
    """Randomly takes `maxn` datapoints from the timeseries to circumvent
    memory limitations.

    :param time_series: The timeseries dictionary.
    :type time_series: dict
    :param maxn: The number of samples to take.
    :type maxn: int
    :return: A downsampled version of `time_series`.
    :rtype: dict
    """
    samples = random.sample(range(len(time_series["dataE"])), maxn)
    time_series["t"] = time_series["t"][samples]
    asorted = np.argsort(time_series["t"])
    time_series["t"] = time_series["t"][asorted]
    for k in ["dataE", "dataN", "dataU"]:
        time_series[k] = time_series[k][samples][asorted]
    return time_series


def print_inversion_results(matrix: np.ndarray, parameters_list: list[str]):
    """Nicely prints the results from the inversion.

    :param matrix: The inverted model parameters.
    :type matrix: np.ndarray
    :param parameters_list: The parameters string list.
    :type parameters_list: list[str]
    """
    print_string = ""
    directions = [
        "--- East-West ---",
        "--- North-South ---",
        "--- Up-Down ---",
    ]
    if len(matrix) == len(parameters_list):
        for val, name in zip(matrix, parameters_list):
            print_string += f"{name} {val:.2f}\n"
    else:
        try:
            matr_resh = np.reshape(matrix, (3, int(len(matrix) / 3)))
            for ii, direct in enumerate(matr_resh):
                print_string += directions[ii] + "\n"
                for jj, val in enumerate(direct):
                    print_string += f"{parameters_list[jj]}: {val:2f}\n"
        except TypeError:
            raise TypeError("Something is wrong with the solution matrix.")
        except IndexError:
            raise IndexError("Matrix longer than expected...")
    return print_string[:-2]


def reformat_dict(dataset: dict) -> dict:
    """Reformats a loaded hdf5 dictionary to match with the one for inversion.

    :param dataset: The dictionary to reformat.
    :type dataset: dict
    :return: The reformatted dictionary.
    :rtype: dict
    """
    data = stack_data(dataset)
    if isinstance(data["t"][0], datetime):
        data["t"] = u4convert.get_floatyear(data["t"])
    data["sigmE"] = np.ones_like(data["dataE"])
    data["sigmN"] = np.ones_like(data["dataE"])
    data["sigmU"] = np.ones_like(data["dataE"])

    if "inversion_results" in dataset.keys():
        data["inversion_results"] = dataset["inversion_results"]

    return data


def reformat_simple_timeseries(
    time: np.ndarray,
    values: np.ndarray,
    station: str = "",
    xmid: float = 0,
    ymid: float = 0,
) -> dict:
    """Creates a dictionary for inversion from a simple time-value timeseries.

    :param time: The numpy array containing timestamps (in datetime)
    :type time: np.ndarray
    :param values: The values as numpy array
    :type values: np.ndarray
    :param station: The name of the station (optional), defaults to ""
    :type station: str, optional
    :param xmid: The x coordinate of the station (optional), defaults to 0
    :type xmid: float, optional
    :param ymid: The y coordinate of the station (optional), defaults to 0
    :return: A dictionary reformatted for inversion with all components = values.
    :rtype: dict
    """

    data = {
        "t": u4convert.get_floatyear(time),
        "dataE": values,
        "dataN": values,
        "dataU": values,
        "sigmE": np.ones_like(values),
        "sigmN": np.ones_like(values),
        "sigmU": np.ones_like(values),
        "station": station,
        "xmid": xmid,
        "ymid": ymid,
    }
    return data


def smooth_stacked_data(data: dict) -> dict:
    """Takes a stacked data set with irregular sampling intervals and generates a "smooth" version using running means and interpolation.

    :param data: The stacked dataset
    :type data: dict
    :return: The "smoothed" dataset.
    :rtype: dict
    """
    print(data["t"])


def invert_test_data():
    """
    Tests the inversion with synthetic data.

        1. First :func:`create_synthetic_data`
        2. Then :func:`invert_time_series`
        3. Finally :func:`u4py.plotting.plots.plot_inversion_results`.
    """

    logging.basicConfig(
        format="[%(levelname)s] %(funcName)s: %(message)s",
        stream=sys.stdout,
        level=u4config.log_level,
    )
    syn_comps = [
        1.1,
        0.6,
        0.8,
        -0.1,
        0.05,
        [
            -32,
        ],
        [
            2010,
        ],
        [
            10,
        ],
        [
            2014,
        ],
        [
            -15,
        ],
        [
            (2001, 2005),
        ],
    ]
    data, t_EQ, t_AT, t_EX, _ = create_synthetic_data(*syn_comps)
    matrix, data, time_vector, parameters_list = invert_time_series(
        data, t_EQ=t_EQ, t_AT=t_AT, t_EX=t_EX, use_sparse=True
    )
    inversion_results = {"matrix_ori": matrix}
    print_string = print_inversion_results(matrix, parameters_list)
    print(print_string)
    fig, ax = u4plots.plot_inversion_results(
        time=time_vector,
        data=data,
        inversion_results=inversion_results,
        single_dim=True,
    )
    ax.set_title("Inversion Test Data")
    y_EQ = data["ori_dhat_data"]["dhatE"][np.argwhere(data["t"] >= t_EQ)[0]]
    y_AT = data["ori_dhat_data"]["dhatE"][np.argwhere(data["t"] >= t_AT)[0]]
    y_EX = data["ori_dhat_data"]["dhatE"][
        np.argwhere(data["t"] >= t_EX[0][0])[0]
    ]
    ax.annotate(
        text=(
            "Earthquake Displacement\n"
            + f"{matrix[7]:.2f} mm ({syn_comps[5][0]:.1f})\n"
            + "(step + exp. decay)"
        ),
        xy=(syn_comps[6][0], y_EQ),
        xytext=(2004, -20),
        horizontalalignment="center",
        verticalalignment="top",
        arrowprops=dict(arrowstyle="->"),
    )
    ax.annotate(
        text=(
            "Antenna Offset\n"
            + f"{matrix[6]:.2f} mm ({syn_comps[7][0]:.1f})\n"
            + "(step)"
        ),
        xy=(syn_comps[8][0], y_AT),
        xytext=(2011, 19),
        horizontalalignment="center",
        verticalalignment="top",
        arrowprops=dict(arrowstyle="->"),
    )
    ax.annotate(
        text=(
            "Water Extraction\n"
            + f"{matrix[9]:.2f} mm ({syn_comps[9][0]:.1f}), {syn_comps[10][0][0]}-{syn_comps[10][0][1]}\n"
            + "(logistic)"
        ),
        xy=(syn_comps[10][0][0], y_EX),
        xytext=(2003, 10),
        horizontalalignment="center",
        arrowprops=dict(arrowstyle="->"),
    )

    inputs = (
        f"linear trend: {syn_comps[0]:.1f}$\\rightarrow$ {matrix[1]:.2f}\n"
        + f"annual sine: {syn_comps[1]:.1f}$\\rightarrow$ {matrix[2]:.2f}\n"
        + f"annual cosine: {syn_comps[2]:.1f}$\\rightarrow$ {matrix[3]:.2f}\n"
        + f"semi-annual sine: {syn_comps[3]:.1f}$\\rightarrow$ {matrix[4]:.2f}\n"
        + f"semi-annual cosine: {syn_comps[4]:.1f}$\\rightarrow$ {matrix[5]:.2f}\n"
        + f"antenna offset: {syn_comps[7][0]:.1f}$\\rightarrow$ {matrix[6]:.2f}\n"
        + f"earthquake offset: {syn_comps[5][0]:.1f}$\\rightarrow$ {matrix[7]:.2f}\n"
        + f"water extraction: {syn_comps[9][0]:.1f}$\\rightarrow$ {matrix[9]:.2f}\n"
    )
    ax.annotate(
        inputs,
        (0.99, 0.01),
        xycoords="axes fraction",
        horizontalalignment="right",
        verticalalignment="bottom",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Displacement (mm)")
    fig.tight_layout()
    fig.savefig(
        Path(
            "~/HESSENBOX-DA/Umwelt_4_privat/2023-07 Treffen/beispiel_zeitreihe.pdf"
        ).expanduser()
    )
    plt.show()


if __name__ == "__main__":
    invert_test_data()
