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

import logging
import random
import sys
from typing import Tuple

import numpy as np
import scipy.linalg as splinalg
import scipy.sparse as spsparse
import scipy.stats as spstats

# import tensorflow as tf
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
        32,
    ],
    t_EQdummy: list = [
        2010.1561643,
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
    :param t_EQdummy: A list of earthquake times matching `eq_disp`, defaults to [ 2010.1561643, ]
    :type t_EQdummy: list, optional
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
    t = np.arange(2007, 2015, 0.003)  # sample time
    d_noise = spstats.norm(1.5).rvs(len(t))  # noise

    f_Heavi = np.zeros_like(t)  # Heaviside function
    f_eq = np.zeros_like(t)
    for eqd, teq in zip(eq_disp, t_EQdummy):
        f_Heavi[t > teq] = f_Heavi[t > teq] + eqd / 8  # EQ at t_EQdummy
        f_post = np.real(3 * f_Heavi * np.log((1.0 + 0.0j) + (t - teq)))
        f_eq += -8 * f_Heavi + f_post

    # Create synthetic data and uncertainties
    dataE = np.real(
        t * lin
        + ann_sin * np.sin(2 * np.pi * t)
        + ann_cos * np.cos(2 * np.pi * t)
        + sem_sin * np.sin(4 * np.pi * t)
        + sem_cos * np.cos(4 * np.pi * t)
        + f_eq
        + d_noise
    )

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
    return data, t_EQdummy


def invert_time_series(
    data: dict,
    t_AT: list = [],
    t_EQ: list = [],
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

    The full G-Matrix contains a linear trend, offset, oscillation and a
    postseismic signal:

        | :math:`A_1 + A2\\cdot(t-t_R) + A_3 \\cdot (t-t_R)^2 + \\dots`
        | :math:`B_1 \\cdot sin(2 \\pi t) + B_2 \\cdot cos(2 \\pi t) + \\dots`
        | :math:`B_3 \\cdot sin(4 \\pi t) + B_4 \\cdot cos(4 \\pi t) + \\dots`
        | :math:`C_1 \\cdot tH(t_{AT}) + C_2 \\cdot tH(t_{EQ}) + \\dots`
        | :math:`D_1 \\cdot log(1+\\frac{t}{dT_1} + D_2 \\cdot log(1+\\frac{t}{dT_2})`

    In this equation, the following model parameters are included:

    - :math:`A_1 \\dots A_3`: General offset and steady rate (can be steady, linear , for interseismic deformation, or quadratic, for GIA adjustment)
    - :math:`B_1 \\dots B_4`: Amplitudes of annual and semi-annual oscillations
    - :math:`C_1 \\dots C_2`: Amplitude of Heaviside function (:math:`d_{EQ}` or :math:`d_{AT}`) starting at :math:`t_{AT}` or :math:`t_{EQ}`.
    - :math:`D_1 \\dots D_2`: Amplitude of postseismic signal starting at :math:`t_{EQ}`.

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

    g_functions = _prepare_g_functions(
        data, ind, t_AT, t_EQ, num_coeffs, t_relative
    )
    if use_sparse:
        g_matrix = spsparse.csr_matrix(_set_g_matrices(*g_functions))
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
    return matrix, data, time_vector


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
    num_coeffs: int = 2,
    t_relative: float = 0,
) -> Tuple[np.ndarray]:
    """Prepares the Green's functions.

    :param data: The data formatted as a dictionary.
    :type data: dict
    :param ind: Slice to use only a certain time window, defaults to 0
    :type ind: slice
    :param t_AT: A list with times of known antenna offsets, defaults to []
    :type t_AT: list, optional
    :param t_EQ: A list with times of known earthquakes, defaults to []
    :type t_EQ: list, optional
    :param num_coeffs: The number of parameters to use for g_TREND, defaults to 2 (linear and square)
    :type num_coeffs: int, optional
    :param t_relative: A time offset used mainly for plotting, defaults to 0
    :type t_relative: float, optional
    :return: The Green's functions as a stacked matrix.
    :rtype: Tuple[np.ndarray]
    """
    logging.info("Setting Mini-g-function == TREND")
    # Mini-g-function == TREND ============ 1-n parameters ==== A1 to Ax ====
    g_TREND = [
        (data["t"][ind] - t_relative) ** ii for ii in range(num_coeffs + 1)
    ]

    logging.info("Setting Mini-g-function == ANNUAL")
    # Mini-g-function == ANNUAL SIGNAL ==== 4 parameters ===== B1 to B4 ===
    g_ANNUAL = [
        np.sin(2 * np.pi * data["t"][ind]),
        np.cos(2 * np.pi * data["t"][ind]),
        np.sin(4 * np.pi * data["t"][ind]),
        np.cos(4 * np.pi * data["t"][ind]),
    ]

    logging.info("Setting Mini-g-function == HEAVISIDE")
    # Mini-g-function == HEAVISIDE ======== 1-2 parameters ===== C1, C2 ===

    # a) HEAVISIDE AT -- Find offsets in the AT-list
    if not t_AT:  # No Heaviside function, if no offset exists
        g_HEAVIS_AT = []
    else:
        # Create a Heaviside vector (or matrix, if t_AT has more than one entry):
        for ii in range(len(t_AT)):
            g_HEAVIS_AT = np.zeros_like(data["t"][ind])
            g_HEAVIS_AT[data["t"][ind] > t_AT[ii], ii] = (
                g_HEAVIS_AT[data["t"][ind] > t_AT[ii], ii] + 1
            )

    # b) HEAVISIDE EQ -- get offsets from external function (above)
    if (
        not t_EQ
        or (np.max(data["t"][ind]) < np.min(t_EQ))
        or (np.min(data["t"][ind]) > np.max(t_EQ))
    ):  # No Heaviside function, if no offset exists
        g_HEAVIS_EQ = []

    else:
        # If you would like to simulate Maule postseismics for data, which
        # only starts AFTER the Maule EQ:

        # if t_EQ[0] > t_MA:
        #     t_EQ = [t_MA, t_EQ]

        if t_EQ[0] < data["t"][0]:
            t_EQ[0] = data["t"][0]

        # Here I check if several earthquakes occurred within the gap of a
        # time-series. If yes, I remove the first earthquakes and keep only
        # the last event in this data gap.
        # spaceind = [
        #     len(np.argwhere(data["t"] > t_EQ[k]) and data["t"] < t_EQ[k + 1]))
        #     for k in range(len(t_EQ))
        # ]
        # t_EQ[spaceind == 0] = []

        # Create a Heaviside vector (or matrix, if t_EQ has more than one entry):
        g_HEAVIS_EQ = []
        for ii in range(len(t_EQ)):
            g_heq = np.zeros_like(data["t"][ind])
            g_heq[data["t"][ind] > t_EQ[ii]] = (
                g_heq[data["t"][ind] > t_EQ[ii]] + 1
            )
            g_HEAVIS_EQ.append(g_heq)

    logging.info("Setting Mini-g-function == POSTS")
    # Mini-g-function == POSTS. SIGNAL === 1- parameters ===== D1 - Dx  ===
    # Attention!!!:
    # The postseismic signal is A-PRIORI-LINEARIZED by assuming dT = 1
    # (Bevis & Brown, 2014). If you want to really study the geophysical
    # properties of the post-seismic signal, you have to estimate dT using a
    # non-linear approximation for each station, or group of stations later
    # on!!!!

    if not t_EQ or len(data["t"]) < 101:
        g_POSTSM = []
    else:
        g_POSTSM = np.zeros_like(data["t"])
        for te in t_EQ:
            iB = data["t"][ind] > te
            g_POSTSM[iB] = g_POSTSM[iB] + np.log(1 + (data["t"][iB] - te))
    return (
        np.asarray(g_TREND),
        np.asarray(g_ANNUAL),
        np.asarray(g_HEAVIS_AT),
        np.asarray(g_HEAVIS_EQ),
        np.asarray(g_POSTSM),
    )


def _set_g_matrices(
    g_TREND, g_ANNUAL, g_HEAVIS_AT, g_HEAVIS_EQ, g_POSTSM
) -> np.ndarray:
    """Set up a G-matrices for each component and then the full, global G-matrix.

    :return: Block diagonal matrix of stacked Green's function for each component.
    :rtype: np.ndarray
    """
    logging.info("Setting up G-Matrices")
    gE = np.array([])
    saved_locals = locals()
    for k in saved_locals:
        if not gE.any():
            gE = saved_locals[k]
            gN = saved_locals[k]
            gU = saved_locals[k]
        elif k.startswith("g_") and saved_locals[k].any():
            gE = np.vstack((gE, saved_locals[k]))
            gN = np.vstack((gN, saved_locals[k]))
            gU = np.vstack((gU, saved_locals[k]))
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


def print_inversion_results(matrix: np.ndarray):
    """Nicely prints the results from the inversion.

    :param matrix: The inverted model parameters.
    :type matrix: np.ndarray
    """

    directions = [
        "--- East-West ---",
        "--- North-South ---",
        "--- Up-Down ---",
    ]

    names = [
        " [0] yaxis-offset (year=0!):",
        " [1]           linear trend:",
        " [2]       semi-annual sine:",
        " [3]     semi-annual cosine:",
        " [4]            annual sine:",
        " [5]          annual cosine:",
        " [6]              eq offset:",
        " [7]           post-seismic:",
    ]
    try:
        matr_resh = np.reshape(matrix, (3, int(len(matrix) / 3)))
        for ii, direct in enumerate(matr_resh):
            print(directions[ii])
            for jj, val in enumerate(direct):
                print(f"{names[jj]} {val:2f}")
    except TypeError:
        raise TypeError("Something is wrong with the solution matrix.")


def reformat_dict(dataset: dict) -> dict:
    """Reformats a loaded hdf5 dictionary to match with the one for inversion.

    :param dataset: The dictionary to reformat.
    :type dataset: dict
    :return: The reformatted dictionary.
    :rtype: dict
    """
    data = stack_data(dataset)
    data["sigmE"] = np.ones_like(data["dataE"])
    data["sigmN"] = np.ones_like(data["dataE"])
    data["sigmU"] = np.ones_like(data["dataE"])

    if "inversion_results" in dataset.keys():
        data["inversion_results"] = dataset["inversion_results"]

    return data


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
            32,
        ],
        [
            2010.1561643,
        ],
    ]
    data, t_EQ = create_synthetic_data(*syn_comps)
    matrix, data, time_vector = invert_time_series(
        data, t_EQ=t_EQ, use_sparse=True
    )
    inversion_results = {"matrix_ori": matrix}
    print_inversion_results(matrix)
    u4plots.plot_inversion_results(
        time=time_vector, data=data, inversion_results=inversion_results
    )


if __name__ == "__main__":
    invert_test_data()
