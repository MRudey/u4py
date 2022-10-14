import numpy as np
import scipy.linalg as splinalg
import scipy.stats as spstats


def create_synthetic_data():
    """Test data for inversion"""
    t = np.arange(2007, 2015, 0.003)  # sample time
    t_EQdummy = 2010.1561643  # simulated Earthquake
    d_noise = spstats.norm(1.5).rvs(len(t))  # noise

    f_Heavi = np.zeros_like(t)  # Heaviside function
    f_Heavi[t > t_EQdummy] = f_Heavi[t > t_EQdummy] + 4  # EQ at t_EQdummy

    # Create synthetic data and uncertainties
    dataE = np.real(
        t * 1.1
        + 0.6 * np.sin(2 * np.pi * t)
        + 0.8 * np.cos(2 * np.pi * t)
        - 0.1 * np.sin(4 * np.pi * t)
        + 0.05 * np.cos(4 * np.pi * t)
        - 8 * f_Heavi
        + 3 * f_Heavi * np.log((1.0 + 0.0j) + (t - t_EQdummy))
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
    return data, [
        t_EQdummy,
    ]


def invert_time_series(
    data, t_AT=[], t_EQ=[], ind=0, num_coeffs=1, t_relative=0
):
    """Inverts the timeseries

    ### Inversion Theory
    NB: The problem must be linear, otherwise you cannot invert!

    (d: data, G: solver matrix, m: model parameters)

      d = G * m            =>      m = np.linalg.inv(G'*G) * G' * d

    and, using weights:

      W * d = W * G * m    ==>     m = np.linalg.inv(G' * np.linalg.inv(S) * G) * G' * np.linalg.inv(S) * d

    with W'* W = np.linalg.inv(S), or, simplified: S = s^2


    ### SET UP THE G-MATRIX
    The full G-Matrix contains a linear trend, offset, oscillation and a
    postseismic signal:

    A1 + A2*(t-tR)     + A3 * (t-tR)^2       + ...
    B1 * sin(2*pi*t)   + B2 * cos(2*pi*t)    + ...
    B3 * sin(4*pi*t)   + B4 * cos(4*pi*t)    + ...
    C1 * tH(ant.off.)  + C2 * tH(EQ.offs.)   + ...
    D1 * log(1+t/dT1)  + D2 * log(1+t/dT2)   + ...

    In this equation, the following model parameters are included:

    A1-...  - General offset and steady rate (can be steady, linear , for
              interseismic deformation, or quadratic, for GIA adjustment)
    B1-...  - Amplitudes of annual and semi-annual oscillations
    C1-...  - Amplitude of Heaviside function (EQ_offsets/AT_offsets)
              starting at t_eq or t_at
    D1-...  - Amplitude of postseismic signal starting at t_EQ

    The G-Matrix needs to be individualized for each station, depending if
    it is affected by these signals. You should create a GLOBAL G-Matrix
    including ALL stations and not solve this for each single station. You do
    that by concatenating all stations and components into one big G-Matrix.
    In this example, I only show how it is done simultaneously for the three
    components of one station.
    """

    if not ind:
        ind = slice(len(data["t"]))
        ori = True
    else:
        ori = False

    g_functions = prepare_g_functions(
        data, ind, t_AT, t_EQ, num_coeffs, t_relative
    )
    g_matrix = set_g_matrices(*g_functions)

    # The G-matrix has now the following dimension:
    # Length: #samples * #stations * #components
    # Width:  #parameters

    # Set up the DATA vector D, TIME vector T and SIGMA-matrix S
    # Dimension:
    #     (#samples * #components * #stations) x 1
    data_vector = np.hstack(
        (data["dataE"][ind], data["dataN"][ind], data["dataU"][ind])
    )

    # Define SIGMA-Matrix
    # In the best case this should be a full-matrix with correlated errors. In
    # reality, you often only have the diagonal components. Use SPARSE to
    # reduce the size of the matrix.
    # Dimension:
    #    (#samples * #components * #stations) x
    #    (#samples * #components * #stations)
    sigma_matrix = np.diag(
        np.hstack(
            (
                data["sigmE"][ind] ** 2,
                data["sigmN"][ind] ** 2,
                data["sigmU"][ind] ** 2,
            )
        )
    )

    # Define TIME-vector
    # Dimension:
    #     (#samples * #components * #stations) x 1
    time_vector = [data["t"][ind], data["t"][ind], data["t"][ind]]

    matrix = invert(g_matrix, data_vector, sigma_matrix)
    data = forward_model(matrix, g_matrix, data, ind, ori=ori)
    return matrix, data, time_vector


def invert(G, D, S=np.array([])):
    """Invert for your model parameters
    Dimension: #parameters x 1
    """
    Gp = G.conj().transpose()
    if S.any():
        iS = np.linalg.inv(S)
        iM1 = np.linalg.inv(np.matmul(np.matmul(G, iS), Gp))
        M = np.matmul(np.matmul(np.matmul(iS, D), Gp), iM1)
    else:
        iM1 = np.linalg.inv(np.matmul(Gp, G))
        M = np.matmul(np.matmul(iM1, D), Gp)

    return M


def forward_model(M, G, data, ind, ori=True):
    """Forward model the data using the model parameters:
    Separation of the big DHAT-vector into different components
    """
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


def prepare_g_functions(
    data, ind, t_AT=[], t_EQ=[], num_coeffs=1, t_relative=0
):
    """Prepares the Green's functions

    Args:
        data (_type_): _description_
        ind (_type_): _description_
        t_AT (list, optional): _description_. Defaults to [].
        t_EQ (list, optional): _description_. Defaults to [].
        num_coeffs (int, optional): _description_. Defaults to 1.
        t_relative (int, optional): _description_. Defaults to 0.
    """
    # Mini-g-function == TREND ============ 1- parameters ==== A1 to Ax ====
    g_TREND = [
        (data["t"][ind] - t_relative) ** ii for ii in range(num_coeffs + 1)
    ]

    # Use annual oscillation only for continuous data
    if len(data["dataE"]) > 360:
        # Mini-g-function == ANNUAL SIGNAL ==== 4 parameters ===== B1 to B4 ===
        g_ANNUAL = [
            np.sin(2 * np.pi * data["t"][ind]),
            np.cos(2 * np.pi * data["t"][ind]),
            np.sin(4 * np.pi * data["t"][ind]),
            np.cos(4 * np.pi * data["t"][ind]),
        ]
    else:
        g_ANNUAL = []

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

    # Mini-g-function == POSTS. SIGNAL === 1- parameters ===== D1 - Dx  ===
    # Attention!!!:
    # The postseismic signal is A-PRIORI-LINEARIZED by assuming dT = 1
    # (Bevis & Brown, 2014). If you want to really study the geophysical
    # properties of the post-seismic signal, you have to estimate dT using a
    # non-linear approximation for each station, or group of stations later
    # on!!!!

    if not t_EQ or len(data["t"]) < 101 or np.sum(t_EQ == 2010.1562) == 0:
        g_POSTSM = []
    else:
        for ii in np.argwhere(t_EQ == 2010.1562):
            iA = data["t"][ind] <= t_EQ[ii]
            iB = data["t"][ind] > t_EQ[ii]
            g_POSTSM[iA, 1] = np.zeros(sum(iA), 1)
            g_POSTSM[iB, 1] = np.log(1 + (data["t"][ind[iB]] - t_EQ[ii]))
    return (
        np.asarray(g_TREND),
        np.asarray(g_ANNUAL),
        np.asarray(g_HEAVIS_AT),
        np.asarray(g_HEAVIS_EQ),
        np.asarray(g_POSTSM),
    )


def set_g_matrices(g_TREND, g_ANNUAL, g_HEAVIS_AT, g_HEAVIS_EQ, g_POSTSM):
    """
    Set up a G-matrices for each component and then the full, global G-matrix
    """
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


def remove_outliers(data, threshold=2.5):
    thrE = threshold * np.std(data["dresE"])
    thrN = threshold * np.std(data["dresN"])
    thrU = threshold * np.std(data["dresU"])
    ind = (
        np.nonzero(np.abs(data["dresE"]) < thrE)
        and np.nonzero(np.abs(data["dresN"]) < thrN)
        and np.nonzero(np.abs(data["dresU"]) < thrU)
    )
    return ind


if __name__ == "__main__":
    main()
