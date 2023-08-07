"""
Illustrates the principle of the inversion algorithm
"""

import matplotlib.pyplot as plt
import numpy as np

import u4py.analysis.inversion as u4invert


def main():
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
    test_data = u4invert.create_synthetic_data(*syn_comps)
    t = test_data[0]["t"]
    ind_g_funcs = test_data[-1]
    fig, axes = plt.subplots(
        nrows=len(ind_g_funcs), sharex=True, figsize=(7, 5)
    )
    for ii, kk in enumerate(ind_g_funcs.keys()):
        axes[ii].plot(t, ind_g_funcs[kk])
        axes[ii].annotate(kk, (1.01, 0.5), xycoords="axes fraction")
        axes[ii].set_yticks([np.min(ind_g_funcs[kk]), np.max(ind_g_funcs[kk])])
    for ax in axes:
        for sp in ["top", "bottom", "left", "right"]:
            ax.spines[sp].set_linewidth(0.25)
    axes[-1].set_xlabel("Time")
    fig.tight_layout(pad=0)
    fig.savefig(r"~\HESSENBOX-DA\Umwelt_4_privat\2023-07 Treffen\lin_comb.pdf")


if __name__ == "__main__":
    main()
