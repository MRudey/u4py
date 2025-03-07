"""
Illustrates the principle of the inversion algorithm
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import u4py.analysis.inversion as u4invert


def main():
    cm = 1 / 2.54
    syn_comps = [
        3,
        1,
        2,
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
    annotations = {
        "lin": "Linear",
        "ann_sin": "Saisonal Sinus",
        "ann_cos": "Saisonal Cosinus",
        "sem_sin": "Halbjährl. Sinus",
        "sem_cos": "Halbjährl. Cosinus",
        "d_noise": "Rauschen",
        "f_at": "Abrupte Bewegung",
        "f_eq": "Erdbeben",
        "f_ex": "Transiente",
    }
    test_data = u4invert.create_synthetic_data(*syn_comps)
    t = test_data[0]["t"]
    ind_g_funcs = test_data[-1]
    ind_g_funcs["lin"] -= np.min(ind_g_funcs["lin"])
    cumulative = np.ones_like(t)
    for ii, kk in enumerate(ind_g_funcs.keys()):
        fig, axes = plt.subplots(
            nrows=2, sharex=True, figsize=(11.22 * cm, 10.88 * cm), dpi=150
        )
        cumulative += ind_g_funcs[kk]
        axes[0].plot(t, cumulative, label=kk)
        axes[1].plot(t, ind_g_funcs[kk], label=kk)
        axes[0].set_ylabel("Gesamtbewegung\n(mm)")
        axes[1].set_ylabel("Anteil Komponente\n(mm)")
        axes[-1].set_xlabel("Zeit")
        fig.suptitle(annotations[kk])
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                Path(
                    r"~\nextHessenbox\Umwelt_4_privat\2023-12 HLNUG Kolloq"
                ).expanduser(),
                f"{kk}.png",
            )
        )


if __name__ == "__main__":
    main()
