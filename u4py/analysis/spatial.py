import numpy as np


def get_features(in_dict: dict, features: list):
    """Gets specied features from input dictionary"""
    ind = []
    for feat in features:
        ind.extend(np.argwhere(in_dict["fclass"] == feat))
    coords = np.squeeze(in_dict["geometry"][ind])
    names = [
        str((n.encode("latin_1")).decode("utf8"))
        for n in np.squeeze(in_dict["name"][ind])
    ]

    return (names, coords)
