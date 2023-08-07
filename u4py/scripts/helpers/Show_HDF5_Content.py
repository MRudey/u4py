""" Shows the content of a h5-file as a tree in the terminal"""

import os

import h5py
from ete3 import Tree

import u4py.utils.files as u4files


def main():
    # /mnt/Raid/Umwelt4/Converted_gpkg/Insar_chunks/L3/merged/PSI_chunk_x412750_y5545000.h5
    file_path = u4files.get_file_paths(filetypes=(("*.h5", "*.h5"),))
    show_h5_contents(file_path)


def show_h5_contents(file_path: os.PathLike):
    """Displays the content of a h5file in a structured format"""
    with h5py.File(file_path, "r") as exp:
        tree_string = extract_keys_asTree(exp)
        tree = Tree(tree_string + ";", format=1)
    print(tree.get_ascii(show_internal=True))


def extract_keys_asTree(f: h5py.File) -> str:
    """Generates a string for use with the ete3 TreeObject"""
    tree_string = "("
    for key in f.keys():
        add = ""
        if isinstance(f[key], h5py.Group):
            tree_string += extract_keys_asTree(f[key])
        else:
            typ = dtype_to_str(str(f[key].dtype))
            shp = shp_to_str(f[key].shape)

            add = f"({shp}->{typ})"
        tree_string += f"{add}--{key},"
    tree_string = tree_string[0:-1] + ")"  # Replaces trailing comma with ')'
    return tree_string


def shp_to_str(shp: tuple) -> str:
    """Formats the shape"""

    shp_str = "1"
    for s in shp:
        shp_str = f"{str(s)}x{shp_str}"
    return f"({shp_str})"


def dtype_to_str(dtype_str: str) -> str:
    """Returns a more meaningful data type string"""

    dtypes = {"object": "string", "|S19": "datetime"}
    try:
        output = dtypes[dtype_str]
    except KeyError:
        output = dtype_str
    return output


if __name__ == "__main__":
    main()
