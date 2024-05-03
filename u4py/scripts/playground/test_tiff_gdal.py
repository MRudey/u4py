"""
Testing the outputs of gdal tiff processing
"""

import matplotlib.pyplot as plt

import u4py.io.tiff as u4tiff


def main():
    test_path = "/mnt/Raid/Umwelt4/DGM1_2021/raster_2021/g_530_5575"

    slope = u4tiff.get_terrain(test_path)
    hillshade = u4tiff.get_terrain(test_path, terrain_feature="hillshade")
    multi_hillshade = u4tiff.get_terrain(
        test_path, terrain_feature="multi_hillshade"
    )
    aspect = u4tiff.get_terrain(test_path, terrain_feature="aspect")


if __name__ == "__main__":
    main()
