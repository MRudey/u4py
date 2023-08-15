import matplotlib.pyplot as plt

import u4py.analysis.spatial as u4spatial
import u4py.plotting.formatting as u4pltfmt
import u4py.utils.config as u4config
import u4py.utils.files as u4files

u4config.start_logger()


def main():
    # merged_file_path = r"D:\Projekte\Umwelt4\DGM-Differenzenplan_MitKorrektur\TB2\dgm1_32_461_5645_1_he_Kor_Diff.tif"
    merged_file_path = r"D:\Projekte\Umwelt4\DGM-Differenzenplan_MitKorrektur\clipped_tiffs_2Kassel_merged.tif"

    threshold = 25
    levels = u4spatial.plus_minus_levels([0.5, 1, 2, 5, 10])
    gdf = u4files.get_thresholded_contours(merged_file_path, levels, threshold)

    fig, ax = plt.subplots()
    gdf.plot(ax=ax)
    u4pltfmt.map_style(ax)
    plt.show()


if __name__ == "__main__":
    main()
