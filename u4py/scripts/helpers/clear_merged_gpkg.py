from pathlib import Path

import geopandas as gp
import numpy as np
from tqdm import tqdm


def main():
    file_path = Path(
        r"~\Documents\ArcGIS\SelectedSites_August24\Filtered_Classified_Shapes_all_merged_unclean.gpkg"
    ).expanduser()
    gdf: gp.GeoDataFrame
    print("Loading File")
    gdf = gp.read_file(file_path, layer="Filtered_Classified_Shapes")
    for ii, kk in tqdm(enumerate(gdf.manual_class_1), desc="Cleaning indices"):
        if kk:
            if "[" in kk:
                man_classes = list(np.unique(eval(kk)))
                man_comments = list(
                    np.unique(eval(gdf.manual_comment.iat[ii]))
                )
                if len(man_comments) > 1:
                    gdf.manual_comment.iat[ii] = " und ".join(man_comments)
                else:
                    gdf.manual_comment.iat[ii] = man_comments[0]

                if len(man_classes) > 2:
                    gdf.manual_class_3.iat[ii] = man_classes[2]
                else:
                    gdf.manual_class_3.iat[ii] = "(empty)"
                if len(man_classes) > 1:
                    gdf.manual_class_2.iat[ii] = man_classes[1]
                else:
                    gdf.manual_class_2.iat[ii] = "(empty)"
                if len(man_classes) > 0:
                    gdf.manual_class_1.iat[ii] = man_classes[0]
                else:
                    gdf.manual_class_1.iat[ii] = "(empty)"
                man_unc = np.sum(eval(gdf.manual_unclear_1.iat[ii]))
                if man_unc > 0:
                    gdf.manual_unclear_1.iat[ii] = 1
                else:
                    gdf.manual_unclear_1.iat[ii] = 0
                gdf.manual_unclear_2.iat[ii] = 0
                gdf.manual_unclear_3.iat[ii] = 0
                man_res = np.sum(eval(gdf.manual_research.iat[ii]))
                if man_res > 0:
                    gdf.manual_research.iat[ii] = 1
                else:
                    gdf.manual_research.iat[ii] = 0
                man_known = np.sum(eval(gdf.manual_known.iat[ii]))
                if man_known > 0:
                    gdf.manual_known.iat[ii] = 1
                else:
                    gdf.manual_known.iat[ii] = 0
    print("Saving File")
    gdf.to_file(file_path.replace("_unclean.gpkg", "_clean.gpkg"))


if __name__ == "__main__":
    main()
