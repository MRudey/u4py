import os

import arcpy.management
from tqdm import tqdm

import u4py.utils.files as u4files


def main():
    base_folder = u4files.get_folder_paths()
    folder_list = [
        os.path.join(base_folder, f)
        for f in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, f))
        and not f.endswith(".gdb")
    ]
    fgdb_name = "DiffPlan_Corrected_Mosaics.gdb"
    fgdb_path = os.path.join(base_folder, fgdb_name)
    if not os.path.exists(fgdb_path):
        arcpy.management.CreateFileGDB(base_folder, fgdb_name)

    for folder_path in tqdm(folder_list):
        add_raster(folder_path, fgdb_path)


def add_raster(folder_path: os.PathLike, workspace):
    base_path, dataset_name = os.path.split(folder_path)
    try:
        Output_Mosaic_Dataset = arcpy.management.CreateMosaicDataset(
            in_workspace=workspace,
            in_mosaicdataset_name=dataset_name,
            coordinate_system=arcpy.SpatialReference(32632),
        )[0]
    except arcpy.ExecuteError:
        Output_Mosaic_Dataset = os.path.join(workspace, dataset_name)

    arcpy.management.AddRastersToMosaicDataset(
        in_mosaic_dataset=Output_Mosaic_Dataset,
        raster_type="Raster Dataset",
        input_path=[folder_path],
        update_overviews="UPDATE_OVERVIEWS",
        build_pyramids="BUILD_PYRAMIDS",
        calculate_statistics="CALCULATE_STATISTICS",
        build_thumbnails="BUILD_THUMBNAILS",
        estimate_statistics="ESTIMATE_STATISTICS",
    )[0]


if __name__ == "__main__":
    main()
