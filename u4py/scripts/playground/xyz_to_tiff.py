import os
import shutil
import zipfile

import osgeo.gdal as gdal

import u4py.analysis.processing as u4proc
import u4py.utils.cmd_args as u4args

gdal.UseExceptions()


def main():
    u4args.load()
    local_folder = "/mnt/Raid/Umwelt4/DGM1_BW/zips"
    tiff_path = "/mnt/Raid/Umwelt4/DGM1_BW/tiffs"
    os.makedirs(local_folder, exist_ok=True)
    os.makedirs(tiff_path, exist_ok=True)

    args = [
        (os.path.join(local_folder, fp), tiff_path)
        for fp in os.listdir(local_folder)
        if fp.endswith(".zip")
    ]
    u4proc.batch_mapping(args, batch_convert, desc="Converting XYZ to Tiff")


def batch_convert(args):
    return convert_file(*args)


def convert_file(file_path, tiff_path):
    with zipfile.ZipFile(file_path, "r") as zip_file:
        temp_path = "/mnt/Raid/Umwelt4/DGM1_BW/temp"
        os.makedirs(temp_path, exist_ok=True)
        zip_file.extractall(temp_path)
        fname = os.path.split(os.path.splitext(file_path)[0])[-1]
        data_folder = os.path.join(temp_path, fname)
        xyz_list = [
            os.path.join(data_folder, fp)
            for fp in os.listdir(data_folder)
            if fp.endswith(".xyz")
        ]
        for xyz_path in xyz_list:
            gdal.Translate(
                os.path.join(tiff_path, fname + ".tiff"),
                xyz_path,
                outputSRS="EPSG:25832",
                creationOptions="COMPRESS=LZW",
                stats=True,
            )
        shutil.rmtree(data_folder)


if __name__ == "__main__":
    main()
