import os
import datetime

import arcpy
import numpy as np
import h5py

def main():
    base_path = r"C:\Users\Michael Rudolf\Documents\ArcGIS\INSAR_Data"
    arcpy.env.workspace = os.path.join(
        base_path, "BBD_Update_2021_Hessen\BBD_Update_2021_Hessen.gdb"
    )

    all_data = False
    out_path = os.path.join(base_path, "BBD_Exports")
    os.makedirs(out_path, exist_ok=True)
    num_fc = len(arcpy.ListFeatureClasses())

    if all_data:
        # Directly converts FeatureClass to h5 via NumPy
        for ii, fc in enumerate(arcpy.ListFeatureClasses()):
            curi = ii+1
            print(f'Converting {fc}...  {curi} of {num_fc}:')
            print('... load and convert to dictionary.')
            output = feature_to_dict(arcpy.da.FeatureClassToNumPyArray(fc, '*'))
            print('... compress and save to h5.')
            h5path = os.path.join(base_path, fc + ".h5")
            dict_to_hdf5(h5path, output)
    else:
        fc =  "BBD_2021_PSI_Vertikal"
        print(f'Converting {fc}...:')
        print('... load and convert to dictionary.')
        output = feature_to_dict(arcpy.da.FeatureClassToNumPyArray(fc, '*'))
        print('... compress and save to h5.')
        h5path = os.path.join(base_path, fc + ".h5")
        dict_to_hdf5(h5path, output)

    # Converts to shapefiles
    # for ii, fc in enumerate(arcpy.ListFeatureClasses()):
    #     print("Exporting %i/%i" % (ii + 1, len(arcpy.ListFeatureClasses())))
    #     out_fname = os.path.join(out_path, fc)
    #     arcpy.conversion.FeatureClassToShapefile(fc, out_path)

def feature_to_dict(arr):
    """Converts an array containing feature classes to a dictionary

    Args:
        arr (np.array): The array obtained from arcpy.da.FeatureClassToNumPyArray
    """

    num_points = len(arr)

    # Define type of file:
    all_keys = arr.dtype.names
    has_time = True
    if "stack_ID" in all_keys:
        has_time = False
    elif "PS_ID" in all_keys:
        non_time_keys = ["X", "Y", "Z", "PS_ID", "Shape", "OBJECTID"]
        id_key = "PS_ID"
    if "Input" in all_keys:
        non_time_keys = [
            "X",
            "Y",
            "Z",
            "ID",
            "Input",
            "mean_velo_east",
            "mean_velo_vert",
            "var_mean_velo_east",
            "var_mean_velo_vert",
            "OBJECTID",
            "Shape",
        ]
        id_key = "ID"
    # Create time axis
    if has_time:
        key_list = [
            k for k in all_keys if k not in non_time_keys
        ]
        time = np.array([datetime.datetime.strptime(k, 'date_%Y%m%d') for k in key_list])
        num_fields = len(key_list)
        timeseries = np.zeros((num_points, num_fields))
        # key_to_num = dict()
        xx = arr['X']
        yy = arr['Y']
        zz = arr['Z']
        ps_id = arr[id_key]

        for jj, k in enumerate(key_list):
            timeseries[:,jj] = arr[k]
    else:
        for ii in range(num_points):
            xx = arr["X"]
            yy = arr["Y"]
            zz = arr["Z"]
            ps_id = arr["PS_ID"]
            mean_vel = arr["mean_velocity"]
            var_mean_vel = arr["var_mean_velocity"]

    if has_time:
        output = {
            "x": xx,
            "y": yy,
            "z": zz,
            "time": time,
            "ps_id": ps_id,
            "timeseries": timeseries.astype('e')  # Converts floats to half-rp,
        }
    else:
        output = {
            "x": xx,
            "y": yy,
            "z": zz,
            "ps_id": ps_id,
            "mean_vel": mean_vel,
            "var_mean_vel": var_mean_vel,
        }
    return output

def convert_file(file_path):
    """
    Converts the given dbf file into a h5 file. The h5 file only contains the
    necessary information and is zipped with gzip.
    """
    base_path, fname_ext = os.path.split(file_path)
    base_path, _ = os.path.split(base_path)
    fname, _ = os.path.splitext(fname_ext)
    data = dbf_to_dict(file_path)
    h5path = os.path.join(base_path, fname + ".h5")
    dict_to_hdf5(h5path, data)



def dict_to_hdf5(
    h5path, data, compression="gzip", compression_opts=9
):
    """
    Saves contents of dictionary into given h5 file. Dates are converted to
    strings following ISO date formatting.
    """
    with h5py.File(h5path, "w") as h5file:
        for k, v in data.items():
            try:
                if v.dtype == "O":
                    v = np.array([val.isoformat().encode() for val in v])
                h5file.create_dataset(
                    k,
                    data=v,
                    compression=compression,
                    compression_opts=compression_opts,
                )
            except AttributeError:
                h5file.create_dataset(
                    k,
                    data=v,
                )
            except TypeError:
                h5file.create_dataset(
                    k,
                    data=v,
                )


if __name__ == "__main__":
    main()
