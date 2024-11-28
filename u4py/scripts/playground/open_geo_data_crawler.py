import os
import shutil
import time
import zipfile
from pathlib import Path
from urllib import error, request

import numpy as np
import osgeo.gdal as gdal
from tqdm import tqdm

gdal.UseExceptions()


def main():
    base_url = "https://opengeodata.lgl-bw.de/data/dgm/"
    local_folder = "/mnt/Raid/Umwelt4/DGM1_BW/zips"
    tiff_path = "/mnt/Raid/Umwelt4/DGM1_BW/tiffs"
    os.makedirs(local_folder, exist_ok=True)
    os.makedirs(tiff_path, exist_ok=True)

    # Minimum coordinates
    min_east = 387
    max_east = 609
    min_north = 5264
    max_north = 5514

    # Build list of possible file names
    eastings = np.arange(min_east, max_east, 2)
    northings = np.arange(min_north, max_north, 2)
    file_list = []
    for easting in eastings:
        for northing in northings:
            file_list.append(f"dgm1_32_{easting}_{northing}_2_bw.zip")

    # Crawl data and convert to tiff
    opener = request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    request.install_opener(opener)

    for file_name in tqdm(file_list):
        local_path = os.path.join(local_folder, file_name)
        if not os.path.exists(local_path):
            retrieve_file(base_url + file_name, local_path)


def retrieve_file(target_url, local_path):
    time.sleep(0.5)
    try:
        request.urlretrieve(target_url, local_path)
        return True
    except error.HTTPError:
        return False


if __name__ == "__main__":
    main()
