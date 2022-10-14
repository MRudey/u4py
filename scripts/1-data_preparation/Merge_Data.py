import os
from multiprocessing import Pool, cpu_count

import u4py.analysis.processing as u4process
import u4py.utils.convert as u4convert
import u4py.utils.files as u4files
from tqdm import tqdm


def main():
    base_path = u4files.get_folder_paths()
    common_files = u4process.get_common_files(base_path, "BBD_EW", "BBD_Vert")
    inputs = [(base_path, fn) for fn in common_files]
    os.makedirs(os.path.join(base_path, "merged"), exist_ok=True)
    # u4convert.merge_data(inputs[0])
    with Pool(cpu_count() - 2) as p:
        list(
            tqdm(
                p.imap_unordered(u4convert.merge_data, inputs),
                total=len(inputs),
                desc="Merging files",
                leave=False,
            )
        )


if __name__ == "__main__":
    main()
