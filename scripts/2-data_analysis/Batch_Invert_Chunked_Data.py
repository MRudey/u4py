import os

import u4py.analysis.processing as u4process
import u4py.utils.files as u4files


def main():
    file_list = u4files.get_file_list(
        folder_path="/mnt/Raid/Umwelt_4/INSAR_chunks/merged"
    )
    output_path, _ = os.path.split(os.path.split(file_list[0])[0])
    results, chunk_size = u4process.get_inversion_results(file_list)
    print(chunk_size)


if __name__ == "__main__":
    main()
