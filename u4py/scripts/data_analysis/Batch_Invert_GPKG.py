"""
Inverts all timeseries in a GPKG file and outputs the results as a pkl file.
"""

import os
import pickle

import u4py.analysis.processing as u4proc
import u4py.io.sql as u4sql
import u4py.utils.projects as u4proj


def main():
    overwrite_intermediate = False
    overwrite_results = False
    project = u4proj.get_project(
        required=["base_path", "psi_path", "output_path"]
    )

    # Set Paths
    input_file = os.path.join(
        project["paths"]["psi_path"], "hessen_l3_clipped.gpkg"
    )
    intermediate_file = os.path.join(
        project["paths"]["output_path"], "extracted_hessen_l3_data.pkl"
    )
    output_file = os.path.join(
        project["paths"]["output_path"], "all_inversion_results.pkl"
    )

    # Loads data from intermediate storage
    if os.path.exists(intermediate_file) and not overwrite_intermediate:
        with open(intermediate_file, "rb") as pkl_file:
            data = pickle.load(pkl_file)
    else:
        data = u4sql.load_tables(input_file)
        with open(intermediate_file, "wb") as pkl_file:
            pickle.dump(data, pkl_file)

    # Start processing
    if not os.path.exists(output_file) or overwrite_results:
        extracts = u4proc.get_extracts(data)

        # Parallel processing
        results = u4proc.batch_mapping(
            extracts, u4proc.inversion_map_worker, "Inverting Extracts"
        )

        # Single processing
        # results = [u4proc.inversion_map_worker(ext) for ext in tqdm(extracts)]

        with open(output_file, "wb") as pkl_file:
            pickle.dump(results, pkl_file)


if __name__ == "__main__":
    main()
