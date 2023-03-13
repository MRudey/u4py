"""
Searches data in the given input folder and inverts the data if not present yet.

#Parallelized
#SLURM
"""
import u4py.analysis.processing as u4process
import u4py.utils.cmd_args as u4cmds
import u4py.utils.config as u4config
import u4py.utils.files as u4files


def main():
    u4cmds.load(module_descript=__doc__)
    file_list = u4files.get_file_list(folder_path=u4config.in_path)
    u4process.get_inversion_results(file_list)


if __name__ == "__main__":
    main()
