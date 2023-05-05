""" Submodule for managing paths as config files (projects)

"""

import configparser
import logging
import os
import tkinter as tk
from functools import partial
from tkinter import messagebox, ttk

import u4py.utils.files as u4files

global PROJECT


def get_project(
    proj_path: os.PathLike = "",
    required: list[str] = [
        "base_path",
    ],
    interactive: bool = True,
) -> configparser.ConfigParser:
    """Interactive project path loader.

    A project file stores all relevant paths for specific plots. Typically used
    paths are interpolated using `configparser.BasicInterpolation`.

    When a script that uses one of the paths below is run:
    1. It tries to load an existing project file from `proj_path`:
        - either hard coded input, or
        - interactive opening of project file
        - if `None` is given a new project file is created.
    2. After loading the paths are shown in a dialog
        - A keyword argument indicates if path existance should be enforced
        - Required paths are given as a list of path names
        - Markups indicate if a path exists
            - :green: folder/file exists
            - :orange: folder/file does not exist but is not required
            - :red: folder/file does not exist but is required for script
        - When `interactive=False` no dialog is shown
    3. Paths are handed to the script as a `ConfigParser` object.
    """
    global PROJECT
    PROJECT, proj_path = load_project(proj_path=proj_path, required=required)
    _, proj_name = os.path.split(proj_path)
    if interactive:
        path_dialog(
            required=required,
            project_name=proj_name,
            proj_path=proj_path,
        )
    return PROJECT


def load_project(
    proj_path: os.PathLike = "", required: list[str] = []
) -> configparser.ConfigParser:
    """Loads the project from a file or generates a new one based on defaults.

    Keyword Arguments:
        proj_path -- Path to the project file (default: {""})
        required -- List of required keys for project (default: {[]})

    Returns:
        `ConfigParser` of the config file
    """
    project = configparser.ConfigParser()

    if proj_path:
        project.read(proj_path)
        logging.info("Config read.")

    else:
        proj_path = u4files.get_file_paths(
            filetypes=((".u4project", ".u4project"),),
            title="Select project file.",
        )
        if proj_path:
            project.read(proj_path)
            logging.info("Config read.")
        else:
            logging.info("No config found, generating defaults.")
            project = generate_default()
            proj_path = u4files.get_save_path(
                title="Save to project file.",
                defaultextension=".u4project",
                filetypes=((".u4project", ".u4project"),),
                initialdir=project.get("DEFAULT", "u4projects_path"),
                initialfile="test.u4project",
                confirmoverwrite=True,
            )
            project["paths"] = {"base_path": project["DEFAULT"]["base_path"]}
            for r in required:
                project["paths"][r] = project["DEFAULT"][r]
            if proj_path:
                with open(proj_path, "wt") as proj_file:
                    project.write(proj_file)
                    logging.info(f"Config file saved to {proj_file}.")
            else:
                logging.info("Config file not saved.")
    return project, proj_path


def generate_default() -> configparser.ConfigParser:
    """
    Generates the default config
    """
    base_path = u4files.get_folder_paths(title="Select base folder.")
    project = configparser.ConfigParser()
    project["DEFAULT"] = {
        # BASE
        "base_path": base_path,
        # First order
        "ext_path": "%(base_path)s/ExternalData",
        "places_path": "%(base_path)s/Places",
        "output_path": "%(base_path)s/INSAR_plots",
        "psi_path": "%(base_path)s/Data_2023",
        "processing_path": "%(base_path)s/INSAR_results",
        "u4projects_path": "%(base_path)s/U4_Projects",
        # Places paths
        "tektonik_path": "%(places_path)s/tektonik_cropped.shp",
        "bld_path": "%(places_path)s/vg2500_bld.shp",
        "piloten_path": "%(places_path)s/Pilotregionen.shp",
        "base_map_path": "%(places_path)s/hessen_map.tif",
        # PSI paths
        "psivert_path": "%(psi_path)s/BBD_Vert",
        "psiew_path": "%(psi_path)s/BBD_EW",
        # Results paths:
        "results_path": "%(processing_path)s/inversion_results_2023_data.pkl",
    }
    return project


def path_dialog(
    required: list,
    proj_path: os.PathLike,
    project_name: str = "Project Name",
):
    """Shows a path dialog for interactive adjustments

    Arguments:
        project -- A loaded project file
    """
    global PROJECT
    # Set up main frame
    root = tk.Tk()
    root.title(f"Paths for {project_name}")
    mainframe = ttk.Frame(root, padding="3 3 12 12")
    mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    root.attributes("-topmost", True)
    # root.bind("<focusOut>", root.attributes(0))

    # Populate UI with entries
    entries = dict()
    # required = [p for p in project["paths"].keys()]
    for ii, k in enumerate(PROJECT.defaults().keys()):
        entry = dict()
        entry["var"] = tk.StringVar()  # Holds the path string
        entry["var"].set(PROJECT["paths"][k])
        entry["var"].trace_add(
            "write",
            lambda *args, entries=entries, required=required: check_paths(
                entries, required, *args
            ),
        )
        entry["label"] = ttk.Label(mainframe, text=k)  # Label with path_name
        entry["label"].grid(column=1, row=ii + 1, sticky=tk.W)
        entry["entry"] = ttk.Entry(  # The entry
            mainframe, width=75, textvariable=entry["var"]
        )
        entry["entry"].grid(column=2, row=ii + 1, sticky=(tk.W, tk.E))
        entry["button"] = ttk.Button(  # Button for opening path
            mainframe, text="...", command=partial(set_path, entry)
        )
        entry["button"].grid(column=3, row=ii + 1, sticky=tk.W)
        entries[k] = entry

    # Start Button
    entries["start_button"] = ttk.Button(
        mainframe,
        text="Save and Continue...",
        command=partial(save_n_go, root, entries, PROJECT, proj_path),
    )
    entries["start_button"].grid(column=4, row=ii + 2, sticky=tk.W)
    entries["start_button"].focus()

    # Start Button
    entries["save_go_button"] = ttk.Button(
        mainframe,
        text="Save As and Continue...",
        command=partial(save_n_go, root, entries, PROJECT),
    )
    entries["save_go_button"].grid(column=3, row=ii + 2, sticky=tk.W)
    entries["save_go_button"].focus()

    # Check if paths are there for first run
    check_paths(entries, required)

    root.bind(
        "<Return>", partial(save_n_go, root, entries, PROJECT, proj_path)
    )

    # Run UI
    root.mainloop()


def check_paths(entries: dict, required: list, *args):
    """Checks the paths in the entry if they are required and colors the text boxes accordingly

    Arguments:
        entries -- The dictionary containing the UI elements
        required -- A list of required paths
    """
    entries["start_button"].state(["!disabled"])
    for k in entries.keys():
        if isinstance(entries[k], dict):
            path = entries[k]["var"].get()
            if os.path.exists(path):
                entries[k]["entry"].configure(foreground="green")
            else:
                if k in required:
                    entries[k]["entry"].configure(foreground="red")
                    entries["start_button"].state(["disabled"])
                else:
                    entries[k]["entry"].configure(foreground="orange")


def set_path(entry):
    """Callback for the open buttons

    Arguments:
        entry -- The entry to change
    """
    old_path = entry["var"].get()
    _, ext = os.path.splitext(old_path)
    if ext:  # ext is empty when old_path is a directory
        path = u4files.get_file_paths(filetypes=((ext, ext),))
    else:
        path = u4files.get_folder_paths()
    if isinstance(path, (list, tuple)):
        messagebox.showwarning(
            title="Wrong Path", message="Only select a single file!"
        )
        path = old_path
    elif not path:
        path = old_path
    entry["var"].set(path)


def save_n_go(
    root: tk.Tk,
    entries: dict,
    project: configparser.ConfigParser,
    file_path: os.PathLike = None,
):
    """Closes the dialog, saves all current entries to the config file and
    then returns to main script

    Arguments:
        root -- The main window with the list of paths.
        entries -- A dictionary containing all entries
        project -- The loaded project config
        file_path -- The path to the config
    """
    changed = False
    for k in entries.keys():
        if isinstance(entries[k], dict) and (
            entries[k]["var"].get() != project["paths"][k]
        ):
            project["paths"][k] = entries[k]["var"].get()
            changed = True

    if file_path and changed:
        with open(file_path, "wt") as project_file:
            project.write(project_file)
    elif not (file_path):
        file_path = u4files.get_save_path(
            title="Save project as:",
            filetypes=(("*.u4project", "*.u4project"),),
        )
        with open(file_path, "wt") as project_file:
            project.write(project_file)
    root.destroy()


# For DEBUGGING PURPOSES


# def main():
#     project = get_project()
#     for k, v in project["paths"].items():
#         print(k, "-->", v)


# if __name__ == "__main__":
#     main()
