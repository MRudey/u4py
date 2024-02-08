"""Downloads data from OSM and saves it as shapefile."""
import shapely
from tqdm import tqdm

import u4py.utils.config as u4config
import u4py.utils.files as u4files
import u4py.utils.projects as u4projects

u4config.start_logger()


def main():
    tags_list = [
        # {"power": "generator", "generator:method": "wind_turbine"},
        # {"parking": "surface"},
        # {"landuse": "construction"},
        {"landuse": "landfill"},
    ]
    types_list = [shapely.Point, shapely.Polygon, shapely.Polygon]

    project = u4projects.get_project(
        required=["places_path"], interactive=False
    )
    for tags, ptype in tqdm(zip(tags_list, types_list), total=len(types_list)):
        u4files.get_osm_as_shp(tags, project, shape_type=ptype)


if __name__ == "__main__":
    main()
