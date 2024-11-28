import geopandas as gp

from u4py.io.tex_report import hlnug_description


def main():
    hld = gp.read_file(
        "/home/rudolf/Documents/umwelt4/Places/HLNUG Daten/SHP/RD_Rutschungen_gesamt.shp"
    )
    print(hlnug_description(hld[hld.AMT_NR_ == 13]))


if __name__ == "__main__":
    main()
