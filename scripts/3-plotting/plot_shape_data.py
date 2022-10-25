import geopandas
import matplotlib.pyplot as plt


def main():
    tektonik_path = (
        r"C:\Users\Michael Rudolf\Documents\ArcGIS\Places\tektonik.dbf"
    )
    bld_path = (
        r"C:\Users\Michael Rudolf\Documents\ArcGIS\Places\vg2500_bld.dbf"
    )

    tektonik = geopandas.read_file(tektonik_path).to_crs("EPSG:32632")
    bld = geopandas.read_file(bld_path).to_crs("EPSG:32632")
    hessen = bld[bld["GEN"] == "Hessen"]
    tektonik_hessen = geopandas.clip(tektonik, hessen)
    fig, ax = plt.subplots()
    tektonik_hessen.plot(ax=ax)
    plt.show()


if __name__ == "__main__":
    main()
