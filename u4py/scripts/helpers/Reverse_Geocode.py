"""
Geolocates the centres of the filtered shapes in the classified shapes database. The results are saved into a csv file for easier access by other scripts.
"""

import geopandas as gp
import numpy as np
from geopy.geocoders import Nominatim
from geopy.point import Point
from tqdm import tqdm


def main():
    # Read Data
    class_shp_gdf = gp.read_file(
        "/home/rudolf/Documents/umwelt4/SelectedSites/Classified_Shapes.gpkg"
    )

    lands_f = class_shp_gdf.landslides_num_inside > 0
    karst_f = class_shp_gdf.karst_num_inside > 0
    rockf_f = class_shp_gdf.rockfall_num_inside > 0

    filter_all = np.logical_or(np.logical_or(lands_f, karst_f), rockf_f)
    gdf_filtered = class_shp_gdf[filter_all]

    # # Locations already extracted for a test
    # locations = [
    #     "B 84, Roßbach, Hünfeld, Landkreis Fulda, Hessen, 36088, Deutschland",
    #     "B 84, Kirchhasel, Hünfeld, Landkreis Fulda, Hessen, 36088, Deutschland",
    #     "3, Am Rücker Hof, Keutzelbuch, Rückers, Flieden, Landkreis Fulda, Hessen, 36103, Deutschland",
    #     "L 3076, Korbach, Landkreis Waldeck-Frankenberg, Hessen, 34497, Deutschland",
    #     "Neue Heimat, Erbach, Eltville am Rhein, Rheingau-Taunus-Kreis, Hessen, 65346, Deutschland",
    #     "21, Am See, Zimmersrode, Neuental, Schwalm-Eder-Kreis, Hessen, 34599, Deutschland",
    #     "Ziegelei, Dachziegelwerk, Angersbach, Wartenberg, Vogelsbergkreis, Hessen, 36367, Deutschland",
    #     "Am Erdbacheinschlupf, Erbach (Odenwald), Erbach, Odenwaldkreis, Hessen, 64711, Deutschland",
    #     "K 80, Frederinghausen, Gembeck, Twistetal, Landkreis Waldeck-Frankenberg, Hessen, 34477, Deutschland",
    #     "Sportplatz, Knickhagen, Herlinghausen, Warburg, Kreis Höxter, Nordrhein-Westfalen, 34414, Deutschland",
    #     "Burgruine Großer Gudenberg, Bärenbergstraße, Zierenberg, Landkreis Kassel, Hessen, 34289, Deutschland",
    #     'Motocross-Strecke "Am Idel", Idelslinie, Kassel, Biebergemünd, Main-Kinzig-Kreis, Hessen, 63599, Deutschland',
    #     "202, Zum Schäferköppel, Nieder-Erlenbach, Frankfurt am Main, Hessen, 60437, Deutschland",
    #     "20, Marienstraße, Kerzell, Eichenzell, Landkreis Fulda, Hessen, 36124, Deutschland",
    #     "3f, Heinrich-Hertz-Straße, Waldau, Kassel, Hessen, 34123, Deutschland",
    #     "Hof Ottersbach, Oberjossa, Breitenbach am Herzberg, Landkreis Hersfeld-Rotenburg, Hessen, 36287, Deutschland",
    #     "55, Brunnenstraße, Oberscheld, Dillenburg, Lahn-Dill-Kreis, Hessen, 35688, Deutschland",
    #     "56357, Kasdorf, Nastätten, Rhein-Lahn-Kreis, Rheinland-Pfalz, Deutschland",
    #     "L 3272, Geisenheim, Rheingau-Taunus-Kreis, Hessen, 65366, Deutschland",
    #     "Heimbacher Straße, Bad Schwalbach, Rheingau-Taunus-Kreis, Hessen, 65307, Deutschland",
    #     "Alte Tränke, Judenpfad, Nentershausen, Landkreis Hersfeld-Rotenburg, Hessen, 36214, Deutschland",
    #     "Am Weinberg, Oberellenbach, Alheim, Landkreis Hersfeld-Rotenburg, Hessen, 36211, Deutschland",
    #     "Riedmühle, Oberellenbach, Alheim, Landkreis Hersfeld-Rotenburg, Hessen, 36211, Deutschland",
    #     "13, Goethestraße, Westuffeln, Calden, Landkreis Kassel, Hessen, 34379, Deutschland",
    #     "L 2304, Sterbfritz, Sinntal, Main-Kinzig-Kreis, Hessen, 36391, Deutschland",
    #     "7, Im Eichgrund, Kellerei, Döngesmühle, Flieden, Landkreis Fulda, Hessen, 36103, Deutschland",
    #     "Gut Mönchhof, Alberode, Abterode, Meißner, Werra-Meißner-Kreis, Hessen, 37290, Deutschland",
    #     "Am Petersbach, Niddawitzhausen, Eschwege, Werra-Meißner-Kreis, Hessen, 37269, Deutschland",
    #     "K 62, Wettesingen, Breuna, Landkreis Kassel, Hessen, 34479, Deutschland",
    #     "Liebenauer Straße, Körbecke, Borgentreich, Kreis Höxter, Nordrhein-Westfalen, 34434, Deutschland",
    #     "Hochbehälter Solz, K 55, Solz, Bebra, Landkreis Hersfeld-Rotenburg, Hessen, 36179, Deutschland",
    #     "Höllgrabenstraße, Herlefeld, Spangenberg, Schwalm-Eder-Kreis, Hessen, 34286, Deutschland",
    #     "11, Uferstraße, Bergshausen, Fuldabrück, Landkreis Kassel, Hessen, 34277, Deutschland",
    #     "Steinkaute bei Holzheim, Butzbacher Weg, Holzheim, Pohlheim, Landkreis Gießen, Hessen, 35415, Deutschland",
    #     "Rhönbergstraße, Knottenhof, Theobaldshof, Tann (Rhön), Landkreis Fulda, Hessen, 36142, Deutschland",
    #     "Veitsteinbach, Kalbach, Landkreis Fulda, Hessen, 36148, Deutschland",
    #     "L 3044, Haus Marianne, Langenaubach, Haiger, Lahn-Dill-Kreis, Hessen, 35708, Deutschland",
    #     "Kleine Wann, Echzell, Wetteraukreis, Hessen, 61209, Deutschland",
    #     "Kolonnenweg, Volkerode, Ershausen/Geismar, Landkreis Eichsfeld, Thüringen, 37308, Deutschland",
    #     "Burgruine Schartenberg, L 3211, Rangen, Zierenberg, Landkreis Kassel, Hessen, 34289, Deutschland",
    #     "4, An der Kippe, Schlüchtern, Main-Kinzig-Kreis, Hessen, 36381, Deutschland",
    #     "Nußbaumweg, Halgehausen, Haina (Kloster), Landkreis Waldeck-Frankenberg, Hessen, 35114, Deutschland",
    #     "Briloner Landstraße, Lelbach, Korbach, Landkreis Waldeck-Frankenberg, Hessen, 34497, Deutschland",
    #     "Borkener See, Nassenerfurth, Borken (Hessen), Schwalm-Eder-Kreis, Hessen, 34582, Deutschland",
    #     "Höllkaute, K 66, Oberellenbach, Alheim, Landkreis Hersfeld-Rotenburg, Hessen, 36211, Deutschland",
    #     "Mühlenstraße, Nentershausen, Landkreis Hersfeld-Rotenburg, Hessen, 36214, Deutschland",
    #     "L 3241, Weidenhausen, Meißner, Werra-Meißner-Kreis, Hessen, 37290, Deutschland",
    #     "Wasserturm Kirchbergweg, Riedmühle, Niederellenbach, Alheim, Landkreis Hersfeld-Rotenburg, Hessen, 36211, Deutschland",
    #     "Lindenhof, Altmorschen, Morschen, Schwalm-Eder-Kreis, Hessen, 34326, Deutschland",
    #     "B 83, Altmorschen, Morschen, Schwalm-Eder-Kreis, Hessen, 34326, Deutschland",
    #     "Schießstand, Carolinenweg, Gutsbezirk Reinhardswald, Landkreis Kassel, Hessen, 34369, Deutschland",
    #     "Westerwaldquerbahn, Lochmühle, Heiligenborn, Driedorf, Lahn-Dill-Kreis, Hessen, 35759, Deutschland",
    #     "23, Forststraße, Rommerz, Neuhof, Landkreis Fulda, Hessen, 36119, Deutschland",
    #     "L 3083, Korbach, Landkreis Waldeck-Frankenberg, Hessen, 34497, Deutschland",
    #     "A 66, Schlüchtern, Main-Kinzig-Kreis, Hessen, 36381, Deutschland",
    #     "Höhe 304, Wilhelmshausen, Fuldatal, Landkreis Kassel, Hessen, 34233, Deutschland",
    #     "Höhe 304, Wilhelmshausen, Fuldatal, Landkreis Kassel, Hessen, 34233, Deutschland",
    #     "4, Am Knöschen, Wallroth, Schlüchtern, Main-Kinzig-Kreis, Hessen, 36381, Deutschland",
    #     "K 928, Drasenberg, Klosterhöfe, Schlüchtern, Main-Kinzig-Kreis, Hessen, 36381, Deutschland",
    #     "Am Schottenberg, Kransberg, Usingen, Hochtaunuskreis, Hessen, 61250, Deutschland",
    #     "L 3179, Steinau an der Straße, Main-Kinzig-Kreis, Hessen, 36396, Deutschland",
    #     "Alseeweg, Udenhain, Brachttal, Main-Kinzig-Kreis, Hessen, 63636, Deutschland",
    #     "Zum Hessenturm, Niedenstein, Schwalm-Eder-Kreis, Hessen, 34305, Deutschland",
    #     "Burg Blideneck, K 625, Espenschied, Lorch, Rheingau-Taunus-Kreis, Hessen, 65391, Deutschland",
    #     "L 3076, Dingeringhausen, Helmscheid, Korbach, Landkreis Waldeck-Frankenberg, Hessen, 34497, Deutschland",
    #     "16, Thüringer Straße, Neustadt, Neuhof, Landkreis Fulda, Hessen, 36119, Deutschland",
    #     "Altenberger Straße, Altenberg, Solms, Lahn-Dill-Kreis, Hessen, 35606, Deutschland",
    #     "Asphaltmischwerke Hauneck, Blaue Liede, Unterhaun, Hauneck, Landkreis Hersfeld-Rotenburg, Hessen, 36282, Deutschland",
    #     "Hübenthal, Berneburg, Sontra, Werra-Meißner-Kreis, Hessen, 36205, Deutschland",
    #     "Hohlweg, Berneburg, Sontra, Werra-Meißner-Kreis, Hessen, 36205, Deutschland",
    #     "31, Am Ringwall, Gudensberg, Schwalm-Eder-Kreis, Hessen, 34281, Deutschland",
    #     "K 50, Rockensüß, Cornberg, Landkreis Hersfeld-Rotenburg, Hessen, 36219, Deutschland",
    #     "L 3249, Heyerode, Sontra, Werra-Meißner-Kreis, Hessen, 36205, Deutschland",
    #     "Im Grund, Untergeis, Neuenstein, Landkreis Hersfeld-Rotenburg, Hessen, 36286, Deutschland",
    #     "L 3170, Leibolz, Eiterfeld, Landkreis Fulda, Hessen, 36132, Deutschland",
    #     "K 18, Pohl-Göns, Butzbach, Wetteraukreis, Hessen, 35510, Deutschland",
    #     "Karl-Förster-Weg, Adenmühle, Flieden, Landkreis Fulda, Hessen, 36103, Deutschland",
    #     "K 50, Rittershain, Rockensüß, Cornberg, Landkreis Hersfeld-Rotenburg, Hessen, 36219, Deutschland",
    #     "K 785, Dürrwiesen, Diedenbergen, Hofheim am Taunus, Main-Taunus-Kreis, Hessen, 65719, Deutschland",
    #     "Willi-Seidel-Weg, Neuholland, Bad Wilhelmshöhe, Kassel, Hessen, 34131, Deutschland",
    #     "L 3241, Strahlshausen, Niederhone, Eschwege, Werra-Meißner-Kreis, Hessen, 37269, Deutschland",
    #     "Hegenhausen, Waldkappel, Werra-Meißner-Kreis, Hessen, 37284, Deutschland",
    #     "K 54, Mönchhoßbach, Nentershausen, Landkreis Hersfeld-Rotenburg, Hessen, 36214, Deutschland",
    #     "Lange Straße, Mönchhoßbach, Nentershausen, Landkreis Hersfeld-Rotenburg, Hessen, 36214, Deutschland",
    #     "34379, Calden, Landkreis Kassel, Hessen, Deutschland",
    #     "Mühlenbergstraße, Westuffeln, Calden, Landkreis Kassel, Hessen, 34379, Deutschland",
    #     "Mellnauer Kreuz, Mellnau, Wetter, Landkreis Marburg-Biedenkopf, Hessen, 35083, Deutschland",
    #     "Stausee von Affoldern, Bataillonsweg, Affoldern, Edertal, Landkreis Waldeck-Frankenberg, Hessen, 34549, Deutschland",
    #     "Stausee von Affoldern, Bataillonsweg, Affoldern, Edertal, Landkreis Waldeck-Frankenberg, Hessen, 34549, Deutschland",
    #     "L 3241, Schwalbenthal, Hausen, Hessisch Lichtenau, Werra-Meißner-Kreis, Hessen, 37235, Deutschland",
    #     "Zimberg, Mengerskirchen, Landkreis Limburg-Weilburg, Hessen, 35794, Deutschland",
    #     "Leimbachshöfer Straße, Rückers, Hünfeld, Landkreis Fulda, Hessen, 36088, Deutschland",
    #     "100, Schützenweg, Großen-Buseck, Buseck, Landkreis Gießen, Hessen, 35418, Deutschland",
    #     "K 80, Frederinghausen, Gembeck, Twistetal, Landkreis Waldeck-Frankenberg, Hessen, 34477, Deutschland",
    #     "54, Eisenbahnstraße, Elm, Schlüchtern, Main-Kinzig-Kreis, Hessen, 36381, Deutschland",
    #     "L 2109, Großburschla, Treffurt, Wartburgkreis, Thüringen, 99830, Deutschland",
    #     "Cornberger Straße, Berneburg, Sontra, Werra-Meißner-Kreis, Hessen, 36205, Deutschland",
    #     "Im Herzbachgrund, Mauers, Haunetal, Landkreis Hersfeld-Rotenburg, Hessen, 36166, Deutschland",
    #     "Frauensteiner Weg, Martinsthal, Eltville am Rhein, Rheingau-Taunus-Kreis, Hessen, 65344, Deutschland",
    #     "K 51, Dudenrode, Bad Sooden-Allendorf, Berkatal, Werra-Meißner-Kreis, Hessen, 37242, Deutschland",
    #     "B 84, Roßbach, Hünfeld, Landkreis Fulda, Hessen, 36088, Deutschland",
    #     "2, Am Vogteigericht, Heuchelheim, Kinzenbach, Heuchelheim an der Lahn, Landkreis Gießen, Hessen, 35452, Deutschland",
    #     "Teufelsee und Pfaffensee zwischen Echzell und Reichelsheim-Weckesheim, L 3412, Gettenau, Echzell, Wetteraukreis, Hessen, 61209, Deutschland",
    #     "Beobachtungshütte, L 3412, Gettenau, Echzell, Wetteraukreis, Hessen, 61209, Deutschland",
    #     "Gottstreuer Straße, Gieselwerder, Wesertal, Landkreis Kassel, Hessen, 34399, Deutschland",
    #     "Steingraben, Blankenbach, Sontra, Werra-Meißner-Kreis, Hessen, 36205, Deutschland",
    #     "Stölzinger Straße, Diemerode, Sontra, Werra-Meißner-Kreis, Hessen, 36205, Deutschland",
    #     "10, Neue Straße, Dens, Nentershausen, Landkreis Hersfeld-Rotenburg, Hessen, 36214, Deutschland",
    #     "Udenhäuser Stock, Gutsbezirk Reinhardswald, Landkreis Kassel, Hessen, 34369, Deutschland",
    #     "K 84, Oberlistingen, Breuna, Landkreis Kassel, Hessen, 34479, Deutschland",
    #     "Nieder-Ofleidener Straße, Ober-Ofleiden, Homberg (Ohm), Vogelsbergkreis, Hessen, 35315, Deutschland",
    #     "Grätz-Markersdorfer-Straße, Michelstadt, Odenwaldkreis, Hessen, 64720, Deutschland",
    #     "Menglers, Mönchhosbach, Nentershausen, Landkreis Hersfeld-Rotenburg, Hessen, 36214, Deutschland",
    #     "Burg Hohenstein, Burgstraße, Burg-Hohenstein, Unterdorf, Burg Hohenstein, Hohenstein, Rheingau-Taunus-Kreis, Hessen, 65329, Deutschland",
    #     "K 1, Birkenhof, Meckbach, Ludwigsau, Landkreis Hersfeld-Rotenburg, Hessen, 36251, Deutschland",
    #     "Auf der Höhe, Cornberg, Landkreis Hersfeld-Rotenburg, Hessen, 36219, Deutschland",
    #     "11, Niederhoner Straße, Niederhone, Eschwege, Werra-Meißner-Kreis, Hessen, 37269, Deutschland",
    #     "Schöne Aussicht, Abgunst, Trendelburg, Landkreis Kassel, Hessen, 34388, Deutschland",
    #     "L 763, Saures Tal, Trendelburg, Landkreis Kassel, Hessen, 34388, Deutschland",
    #     "B 80, Hann. Münden, Gutsbezirk Reinhardswald, Landkreis Kassel, Hessen, 34369, Deutschland",
    #     "Röstweg, Donnershag, Sontra, Werra-Meißner-Kreis, Hessen, 36205, Deutschland",
    #     "Hans-Bredow-Straße, Südost, Wiesbaden, Hessen, 65189, Deutschland",
    #     "Widdershausen, Heringen (Werra), Landkreis Hersfeld-Rotenburg, Hessen, 36266, Deutschland",
    #     "A 7, Thalau, Ebersburg, Landkreis Fulda, Hessen, 36157, Deutschland",
    #     "Hohe Warte bei Gießen, B 457, Gießen, Fernwald, Landkreis Gießen, Hessen, 35394, Deutschland",
    #     "Amazon-Straße, Petersberg, Bad Hersfeld, Landkreis Hersfeld-Rotenburg, Hessen, 36251, Deutschland",
    #     "6, Schlesierstraße, Lämmerspiel, Mühlheim am Main, Landkreis Offenbach, Hessen, 63165, Deutschland",
    #     "1, Fliederweg, Weidenhausen, Meißner, Werra-Meißner-Kreis, Hessen, 37290, Deutschland",
    #     "Prof. Fröhlich Eiche, Kohlenstraße, Gutsbezirk Reinhardswald, Landkreis Kassel, Hessen, 34369, Deutschland",
    #     "L 3329, Am Heiligenborn, Hutten, Schlüchtern, Main-Kinzig-Kreis, Hessen, 36381, Deutschland",
    #     "Wakenfeld, Vor dem Musenberge, Wakenfeld, Usseln, Willingen (Upland), Landkreis Waldeck-Frankenberg, Hessen, 34508, Deutschland",
    #     "Schwarzbachstraße, Gundhelm, Schlüchtern, Main-Kinzig-Kreis, Hessen, 36381, Deutschland",
    #     "Am Stechkopf, Neue Mitte, Oberkaufungen, Kaufungen, Landkreis Kassel, Hessen, 34260, Deutschland",
    #     "Steinköhlerspfad, Gutsbezirk Reinhardswald, Landkreis Kassel, Hessen, 34369, Deutschland",
    #     "2, Am Brückenrasen, Tiefengruben, Neuhof, Landkreis Fulda, Hessen, 36119, Deutschland",
    #     "K 141, Kaulstoß, Schotten, Vogelsbergkreis, Hessen, 63679, Deutschland",
    #     "Wagenweg, Bergen-Enkheim, Frankfurt am Main, Hessen, 60388, Deutschland",
    #     "K 63, Rückerode, Hundelshausen, Witzenhausen, Werra-Meißner-Kreis, Hessen, 37215, Deutschland",
    #     "Bei der Oberdorfer Mühle, Frankershausen, Berkatal, Werra-Meißner-Kreis, Hessen, 37297, Deutschland",
    #     "37242, Bad Sooden, Bad Sooden-Allendorf, Werra-Meißner-Kreis, Hessen, Deutschland",
    #     "K 50, Rockensüß, Cornberg, Landkreis Hersfeld-Rotenburg, Hessen, 36219, Deutschland",
    #     "2, Am Katzkopf, Rockensüß, Cornberg, Landkreis Hersfeld-Rotenburg, Hessen, 36219, Deutschland",
    #     "10, Troßbachtal, Rimbach, Schlitz, Vogelsbergkreis, Hessen, 36110, Deutschland",
    #     "35767, Breitscheid, Lahn-Dill-Kreis, Hessen, Deutschland",
    #     "K 50, Rockensüß, Cornberg, Landkreis Hersfeld-Rotenburg, Hessen, 36219, Deutschland",
    #     "6, Nußhecke, Arzell, Eiterfeld, Landkreis Fulda, Hessen, 36132, Deutschland",
    #     "A 4, Friedewald, Landkreis Hersfeld-Rotenburg, Hessen, 36289, Deutschland",
    #     "35753, Rodenroth, Greifenstein, Lahn-Dill-Kreis, Hessen, Deutschland",
    #     "Eichmühle, Königswald, Cornberg, Landkreis Hersfeld-Rotenburg, Hessen, 36219, Deutschland",
    #     "Eichmühle, Königswald, Cornberg, Landkreis Hersfeld-Rotenburg, Hessen, 36219, Deutschland",
    #     "Obermeiser, Calden, Landkreis Kassel, Hessen, 34379, Deutschland",
    #     "A 7, Büchenberg, Eichenzell, Landkreis Fulda, Hessen, 36124, Deutschland",
    #     "1a, Talstraße, Heyerode, Sontra, Werra-Meißner-Kreis, Hessen, 36205, Deutschland",
    #     "Baunsberg, A 44, Altenbauna, Baunatal, Landkreis Kassel, Hessen, 34225, Deutschland",
    #     "1b, Gewerbering, Gewerbegebiet Ost, Roter Rain, Fritzlar, Schwalm-Eder-Kreis, Hessen, 34560, Deutschland",
    #     "B 27, Rautenhausen, Bebra, Landkreis Hersfeld-Rotenburg, Hessen, 36179, Deutschland",
    #     "Schlüsselgrund, Wettesingen, Breuna, Landkreis Kassel, Hessen, 34479, Deutschland",
    # ]

    locations = []
    geolocator = Nominatim(user_agent="test")
    centroids = gdf_filtered.to_crs("EPSG:4326").geometry.centroid.to_list()
    for point in tqdm(centroids, "Reverse Geolocation"):
        try:
            lon = point.x
            lat = point.y
            location = geolocator.reverse(Point(lat, lon)).raw["display_name"]
        except:
            location = ""
        locations.append(location)

    with open(
        "/home/rudolf/Documents/umwelt4/SelectedSites/cached_locations.txt",
        "wt",
        encoding="utf-8",
    ) as cache:
        for loc in locations:
            cache.write(loc + "\n")


if __name__ == "__main__":
    main()
