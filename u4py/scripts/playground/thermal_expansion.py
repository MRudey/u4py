"""
Calculates estimated elevation change of various structures due to thermal
expansion. Including a rough error estimation.

"""
import uncertainties as unc

import u4py.addons.climate as u4climate


def main():
    dT = {"Freiland": 40, "Stadt": 25, "Isoliert": 5}  # K Temperaturdifferenz

    alpha = {  # Ausdehnungskoeffizienten verschiedener Materialien
        "Aluminium": unc.ufloat(23.1, 23.1 * 0.05),
        "Stahl": unc.ufloat(12, 1),
        "Grauguss": unc.ufloat(10.4, 10.4 * 0.05),
        "Beton": unc.ufloat(10, 10 * 0.05),
        "Ziegel": unc.ufloat(5, 5 * 0.05),
        "Holz": unc.ufloat(8, 8 * 0.05),
        "Klinker": unc.ufloat(3.8, 1),
    }

    structures = [
        {
            "Typ": "Freileitungsmast 220 - 380 kV",
            "Höhe": 50,
            "Material": "Stahl",
            "Umgebung": "Freiland",
        },
        {
            "Typ": "Freileitungsmast 50 - 110 kV",
            "Höhe": unc.ufloat(25, 5),
            "Material": "Stahl",
            "Umgebung": "Freiland",
        },
        {
            "Typ": "Strommast 50 - 110 kV",
            "Höhe": unc.ufloat(25, 5),
            "Material": "Beton",
            "Umgebung": "Freiland",
        },
        {
            "Typ": "Strommast bis 36 kV",
            "Höhe": unc.ufloat(15, 5),
            "Material": "Beton",
            "Umgebung": "Freiland",
        },
        {
            "Typ": "Strommast bis 36 kV",
            "Höhe": unc.ufloat(15, 5),
            "Material": "Holz",
            "Umgebung": "Freiland",
        },
        {
            "Typ": "Laternenmast",
            "Höhe": unc.ufloat(3, 1),
            "Material": "Grauguss",
            "Umgebung": "Stadt",
        },
        {
            "Typ": "Schornstein",
            "Höhe": unc.ufloat(60, 15),
            "Material": "Ziegel",
            "Umgebung": "Freiland",
        },
        {
            "Typ": "Schornstein",
            "Höhe": unc.ufloat(60, 15),
            "Material": "Klinker",
            "Umgebung": "Freiland",
        },
        {
            "Typ": "Einfamilienhaus",
            "Höhe": unc.ufloat(10, 1),
            "Material": "Ziegel",
            "Umgebung": "Isoliert",
        },
        {
            "Typ": "Mehrfamilienhaus",
            "Höhe": unc.ufloat(18, 2),
            "Material": "Ziegel",
            "Umgebung": "Isoliert",
        },
        {
            "Typ": "Hochhaus 30er",
            "Höhe": unc.ufloat(34, 2),
            "Material": "Ziegel",
            "Umgebung": "Isoliert",
        },
        {
            "Typ": "Hochhaus",
            "Höhe": unc.ufloat(50, 2),
            "Material": "Beton",
            "Umgebung": "Isoliert",
        },
        {
            "Typ": "Hochhaus",
            "Höhe": unc.ufloat(100, 5),
            "Material": "Beton",
            "Umgebung": "Isoliert",
        },
        {
            "Typ": "Wolkenkratzer",
            "Höhe": unc.ufloat(150, 20),
            "Material": "Beton",
            "Umgebung": "Isoliert",
        },
        {
            "Typ": "Wolkenkratzer",
            "Höhe": unc.ufloat(200, 20),
            "Material": "Beton",
            "Umgebung": "Isoliert",
        },
    ]

    print(
        f"Theoretische Ausdehung verschiedener Strukturen bei {dT} K Schwankung."
    )
    for struc in structures:
        d_len = (
            u4climate.thermal_expansion(
                struc["Höhe"],
                alpha[struc["Material"]] * 10**-6,
                dT[struc["Umgebung"]],
            )
            * 1000
        )
        print(
            f"{struc['Typ']} ({struc['Höhe']}m, {dT[struc['Umgebung']]}K) aus {struc['Material']}: {d_len} mm"
        )


if __name__ == "__main__":
    main()
