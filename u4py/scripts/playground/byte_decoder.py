import struct

from osgeo import ogr


def main():
    binary_polygon = b'GP\x00\x03\xe6\x10\x00\x00e\xd2\x95\xbe\xc6L!@\x97c\xc3\xe4WM!@\xd9r\xd3\x0b\xe3\xefH@E\xff\xba\x89\x10\xf0H@\x01\x03\x00\x00\x00\x01\x00\x00\x00\t\x00\x00\x00e\xd2\x95\xbe\xc6L!@#\xbf~\x88\r\xf0H@\xcd\x84\x15]\xcdL!@\x17\xe3\xb2\xc0\r\xf0H@\nz\xb9\x99\x1bM!@3\x94\xc9_\x10\xf0H@\x8fE\xe4\x05"M!@E\xff\xba\x89\x10\xf0H@\x97c\xc3\xe4WM!@1j\xad\x1e\xe6\xefH@G}H\xaeQM!@\x01\x8a\xec\xde\xe5\xefH@\xdb\xef\x9a\xa1\x02M!@\x80\x9d\x9b6\xe3\xefH@\x9c\x88~m\xfdL!@\xd9r\xd3\x0b\xe3\xefH@e\xd2\x95\xbe\xc6L!@#\xbf~\x88\r\xf0H@'
    binary_point = b"GP\x00\x01\xe6\x10\x00\x00\x01\x01\x00\x00\x00\xa1\xb8\x88\x94\x0bz!@]a\x0b\x8b\x9b\rI@"

    print(decode_geom(binary_polygon))
    print(decode_geom(binary_point))


def decode_geom(stream: str) -> ogr.Geometry:
    """Primitive decoder for geometry blobs in a gpkg file. See http://www.geopackage.org./spec/#gpb_format.

    :param stream: The blob as a bytestring
    :type stream: str
    :return: The geometry geocoded in the data
    :rtype: ogr.Geometry

    The geometry blob contains a header, which may include the envelope of the features, and a well known binary (WKB) encoded geometry. We first decode the first 8 bytes to get some more information on what is stored in the blob:
        - 2 bytes: should be "GP" in ASCII
        - 1 byte: 8-bit unsigned Integer for version (0=v.1)
        - 1 byte: GeoPackageBinary flags byte -> has to be decoded to binary
        - 4 byte: 32-bit unsigned Integer with SRS ID.

    The GeoPackageBinary flag contains information about the length of the envelope that follows the first part of the header. This is needed to know where the WKB geometry starts. When we know that we can start to read the rest of the bytestring and feed it to `ogr.CreateGeometryFromWkb` that creates the geometry.
    """
    first_header = struct.unpack("ccBcI", stream[:8])
    magic = first_header[0].decode() + first_header[1].decode()
    if magic == "GP":
        # version = first_header[2]
        flags = binary_flag(first_header[3])
        env_len = get_envlen(flags[-4:-1])
        # srs_id = first_header[4]
        wkb_start = env_len + 8
        return ogr.CreateGeometryFromWkb(stream[wkb_start:])
    else:
        UnicodeEncodeError("Not a valid GPKG enconding")


def binary_flag(inp: str) -> str:
    """Input GeoPackageBinary flag as an encoded bytestring.

    :param inp: The bytestring (single byte)
    :type inp: str
    :return: The bytestring in binary representation
    :rtype: str
    """
    bin_flags = f"{int(inp.hex()):0>8b}"
    return bin_flags


def get_envlen(eee: str) -> int:
    """Gets the length of an envelope as specified in the 5 to 7th bit of the binary flag.

    :param eee: The three relevant bits from the binary flag
    :type eee: str
    :return: The length of the envelope
    :rtype: int
    """
    envlen = [0, 32, 48, 48, 64]
    return envlen[int(eee, base=3)]


if __name__ == "__main__":
    main()
