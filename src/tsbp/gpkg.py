"""Split a GeoPackage layer of digitised wavefronts into one GeoJSON per feature.

QGIS digitising naturally produces ONE GeoPackage layer holding several
wavefront line features (distinguished by an id attribute, e.g. ``wf_id``), but
``tsbp`` reads one GeoJSON polyline per wavefront.  This module bridges the two
with no external geo dependencies: a GeoPackage is a SQLite database and its
geometries are a GeoPackage-binary header followed by standard WKB, both of
which we parse with the standard library (``sqlite3`` + ``struct``).

Coordinates are assumed geographic lon/lat (EPSG:4326 / undefined-geographic,
srs_id 4326 or 0), which is what the back-projection expects; a projected layer
raises (reproject to EPSG:4326 in QGIS or with ``ogr2ogr -t_srs EPSG:4326``
first).  Only LineString / MultiLineString geometries are supported; Z/M
ordinates are read and dropped.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import struct

_ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


def _parse_wkb(buf, off):
    """Parse a WKB LineString/MultiLineString at ``buf[off:]`` into a flat list
    of (lon, lat) tuples (concatenating multi-parts).  Returns (points, new_off)."""
    byte_order = buf[off]
    fmt = "<" if byte_order == 1 else ">"
    off += 1
    gtype = struct.unpack_from(fmt + "I", buf, off)[0]
    off += 4
    dim = (gtype // 1000) % 10          # 0=XY, 1=XYZ, 2=XYM, 3=XYZM (ISO WKB)
    base = gtype % 1000                 # 2=LineString, 5=MultiLineString
    n_ord = 2 + (1 if dim in (1, 3) else 0) + (1 if dim in (2, 3) else 0)

    pts = []
    if base == 2:                       # LineString
        npts = struct.unpack_from(fmt + "I", buf, off)[0]
        off += 4
        for _ in range(npts):
            x, y = struct.unpack_from(fmt + "dd", buf, off)
            pts.append((x, y))
            off += 8 * n_ord
    elif base == 5:                     # MultiLineString
        nlines = struct.unpack_from(fmt + "I", buf, off)[0]
        off += 4
        for _ in range(nlines):
            sub, off = _parse_wkb(buf, off)
            pts.extend(sub)
    else:
        raise ValueError(
            f"unsupported WKB geometry type {gtype} "
            "(only LineString / MultiLineString)")
    return pts, off


def _parse_gpkg_geometry(blob):
    """Extract (lon, lat) points from a GeoPackage geometry blob.
    Returns (points, srs_id)."""
    if blob[:2] != b"GP":
        raise ValueError("not a GeoPackage geometry blob (bad magic)")
    flags = blob[3]
    hdr_fmt = "<i" if (flags & 0x01) else ">i"
    srs_id = struct.unpack_from(hdr_fmt, blob, 4)[0]
    env_code = (flags >> 1) & 0x07
    if env_code not in _ENVELOPE_BYTES:
        raise ValueError(f"reserved GeoPackage envelope code {env_code}")
    wkb_off = 8 + _ENVELOPE_BYTES[env_code]
    pts, _ = _parse_wkb(blob, wkb_off)
    return pts, srs_id


def _resolve_layer(con, layer):
    """Return (table_name, geom_column) for the requested layer, or the sole
    geometry layer if ``layer`` is None."""
    rows = con.execute(
        "SELECT table_name, column_name FROM gpkg_geometry_columns"
    ).fetchall()
    if not rows:
        raise ValueError("no geometry columns found in this GeoPackage")
    if layer is None:
        if len(rows) != 1:
            names = ", ".join(r[0] for r in rows)
            raise ValueError(f"multiple layers ({names}); pass layer=...")
        return rows[0]
    for tbl, col in rows:
        if tbl == layer:
            return tbl, col
    raise ValueError(f"layer {layer!r} not found; have "
                     f"{', '.join(r[0] for r in rows)}")


def gpkg_to_geojson(gpkg_path, out_dir, layer=None, id_field="wf_id",
                    name_prefix=None):
    """Write one GeoJSON polyline per feature of a GeoPackage line layer.

    Parameters
    ----------
    gpkg_path : str
        Path to the ``.gpkg`` file.
    out_dir : str
        Directory to write the GeoJSON files into (created if needed).
    layer : str or None
        Layer/table name.  If None and the file has exactly one geometry layer,
        that one is used.
    id_field : str
        Integer/text attribute whose value names each output file and is copied
        into the feature's properties.  If absent, the SQLite ``rowid`` is used.
    name_prefix : str or None
        Output filename stem.  Files are ``<prefix>_<id>.geojson``.  Defaults to
        the layer name.

    Returns
    -------
    list[str]
        Paths written, ordered by ``id_field``.
    """
    os.makedirs(out_dir, exist_ok=True)
    con = sqlite3.connect(gpkg_path)
    try:
        table, geom_col = _resolve_layer(con, layer)
        prefix = name_prefix or table

        cols = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]
        id_expr = id_field if id_field in cols else "rowid"
        if id_expr == "rowid" and id_field != "rowid":
            print(f"  note: '{id_field}' not found; using rowid to name files")

        rows = con.execute(
            f'SELECT "{id_expr}", "{geom_col}" FROM "{table}" '
            f'ORDER BY "{id_expr}"'
        ).fetchall()

        written = []
        for idval, blob in rows:
            if blob is None:
                print(f"  skip {idval}: NULL geometry")
                continue
            pts, srs_id = _parse_gpkg_geometry(blob)
            if srs_id not in (4326, 0):
                raise ValueError(
                    f"layer srs_id={srs_id} is not geographic lon/lat (EPSG:4326)."
                    " Reproject to EPSG:4326 before exporting.")
            if len(pts) < 2:
                print(f"  skip {idval}: <2 vertices")
                continue
            doc = {"type": "FeatureCollection", "features": [{
                "type": "Feature",
                "properties": {id_field: idval},
                "geometry": {"type": "LineString",
                             "coordinates": [[x, y] for x, y in pts]},
            }]}
            path = os.path.join(out_dir, f"{prefix}_{idval}.geojson")
            with open(path, "w") as fh:
                json.dump(doc, fh)
            written.append(path)
            print(f"  wrote {path}  ({len(pts)} points)")
        return written
    finally:
        con.close()


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Split a GeoPackage wavefront layer into one GeoJSON per feature")
    p.add_argument("gpkg", help="input .gpkg file")
    p.add_argument("out_dir", help="directory to write the GeoJSON files into")
    p.add_argument("--layer", help="layer/table name (default: the sole layer)")
    p.add_argument("--id-field", default="wf_id",
                   help="attribute naming each file (default: wf_id)")
    p.add_argument("--prefix", dest="name_prefix",
                   help="output filename stem (default: layer name)")
    args = p.parse_args(argv)
    paths = gpkg_to_geojson(args.gpkg, args.out_dir, layer=args.layer,
                            id_field=args.id_field, name_prefix=args.name_prefix)
    print(f"\n{len(paths)} wavefront(s) written to {args.out_dir}")


if __name__ == "__main__":
    main()
