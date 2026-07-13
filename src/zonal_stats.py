"""
Zonal statistics: for each polygon in a GeoJSON, count how many pixels of
each class fall inside it, for a given classified (H, W) map.
"""

from __future__ import annotations

import io

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from rasterio.features import geometry_mask
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds

from .model_utils import CLASSES

CHECK_CRS = "EPSG:4326"

_ID_CANDIDATE_COLUMNS = ["id", "ID", "name", "Name", "NAME", "platja", "platges_id"]


def _polygon_id(gdf: gpd.GeoDataFrame, idx: int) -> str:
    """Prefer a natural identifier column if present, else fall back to
    a 1-based positional index."""
    for col in _ID_CANDIDATE_COLUMNS:
        if col in gdf.columns and pd.notna(gdf.iloc[idx][col]):
            return str(gdf.iloc[idx][col])
    return f"polygon_{idx + 1}"


def _crs_equals(crs_a, crs_b) -> bool:
    """
    Robustly compare two CRS objects that may come from different
    libraries (e.g. a pyproj.CRS from geopandas vs. a rasterio.crs.CRS
    from a raster profile). A plain `==` between those two types is not
    reliable -- it can incorrectly report a match (skipping a reprojection
    that was actually needed) or a mismatch. Wrapping both through
    pyproj.CRS.from_user_input() first normalizes them to the same type
    before comparing.
    """
    try:
        return pyproj.CRS.from_user_input(crs_a).equals(pyproj.CRS.from_user_input(crs_b))
    except Exception:
        return False


def check_crs_compatibility(gdf: gpd.GeoDataFrame, profile: dict) -> dict:
    """
    Check whether a GeoJSON's geometries actually line up with a raster,
    independent of whether their CRS labels nominally match.

    Both files' bounds are converted to EPSG:4326 (plain lat/lon) for this
    check, rather than comparing in the raster's own declared CRS. That
    matters because the raster's CRS *metadata itself* can be the thing
    that's mislabeled (e.g. UTM data tagged as EPSG:4326/3857) -- trusting
    it as the reference frame would just reproduce the same bug. WGS84
    lat/lon is also easy to sanity-check by eye (e.g. Barcelona should read
    roughly lon~2.1, lat~41.3), unlike arbitrary UTM eastings/northings.

    Note: this EPSG:4326 conversion is only for this human-facing check.
    The actual pixel masking in compute_zonal_stats() still has to work in
    the raster's native CRS/transform, since that's what its pixel grid is
    indexed in -- there's no way around trusting that for the computation
    itself, only for this sanity check.

    Returns a dict with:
        raster_crs, geojson_crs: original CRS objects (or None)
        crs_labels_match: bool -- informational only
        raster_bounds, geojson_bounds: (left, bottom, right, top) in
            EPSG:4326 (lon/lat)
        bounds_overlap: bool -- the real compatibility signal
        reprojected: bool -- always True once CRS are confirmed present
        check_crs: the common comparison CRS used (EPSG:4326)
        error: str or None -- set if the check itself couldn't be completed
    """
    raster_crs = profile.get("crs")
    geojson_crs = gdf.crs

    result = {
        "raster_crs": raster_crs,
        "geojson_crs": geojson_crs,
        "crs_labels_match": None,
        "raster_bounds": None,
        "geojson_bounds": None,
        "bounds_overlap": None,
        "reprojected": False,
        "check_crs": CHECK_CRS,
        "error": None,
    }

    if geojson_crs is None:
        result["error"] = (
            "The GeoJSON has no CRS defined. Can't verify or reproject it -- "
            "please set a CRS on the file before running zonal stats."
        )
        return result

    if raster_crs is None:
        result["error"] = "The raster has no CRS defined in its metadata."
        return result

    result["crs_labels_match"] = _crs_equals(geojson_crs, raster_crs)

    raster_bounds_native = array_bounds(profile["height"], profile["width"], profile["transform"])
    raster_bounds_4326 = transform_bounds(raster_crs, CHECK_CRS, *raster_bounds_native)

    gdf_4326 = gdf.to_crs(CHECK_CRS)
    geojson_bounds_4326 = tuple(gdf_4326.total_bounds)  # (minx, miny, maxx, maxy)

    result["reprojected"] = True
    result["raster_bounds"] = raster_bounds_4326
    result["geojson_bounds"] = geojson_bounds_4326

    r_left, r_bottom, r_right, r_top = raster_bounds_4326
    g_left, g_bottom, g_right, g_top = geojson_bounds_4326
    result["bounds_overlap"] = not (
        g_right < r_left or g_left > r_right or g_top < r_bottom or g_bottom > r_top
    )

    return result


def compute_zonal_stats(
    class_map: np.ndarray,
    profile: dict,
    geojson_bytes: bytes,
    model_name: str,
) -> pd.DataFrame:
    """
    Args:
        class_map: (H, W) int array of predicted class indices.
        profile: rasterio profile dict for the classified map (must include
            'transform' and 'crs').
        geojson_bytes: raw bytes of an uploaded .geojson file.
        model_name: label to tag rows with (e.g. "CNN" or "RF").

    Returns:
        Tidy DataFrame with one row per (polygon, class): polygon_id,
        model, class, pixel_count, area_m2, pct_within_polygon.
    """
    gdf = gpd.read_file(io.BytesIO(geojson_bytes))

    if gdf.crs is None:
        raise ValueError(
            "The uploaded GeoJSON has no CRS defined, so it can't be safely "
            "reprojected to match the raster. Please set a CRS on the file."
        )
    if profile.get("crs") is None:
        raise ValueError("The raster has no CRS defined in its metadata.")

    # Always reproject unconditionally -- do not gate this behind a CRS
    # equality check (see _crs_equals docstring for why that's unreliable
    # across pyproj/rasterio CRS objects). to_crs() is a safe no-op if the
    # GeoJSON is already in the raster's CRS.
    gdf = gdf.to_crs(profile["crs"])

    transform = profile["transform"]
    h, w = class_map.shape
    pixel_area_m2 = abs(transform.a * transform.e)

    rows = []
    for idx, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty:
            continue

        pid = _polygon_id(gdf, idx)
        mask = geometry_mask(
            [geom], out_shape=(h, w), transform=transform, invert=True
        )
        pixels = class_map[mask]
        total = pixels.size

        if total == 0:
            # Polygon doesn't overlap the raster at all -- skip silently
            # but still worth knowing about, so surface a zero-row.
            for class_idx, class_name in enumerate(CLASSES):
                rows.append(
                    {
                        "polygon_id": pid,
                        "model": model_name,
                        "class": class_name,
                        "pixel_count": 0,
                        "area_m2": 0.0,
                        "pct_within_polygon": 0.0,
                    }
                )
            continue

        counts = np.bincount(pixels, minlength=len(CLASSES))
        for class_idx, class_name in enumerate(CLASSES):
            count = int(counts[class_idx])
            rows.append(
                {
                    "polygon_id": pid,
                    "model": model_name,
                    "class": class_name,
                    "pixel_count": count,
                    "area_m2": count * pixel_area_m2,
                    "pct_within_polygon": count / total * 100,
                }
            )

    return pd.DataFrame(rows)