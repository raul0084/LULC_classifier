"""
Sentinel-2 Beach Land Cover Classifier
---------------------------------------
Upload a 10-band Sentinel-2 GeoTIFF (bands B02, B03, B04, B05, B06, B07,
B08, B8A, B11, B12, in that order), run a patch-based CNN and/or a
pixel-based Random Forest on demand, and optionally extract per-class
pixel/area stats within GeoJSON polygons.

Run with:
    streamlit run app.py
"""

import io

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import streamlit as st

from src.model_utils import (
    CLASSES,
    PATCH_SIZE,
    classify_cnn_patchwise,
    classify_rf_pixelwise,
    load_cnn,
    load_rf,
)
from src.preprocessing import BAND_ORDER, add_bands, stack_to_tensor
from src.visualization import agreement_map, legend_figure, plot_class_map
from src.zonal_stats import check_crs_compatibility, compute_zonal_stats

st.set_page_config(page_title="Sentinel-2 Beach Classifier", layout="wide")

CNN_PATH = "models/cnn_model.pth"
RF_PATH = "models/rf_model.pkl"


@st.cache_resource
def get_models():
    cnn = load_cnn(CNN_PATH)
    rf = load_rf(RF_PATH)
    return cnn, rf


def read_geotiff(file_bytes: bytes):
    """Read all bands + georeferencing metadata from an in-memory GeoTIFF."""
    with rasterio.MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            bands = src.read()  # (n_bands, H, W)
            profile = src.profile
    return bands, profile


def write_geotiff(class_map: np.ndarray, profile: dict) -> bytes:
    """Encode a classified (H, W) map as a single-band GeoTIFF, in memory."""
    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8", compress="lzw")
    buf = io.BytesIO()
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**out_profile) as dst:
            dst.write(class_map.astype("uint8")[np.newaxis, :, :])
        buf.write(memfile.read())
    buf.seek(0)
    return buf.getvalue()


def reset_results_if_new_file(file_id):
    """Clear any previously computed maps when a different file is uploaded."""
    if st.session_state.get("file_id") != file_id:
        st.session_state["file_id"] = file_id
        st.session_state.pop("cnn_map", None)
        st.session_state.pop("rf_map", None)


def main():
    st.title("🛰️ Sentinel-2 Beach Land Cover Classifier")
    st.caption(
        f"Expected band order: {', '.join(BAND_ORDER)} · "
        f"Classes: {', '.join(CLASSES)}"
    )

    uploaded = st.file_uploader("Upload a Sentinel-2 GeoTIFF (10 bands)", type=["tif", "tiff"])

    stride = st.sidebar.slider(
        "CNN stride (pixels)",
        min_value=1,
        max_value=PATCH_SIZE,
        value=PATCH_SIZE,
        help=(
            f"Step size for sliding the {PATCH_SIZE}x{PATCH_SIZE} CNN patch "
            f"window. {PATCH_SIZE} = non-overlapping tiles (fastest). Smaller "
            "values overlap patches and resolve each pixel by majority vote "
            "across all patches covering it (smoother, more compute)."
        ),
    )

    if uploaded is None:
        st.info("Upload a .tif to get started.")
        return

    file_bytes = uploaded.getvalue()
    reset_results_if_new_file(uploaded.file_id)

    try:
        bands, profile = read_geotiff(file_bytes)
    except Exception as e:
        st.error(f"Could not read the GeoTIFF: {e}")
        return

    if bands.shape[0] != len(BAND_ORDER):
        st.error(
            f"Expected {len(BAND_ORDER)} bands ({', '.join(BAND_ORDER)}), "
            f"but the uploaded file has {bands.shape[0]}."
        )
        return

    with st.spinner("Loading models..."):
        try:
            cnn, rf = get_models()
        except FileNotFoundError as e:
            st.error(
                f"Model file not found: {e}. Place your models at "
                f"'{CNN_PATH}' and '{RF_PATH}' relative to app.py."
            )
            return

    with st.spinner("Computing spectral indices and standardizing..."):
        raw_tensor = stack_to_tensor(bands)
        standardized, n_channels, means, stds = add_bands(raw_tensor)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶️ Run CNN", use_container_width=True):
            with st.spinner(f"Running CNN (patch-based, stride={stride})..."):
                st.session_state["cnn_map"] = classify_cnn_patchwise(
                    cnn, standardized, stride=stride
                )
        if "cnn_map" in st.session_state:
            st.pyplot(
                plot_class_map(
                    st.session_state["cnn_map"],
                    f"CNN (patch={PATCH_SIZE}, stride={stride})",
                )
            )

    with col2:
        if st.button("▶️ Run RF", use_container_width=True):
            with st.spinner("Running Random Forest (pixel-based)..."):
                st.session_state["rf_map"] = classify_rf_pixelwise(rf, standardized)
        if "rf_map" in st.session_state:
            st.pyplot(plot_class_map(st.session_state["rf_map"], "Random Forest (pixel-based)"))

    cnn_map = st.session_state.get("cnn_map")
    rf_map = st.session_state.get("rf_map")

    if cnn_map is None and rf_map is None:
        st.info("Click 'Run CNN' and/or 'Run RF' above to classify this image.")
        return

    st.pyplot(legend_figure())

    if cnn_map is not None and rf_map is not None:
        st.subheader("Model agreement")
        st.pyplot(agreement_map(cnn_map, rf_map))

    st.subheader("Class distribution")
    available = [(cnn_map, "CNN"), (rf_map, "RF")]
    available = [(m, n) for m, n in available if m is not None]
    dist_cols = st.columns(len(available))
    for col, (class_map, name) in zip(dist_cols, available):
        with col:
            st.write(f"**{name}**")
            counts = np.bincount(class_map.ravel(), minlength=len(CLASSES))
            pct = counts / counts.sum() * 100
            st.bar_chart({CLASSES[i]: pct[i] for i in range(len(CLASSES))})

    st.subheader("Download classified rasters")
    dl_cols = st.columns(len(available))
    for col, (class_map, name) in zip(dl_cols, available):
        with col:
            st.download_button(
                f"Download {name} classification (.tif)",
                data=write_geotiff(class_map, profile),
                file_name=f"{name.lower()}_classification.tif",
                mime="image/tiff",
            )

    st.divider()
    st.subheader("Zonal stats from a GeoJSON")
    st.caption(
        "Upload polygons (e.g. beach/transect boundaries) to count pixels "
        "of each class inside each polygon, for whichever model(s) you've run."
    )
    geojson_file = st.file_uploader("Upload a GeoJSON", type=["geojson", "json"])

    if geojson_file is not None:
        geojson_bytes = geojson_file.getvalue()

        try:
            gdf_preview = gpd.read_file(io.BytesIO(geojson_bytes))
            crs_check = check_crs_compatibility(gdf_preview, profile)
        except Exception as e:
            st.error(f"Could not read the GeoJSON: {e}")
            return

        if crs_check["error"]:
            st.error(crs_check["error"])
            return

        if not crs_check["bounds_overlap"]:
            st.error(
                "⚠️ The GeoJSON's extent does not overlap the raster's extent "
                "at all, when both are converted to EPSG:4326 (lon/lat) for "
                "comparison. This usually means one file's CRS metadata is "
                "mislabeled -- extracting stats would just produce an "
                "all-zero CSV, so fix the CRS before continuing."
            )
            with st.expander("Show raster vs. GeoJSON extents (EPSG:4326, lon/lat)"):
                st.write(f"Raster bounds (declared as {crs_check['raster_crs']}): {crs_check['raster_bounds']}")
                st.write(f"GeoJSON bounds (declared as {crs_check['geojson_crs']}): {crs_check['geojson_bounds']}")
                st.caption(
                    "Both converted to lon/lat for this comparison, since "
                    "the raster's own CRS metadata could itself be wrong -- "
                    "trusting it as the reference frame wouldn't catch that."
                )
            return

        with st.expander("Show raster vs. GeoJSON extents (EPSG:4326, lon/lat)"):
            st.write(f"Raster bounds (declared as {crs_check['raster_crs']}): {crs_check['raster_bounds']}")
            st.write(f"GeoJSON bounds (declared as {crs_check['geojson_crs']}): {crs_check['geojson_bounds']}")

        if not crs_check["crs_labels_match"]:
            st.warning(
                f"CRS labels differ (raster: {crs_check['raster_crs']}, "
                f"GeoJSON: {crs_check['geojson_crs']}), but both reproject "
                "to overlapping extents in lon/lat, so this looks fine. "
                "The GeoJSON will be reprojected to the raster's native CRS "
                "for the actual pixel extraction below."
            )

        if st.button("📊 Extract zonal stats"):
            all_stats = []
            try:
                for class_map, name in available:
                    with st.spinner(f"Computing zonal stats for {name}..."):
                        df = compute_zonal_stats(class_map, profile, geojson_bytes, name)
                        all_stats.append(df)
            except Exception as e:
                st.error(f"Could not compute zonal stats: {e}")
            else:
                stats_df = pd.concat(all_stats, ignore_index=True)
                st.session_state["stats_df"] = stats_df

        if "stats_df" in st.session_state:
            st.dataframe(st.session_state["stats_df"], use_container_width=True)
            st.download_button(
                "Download zonal stats (.csv)",
                data=st.session_state["stats_df"].to_csv(index=False).encode("utf-8"),
                file_name="zonal_stats.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()