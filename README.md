# Sentinel-2 Beach Land Cover Classifier (Demo)

> **This is a lightweight demo version of my MSc thesis work**, built for
> portfolio/display purposes — not the full research pipeline. It's meant
> to give recruiters, collaborators, or anyone curious a hands-on way to
> see the classification approach in action, without needing the thesis
> database, validation pipeline, or GIS stack behind it.

Streamlit app that classifies an uploaded Sentinel-2 GeoTIFF using both
a patch-based CNN and a pixel-based Random Forest, side by side.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Input requirements

- GeoTIFF with exactly 10 bands, in this order:
  `B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12`
- If your raster stores bands in a different order, reorder them
  (e.g. with `rasterio` or GDAL) before uploading.

## What it does

1. Reads the 10 raw bands.
2. Computes 5 spectral indices (NDVI, MNDWI, BSI, MBI, NDBI) using the
   exact same band indices as training (note: "nir" in the indices
   uses B8A, not B08 — this matches the original training code).
3. Standardizes all 15 channels using the image's own per-band mean/std.
4. **CNN**: tiles the image into patches and predicts one class per
   patch, painting that class across the whole patch. Patch stride is
   adjustable in the UI (1–16 px) so you can trade off speed vs. spatial
   detail — stride 16 is fast and blocky (non-overlapping tiles), lower
   strides give smoother, more granular output at the cost of more
   inference passes.
5. **RF**: predicts one class per pixel directly from the 15-channel
   feature vector.
6. Displays both classified maps, a legend, an agreement map, class
   distribution bar charts, download each classification as a georeferenced
   GeoTIFF, and counts the pixels per class in an AOI when you upload
   a GeoJSON with polygons.

## What's simplified vs. the full thesis pipeline

- No PostGIS/database connection — this app works on a single uploaded
  image at a time, with no persistence between sessions.
- No zonal statistics, transect analysis, or multi-date change tracking
  (that lives in the separate thesis validation/monitoring pipeline).
- Standardization stats and model weights are baked in; there's no
  retraining or fine-tuning UI.

## Known limitations of this version

- Standardization is computed per-uploaded-image (matching the training
  function), not from stored training-set statistics.
- The CNN assumes one label per patch (image classification, not
  per-pixel segmentation) — smaller strides interpolate this visually
  but don't add true per-pixel resolution.
- Edge patches are reflect-padded to reach a multiple of the patch size,
  then cropped back — this only affects the last row/column of patches
  at image edges.
