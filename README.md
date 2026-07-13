# Sentinel-2 Beach Land Cover Classifier

Streamlit app that classifies an uploaded Sentinel-2 GeoTIFF using both
your patch-based CNN and pixel-based Random Forest, side by side.

## Setup

```bash
pip install -r requirements.txt
```

Place your trained models here:

```
sentinel2_app/
├── models/
│   ├── cnn_model.pth   # full model, saved with torch.save(model, ...)
│   └── rf_model.pkl    # saved with joblib.dump(rf, ...)
├── app.py
└── src/
```

Update `CNN_PATH` / `RF_PATH` in `app.py` if your filenames differ.

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
   uses B8A, not B08 -- this matches the original training code).
3. Standardizes all 15 channels using the image's own per-band mean/std.
4. **CNN**: tiles the image into non-overlapping 16x16 patches, predicts
   one class per patch, and paints that class across the whole patch.
5. **RF**: predicts one class per pixel directly from the 15-channel
   feature vector.
6. Displays both classified maps, a legend, an agreement map, class
   distribution bar charts, and lets you download each classification
   as a georeferenced GeoTIFF.

## Known limitations / things to double check

- Standardization is computed per-uploaded-image (matching the training
  function), not from stored training-set statistics. If your models
  were actually trained with fixed global mean/std (not per-image), swap
  the standardization step in `src/preprocessing.py` accordingly.
- The CNN assumes one label per 16x16 patch (image classification, not
  per-pixel segmentation). If your CNN instead outputs a per-pixel mask
  per patch, `classify_cnn_patchwise` in `src/model_utils.py` needs a
  small change to keep the full per-pixel output instead of an argmax
  per patch.
- Edge patches are reflect-padded to reach a multiple of 16, then cropped
  back — this only affects the last row/column of patches at image edges.
- No nodata masking is applied; pixels outside the actual beach/AOI will
  still get classified.
