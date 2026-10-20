# Sentinel-2 Beach Land Cover Classification

A lightweight Streamlit application for classifying coastal land cover from Sentinel-2 satellite imagery using Convolutional Neural Networks (CNNs) and Random Forest (RF) models.

## Overview

This project is a simplified demonstration of the land-cover classification models developed as part of my MSc thesis on coastal monitoring and shoreline evolution in Barcelona.

The application packages pretrained model artifacts in an interactive interface where users can upload satellite imagery, visualize classification results, compare model predictions, and export georeferenced maps.

This repository contains a **lite version of the models and inference workflow**, designed for demonstration and reproducibility. It does not necessarily include the latest or final models from the thesis. The full thesis covers the broader research methodology and processing pipeline.

## Objectives

- Demonstrate CNN- and Random Forest-based land-cover classification using Sentinel-2 imagery.
- Explore spectral feature engineering through NDVI, MNDWI, BSI, MBI, and NDBI.
- Compare classification outputs through interactive maps, class distributions, and model agreement.
- Generate georeferenced GeoTIFF outputs for further analysis in GIS software.
- Provide a reproducible, containerized application that can be run locally.

## Using the Project

### Requirements

- Docker Desktop or another working Docker installation.
- A Sentinel-2 GeoTIFF containing 10 bands in the expected order.
- Optionally, a GeoJSON file containing areas of interest for zonal statistics.

The required band order is:

`B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12`

### Running the application

Clone or download the repository and open a terminal in its root directory.

Build the Docker image:

```bash
./build.sh
```

Run the application:

```bash
./run.sh
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

To stop the application, press `Ctrl+C` in the terminal.

### Application workflow

1. Upload a compatible 10-band Sentinel-2 GeoTIFF.
2. Configure the available CNN inference settings.
3. Run the CNN and Random Forest classifications.
4. Explore the classification maps, class distributions, and agreement between predictions.
5. Download the georeferenced classification outputs.
6. Optionally, upload GeoJSON polygons to summarize predicted class counts within areas of interest.

### Repository structure

```text
classifier_app/
├── app.py
├── models/
│   ├── cnn_model.pth
│   └── rf_model.pkl
├── src/
│   ├── architectures.py
│   ├── model_utils.py
│   ├── preprocessing.py
│   ├── visualization.py
│   └── zonal_stats.py
├── pyproject.toml
├── Dockerfile
├── .dockerignore
├── build.sh
├── run.sh
└── README.md
```

## Building on the Project

### Technology stack

- **Python:** application logic and model inference.
- **PyTorch:** CNN architecture and inference.
- **scikit-learn:** Random Forest classification.
- **Rasterio and GeoPandas:** raster processing and geospatial analysis.
- **NumPy and pandas:** numerical processing and data manipulation.
- **Matplotlib and Streamlit:** visualization and interactive interface.
- **Docker:** reproducible application deployment.

### Extending the application

The application provides a starting point for further development, including:

- Integrating updated or retrained models from the full research pipeline.
- Adding independent validation datasets and quantitative accuracy metrics.
- Supporting multi-temporal image comparisons and coastal change analysis.
- Extending geospatial summaries and exporting results for downstream GIS workflows.

### Relationship to the MSc thesis

The full MSc thesis investigates coastal monitoring and shoreline evolution in Barcelona using satellite imagery and geospatial analysis. It contains the broader research methodology and processing pipeline beyond the simplified inference workflow provided here.

This repository complements that work by offering an accessible, interactive demonstration of land-cover classification. Its included models and outputs should be treated as exploratory and should not be assumed to represent the final validated thesis results.
