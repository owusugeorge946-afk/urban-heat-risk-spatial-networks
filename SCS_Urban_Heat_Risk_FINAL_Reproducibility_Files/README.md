# FINAL SCS Reproducibility Files

Manuscript:
**Spatial Networks of Urban Surface Heat Risk: Connectivity, Communities and Neighbourhood Reinforcement across Four West African Coastal Cities**

This package contains the reconstructed final analysis workflow for the manuscript.

## Main files
1. `code/01_GEE_FINAL_200_TREES.js`
   - Uses the original 20 Earth Engine Code Editor training Imports.
   - Uses the four confirmed city-boundary assets.
   - Uses a 200-tree Random Forest, as specified for the final reproducibility workflow.
   - Produces 2024 LULC, validation statistics, Landsat LST/NDVI/NDBI, HRI and 500 m node exports.

2. `code/02_COLAB_NETWORK_ANALYSIS.py`
   - Builds the environmental-similarity spatial networks.
   - Computes topology and centrality.
   - Runs Leiden community detection.
   - Runs HRI propagation.
   - Runs the reported sensitivity tests.
   - Writes a manuscript benchmark audit.

## Critical GEE instruction
Do NOT create new training samples. Open the existing Earth Engine script that currently shows the 20 Imports, retain those Imports, and replace/paste the JavaScript body with `01_GEE_FINAL_200_TREES.js`. The JavaScript expects the imported variable names exactly as they currently appear.

## Execution
Run GEE separately for Accra, Lagos, Abidjan and Freetown. Download/export the four node GeoJSON files to Google Drive, then run the Colab script.

## Provenance
The original training geometries survive in the user's Earth Engine Code Editor Imports. Their coordinates are not embedded in this ZIP because the screenshot/code export does not contain the import geometry metadata. Therefore the existing GEE script with its Imports is part of the reproducibility record and should be preserved.
