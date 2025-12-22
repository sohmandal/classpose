# QuPath Classpose Extension

Run Classpose whole-slide inference from QuPath and import the resulting annotations. We support automatic tissue and artefact detection as well as inference using a user-defined ROI.

## Overview

- Adds a menu entry: `Extensions > Classpose > Predict WSI...`
- Runs the Classpose WSI predictor (Python) on the currently open slide
- Imports generated GeoJSONs (cells, centroids, tissue, artefacts) back into QuPath
- Provides a live log window and a Cancel button while inference runs
- Caches the Python executable path for convenience
- Auto-resolves fixed GrandQC model locations next to the installed JAR

## Requirements

- QuPath 0.6.x
- A Python environment with:
  - `classpose` (your repository installed)
  - `torch`, `torchvision`
  - `openslide-python`
  - `grandqc` dependencies as used by `classpose`
- A trained Classpose model file (path provided in the dialog)

## Installation

Requirements:
- JDK 21+
- Gradle (or use wrapper)
- Cloning the [Classpose](https://github.com/sohmandal/classpose) repo

1) Clone the Classpose repo:
```bash
git clone https://github.com/sohmandal/classpose.git
cd classpose
```

2) Go to the QuPath extension directory:
```bash
cd qupath-extension-classpose
```

3) Build the extension (or use the provided JAR if available) with Gradle:
- `gradle build` if Gradle is installed
- **Wrapper**: `./gradlew build` (or `gradlew.bat build` on Windows) otherwise

This produces `build/libs/qupath-extension-classpose-<version>.jar`.

4) Install the JAR:

- Inside QuPath, go to `Extensions > Manage extensions`
- Click "Open extension directory"
- Copy `build/libs/qupath-extension-classpose-<version>.jar` to the extensions directory.
- Restart QuPath.

5) Verify it appears under `Extensions > Classpose > Predict WSI...`

For your convenience, we provide a pre-built JAR file at [`build/libs/qupath-extension-classpose-0.1.0-SNAPSHOT.jar`](build/libs/qupath-extension-classpose-0.1.0-SNAPSHOT.jar).

## Usage

### Predict WSI

1) Open a local WSI in QuPath.
2) Choose `Extensions > Classpose > Predict WSI...`.
3) Fill the following:
   - Python executable (cached after first run)
   - Model path (your trained Classpose weights)
   - Output folder
   - Optional: output types (CSV for cellular density statistics, SpatialData for a unified Zarr store), artefact filtering, TTA (test-time augmentation), device (e.g., `cuda:0`, `mps`, `cpu`), batch size, MPP, tile size, overlap, min tissue area, labels
4) The extension automatically:
   - Resolves the slide path from the current image
   - Uses fixed GrandQC model paths under `<QuPath extensions folder>/classpose-models/`
   - Downloads GrandQC weights via your Python code if missing
5) Click Run. A live log window opens; you can Cancel at any time.
6) On success, cell, centroid, tissue, and artefact annotations (if requested) are imported automatically. Additionally, CSV density statistics and/or a SpatialData Zarr store are generated based on the selected output types.

### Tissue detection

1) Open a local WSI in QuPath.
2) Choose `Extensions > Classpose > Tissue detection...`.
3) Fill the following:
   - Python executable (cached after first run)
   - Model path (your trained Classpose weights)
   - Output folder
   - Optional: device, min tissue area
4) Click Run. A live log window opens; you can Cancel at any time.
5) On success, tissue annotations are imported automatically.

### Artefact detection

1) Open a local WSI in QuPath.
2) Choose `Extensions > Classpose > Artefact detection...`.
3) Fill the following:
   - Python executable (cached after first run)
   - Model path (your trained Classpose weights)
   - Output folder
   - Optional: device (e.g., `cuda:0`, `mps`, `cpu`), MPP, min tissue area
4) Click Run. A live log window opens; you can Cancel at any time.
5) On success, artefact annotations are imported automatically.

### Notes

- **Output types:** The extension supports generating additional outputs beyond GeoJSON annotations. Select "CSV" to get cellular density statistics (cells/mm² per class), and/or "SpatialData" for a unified Zarr store containing all results. These require tissue detection to be enabled.
- **Hierarchy:** if you want to have the objects contained in their respective tissue fragments, you can use QuPath's built-in methods in `Objects > Annotations > Expand annotations`.
- **Performance:** while ROI inference is _possible_ on CPU chips it is not recommended as it may take a significant amount of time to complete. We recommend using either MPS (Apple Silicon) for small ROIs or CUDA for large ROIs/workflows segmenting cells in large tissue sections.

### Potential issues

- **Tissue detection:** tissue detection (from GrandQC) works remarkable well. However, users may want to try running it for a few samples before running Classpose to avoid running Classpose on poorly inferred tissues.
   - **Solution:** if tissue detection performs poorly (extremely rare in our experience), we recommend using user-defined ROI instead. While more laborious, it guarantees that your results are accurate.
- **Artefact detection:** artefacts are detected using the GrandQC model, but cells are not filtered using artefacts - we only use tissue segmentations (also from GrandQC) to this effect. Artefacts, while detected, oftentimes contain false positives
   - **Solution:** we recommend that users use their judgement on including/removing them from their analysis. Using hierarchies (as described above), removing cells contained within confirmed artefacts becomes trivial.

## Troubleshooting

- Live log window streams Python stdout/stderr
- A log file is written to the output folder: `classpose_predict.log`
