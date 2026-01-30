<p align="center">
  <img src="assets/classpose_logo.png" alt="Classpose logo" width="720" />
</p>

# Classpose: foundation model-driven cell phenotyping in WSI

Semantic cell classification built on top of Cellpose, with a whole slide image (WSI) workflow and a QuPath extension for integrated inference and import.

This project:

- Extends the [Cellpose](https://github.com/MouseLand/cellpose) stack to output an additional semantic segmentation mask alongside standard Cellpose outputs.
- Beefs up the training routine with improved augmentations, improved loss function with a classification objective and adaptive loss weights, sparse annotation support and minority class over-sampling.
- Provides a WSI inference entrypoint that handles tissue/artefact detection and produces outputs (GeoJSON and SpatialData) for easy downstream spatial analysis.
- Includes a [QuPath extension](./qupath-extension-classpose) to run the Python predictor on the currently opened slide and import results into QuPath.
- Extensively benchmarks Classpose against state-of-the-art approaches (Semantic Cellpose-SAM, StarDist, and CellViT++) across multiple datasets and demonstrate its detection superiority [here](https://www.biorxiv.org/content/10.64898/2025.12.18.695211v1).

## Features

- Adds a semantic head to leverage generalist Cellpose weights for classification.
- WSI pipeline with tissue and optional artefact detection (GrandQC-based components in `src/classpose/grandqc/`).
- CLI entrypoint: `classpose-predict-wsi` (also available via `python -m classpose.entrypoints.predict_wsi`).
- QuPath integration: [qupath-extension-classpose](./qupath-extension-classpose) with UI, live logging, and auto-import.

## Installation

Prerequisites:

- [`uv`](https://docs.astral.sh/uv/)
- [`git`](https://git-scm.com/install/)

To use this as a CLI you just need to migrate to this directory and append `uv run` to every command you run. This ensures that the correct Python version and dependencies are used. To simply install all dependencies in a virtual environment, run `uv sync`. This will create a virtual environment in `./.venv` and install all dependencies.

## Quick start (CLI)

The WSI predictor is exposed as a console script and a module entrypoint (`uv run classpose-predict-wsi`).

Required arguments:

```
  --model_config MODEL_CONFIG
                        One of 'conic', 'consep', 'glysac', 'monusac', 'nucls', 'puma' or a path to a model config YAML file.
  --slide_path SLIDE_PATH
                        Path to the whole-slide image to process (e.g. .svs, .tiff; any
                        format supported by OpenSlide).
  --output_folder OUTPUT_FOLDER
                        Path to save the output files (basename_cell_contours.geojson,
                        basename_cell_centroids.geojson, basename_tissue_contours.geojson, and
                        basename_artefact_contours.geojson).
```

Optional arguments:

```
  --tissue_detection_model_path TISSUE_DETECTION_MODEL_PATH
                        Path to the GrandQC tissue detection model weights. If specified but not
                        found it will be downloaded.
  --artefact_detection_model_path ARTEFACT_DETECTION_MODEL_PATH
                        Path to GrandQC artefact detection model. If specified, detects artefact
                        regions and optionally filters cells in those regions.
  --filter_artefacts, --no-filter_artefacts
                        If enabled and --artefact_detection_model_path is provided, removes
                        cells detected in artefact regions from the outputs.
  --roi_geojson ROI_GEOJSON
                        FeatureCollection with (Multi)Polygon(s) in level-0 coordinates used
                        to restrict inference and (optionally) compute per-ROI-class densities.
  --roi_class_priority ROI_CLASS_PRIORITY [ROI_CLASS_PRIORITY ...]
                        Space-separated list of ROI class names in priority order for
                        overlapping regions (used for density calculations only). Cells in
                        overlapping ROIs will be assigned to the first matching class in this
                        list; classes not in the list are checked after, in their natural order.
  --min_area MIN_AREA   Minimum area of the tissue polygons.
  --tta, --no-tta       Triggers test time augmentation.
  --batch_size BATCH_SIZE
                        Batch size for inference.
  --device DEVICE       Device to use for inference. If None, automatically infers the device
                        and supports multi-device execution when available.
                        Multi-GPU execution can be specified as 'cuda:0,1' or 'cuda:0,1,2,3'.
  --bf16, --no-bf16     Enables bfloat16 inference for supported devices.
  --tile_size TILE_SIZE
                        Tile size for inference.
  --overlap OVERLAP     Tile overlap for inference.
  --output_type {csv,spatialdata} [{csv,spatialdata} ...]
                        Optional output type(s). 'csv' generates cellular density statistics
                        (cells/mm²) per class. 'spatialdata' generates a unified SpatialData Zarr
                        store containing all outputs. Can specify multiple types separated by
                        spaces (e.g. --output_type csv spatialdata).
```

**Important:**

- **Tissue / artefact models:** specifying either `--tissue_detection_model_path` or
  `--artefact_detection_model_path` will download the corresponding GrandQC models to that
  path if they are not present. This makes use of the GrandQC models available in Zenodo
  ([tissue model](https://zenodo.org/records/14507273) and
  [artefact model](https://zenodo.org/records/14041538)). Please note that if you use any part of Classpose which makes use of GrandQC please follow the instructions at [here](https://github.com/cpath-ukk/grandqc/tree/main) to cite them appropriately. Similarly to Classpose, GrandQC is under a non-commercial license whose terms can be found at [here](https://github.com/cpath-ukk/grandqc/blob/main/LICENSE).
- **Output types:** if you request `--output_type csv` or `--output_type spatialdata` (or
  both), `--tissue_detection_model_path` **must** be provided; otherwise the CLI will error.
- **ROI-aware densities:** when `--roi_geojson` and `--output_type` include `csv` and/or
  `spatialdata`, densities are computed per ROI class. If `--roi_class_priority` is given,
  it is used to resolve cells that fall into overlapping ROIs.

Examples:

```bash
# Using the console script (declared in pyproject.toml)
classpose-predict-wsi \
  --model_config conic \
  --slide_path /path/to/slide.svs \
  --output_folder /tmp/classpose_out \
  --output_type csv spatialdata \
  --device mps  # or cuda:0 or cpu

# Using the module entrypoint
python -m classpose.entrypoints.predict_wsi \
  --model_config conic \
  --slide_path /path/to/slide.svs \
  --output_folder /tmp/classpose_out \
  --output_type csv
```

Outputs include raster masks and GeoJSONs (e.g., cells, centroids, tissue; artefacts if enabled). Optionally, CSV files with cellular density statistics and/or a SpatialData Zarr store can be generated when the tissue detection model is provided.

### Model configurations

Model configurations are specified using the `--model_config` argument. There are two ways of specifying model configurations: the "plug-and-play" way and the "local/custom" way.

#### Plug-and-play

The following model configurations are available:

- `conic`
- `consep`
- `glysac`
- `monusac`
- `nucls`
- `puma`

These will automatically download the model to the path defined in the `CLASSPOSE_MODEL_DIR` environment variable (default: `~/.classpose_models`). The `nucls` model may produce lower-quality results compared to other models. Take extra caution when using this model.

#### Local/custom

If users want to use their own model configurations: they can! The model configurations must be specified as a path to a YAML file. The YAML file must contain the following keys:

- `path`: Path to the model weights.
- `mpp`: MPP of the slide.
- `url`: URL to the model weights (optional).
- `hf`: HuggingFace model identifier (optional).
- `cell_types`: List of cell types (optional). 

**Example:**

```yaml
path: /path/to/model.pt
mpp: 0.5
url: https://example.com/model.pt
hf:
  repo_id: example/model
  filename: model.pt
cell_types:
  - "Cell Type 1"
  - "Cell Type 2"
```

## Training

Classpose supports custom model training with features like sparse annotation support, class weighting, and uncertainty-based loss balancing.

**Basic example:**

```python
from classpose.models import ClassposeModel
from classpose.train import train_class_seg

# Initialize model
model = ClassposeModel(gpu=True, pretrained_model="cpsam", nclasses=6)

# Train (expects data as lists of numpy arrays with shape (C, H, W))
# Labels should have 2 channels: [instances, class] where the last channel contains class labels
# Optionally 4 channels: [instances, flow_y, flow_x, class] if flows are pre-computed
train_class_seg(
    net=model.net,
    train_data=train_images,      # List of image arrays
    train_labels=train_labels,    # List of label arrays 
    test_data=test_images,        # Optional validation data
    test_labels=test_labels,
    n_epochs=100,
    batch_size=4,
    save_path="./models",
    model_name="my_classpose_model"
)
```

For a complete training example with data loading, oversampling, and advanced options, see [`paper_experiments/run_training.py`](./paper_experiments/run_training.py).

## QuPath extension

The companion QuPath extension lives in `qupath-extension-classpose/` and makes Classpose directly accessible from QuPath with only a few configuration steps.

Build and installation instructions are in [`qupath-extension-classpose`](./qupath-extension-classpose).

https://github.com/user-attachments/assets/1a49a140-e6fe-4e3e-9c8f-49d0ed785ed3

## Citation

Soham Mandal*, José Guilherme de Almeida*, Nickolas Papanikolaou, and Trevor A. Graham. *“Classpose: Foundation Model-Driven Whole Slide Image-Scale Cell Phenotyping in H&E.”* bioRxiv. bioRxiv, December 22, 2025. https://doi.org/10.64898/2025.12.18.695211.

*joint first authors

If you use this project in your work, please cite Cellpose and Classpose.
```bibtex
@article {Mandal2025.12.18.695211,
	author = {Mandal, Soham and de Almeida, Jos{\'e} Guilherme and Papanikolaou, Nickolas and Graham, Trevor A},
	title = {Classpose: foundation model-driven whole slide image-scale cell phenotyping in H\&E},
	elocation-id = {2025.12.18.695211},
	year = {2025},
	doi = {10.64898/2025.12.18.695211}
}
```
