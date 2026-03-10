import os
import json
import pytest
import shutil
import torch
from pathlib import Path
from classpose.entrypoints.predict_wsi import main as predict_wsi_main
from classpose.entrypoints.predict_wsi_cpsam import (
    main as predict_wsi_cpsam_main,
)
from classpose.utils import download_if_unavailable


@pytest.fixture
def test_data_dir():
    path = Path("tests/data")
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def small_region_svs(test_data_dir):
    path = test_data_dir / "CMU-1-Small-Region.svs"
    url = "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs"
    download_if_unavailable(str(path), url)
    return path


@pytest.fixture
def jp2k_33005_svs(test_data_dir):
    path = test_data_dir / "CMU-1-JP2K-33005.svs"
    url = "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-JP2K-33005.svs"
    download_if_unavailable(str(path), url)
    return path


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "output"


def test_predict_wsi_integration(small_region_svs, output_dir):
    """
    Test predict_wsi.py with CMU-1-Small-Region.svs
    """
    assert small_region_svs.exists(), f"Test slide {small_region_svs} not found"

    output_dir.mkdir(parents=True, exist_ok=True)

    args = type(
        "Args",
        (),
        {
            "model_config": "conic",
            "slide_path": str(small_region_svs),
            "output_folder": str(output_dir),
            "tissue_detection_model_path": None,
            "artefact_detection_model_path": None,
            "filter_artefacts": False,
            "roi_geojson": None,
            "roi_class_priority": None,
            "min_area": 0,
            "tta": False,
            "batch_size": 1,
            "device": "cuda",
            "tile_size": 256,
            "bf16": False,
            "overlap": 64,
            "output_type": None,
        },
    )

    predict_wsi_main(args)

    # Verify outputs
    basename = small_region_svs.stem
    assert (output_dir / f"{basename}_cell_contours.geojson").exists()
    assert (output_dir / f"{basename}_cell_centroids.geojson").exists()
    with open(output_dir / f"{basename}_cell_contours.geojson") as f:
        data = json.load(f)
        assert data["type"] == "FeatureCollection"


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Needs at least 2 GPUs"
)
def test_predict_wsi_multi_gpu_integration(small_region_svs, output_dir):
    """
    Test predict_wsi.py with 2 GPUs
    """
    assert small_region_svs.exists()

    output_dir.mkdir(parents=True, exist_ok=True)

    args = type(
        "Args",
        (),
        {
            "model_config": "conic",
            "slide_path": str(small_region_svs),
            "output_folder": str(output_dir),
            "tissue_detection_model_path": None,
            "artefact_detection_model_path": None,
            "filter_artefacts": False,
            "roi_geojson": None,
            "roi_class_priority": None,
            "min_area": 0,
            "tta": False,
            "batch_size": 1,
            "device": "cuda:0,1",
            "tile_size": 256,
            "bf16": False,
            "overlap": 64,
            "output_type": None,
        },
    )

    predict_wsi_main(args)

    basename = small_region_svs.stem
    assert (output_dir / f"{basename}_cell_contours.geojson").exists()
    assert (output_dir / f"{basename}_cell_centroids.geojson").exists()


def test_predict_wsi_cpsam_integration(small_region_svs, output_dir):
    """
    Test predict_wsi_cpsam.py with CMU-1-Small-Region.svs
    """
    assert small_region_svs.exists()

    output_dir.mkdir(parents=True, exist_ok=True)

    args = type(
        "Args",
        (),
        {
            "model_path": "cpsam",
            "slide_path": str(small_region_svs),
            "output_folder": str(output_dir),
            "train_mpp": 0.5,
            "tissue_detection_model_path": None,
            "artefact_detection_model_path": None,
            "filter_artefacts": False,
            "roi_geojson": None,
            "roi_class_priority": None,
            "min_area": 0,
            "tta": False,
            "batch_size": 1,
            "device": "cuda",
            "tile_size": 256,
            "bf16": False,
            "overlap": 64,
            "output_type": None,
        },
    )

    predict_wsi_cpsam_main(args)

    basename = small_region_svs.stem
    assert (output_dir / f"{basename}_cell_contours.geojson").exists()
    assert (output_dir / f"{basename}_cell_centroids.geojson").exists()
