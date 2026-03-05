import os
import pytest
import numpy as np
from pathlib import Path
from openslide import OpenSlide
from classpose.grandqc.wsi_tissue_detection import detect_tissue_wsi
from classpose.grandqc.wsi_artefact_detection import detect_artefacts_wsi
from classpose.utils import download_if_unavailable


@pytest.fixture
def test_data_dir():
    path = Path("tests/data")
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def jp2k_33005_svs(test_data_dir):
    path = test_data_dir / "CMU-1-JP2K-33005.svs"
    url = "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-JP2K-33005.svs"
    download_if_unavailable(str(path), url)
    return path


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "output_grandqc"


def test_tissue_detection_integration(jp2k_33005_svs, output_dir):
    """
    Test GrandQC tissue detection with CMU-1-JP2K-33005.svs
    """
    assert jp2k_33005_svs.exists(), f"Test slide {jp2k_33005_svs} not found"

    output_dir.mkdir(parents=True, exist_ok=True)

    slide = OpenSlide(str(jp2k_33005_svs))

    # Run tissue detection
    results = detect_tissue_wsi(
        slide=slide, device="cpu", min_area=100  # Force CPU for tests
    )

    image, mask, filled_class_map, output_cnts, geojson, mpp = results

    assert image is not None
    assert isinstance(mask, np.ndarray)
    assert len(output_cnts) >= 0
    assert geojson["type"] == "FeatureCollection"
    assert mpp > 0


def test_artefact_detection_integration(jp2k_33005_svs, output_dir):
    """
    Test GrandQC artefact detection with CMU-1-JP2K-33005.svs
    """
    assert jp2k_33005_svs.exists()

    output_dir.mkdir(parents=True, exist_ok=True)

    slide = OpenSlide(str(jp2k_33005_svs))

    # Run artefact detection
    results = detect_artefacts_wsi(slide=slide, device="cuda", min_area=100)

    artefact_mask, artefact_map, artefact_cnts, geojson = results

    assert isinstance(artefact_mask, np.ndarray)
    assert isinstance(artefact_map, np.ndarray)
    assert isinstance(artefact_cnts, dict)
    assert geojson["type"] == "FeatureCollection"
