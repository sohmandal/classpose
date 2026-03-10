import numpy as np
import pytest
from pathlib import Path

from classpose.utils import download_if_unavailable
from classpose.wsi_utils import CZISlide


@pytest.fixture
def test_data_dir():
    path = Path("tests/data")
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def zeiss_czi(test_data_dir):
    path = test_data_dir / "Zeiss-5-Cropped.czi"
    url = "https://openslide.cs.cmu.edu/download/openslide-testdata/Zeiss/Zeiss-5-Cropped.czi"
    download_if_unavailable(str(path), url)
    return path


def test_czi_slide_properties_and_levels(zeiss_czi):
    with CZISlide(str(zeiss_czi)) as slide:
        assert slide.level_count == 5
        assert slide.level_downsamples == [1, 2, 4, 8, 16]
        assert slide.level_dimensions[0][0] > 0
        assert slide.level_dimensions[0][1] > 0
        assert slide.properties["openslide.vendor"] == "zeiss"
        assert slide.properties["openslide.mpp-x"] > 0
        assert slide.properties["openslide.mpp-y"] > 0


def test_czi_slide_read_region(zeiss_czi):
    with CZISlide(str(zeiss_czi)) as slide:
        tile = slide.read_region((0, 0), level=0, size=(256, 128))

    assert isinstance(tile, np.ndarray)
    assert tile.shape == (128, 256, 3)
    assert tile.dtype == np.uint8


def test_czi_slide_thumbnail(zeiss_czi):
    with CZISlide(str(zeiss_czi)) as slide:
        thumb = slide.get_thumbnail((256, 256))

    assert isinstance(thumb, np.ndarray)
    assert thumb.ndim == 3
    assert thumb.shape[0] > 0
    assert thumb.shape[1] > 0
    assert thumb.shape[2] >= 3
