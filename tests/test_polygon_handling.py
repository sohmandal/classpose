import pytest
import shapely
from classpose.entrypoints.predict_wsi import make_valid, get_maximum_lengths

polygon = {
    "type": "MultiPolygon",
    "coordinates": [
        [
            [
                [9520, 14217],
                [12017, 17987],
                [14620.19, 15975.51],
                [13087, 11312],
                [9520, 14217],
                [14620.19, 15975.51],
                [15533, 18752],
                [15992, 16968],
                [15735.36, 15113.82],
                [14620.19, 15975.51],
                [15329, 12178],
                [15735.36, 15113.82],
                [17622, 13656],
                [15329, 12178],
            ]
        ],
    ],
}


@pytest.fixture
def invalid_polygon_shapely():
    return shapely.geometry.shape(polygon)


def test_get_maximum_lengths_error(invalid_polygon_shapely):
    with pytest.raises(shapely.errors.GEOSException):
        get_maximum_lengths(invalid_polygon_shapely)


def test_get_maximum_lengths_make_valid(invalid_polygon_shapely):
    get_maximum_lengths(make_valid(invalid_polygon_shapely))
