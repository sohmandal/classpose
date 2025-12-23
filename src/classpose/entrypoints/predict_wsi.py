"""
Predict Classpose cells and centroids for a whole-slide image.

The workflow is as follows:
1. A tile loader is instantiated. This tile loader filters out tiles
    which are unlikely to contain tissue. This works as a background process.
2. A post processor is instantiated. This post processor processes the tiles
    in parallel and stores the results in a shared memory list. This works as a
    background process.
3. Begin the prediction using Classpose. This supports multi-GPU inference using
    ``torch.multiprocessing`` and consumes tiles from 1 and inputs predictions
    into 2.
4. Given that predictions are performed with an overlap (avoiding edge artifacts)
    the final stage is cell de-duplication. This is performed with the final list
    of cellular coordinates by identifying cells whose centroids are very close to
    each other and keeping the largest one (heuristic: if a cell prediction happens
    at the edge of an image, it is likely to be smaller)
5. Optional tissue detection using GrandQC to identify tissue regions.
6. Optional artefact detection to filter out cells in artefact regions.
7. Optional ROI-based filtering to restrict analysis to specific regions.
8. Filter detected cells based on tissue and artefact contours.
9. Generate outputs including cell contours and centroids as GeoJSON files,
    tissue contours, artefact contours, and optionally cellular densities as CSV
    and/or a unified SpatialData Zarr store.
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="dask")
warnings.filterwarnings("ignore", category=UserWarning, module="xarray_schema")
warnings.filterwarnings("ignore", category=FutureWarning, module="spatialdata")
warnings.filterwarnings("ignore", message=".*ImplicitModification.*")
warnings.filterwarnings("ignore", message=".*Transforming to str index.*")

import argparse
import json
import logging
import re
import time
import uuid
from pathlib import Path
from multiprocessing.managers import SyncManager

import cv2
import numpy as np
import shapely
import torch
import torch.multiprocessing as tmproc
from matplotlib import colormaps
from openslide import OpenSlide
from scipy.spatial import KDTree
from skimage import color, measure
from tqdm import tqdm

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

from classpose.log import get_logger
from classpose.grandqc.wsi_artefact_detection import detect_artefacts_wsi
from classpose.grandqc.wsi_tissue_detection import detect_tissue_wsi
from classpose.model_configs import DEFAULT_MODEL_CONFIGS, ModelConfig
from classpose.models import ClassposeModel
from classpose.utils import get_device, get_slide_resolution
from classpose.entrypoints.outputs import (
    calculate_cellular_densities,
    create_spatialdata_output,
    create_valid_polygon,
    map_cells_to_roi_classes,
)

logger = get_logger("classpose")


DEFAULT_TRAIN_MPP = 0.5
DEFAULT_TILE_SIZE = 1024
DEFAULT_OVERLAP = 64
MAX_QUEUE_SIZE = 2048
MIN_TILE_SIZE = 256
COLORMAP = [[int(y * 255) for y in x] for x in colormaps["Set3"].colors]


class SlideLoader:
    """
    SlideLoader is a class that loads a slide and returns tiles for inference.
    """

    def __init__(
        self,
        slide_path: str,
        tile_size: int = DEFAULT_TILE_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        train_mpp: float = DEFAULT_TRAIN_MPP,
        manager: SyncManager | None = None,
        n_none: int = 1,
        tissue_detection_model_path: str | None = None,
        min_area: int = 0,
        roi_tree: shapely.STRtree | None = None,
        device: str | None = None,
    ):
        """
        Args:
            slide_path (str): Path to the slide.
            tile_size (int): Size of the tiles.
            overlap (int): Overlap between tiles.
            train_mpp (float): microns per pixel of the training data.
            manager (tmproc.Manager, optional): Manager to use for shared memory.
            n_none (int, optional): Number of None items to put in the queue. This
                is used to signal the end of the queue to other multiprocessing
                objects.
            tissue_detection_model_path (str, optional): Path to the GrandQC
                tissue detection model weights. If None, tissue detection is not
                performed.
            min_area (int, optional): Minimum area of the tissue polygons. Defaults
                to 0.
            roi_tree (shapely.STRtree, optional): STRtree of the ROI polygons.
                Defaults to None.
            device (str, optional): Device to use for inference. Defaults to None.
        """
        self.slide_path = slide_path
        self.tile_size = tile_size
        self.overlap = overlap
        self.train_mpp = train_mpp
        self.n_none = n_none
        self.tissue_detection_model_path = tissue_detection_model_path
        self.min_area = min_area
        self.roi_tree = roi_tree
        self.device = device

        if manager is None:
            manager = tmproc.Manager()

        self.n = manager.Value("i", 0)
        self.q = manager.Queue(maxsize=MAX_QUEUE_SIZE)
        self.ts = manager.Value("f", 0)
        self.mpp_x = manager.Value("f", 0)
        self.mpp_y = manager.Value("f", 0)
        self.tissue_cnts = manager.list()
        self.roi_cnts = manager.list()

        self.p = tmproc.Process(target=self.fill_queue)
        self.p.start()

    def _init_slide(self):
        self.slide = OpenSlide(self.slide_path)
        self.mpp = get_slide_resolution(self.slide)
        self.mpp_x.value = self.mpp[0]
        self.mpp_y.value = self.mpp[1]
        target_downsample = min(
            self.train_mpp / self.mpp[0], self.train_mpp / self.mpp[1]
        )
        self.level = self.slide.get_best_level_for_downsample(target_downsample)
        self.slide_dim = self.slide.level_dimensions[self.level]
        self.ts.value = self.slide.level_downsamples[self.level]
        if self.roi_tree is not None:
            self.coords = list(
                self._get_coords_roi(
                    self.tile_size, self.overlap, self.slide_dim, self.ts.value
                )
            )
        else:
            self.coords = list(
                self._get_coords(
                    self.tile_size, self.overlap, self.slide_dim, self.ts.value
                )
            )
        logger.info(f"Slide mpp: {self.mpp}")
        logger.info(f"Number of tiles: {len(self.coords)}")
        logger.info(f"Slide dimensions: {self.slide_dim}")
        logger.info(f"Tile size: {self.tile_size}")
        logger.info(f"Overlap: {self.overlap}")
        logger.info(f"Target downsample: {target_downsample}")
        logger.info(f"Slide downsample: {self.ts.value}")
        logger.info(f"Level: {self.level}")

    def _get_tissue_contours(self):
        if self.tissue_detection_model_path is not None:
            logger.info("Detecting tissue contours using GrandQC")
            _, _, _, tissue_cnts, _, _ = detect_tissue_wsi(
                slide=OpenSlide(self.slide_path),
                model_td_path=self.tissue_detection_model_path,
                min_area=self.min_area,
                device=self.device,
            )
            self.tissue_cnts.extend(
                [
                    shapely.Polygon(cnt["contour"], cnt["holes"])
                    for cnt in tissue_cnts.values()
                ]
            )
            logger.info(f"Number of tissue contours: {len(self.tissue_cnts)}")
        else:
            logger.info("Tissue detection not performed")

    def _get_coords_roi(
        self,
        tile_size: int,
        overlap: int,
        slide_dim: tuple[int, int],
        ts: float,
    ):
        """
        Extracts all tile coordinates given a tile size, overlap and slide dimensions.

        Args:
            tile_size (int): Size of the tiles.
            overlap (int): Overlap between tiles.
            slide_dim (tuple[int, int]): Dimensions of the slide.
            ts (float): Target downsample.

        Yields:
            tuple[int, int]: Tuple of coordinates (x, y).
        """
        if self.roi_tree is not None:
            logger.info(
                "Selecting tiles using ROI with %s polygons",
                len(self.roi_tree.geometries),
            )
            adj = self.overlap // 2
            for geom in self.roi_tree.geometries:
                self.roi_cnts.append(geom)
                coords = np.array(geom.exterior.coords.xy).T.astype(int)
                coords = (coords / ts).astype(int)
                cmin, cmax = coords.min(axis=0), coords.max(axis=0)
                cmin -= adj
                cmax += adj
                min_max_lens = int(min(get_maximum_lengths(geom)) / ts)
                cts = min(max(min_max_lens, MIN_TILE_SIZE), tile_size)
                for i in range(cmin[0], cmax[0], cts - overlap):
                    if (i + cts) > cmax[0]:
                        i = cmax[0] - cts
                    for j in range(cmin[1], cmax[1], cts - overlap):
                        if (j + cts) > cmax[1]:
                            j = cmax[1] - cts
                        yield ((int(i * ts), int(j * ts)), cts)

    def _get_coords(
        self,
        tile_size: int,
        overlap: int,
        slide_dim: tuple[int, int],
        ts: float,
    ):
        """
        Extracts all tile coordinates given a tile size, overlap and slide dimensions.

        Args:
            tile_size (int): Size of the tiles.
            overlap (int): Overlap between tiles.
            slide_dim (tuple[int, int]): Dimensions of the slide.
            ts (float): Target downsample.

        Yields:
            tuple[int, int]: Tuple of coordinates (x, y).
        """
        for i in range(0, slide_dim[0], tile_size - overlap):
            if (i + tile_size) > slide_dim[0]:
                break
            for j in range(0, slide_dim[1], tile_size - overlap):
                if (j + tile_size) > slide_dim[1]:
                    break
                yield ((int(i * ts), int(j * ts)), tile_size)

    def _point_to_square_polygon(
        self, point: tuple[int, int], tile_size: int
    ) -> shapely.Polygon:
        return shapely.Polygon(
            [
                (point[0], point[1]),
                (point[0] + tile_size, point[1]),
                (point[0] + tile_size, point[1] + tile_size),
                (point[0], point[1] + tile_size),
                (point[0], point[1]),
            ]
        )

    def fill_queue(self):
        """
        Fills the queue with tiles for inference.
        """
        self._init_slide()
        self._get_tissue_contours()
        n = 0
        with tqdm(self.coords, desc="Tiles to predict: 0", position=0) as pbar:
            for coords, tile_size in pbar:
                if self.tissue_cnts:
                    tile = self._point_to_square_polygon(coords, tile_size)
                    has_tissue = any(
                        [cnt.intersects(tile) for cnt in self.tissue_cnts]
                    )
                    if not has_tissue:
                        continue
                if self.roi_cnts:
                    tile = self._point_to_square_polygon(coords, tile_size)
                    has_roi = any(
                        [cnt.intersects(tile) for cnt in self.roi_cnts]
                    )
                    if not has_roi:
                        continue
                tile = self.slide.read_region(
                    coords, self.level, (tile_size, tile_size)
                )
                tile_array = np.array(tile)
                if tile_array.shape[-1] == 4:
                    tile_array = tile_array[:, :, :3]

                self.q.put((tile_array, coords))
                n += 1
                self.n.value += 1
                pbar.set_description(f"Tiles to predict: {n}")
        for _ in range(self.n_none):
            self.q.put((None, None))

    def __iter__(self):
        """
        Generic iterator for the queue.

        Yields:
            tuple[np.ndarray, tuple[int, int]]: Tuple of tile and coordinates.
        """
        while True:
            item = self.q.get()
            if item is None:
                break
            yield item

    def close(self):
        """
        Closes the queue and joins the process.
        """
        self.p.join()


class PostProcessor:
    """
    Post processor class for parallel post processing of tiles.

    The workflow consists in:

    1. Using the instance prediction to identify all cells in the tile
    2. Using the class prediction to identify the class of each cell
    3. Using the cell coordinates to create a polygon for each cell
    4. Filtering out invalid polygons as these represent unlikely predictions
    (i.e. the contour crosses itself at a given point)
    5. Extraction of some minimal features (area, perimeter, centroid) which
    can be used for deduplicating cells at a later stage
    6. Adding the polygons to a list for later writing to a GeoJSON file.
    """

    def __init__(self, labels: list[str], manager: tmproc.Manager = None):
        """
        Args:
            labels (list[str]): Ordered list of labels for the classes.
                Background (0) should not be included.
            manager (tmproc.Manager, optional): Manager to use for shared memory.
        """
        self.labels = labels

        if manager is None:
            manager = tmproc.Manager()

        self.n = manager.Value("i", 0)
        self.polygons = manager.Queue()
        self.value = manager.Value("i", 0)
        self.n_cells = manager.Value("i", 0)
        self.n_invalid_cells = manager.Value("i", 0)
        self.q = manager.Queue()
        self.p = tmproc.Process(target=self.run)
        self.p.start()

    def run(self):
        """
        Generic run method for the process.
        """
        while True:
            item = self.q.get()
            if item is None:
                break
            self(*item)

    def put(self, data: list[tuple]):
        """
        Puts data in the queue.

        Args:
            data (list[tuple]): Data to put in the queue.
        """
        self.q.put(data)

    def __call__(
        self,
        data: list[tuple[np.ndarray, np.ndarray]],
        batch_coords: list[tuple[int, int]],
        ts: float,
    ):
        """
        Data preprocessing following the aforementioned protocol. Appends everything
        to the polygons list, which is a shared memory list.

        Args:
            data (list[tuple]): Data to process. Should be a list of tuples, where each
                tuple contains the instance mask and class mask.
            batch_coords (list[tuple[int, int]]): List of coordinates for the batch.
            ts (float): Target downsample.
        """
        for datum, coords in zip(data, batch_coords):
            masks, class_masks = datum
            u = np.unique(masks)
            u = u[u > 0]
            curr_cells = []
            for i in u:
                cell_mask = masks == i
                curr_coords = (
                    cv2.findContours(
                        np.uint8(cell_mask),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )[0][0][:, 0]
                    * ts
                    + coords
                )
                # discard invalid polygons as these cannot be read in QuPath
                if curr_coords.shape[0] < 4:
                    self.n_invalid_cells.value += 1
                    continue
                polygon = shapely.Polygon(curr_coords)
                if not polygon.is_valid:
                    self.n_invalid_cells.value += 1
                    continue
                center = np.round(polygon.centroid.coords[0], 2).tolist()
                curr_coords = curr_coords.tolist()
                curr_coords.append(curr_coords[0])
                cl = class_masks[cell_mask][0]
                curr_cell = {
                    "type": "Feature",
                    "id": str(uuid.uuid4()),
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [curr_coords],
                    },
                    "properties": {
                        "objectType": "annotation",
                        "isLocked": False,
                        "classification": {
                            "name": self.labels[int(cl) - 1],
                            "color": COLORMAP[int(cl) - 1],
                        },
                        "measurements": [
                            {"name": "area", "value": polygon.area},
                            {"name": "perimeter", "value": polygon.length},
                            {
                                "name": "centroidX",
                                "value": center[0],
                            },
                            {
                                "name": "centroidY",
                                "value": center[1],
                            },
                        ],
                    },
                }
                curr_cells.append(curr_cell)
            self.polygons.put(curr_cells)
            self.n_cells.value += len(curr_cells)
            self.value.value += 1


def worker(
    dev: str,
    model_path: str,
    n_classes: int,
    fts: list[str],
    batch_size: int,
    tta: bool,
    slide_queue: tmproc.Queue,
    postproc_queue: tmproc.Queue,
    predicted_tiles: tmproc.Value,
    slide_queue_size: tmproc.Value,
    n_cells: tmproc.Value,
    n_invalid_cells: tmproc.Value,
    bsize: int = 256,
    target_downsample: float = 1,
    bf16: bool = False,
):
    """
    Worker function for parallel prediction of tiles. Takes a number of shared
    memory objects which are used to retrieve elements in a queue or iteratively
    updated. This makes heavy use of the ``torch.multiprocessing`` (tmp) module.

    Args:
        dev (str): Device to use for inference.
        model_path (str): Path to the model.
        n_classes (int): Number of classes.
        fts (list[str]): Feature transformation structure.
        batch_size (int): Batch size.
        tta (bool): Whether to use test time augmentation.
        slide_queue (tmproc.Queue): tmp Queue to retrieve tiles from.
        postproc_queue (tmproc.Queue): tmp Queue to send results to.
        predicted_tiles (tmproc.Value): tmp Value to count predicted tiles.
        slide_queue_size (tmproc.Value): tmp Value to count total number of tiles.
        n_cells (tmproc.Value): tmp Value to count number of cells.
        n_invalid_cells (tmproc.Value): tmp Value to count number of invalid cells.
        bsize (int): Batch size.
        target_downsample (float): Target downsample.
    """
    model = ClassposeModel(
        gpu=dev.type == "cuda",
        pretrained_model=model_path,
        device=dev,
        nclasses=n_classes,
        feature_transformation_structure=fts,
    )
    model.net.eval()
    if bf16:
        model.net = model.net.half()
    if isinstance(dev, str):
        dev = torch.device(dev)
    if dev.type == "cuda":
        model.net = torch.compile(model.net)

    with tqdm(
        None,
        desc="Predicted tiles (detected cells: 0)",
        position=1,
        total=0,
    ) as pbar:
        while True:
            tile, coords = slide_queue.get()
            if tile is None:
                break
            masks, raw_data, class_masks, styles = model.eval(
                [tile],
                batch_size=batch_size,
                augment=tta,
                bsize=bsize,
                compute_masks=True,
            )
            postproc_queue.put(
                (list(zip(masks, class_masks)), [coords], target_downsample)
            )
            predicted_tiles.value += 1

            pbar.n = predicted_tiles.value
            pbar.total = slide_queue_size.value
            pbar.set_description(
                f"Predicted tiles (detected cells: {n_cells.value}; invalid: {n_invalid_cells.value})"
            )
            pbar.refresh()

        postproc_queue.put(None)
        pbar.set_description(
            f"Predicted tiles (detected cells: {n_cells.value}; invalid: {n_invalid_cells.value})"
        )
        print()


def deduplicate(features: list[dict], max_dist: float = 15 / 2) -> list[dict]:
    """
    Deduplicate cells based on their size and location.

    Args:
        features (list[dict]): List of features (cells) to deduplicate.
        max_dist (float): Maximum distance between centers to consider as duplicates.

    Returns:
        list[dict]: Deduplicated list of features.
    """
    centers = []
    sizes = []
    logger.info("Deduplicating cells")
    logger.debug("Beginning deduplication")
    logger.debug("Extracting centers")
    for feature in features:
        measurements = feature["properties"]["measurements"]
        size = [x for x in measurements if x["name"] == "area"][0]["value"]
        center = [
            [x for x in measurements if x["name"] == "centroidX"][0]["value"],
            [x for x in measurements if x["name"] == "centroidY"][0]["value"],
        ]
        centers.append(center)
        sizes.append(size)

    logger.debug("Building KDTree")
    tree = KDTree(centers)
    logger.debug("KDTree built")

    logger.debug("Finding neighbours")
    neighbours = tree.query_pairs(max_dist)

    groups: dict[str, list] = {}
    member_to_group = {}
    for pair in neighbours:
        if (pair[0] not in member_to_group) and (
            pair[1] not in member_to_group
        ):
            group_idx = len(groups)
            groups[group_idx] = []
            member_to_group[pair[0]] = group_idx
            member_to_group[pair[1]] = group_idx
        else:
            if pair[0] in member_to_group:
                group_idx = member_to_group[pair[0]]
            else:
                group_idx = member_to_group[pair[1]]

        if pair[0] not in groups[group_idx]:
            groups[group_idx].append(pair[0])
        if pair[1] not in groups[group_idx]:
            groups[group_idx].append(pair[1])

    logger.debug("Removing based on size (keeping largest)")
    to_remove = {}
    for k in groups:
        group = groups[k]
        if len(group) > 1:
            curr_sizes = [sizes[i] for i in group]
            largest = group[np.argmax(curr_sizes)]
            for i in group:
                if i != largest and i not in to_remove:
                    to_remove[i] = True

    logger.debug("Generating final list of features (cells)")
    output = [features[i] for i in range(len(features)) if i not in to_remove]
    logger.info(f"Removed {len(to_remove)} duplicates.")
    logger.info(f"Number of cells: {len(output)}")
    return output


def shapely_polygon_to_geojson(
    polygon: shapely.Polygon | shapely.MultiPolygon,
    id: str = None,
    object_type: str = "annotation",
    additional_properties: dict | None = None,
) -> list[dict]:
    if isinstance(polygon, shapely.MultiPolygon):
        features = []
        for poly in polygon.geoms:
            features.extend(
                shapely_polygon_to_geojson(
                    poly,
                    id=None,
                    object_type=object_type,
                    additional_properties=additional_properties,
                )
            )
        return features

    coords = np.array(polygon.exterior.coords.xy).T
    center = coords.mean(axis=1).tolist()
    coords = coords.tolist()
    coords.append(coords[0])
    feature_id = str(uuid.uuid4())
    properties = {
        "objectType": object_type,
        "isLocked": False,
        "measurements": [
            {"name": "area", "value": polygon.area},
            {"name": "perimeter", "value": polygon.length},
            {
                "name": "centroidX",
                "value": center[0],
            },
            {
                "name": "centroidY",
                "value": center[1],
            },
        ],
    }
    if additional_properties is not None:
        properties.update(additional_properties)
    return [
        {
            "type": "Feature",
            "id": feature_id,
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
            "properties": properties,
        }
    ]


def load_roi_polygons(
    roi_geojson_path: str, group_by_class: bool = False
) -> (
    shapely.STRtree
    | tuple[shapely.STRtree, dict[str, list[shapely.Polygon]]]
    | None
):
    """
    Load ROI polygons from a GeoJSON file (FeatureCollection).

    Args:
        roi_geojson_path (str): Path to the GeoJSON file.
        group_by_class (bool): If True, returns tuple of (STRtree, dict mapping
            class names to polygon lists). If False, returns only STRtree.

    Returns:
        shapely.STRtree | tuple[shapely.STRtree, dict[str, list[shapely.Polygon]]] | None:
            STRtree of the polygons or None if the file is not found.
            If group_by_class=True, returns tuple of (STRtree, class_dict).
    """
    with open(roi_geojson_path, "r") as f:
        data = json.load(f)

    polys = []
    class_dict = {}

    for feat in data.get("features", []):
        geom = feat.get("geometry")
        if not geom:
            continue
        shp = shapely.geometry.shape(geom)

        class_name = None
        if group_by_class:
            props = feat.get("properties", {})
            classification = props.get("classification", {})
            class_name = classification.get("name", "unknown")

        if isinstance(shp, shapely.Polygon):
            polys.append(shp)
            if group_by_class:
                if class_name not in class_dict:
                    class_dict[class_name] = []
                class_dict[class_name].append(shp)
        elif isinstance(shp, shapely.MultiPolygon):
            for poly in shp.geoms:
                polys.append(poly)
                if group_by_class:
                    if class_name not in class_dict:
                        class_dict[class_name] = []
                    class_dict[class_name].append(poly)

    if group_by_class:
        class_counts = {k: len(v) for k, v in class_dict.items()}
        logger.info(
            f"Loaded ROI polygons per class: {class_counts} (total polygons: {len(polys)})"
        )

    if not polys:
        return None

    tree = shapely.STRtree(polys)

    if group_by_class:
        return tree, class_dict
    return tree


def get_maximum_lengths(
    polygon: shapely.Polygon, n_samples: int = 100
) -> tuple[int, int]:
    """
    Get the maximum spans of a polygon along the x and y axes.

    Args:
        polygon (shapely.Polygon): The polygon to get the maximum spans of.
        n_samples (int, optional): Number of samples to use for the calculation. Defaults to 100.

    Returns:
        tuple: A tuple containing the maximum spans of the polygon along the x and y axes.
    """
    x, y, x_max, y_max = polygon.bounds
    line_lengths_x: list[int] = []
    line_lengths_y: list[int] = []
    for i in np.linspace(x, x_max, n_samples):
        intersection = polygon.intersection(
            shapely.LineString([(i, y), (i, y_max)])
        )
        if isinstance(intersection, shapely.LineString):
            line_lengths_x.append(intersection.length)
        elif isinstance(intersection, shapely.MultiLineString):
            for line in intersection.geoms:
                if line.length > 0:
                    line_lengths_x.append(line.length)
    for i in np.linspace(y, y_max, n_samples):
        intersection = polygon.intersection(
            shapely.LineString([(x, i), (x_max, i)])
        )
        if isinstance(intersection, shapely.LineString):
            line_lengths_y.append(intersection.length)
        elif isinstance(intersection, shapely.MultiLineString):
            for line in intersection.geoms:
                line_lengths_y.append(line.length)

    return max(line_lengths_x), max(line_lengths_y)


def get_artefact_class_id(class_name: str) -> int:
    """
    Map artefact class name to ID for filtering.

    Args:
        class_name (str): Name of the artefact class.

    Returns:
        int: ID of the artefact class.
    """
    mapping = {
        "Fold": 2,
        "Darkspot & Foreign Object": 3,
        "PenMarking": 4,
        "Edge & Air Bubble": 5,
        "OOF": 6,
    }
    return mapping.get(class_name, 0)  # Return 0 for unknown classes


def get_cell_centroid(cell: dict) -> list[float]:
    """
    Extract the centroid coordinates from a cell feature.

    Args:
        cell (dict): Cell feature containing measurements.

    Returns:
        list[float]: List containing [x, y] centroid coordinates.
    """
    centroid_x = [
        x
        for x in cell["properties"]["measurements"]
        if x["name"] == "centroidX"
    ][0]["value"]
    centroid_y = [
        x
        for x in cell["properties"]["measurements"]
        if x["name"] == "centroidY"
    ][0]["value"]
    return [centroid_x, centroid_y]


def filter_cells_by_contours(
    polygons: list[dict], contours: list[shapely.Polygon]
) -> list[dict]:
    """
    Filter cells based on containment within given contours.

    Args:
        polygons (list[dict]): List of cell polygons in GeoJSON format.
        contours (list[shapely.Polygon]): List of contour polygons to filter against.

    Returns:
        list[dict]: Filtered list of polygons.
    """
    valid_contours = []
    for contour in contours:
        if not contour.is_valid:
            contour = contour.buffer(0)
            if not contour.is_valid:
                continue
        valid_contours.append(contour)

    if len(valid_contours) == 0:
        logger.warning("No valid contours found")
        return polygons

    contour_tree = shapely.STRtree(valid_contours)
    cell_centroids = [
        shapely.Point(get_cell_centroid(cell)) for cell in polygons
    ]
    keep, _ = contour_tree.query(cell_centroids, predicate="within")

    return [polygons[i] for i in keep]


def filter_cells_by_artefacts(
    cells: list[dict], artefact_cnts: dict
) -> tuple[list[dict], int, list[shapely.Polygon]]:
    """
    Filter cells that are contained within artefact regions.

    Args:
        cells (list[dict]): List of cell polygons in GeoJSON format.
        artefact_cnts (dict): Dictionary of artefact contours with format:
            {"id": {"contour": [...], "holes": [...]}}

    Returns:
        tuple[list[dict], int, list[shapely.Polygon]]:
            - Filtered cells
            - Count of removed cells
            - List of valid artefact polygons
    """
    # Convert artefact contours to Shapely polygons with holes
    artefact_contours = []
    for cnt_data in artefact_cnts.values():
        holes = cnt_data.get("holes", [])
        polygon = create_valid_polygon(cnt_data["contour"], holes=holes)
        if polygon is not None:
            artefact_contours.append(polygon)

    if not artefact_contours:
        return cells, 0, []

    artefact_tree = shapely.STRtree(artefact_contours)

    # Find cells contained in artefacts
    cells_in_artefacts_indices = set()
    for i, cell_data in enumerate(
        tqdm(cells, desc="Filtering cells by artefacts")
    ):
        cell_centroid = shapely.Point(get_cell_centroid(cell_data))

        try:
            if len(artefact_tree.query(cell_centroid, predicate="within")) > 0:
                cells_in_artefacts_indices.add(i)
        except Exception as e:
            logger.warning(f"Error checking cell {i}: {e}")
            continue

    # Keep cells that are not in artefacts
    filtered_cells = [
        cell
        for i, cell in enumerate(cells)
        if i not in cells_in_artefacts_indices
    ]
    removed_count = len(cells_in_artefacts_indices)

    return filtered_cells, removed_count, artefact_contours


def polygons_to_centroids(cells: list[dict]) -> list[dict]:
    """
    Convert a list of cell polygons to a list of centroids using the centroid properties.

    Args:
        cells (list[dict]): List of cells to convert.

    Returns:
        list[dict]: List of centroids.
    """
    output = []
    for cell in cells:
        centroidX = [
            x
            for x in cell["properties"]["measurements"]
            if x["name"] == "centroidX"
        ][0]["value"]
        centroidY = [
            x
            for x in cell["properties"]["measurements"]
            if x["name"] == "centroidY"
        ][0]["value"]
        output.append(
            {
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "geometry": {
                    "type": "Point",
                    "coordinates": [centroidX, centroidY],
                },
                "properties": {
                    "objectType": "annotation",
                    "isLocked": False,
                    "classification": cell["properties"]["classification"],
                    "measurements": cell["properties"]["measurements"],
                },
            }
        )
    return output


def infer_structure(
    model_path: str,
) -> tuple[list[int] | None, int]:
    """
    Infer the feature transformation network structure and the number of classes
    using the model weights.

    Args:
        model_path (str): Path to the model weights.

    Returns:
        tuple[dict[str, torch.nn.Parameter], list[int] | None, int]:
            - Model weights.
            - Feature transformation network structure.
            - Number of classes.
    """
    model_weights = torch.load(model_path, map_location="cpu")

    feature_transformation_structure = [
        model_weights[k].shape[0]
        for k in model_weights
        if re.search(
            r"out_class\.encoder_blocks\.[0-9]+\.block.conv1.weight", k
        )
    ]

    if len(feature_transformation_structure) == 0:
        feature_transformation_structure = None
    n_classes = model_weights["W3"].shape[1]

    logger.info(f"Number of classes: {n_classes}")

    if feature_transformation_structure:
        logger.info(
            f"Using UNet classification head with structure: {feature_transformation_structure}"
        )
    else:
        logger.info("Using one Conv2d classification head")

    del model_weights
    torch.cuda.empty_cache()

    return feature_transformation_structure, n_classes


def filter_tile(tile: np.ndarray) -> bool:
    """
    Filter out tiles that are not relevant for inference.

    Args:
        tile (np.ndarray): Tile to filter.

    Returns:
        bool: True if the tile is relevant for inference, False otherwise.
    """
    grey_level = tile.mean(-1)
    grey_level_hist, _ = np.histogram(grey_level, bins=25, range=[0, 255])
    grey_level_hist = grey_level_hist / grey_level_hist.sum()
    am = grey_level_hist.argmax()
    if np.all(
        [
            grey_level_hist[-1] < 0.25,
            grey_level_hist[0] < 0.25,
            grey_level_hist.max() < 0.9,
            am <= 23,
        ]
    ):
        blur = measure.blur_effect(grey_level)
        hed = color.rgb2hed(tile).reshape(-1, 3).max(0).tolist()

        return all([(blur < 0.5), (hed[0] > 0.01), (hed[1] > 0.01)])
    return False


def main(args):
    tmproc.set_start_method("spawn", force=True)

    if args.tile_size < MIN_TILE_SIZE:
        raise ValueError(
            f"Tile size must be at least {MIN_TILE_SIZE}, got {args.tile_size}"
        )

    if args.model_config in DEFAULT_MODEL_CONFIGS:
        model_config = ModelConfig(**DEFAULT_MODEL_CONFIGS[args.model_config])
    else:
        model_config = ModelConfig.load_from_yaml(args.model_config)
    model_config.download_if_necessary()

    if args.roi_geojson:
        output_types = args.output_type if args.output_type is not None else []
        need_class_grouping = any(
            ot in ["csv", "spatialdata"] for ot in output_types
        )
        roi_result = load_roi_polygons(
            args.roi_geojson, group_by_class=need_class_grouping
        )
        if need_class_grouping:
            roi_tree, roi_class_dict = roi_result
        else:
            roi_tree = roi_result
            roi_class_dict = None
    else:
        roi_tree = None
        roi_class_dict = None

    manager = tmproc.Manager()
    predicted_tiles_value = manager.Value("i", 0)

    devices = get_device(args.device)

    logger.info(f"Starting inference with model: {model_config.path}")
    fts, n_classes = infer_structure(model_config.path)

    if model_config.cell_types:
        if len(model_config.cell_types) != n_classes - 1:
            raise ValueError(
                f"Number of labels ({len(model_config.cell_types)}) does not match number of classes ({n_classes})"
            )
        logger.info(f"Using labels: {model_config.cell_types}")
        labels = model_config.cell_types
    else:
        labels = [str(i) for i in range(1, n_classes)]

    slide = SlideLoader(
        args.slide_path,
        tile_size=args.tile_size,
        overlap=args.overlap,
        train_mpp=model_config.mpp,
        manager=manager,
        n_none=len(devices),
        tissue_detection_model_path=args.tissue_detection_model_path,
        min_area=args.min_area,
        roi_tree=roi_tree,
        device=devices[0],
    )
    pp = PostProcessor(labels=labels, manager=manager)
    # Wait for slide to be initialized so that the target downsample is known
    while slide.ts.value == 0:
        time.sleep(0.1)
    ts = float(slide.ts.value)

    if len(devices) > 1:
        workers = []
        logger.info(f"Starting workers on devices: {devices}")
        for device in devices:
            logger.info(f"Starting worker on device: {device}")
            p = tmproc.Process(
                target=worker,
                args=(
                    device,
                    model_config.path,
                    n_classes,
                    fts,
                    args.batch_size,
                    args.tta,
                    slide.q,
                    pp.q,
                    predicted_tiles_value,
                    slide.n,
                    pp.n_cells,
                    pp.n_invalid_cells,
                    256,
                    ts,
                    args.bf16,
                ),
            )
            p.start()
            workers.append(p)
        for p in workers:
            p.join()
    else:
        worker(
            dev=devices[0],
            model_path=model_config.path,
            n_classes=n_classes,
            fts=fts,
            batch_size=args.batch_size,
            tta=args.tta,
            slide_queue=slide.q,
            postproc_queue=pp.q,
            predicted_tiles=predicted_tiles_value,
            slide_queue_size=slide.n,
            n_cells=pp.n_cells,
            n_invalid_cells=pp.n_invalid_cells,
            bsize=256,
            target_downsample=ts,
            bf16=args.bf16,
        )
    pp.p.join()
    slide.close()

    polygons = []
    with tqdm(desc="Collecting polygons") as pbar:
        while not pp.polygons.empty():
            polygons.extend(pp.polygons.get())
            pbar.update()
    logger.info(f"Number of detected cells: {len(polygons)}")
    logger.info(f"Number of invalid cells: {pp.n_invalid_cells.value}")
    if len(polygons) == 0:
        logger.warning("No cells detected")
        logger.info("Exiting")
        return

    logger.info("Creating GeoJSON file")
    polygons = deduplicate(list(polygons))

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    slide_basename = Path(args.slide_path).stem
    cell_contours_filename = f"{slide_basename}_cell_contours.geojson"
    cell_centroids_filename = f"{slide_basename}_cell_centroids.geojson"
    tissue_contours_filename = f"{slide_basename}_tissue_contours.geojson"
    artefact_contours_filename = f"{slide_basename}_artefact_contours.geojson"

    if args.roi_geojson:
        logger.info("Filtering cells based on ROI contours")
        roi_cnts = list(slide.roi_cnts)
        polygons = filter_cells_by_contours(polygons, roi_cnts)
        logger.info(f"Number of cells after filtering: {len(polygons)}")

    if args.tissue_detection_model_path is not None:
        logger.info("Filtering cells based on tissue contours")
        tissue_cnts = list(slide.tissue_cnts)
        polygons = filter_cells_by_contours(polygons, tissue_cnts)
        tissue_features = []
        for i, cnt in enumerate(tissue_cnts):
            tissue_features.extend(
                shapely_polygon_to_geojson(
                    cnt,
                    id=f"tissue_{i}",
                    object_type="annotation",
                    additional_properties={
                        "classification": {"name": "tissue", "color": [0, 0, 0]}
                    },
                )
            )
        total_tissue_area = sum(cnt.area for cnt in tissue_cnts)
        logger.info(f"Total tissue area: {total_tissue_area}")
        tissue_cnts_fmt = {
            "type": "FeatureCollection",
            "features": tissue_features,
        }
        logger.info(f"Number of cells after filtering: {len(polygons)}")
        logger.info(
            f"Saving tissue contours to {output_folder}/{tissue_contours_filename}"
        )
        with open(output_folder / tissue_contours_filename, "w") as f:
            json.dump(tissue_cnts_fmt, f)

    artefact_polygons = []
    if args.artefact_detection_model_path is not None:
        if args.tissue_detection_model_path is None:
            logger.warning(
                "Skipping artefact detection as --tissue_detection_model_path was not provided."
            )
        else:
            logger.info("Running artefact detection")

            (
                artefact_mask,
                artefact_map,
                artefact_cnts,
                artefact_geojson,
            ) = detect_artefacts_wsi(
                slide=OpenSlide(args.slide_path),
                model_art_path=args.artefact_detection_model_path,
                model_td_path=args.tissue_detection_model_path,
                device=devices[0],
            )

            logger.info(f"Found {len(artefact_cnts)} artefact contours")

            if args.filter_artefacts:
                (
                    polygons,
                    artefact_filtered_count,
                    artefact_polygons,
                ) = filter_cells_by_artefacts(polygons, artefact_cnts)

                logger.info(
                    f"Removed {artefact_filtered_count} cells in artefact regions"
                )
                logger.info(
                    f"Cells remaining after artefact filtering: {len(polygons)}"
                )
            else:
                for cnt_data in artefact_cnts.values():
                    holes = cnt_data.get("holes", [])
                    polygon = create_valid_polygon(
                        cnt_data["contour"], holes=holes
                    )
                    if polygon is not None:
                        artefact_polygons.append(polygon)

        artefact_features = []
        for i, poly in enumerate(artefact_polygons):
            artefact_features.extend(
                shapely_polygon_to_geojson(
                    poly,
                    id=f"artefact_{i}",
                    object_type="annotation",
                    additional_properties={
                        "classification": {
                            "name": "artefact",
                            "color": [255, 0, 0],
                        }
                    },
                )
            )

        total_artefact_area = sum(poly.area for poly in artefact_polygons)
        logger.info(f"Total artefact area: {total_artefact_area}")

        artefact_contours_fmt = {
            "type": "FeatureCollection",
            "features": artefact_features,
        }
        logger.info(
            f"Saving artefact contours to {output_folder}/{artefact_contours_filename}"
        )
        with open(output_folder / artefact_contours_filename, "w") as f:
            json.dump(artefact_contours_fmt, f)

    geojson_fmt = {
        "type": "FeatureCollection",
        "features": polygons,
    }
    centroids_fmt = {
        "type": "FeatureCollection",
        "features": polygons_to_centroids(polygons),
    }

    with open(output_folder / cell_contours_filename, "w") as f:
        json.dump(geojson_fmt, f)

    with open(output_folder / cell_centroids_filename, "w") as f:
        json.dump(centroids_fmt, f)

    densities_df = None
    output_types = args.output_type if args.output_type is not None else []
    if any(ot in ["csv", "spatialdata"] for ot in output_types):
        if args.tissue_detection_model_path is None:
            raise ValueError(
                f"Tissue detection model path must be provided when using --output_type {args.output_type}"
            )

        mpp_x = float(slide.mpp_x.value)
        mpp_y = float(slide.mpp_y.value)

        if args.roi_geojson and roi_class_dict is not None:
            logger.info("Calculating cellular densities per ROI class")

            roi_priority = None
            if args.roi_class_priority:
                roi_priority = [c.strip() for c in args.roi_class_priority]
                logger.info(f"Using ROI class priority: {roi_priority}")

            cells_by_roi_class = map_cells_to_roi_classes(
                polygons, roi_class_dict, priority_list=roi_priority
            )

            tissue_areas_by_roi = {}
            artefact_areas_by_roi = {}

            for roi_class_name, roi_polygons in roi_class_dict.items():
                roi_total_area = sum(poly.area for poly in roi_polygons)
                tissue_areas_by_roi[roi_class_name] = roi_total_area

                if args.artefact_detection_model_path:
                    roi_artefact_area = 0
                    for artefact_poly in artefact_polygons:
                        for roi_poly in roi_polygons:
                            intersection = artefact_poly.intersection(roi_poly)
                            if not intersection.is_empty:
                                roi_artefact_area += intersection.area
                    artefact_areas_by_roi[roi_class_name] = roi_artefact_area
                else:
                    artefact_areas_by_roi[roi_class_name] = 0

            densities_df = calculate_cellular_densities(
                cells=cells_by_roi_class,
                tissue_area_pixels=tissue_areas_by_roi,
                artefact_area_pixels=artefact_areas_by_roi,
                mpp_x=mpp_x,
                mpp_y=mpp_y,
                labels=labels,
            )
        else:
            densities_df = calculate_cellular_densities(
                cells=polygons,
                tissue_area_pixels=total_tissue_area,
                artefact_area_pixels=(
                    total_artefact_area
                    if args.artefact_detection_model_path
                    else 0
                ),
                mpp_x=mpp_x,
                mpp_y=mpp_y,
                labels=labels,
            )

    if "csv" in output_types:
        cell_densities_filename = f"{slide_basename}_cell_densities.csv"
        logger.info(
            f"Saving cellular densities to {output_folder}/{cell_densities_filename}"
        )
        densities_df.to_csv(
            output_folder / cell_densities_filename, index=False
        )

    if "spatialdata" in output_types:
        tissue_contours_geojson_path = None
        if args.tissue_detection_model_path is not None:
            tissue_contours_geojson_path = (
                output_folder / tissue_contours_filename
            )

        artefact_contours_geojson_path = None
        if args.artefact_detection_model_path is not None:
            artefact_contours_geojson_path = (
                output_folder / artefact_contours_filename
            )

        spatialdata_filename = f"{slide_basename}_spatialdata.zarr"
        create_spatialdata_output(
            cell_contours_geojson_path=output_folder / cell_contours_filename,
            cell_centroids_geojson_path=output_folder / cell_centroids_filename,
            tissue_contours_geojson_path=tissue_contours_geojson_path,
            artefact_contours_geojson_path=artefact_contours_geojson_path,
            densities_df=densities_df,
            output_path=output_folder / spatialdata_filename,
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            slide_basename=slide_basename,
            model_config=args.model_config,
            n_cells=len(polygons),
            roi_geojson_path=args.roi_geojson,
        )


def main_with_args():
    parser = argparse.ArgumentParser(description="Run Classpose WSI inference.")

    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="One of 'conic', 'consep', 'glysac', 'monusac', 'puma' or "
        "a path to the Classpose model config.",
    )
    parser.add_argument(
        "--slide_path",
        type=str,
        required=True,
        help="Path to the test dataset directory (must contain images.npy and labels.npy).",
    )
    parser.add_argument(
        "--tissue_detection_model_path",
        type=str,
        default=None,
        help="Path to the GrandQC tissue detection model weights. If specified but not found"
        " it will be downloaded.",
    )
    parser.add_argument(
        "--artefact_detection_model_path",
        type=str,
        default=None,
        help="Path to GrandQC artefact detection model. If specified, detects artefact regions.",
    )
    parser.add_argument(
        "--filter_artefacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If True, filters cells detected in artefact regions. Requires --artefact_detection_model_path.",
    )
    parser.add_argument(
        "--roi_geojson",
        type=str,
        default=None,
        help="FeatureCollection with (Multi)Polygon(s) in level-0 coordinates",
    )
    parser.add_argument(
        "--roi_class_priority",
        type=str,
        default=None,
        nargs="+",
        help="Space-separated list of ROI class names in priority order for overlapping regions "
        "(used for density calculations only). "
        "Cells in overlapping ROIs will be assigned to the first matching class in this list. "
        "Classes not in this list will be checked after, in their natural order. "
        "Example: Tumour Stroma Necrosis",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=0,
        help="Minimum area of the tissue polygons.",
    )
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Triggers test time augmentation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference. If None, automatically infers the device. "
        "Multi-GPU execution can be specified as 'cuda:0,1' or 'cuda:0,1,2,3'.",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Triggers bf16 inference.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help="Tile size for inference.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="Tile overlap for inference.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Path to save the output files (basename_cell_contours.geojson, basename_cell_centroids.geojson, basename_tissue_contours.geojson, and basename_artefact_contours.geojson).",
        required=True,
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default=None,
        nargs="+",
        choices=["csv", "spatialdata"],
        help="Optional output type(s). Accepts one or more of: 'csv', 'spatialdata'. "
        "'csv' generates cellular density statistics (cells/mm²) per class. "
        "'spatialdata' generates a unified SpatialData Zarr store containing all outputs. "
        "Can specify multiple types separated by spaces (e.g., --output_type csv spatialdata).",
    )
    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    main_with_args()
