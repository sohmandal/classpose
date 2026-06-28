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
6. Optional artefact detection using GrandQC to filter out cells in artefact regions.
7. Optional ROI-based filtering to restrict analysis to specific regions.
8. Filter detected cells based on tissue and artefact contours.
9. Generate outputs including cell contours and centroids as GeoJSON files,
    tissue contours, artefact contours, and optionally cellular densities as CSV
    and/or a unified SpatialData Zarr store.

Please note that if you use any part of Classpose which makes use of GrandQC please
follow the instructions at https://github.com/cpath-ukk/grandqc/tree/main to cite them
appropriately. Similarly to Classpose, GrandQC is under a non-commercial license
whose terms can be found at https://github.com/cpath-ukk/grandqc/blob/main/LICENSE.
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="dask")
warnings.filterwarnings("ignore", category=UserWarning, module="xarray_schema")
warnings.filterwarnings("ignore", category=FutureWarning, module="spatialdata")
warnings.filterwarnings("ignore", message=".*ImplicitModification.*")
warnings.filterwarnings("ignore", message=".*Transforming to str index.*")

import os
import argparse
import gc
import json
import logging
import queue
import re
import threading
import time
import uuid
from pathlib import Path
from multiprocessing.managers import SyncManager
from multiprocessing import Event
from multiprocessing.synchronize import Event as EventType

import cv2
import numpy as np
import shapely
import torch
import torch.multiprocessing as tmproc
from matplotlib import colormaps
from scipy import ndimage
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
from classpose.utils import (
    get_device,
    get_geojson_output_filename,
    get_slide_resolution,
    download_if_unavailable,
)
from classpose.entrypoints.outputs import (
    calculate_cellular_densities,
    create_spatialdata_output,
    create_valid_polygon,
    map_cells_to_roi_classes,
)
from classpose import WSIReader

logger = get_logger("classpose")


DEFAULT_TRAIN_MPP = 0.5
DEFAULT_TILE_SIZE = 1024
DEFAULT_OVERLAP = 64
MAX_QUEUE_SIZE = 2048
MIN_TILE_SIZE = 256
DEFAULT_INFERENCE_THREADS = 2
COLORMAP = [[int(y * 255) for y in x] for x in colormaps["Set3"].colors]


def resize_tile_to_target_mpp(
    tile: np.ndarray, resize_factor: float
) -> np.ndarray:
    """
    Resize a tile so that its apparent MPP matches the model MPP.

    Args:
        tile (np.ndarray): Tile read from the selected pyramid level.
        resize_factor (float): Resize factor.

    Returns:
        np.ndarray: Resized tile.
    """
    if resize_factor == 1.0:
        return tile
    new_width = max(1, int(round(tile.shape[1] * resize_factor)))
    new_height = max(1, int(round(tile.shape[0] * resize_factor)))
    return cv2.resize(
        tile,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )


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
        termination_event: EventType | None = None,
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
            termination_event (Event, optional): Event to signal termination.
                Defaults to None.
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
        self.termination_event = termination_event

        self.downloaded_slide = None

        if manager is None:
            manager = tmproc.Manager()

        self.n = manager.Value("i", 0)
        self.q = tmproc.Queue(maxsize=MAX_QUEUE_SIZE)
        self.ts = manager.Value("f", 0)
        self.mpp_x = manager.Value("f", 0)
        self.mpp_y = manager.Value("f", 0)
        self.bounds_x = manager.Value("f", 0)
        self.bounds_y = manager.Value("f", 0)
        self.tissue_cnts = manager.list()
        self.roi_cnts = manager.list()
        self.resize_factor = manager.Value("f", 1.0)

        self.p = tmproc.Process(target=self.fill_queue)
        self.p.start()

    def get_real_slide_path(self) -> str:
        """
        Resolve the local path OpenSlide should read from.
        """
        real_slide_path = getattr(self, "real_slide_path", None)
        if real_slide_path is not None:
            return real_slide_path

        if self.slide_path.startswith("http"):
            slide_name = self.slide_path.split("/")[-1].split("?")[0]
            real_slide_path = f".tmp/{slide_name}"
            logger.info(
                f"Downloading slide from {self.slide_path} to {real_slide_path}"
            )
            download_if_unavailable(
                real_slide_path, self.slide_path, "Downloading slide data"
            )
            self.downloaded_slide = real_slide_path
        else:
            real_slide_path = self.slide_path

        self.real_slide_path = real_slide_path
        return self.real_slide_path

    def _init_slide(self):
        self.slide = WSIReader(self.get_real_slide_path())
        self.mpp = get_slide_resolution(self.slide)
        self.mpp_x.value = self.mpp[0]
        self.mpp_y.value = self.mpp[1]
        bounds_x = self.slide.properties.get("openslide.bounds-x")
        bounds_y = self.slide.properties.get("openslide.bounds-y")
        self.bounds_x.value = float(bounds_x) if bounds_x is not None else 0.0
        self.bounds_y.value = float(bounds_y) if bounds_y is not None else 0.0
        logger.info(
            f"Slide bounds offset: x={self.bounds_x.value}, y={self.bounds_y.value}"
        )

        if self.roi_tree is not None and (
            self.bounds_x.value != 0 or self.bounds_y.value != 0
        ):
            self._align_roi_tree_to_slide_bounds()

        prediction_to_slide_scale = min(
            self.train_mpp / self.mpp[0], self.train_mpp / self.mpp[1]
        )
        self.level = self.slide.get_best_level_for_downsample(
            prediction_to_slide_scale
        )
        self.slide_dim = self.slide.level_dimensions[self.level]
        self.ts.value = self.slide.level_downsamples[self.level]
        self.resize_factor.value = self.ts.value / prediction_to_slide_scale
        read_tile_size = max(
            1, round(self.tile_size / self.resize_factor.value)
        )
        read_overlap = max(0, round(self.overlap / self.resize_factor.value))
        if self.roi_tree is not None:
            self.coords = list(
                self._get_coords_roi(
                    read_tile_size, read_overlap, self.slide_dim, self.ts.value
                )
            )
        else:
            self.coords = list(
                self._get_coords(
                    read_tile_size, read_overlap, self.slide_dim, self.ts.value
                )
            )
        logger.info(f"Slide MPP: {self.mpp}")
        logger.info(f"Model MPP: {self.train_mpp}")
        logger.info(f"Number of tiles: {len(self.coords)}")
        logger.info(f"Slide dimensions: {self.slide_dim}")
        logger.info(f"Tile size: {self.tile_size}")
        logger.info(f"Overlap: {self.overlap}")
        logger.info(
            "Prediction-to-slide scale from MPP: %s",
            prediction_to_slide_scale,
        )
        logger.info(f"Selected level: {self.level}")
        logger.info(f"Selected level downsample: {self.ts.value}")
        logger.info(
            "Residual resize factor before inference: %s",
            self.ts.value / prediction_to_slide_scale,
        )

    def _align_roi_tree_to_slide_bounds(self):
        """
        Align ROI polygons to slide coordinates when OpenSlide bounds offsets are present.

        QuPath ROI annotations are exported in coordinates relative to the displayed image
        origin (i.e. bounds-offset corrected). Internal OpenSlide tile coordinates are in
        level-0 slide space. So if needed, shift ROI polygons by +bounds offsets so
        ROI-based tile selection/filtering uses the same frame.
        """
        bounds_x_val = float(self.bounds_x.value)
        bounds_y_val = float(self.bounds_y.value)
        geometries = list(self.roi_tree.geometries)

        logger.info(
            f"Applying bounds offset to ROI polygons: x={bounds_x_val}, y={bounds_y_val}"
        )

        shifted = [
            shapely.affinity.translate(
                geom, xoff=bounds_x_val, yoff=bounds_y_val
            )
            for geom in geometries
        ]
        self.roi_tree = shapely.STRtree(shifted)

    def _get_tissue_contours(self):
        if self.tissue_detection_model_path is not None:
            logger.info("Detecting tissue contours using GrandQC")
            _, _, _, tissue_cnts, _, _ = detect_tissue_wsi(
                slide=WSIReader(self.get_real_slide_path()),
                model_td_path=self.tissue_detection_model_path,
                min_area=self.min_area,
                device=self.device,
            )
            self.tissue_cnts.extend(
                [
                    make_valid(shapely.Polygon(cnt["contour"], cnt["holes"]))
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
            ts (float): Selected level downsample.

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
            ts (float): Selected level downsample.

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
        self, point: tuple[int, int], tile_size: int | float
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

    def _check_tile_in_cnts(
        self,
        coords: tuple[int, int],
        tile_size: int,
        cnts: list[shapely.Geometry],
    ):
        tile_size_level0 = tile_size * float(self.ts.value)
        tile = self._point_to_square_polygon(coords, tile_size_level0)
        has_roi = any([cnt.intersects(tile) for cnt in cnts])
        if not has_roi:
            return False
        return True

    def fill_queue(self):
        """
        Fills the queue with tiles for inference.
        """
        self._init_slide()
        self._get_tissue_contours()
        if (
            self.tissue_detection_model_path is not None
            and not self.tissue_cnts
        ):
            logger.warning("No tissue detected in slide. Skipping inference.")
            for _ in range(self.n_none):
                self.q.put((None, None))
            return
        n = 0
        with tqdm(self.coords, desc="Tiles to predict: 0", position=0) as pbar:
            for coords, tile_size in pbar:
                if self.tissue_cnts:
                    if not self._check_tile_in_cnts(
                        coords, tile_size, self.tissue_cnts
                    ):
                        continue
                if self.roi_cnts:
                    if not self._check_tile_in_cnts(
                        coords, tile_size, self.roi_cnts
                    ):
                        continue
                tile = self.slide.read_region(
                    coords, self.level, (tile_size, tile_size)
                )
                tile_array = np.array(tile)
                if tile_array.shape[-1] == 4:
                    tile_array = tile_array[:, :, :3]

                tile_array = self._resize_tile_to_target_mpp(tile_array)
                self.q.put((tile_array, coords))
                n += 1
                self.n.value += 1
                pbar.set_description(f"Tiles to predict: {n}")
        for _ in range(self.n_none):
            self.q.put((None, None))
        if self.termination_event is not None:
            self.termination_event.wait()

    def _resize_tile_to_target_mpp(self, tile: np.ndarray) -> np.ndarray:
        """
        Resize a tile so that its apparent MPP matches the model MPP.

        Args:
            tile (np.ndarray): Tile read from the selected pyramid level.

        Returns:
            np.ndarray: Resized tile.
        """
        resize_factor = self.resize_factor.value
        if resize_factor == 1.0:
            return tile
        new_width = max(1, int(round(tile.shape[1] * resize_factor)))
        new_height = max(1, int(round(tile.shape[0] * resize_factor)))
        return cv2.resize(
            tile,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR,
        )

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
        if self.termination_event is not None:
            self.termination_event.set()
        self.p.join()
        if self.downloaded_slide is not None:
            logger.info(f"Removing downloaded slide {self.downloaded_slide}")
            os.remove(self.downloaded_slide)


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

    def __init__(
        self,
        manager: tmproc.Manager = None,
        n_workers: int = 1,
        labels: list[str] | None = None,
    ):
        """
        Args:
            manager (tmproc.Manager, optional): Manager to use for shared memory.
            n_workers (int, optional): Number of inference workers that will send a None
                sentinel when done. Defaults to 1.
            labels (list[str], optional): List of class labels. If provided, enables
                multi-class mode where data should be (masks, class_masks) tuples.
                If None, single-class mode is used with default "cell" label.
        """
        self.n_workers = n_workers
        self.labels = labels

        if manager is None:
            manager = tmproc.Manager()

        self.n = manager.Value("i", 0)
        self.polygons = tmproc.Queue()
        self.value = manager.Value("i", 0)
        self.n_cells = manager.Value("i", 0)
        self.n_invalid_cells = manager.Value("i", 0)
        self.q = tmproc.Queue(maxsize=MAX_QUEUE_SIZE)
        self.p = tmproc.Process(target=self.run, args=(self.n_workers,))
        self.p.start()

    def run(self, n_workers: int = 1):
        """
        Generic run method for the process.
        """
        sentinels_remaining = n_workers
        while True:
            item = self.q.get()
            if item is None:
                sentinels_remaining -= 1
                if sentinels_remaining == 0:
                    break
                continue
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
        data: list[tuple[np.ndarray, np.ndarray] | np.ndarray],
        batch_coords: list[tuple[int, int]],
        prediction_to_slide_scale: float,
    ):
        """
        Data preprocessing following the aforementioned protocol. Appends everything
        to the polygons list, which is a shared memory list.

        Args:
            data (list[tuple | np.ndarray]): Data to process. For multi-class mode,
                should be a list of (masks, class_masks) tuples. For single-class mode,
                should be a list of masks arrays.
            batch_coords (list[tuple[int, int]]): List of coordinates for the batch.
            prediction_to_slide_scale (float): Level-0 pixels per prediction pixel.
        """
        for datum, coords in zip(data, batch_coords):
            if self.labels is not None:
                masks, class_masks = datum
            else:
                masks = datum
                class_masks = None
            object_slices = ndimage.find_objects(masks)
            curr_cells = []
            for label_idx, object_slice in enumerate(object_slices, start=1):
                if object_slice is None:
                    continue
                y_slice, x_slice = object_slice
                cell_mask = masks[y_slice, x_slice] == label_idx
                contours = cv2.findContours(
                    np.uint8(cell_mask),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )[0]
                if len(contours) == 0:
                    self.n_invalid_cells.value += 1
                    continue
                # Contour only the object bbox, then shift back to tile coordinates.
                curr_coords = contours[0][:, 0] + np.array(
                    [x_slice.start, y_slice.start]
                )
                curr_coords = curr_coords * prediction_to_slide_scale + coords
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
                curr_coords.append(curr_coords[0].copy())

                if class_masks is not None:
                    cl = class_masks[y_slice, x_slice][cell_mask][0]
                    label = self.labels[int(cl) - 1]
                    color = COLORMAP[int(cl) - 1]
                    class_int = int(cl) - 1
                else:
                    label = "cell"
                    color = [0, 168, 132]
                    class_int = 0

                curr_cell = {
                    "id": str(uuid.uuid4()),
                    "coords": curr_coords,
                    "class_int": class_int,
                    "area": polygon.area,
                    "label": label,
                    "color": color,
                    "perimeter": polygon.length,
                    "centroid": center,
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
    slide_downsample: float = 1,
    bsize: int = 256,
    prediction_to_slide_scale: float = 1,
    precision: str = "bf16",
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
        slide_downsample (float): Pyramid downsample used to read tiles.
        bsize (int): Batch size.
        prediction_to_slide_scale (float): Level-0 pixels per prediction pixel.
    """
    if isinstance(dev, str):
        dev = torch.device(dev)
    model = None
    tile = masks = raw_data = class_masks = styles = None

    try:
        model = ClassposeModel(
            gpu=dev.type == "cuda",
            pretrained_model=model_path,
            device=dev,
            nclasses=n_classes,
            feature_transformation_structure=fts,
            precision=precision,
        )
        model.net.eval()
        if dev.type == "cuda":
            model.net = torch.compile(model.net)

        n_threads = max(1, DEFAULT_INFERENCE_THREADS)
        local_q: queue.Queue = queue.Queue(maxsize=n_threads * 2)
        update_lock = threading.Lock()

        def _feeder():
            """Move tiles from the shared queue into the local queue."""
            while True:
                tile, coords = slide_queue.get()
                if tile is None:
                    break
                local_q.put((tile, coords))
            for _ in range(n_threads):
                local_q.put(None)

        with tqdm(
            None,
            desc="Predicted tiles (detected cells: 0)",
            position=1,
            total=0,
        ) as pbar:

            def _process(tile, coords):
                masks, raw_data, class_masks, styles = model.eval(
                    [tile],
                    batch_size=batch_size,
                    augment=tta,
                    bsize=bsize,
                    compute_masks=True,
                )
                postproc_queue.put(
                    (
                        list(zip(masks, class_masks)),
                        [coords],
                        prediction_to_slide_scale,
                    )
                )
                with update_lock:
                    predicted_tiles.value += 1
                    pbar.n = predicted_tiles.value
                    pbar.total = slide_queue_size.value
                    pbar.set_description(
                        f"Predicted tiles (detected cells: {n_cells.value}; invalid: {n_invalid_cells.value})"
                    )
                    pbar.refresh()

            def _run_inference():
                while True:
                    item = local_q.get()
                    if item is None:
                        break
                    _process(*item)

            feeder = threading.Thread(target=_feeder, daemon=True)
            feeder.start()

            # Compile on the first tile single-threaded to avoid concurrent first-compilation.
            first = local_q.get()
            if first is None:
                local_q.put(None)
            else:
                _process(*first)

            inference_threads = [
                threading.Thread(target=_run_inference, daemon=True)
                for _ in range(n_threads)
            ]
            for t in inference_threads:
                t.start()
            for t in inference_threads:
                t.join()
            feeder.join()

            pbar.set_description(
                f"Predicted tiles (detected cells: {n_cells.value}; invalid: {n_invalid_cells.value})"
            )
            print()
    finally:
        tile = masks = raw_data = class_masks = styles = None
        model = None
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        postproc_queue.put(None)


def to_geojson_polygon(curr_cell: dict) -> dict:
    """
    Formats ``curr_cell`` outputs as polygons for GeoJSON.

    Args:
        curr_cell (dict): cell outputs. These should feature an id (str),
            coords (a list), a label (str or int), a color (list of int), an
            area (float), a perimeter (float) and a centroid (a tuple with
            ints).

    Returns:
        A cellular polygon for GeoJSON.
    """
    curr_cell = {
        "type": "Feature",
        "id": curr_cell["id"],
        "geometry": {
            "type": "Polygon",
            "coordinates": [curr_cell["coords"]],
        },
        "properties": {
            "objectType": "annotation",
            "isLocked": False,
            "classification": {
                "name": curr_cell["label"],
                "color": curr_cell["color"],
            },
            "measurements": [
                {"name": "area", "value": curr_cell["area"]},
                {"name": "perimeter", "value": curr_cell["perimeter"]},
                {
                    "name": "centroidX",
                    "value": curr_cell["centroid"][0],
                },
                {
                    "name": "centroidY",
                    "value": curr_cell["centroid"][1],
                },
            ],
        },
    }
    return curr_cell


def apply_bounds_offset_to_feature(
    feature: dict, bounds_x: float, bounds_y: float
) -> dict:
    """
    Apply bounds offset to a GeoJSON Polygon feature's geometry and centroid measurements.

    Args:
        feature: GeoJSON Feature dict with Polygon geometry
        bounds_x: X offset from openslide.bounds-x property
        bounds_y: Y offset from openslide.bounds-y property

    Returns:
        Feature with shifted polygon coordinates and updated centroid measurements.
    """
    if not feature or "geometry" not in feature:
        return feature

    geometry = feature["geometry"]
    if "coordinates" not in geometry:
        return feature

    shifted_coordinates = []
    for ring in geometry["coordinates"]:
        shifted_ring = [
            [point[0] - bounds_x, point[1] - bounds_y] for point in ring
        ]
        shifted_coordinates.append(shifted_ring)
    geometry["coordinates"] = shifted_coordinates

    if "properties" in feature and "measurements" in feature["properties"]:
        for measurement in feature["properties"]["measurements"]:
            if measurement["name"] == "centroidX":
                measurement["value"] -= bounds_x
            elif measurement["name"] == "centroidY":
                measurement["value"] -= bounds_y

    return feature


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
    if isinstance(polygon, (shapely.MultiPolygon, shapely.GeometryCollection)):
        features = []
        for poly in polygon.geoms:
            # skip linestrings as they lead to problems down the line
            if isinstance(poly, shapely.LineString):
                continue
            features.extend(
                shapely_polygon_to_geojson(
                    poly,
                    id=None,
                    object_type=object_type,
                    additional_properties=additional_properties,
                )
            )
        return features

    if isinstance(polygon, shapely.LineString):
        return []
    exterior = [list(pt) for pt in polygon.exterior.coords]
    interiors = [[list(pt) for pt in ring.coords] for ring in polygon.interiors]
    coordinates = [exterior, *interiors]

    center = list(polygon.centroid.coords[0])
    feature_id = id if id is not None else str(uuid.uuid4())
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
                "coordinates": coordinates,
            },
            "properties": properties,
        }
    ]


def _repair_geometry(
    polygon: shapely.Geometry,
) -> shapely.Geometry:
    """
    Repair an invalid geometry, preserving holes where possible.

    Args:
        polygon (shapely.Geometry): The geometry to repair.

    Returns:
        shapely.Geometry: A valid geometry.
    """
    try:
        repaired = shapely.make_valid(polygon, method="structure")
        if repaired.is_valid and not repaired.is_empty:
            return repaired
    except shapely.errors.GEOSException:
        pass
    return polygon.buffer(0)


def make_valid(
    polygon: shapely.Polygon | shapely.MultiPolygon | shapely.Geometry,
) -> shapely.Polygon | shapely.MultiPolygon | shapely.GeometryCollection:
    """
    Makes a polygon valid and returns either a polygon or a multi-polygon,
    both ready for use in downstream functions.

    Args:
        polygon (shapely.Polygon | shapely.MultiPolygon | shapely.Geometry): The
            polygon to make valid.

    Returns:
        shapely.Polygon | shapely.MultiPolygon | shapely.GeometryCollection: The valid polygon.
    """
    if isinstance(polygon, shapely.Polygon):
        if polygon.is_valid:
            return polygon
        return _repair_geometry(polygon)

    if isinstance(polygon, shapely.MultiPolygon):
        geoms = []
        for geom in polygon.geoms:
            geom = make_valid(geom)
            if isinstance(geom, shapely.MultiPolygon):
                geoms.extend(geom.geoms)
            else:
                geoms.append(geom)
        if not geoms:
            return shapely.MultiPolygon([])
        if len(geoms) == 1:
            return geoms[0]
        return shapely.MultiPolygon(geoms)

    return _repair_geometry(polygon)


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

    if isinstance(data, list):
        data = {"features": data}
    if "features" not in data and "geometry" in data:
        data["features"] = [data]

    for feat in data.get("features", []):
        geom = feat.get("geometry")
        if not geom:
            continue
        shp = shapely.geometry.shape(geom)
        shp = make_valid(shp)

        class_name = None
        if group_by_class:
            props = feat.get("properties", {})
            classification = props.get("classification", {})
            class_name = classification.get("name", "unknown")

        if isinstance(shp, shapely.LineString):
            coords = [(x, y) for x, y in zip(*shp.coords.xy)]
            coords = coords + [coords[0]]
            shp = shapely.Polygon(coords)
            shp = shapely.make_valid(shp)
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
    termination_event = Event()

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
                f"Number of labels ({len(model_config.cell_types)}) does not match number of classes ({n_classes - 1})"
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
        termination_event=termination_event,
    )
    pp = PostProcessor(labels=labels, manager=manager, n_workers=len(devices))
    # Wait for slide to be initialized so that the target scale is known
    while slide.ts.value == 0:
        time.sleep(0.1)
    ts = float(slide.ts.value)
    mpp_x = float(slide.mpp_x.value)
    mpp_y = float(slide.mpp_y.value)
    prediction_to_slide_scale = min(
        model_config.mpp / mpp_x,
        model_config.mpp / mpp_y,
    )
    logger.info(
        "Prediction-to-slide coordinate scale: %s",
        prediction_to_slide_scale,
    )

    collected_batches: list = []

    def _drain_polygons():
        while True:
            item = pp.polygons.get()
            if item is None:
                break
            collected_batches.append(item)

    drain_thread = threading.Thread(target=_drain_polygons, daemon=True)
    drain_thread.start()

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
                    ts,
                    256,
                    prediction_to_slide_scale,
                    args.precision,
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
            slide_downsample=ts,
            bsize=256,
            prediction_to_slide_scale=prediction_to_slide_scale,
            precision=args.precision,
        )

    pp.p.join()
    slide.close()
    pp.polygons.put(None)
    drain_thread.join()

    polygons = []
    with tqdm(desc="Collecting polygons") as pbar:
        for batch in collected_batches:
            polygons.extend([to_geojson_polygon(x) for x in batch])
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
    cell_contours_filename = get_geojson_output_filename(
        "cell_contours", slide_basename
    )
    cell_centroids_filename = get_geojson_output_filename(
        "cell_centroids", slide_basename
    )
    tissue_contours_filename = get_geojson_output_filename(
        "tissue_contours", slide_basename
    )
    artefact_contours_filename = get_geojson_output_filename(
        "artefact_contours", slide_basename
    )

    if args.roi_geojson:
        logger.info("Filtering cells based on ROI contours")
        roi_cnts = list(slide.roi_cnts)
        polygons = filter_cells_by_contours(polygons, roi_cnts)
        logger.info(f"Number of cells after filtering: {len(polygons)}")

    if args.tissue_detection_model_path is not None:
        logger.info("Filtering cells based on tissue contours")
        tissue_cnts = list(slide.tissue_cnts)
        polygons = filter_cells_by_contours(polygons, tissue_cnts)

        bounds_x_val = float(slide.bounds_x.value)
        bounds_y_val = float(slide.bounds_y.value)
        if bounds_x_val != 0 or bounds_y_val != 0:
            tissue_cnts = [
                shapely.affinity.translate(
                    cnt, xoff=-bounds_x_val, yoff=-bounds_y_val
                )
                for cnt in tissue_cnts
            ]

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
                slide=WSIReader(slide.get_real_slide_path()),
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

        bounds_x_val = float(slide.bounds_x.value)
        bounds_y_val = float(slide.bounds_y.value)
        if bounds_x_val != 0 or bounds_y_val != 0:
            artefact_polygons = [
                shapely.affinity.translate(
                    poly, xoff=-bounds_x_val, yoff=-bounds_y_val
                )
                for poly in artefact_polygons
            ]

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

    bounds_x_val = float(slide.bounds_x.value)
    bounds_y_val = float(slide.bounds_y.value)
    if bounds_x_val != 0 or bounds_y_val != 0:
        logger.info(
            f"Applying bounds offset to output coordinates: x={bounds_x_val}, y={bounds_y_val}"
        )
        polygons = [
            apply_bounds_offset_to_feature(f, bounds_x_val, bounds_y_val)
            for f in polygons
        ]

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

    slide.close()


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
        help="Path to the whole-slide image to process (e.g. .svs, .tiff; any "
        "format supported by OpenSlide). If the slide is an HTTP/HTTPS URL, "
        "it will be downloaded to a temporary directory (`.tmp`).",
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
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Inference precision. 'bf16' falls back to 'fp16' on GPUs without "
        "hardware bf16 support.",
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
        help=(
            "Path to save the output files "
            "(basename_cell_contours.geojson, basename_cell_centroids.geojson, "
            "basename_tissue_contours.geojson, basename_artefact_contours.geojson)."
        ),
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
