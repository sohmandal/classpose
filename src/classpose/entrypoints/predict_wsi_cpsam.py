"""
Predict CellposeSAM cells and centroids for a whole-slide image.

The workflow is as follows:
1. A tile loader is instantiated. This tile loader filters out tiles
    which are unlikely to contain tissue. This works as a background process.
2. A post processor is instantiated. This post processor processes the tiles
    in parallel and stores the results in a shared memory list. This works as a
    background process.
3. Begin the prediction using CellposeSAM. This supports multi-GPU inference using
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

Please cite CellposeSAM (https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1)
if you use this tool.

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
import json
import logging
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import shapely
import torch
import torch.multiprocessing as tmproc
from cellpose.models import CellposeModel
from openslide import OpenSlide
from tqdm import tqdm

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

from classpose.entrypoints.outputs import (
    calculate_cellular_densities,
    create_spatialdata_output,
    create_valid_polygon,
    map_cells_to_roi_classes,
)
from classpose.entrypoints.predict_wsi import (
    DEFAULT_OVERLAP,
    DEFAULT_TILE_SIZE,
    DEFAULT_TRAIN_MPP,
    MIN_TILE_SIZE,
    SlideLoader,
    deduplicate,
    filter_cells_by_artefacts,
    filter_cells_by_contours,
    load_roi_polygons,
    polygons_to_centroids,
    shapely_polygon_to_geojson,
)
from classpose.grandqc.wsi_artefact_detection import detect_artefacts_wsi
from classpose.log import get_logger
from classpose.utils import get_device

logger = get_logger("classpose")


class PostProcessor:
    """
    Post processor class for parallel post processing of tiles.

    The workflow consists in:

    1. Using the instance prediction to identify all cells in the tile
    2. Using the cell coordinates to create a polygon for each cell
    3. Filtering out invalid polygons as these represent unlikely predictions
    (i.e. the contour crosses itself at a given point)
    4. Extraction of some minimal features (area, perimeter, centroid) which
    can be used for deduplicating cells at a later stage
    5. Adding the polygons to a list for later writing to a GeoJSON file.
    """

    def __init__(self, manager: tmproc.Manager = None):
        """
        Args:
            manager (tmproc.Manager, optional): Manager to use for shared memory.
        """

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
        data: list[np.ndarray],
        batch_coords: list[tuple[int, int]],
        ts: float,
    ):
        """
        Data preprocessing following the aforementioned protocol. Appends everything
        to the polygons list, which is a shared memory list.

        Args:
            data (list[np.ndarray]): Data to process. Should be a list of instance masks.
            batch_coords (list[tuple[int, int]]): List of coordinates for the batch.
            ts (float): Target downsample.
        """
        class_name = "cell"
        class_color = [0, 168, 132]
        for masks, coords in zip(data, batch_coords):
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
                            "name": class_name,
                            "color": class_color,
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
        model_path (str): Path to the CellposeSAM model (defaults to cpsam if not found).
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
        bf16 (bool): Whether to use bfloat16.
    """
    model = CellposeModel(
        gpu=dev.type == "cuda",
        pretrained_model=model_path,
        device=dev,
    )
    if bf16:
        model.net = model.net.to(torch.bfloat16)
    if isinstance(dev, str):
        dev = torch.device(dev)
    if dev.type == "cuda":
        model.net = model.net.to(dev)
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
            masks, raw_data, styles = model.eval(
                [tile],
                batch_size=batch_size,
                augment=tta,
                bsize=bsize,
                compute_masks=True,
            )
            if isinstance(masks, list):
                batch_masks = masks
            else:
                batch_masks = [masks]
            postproc_queue.put((batch_masks, [coords], target_downsample))
            predicted_tiles.value += 1

            pbar.n = predicted_tiles.value
            pbar.total = slide_queue_size.value
            pbar.set_description(
                "Predicted tiles (detected cells: %s; invalid: %s)"
                % (n_cells.value, n_invalid_cells.value)
            )
            pbar.refresh()

        postproc_queue.put(None)
        pbar.set_description(
            "Predicted tiles (detected cells: %s; invalid: %s)"
            % (n_cells.value, n_invalid_cells.value)
        )
        print()


def main(args):
    tmproc.set_start_method("spawn", force=True)

    if args.tile_size < MIN_TILE_SIZE:
        raise ValueError(
            f"Tile size must be at least {MIN_TILE_SIZE}, got {args.tile_size}"
        )

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

    logger.info(
        "Starting inference with CellposeSAM model: %s", args.model_path
    )
    slide = SlideLoader(
        args.slide_path,
        tile_size=args.tile_size,
        overlap=args.overlap,
        train_mpp=args.train_mpp,
        manager=manager,
        n_none=len(devices),
        tissue_detection_model_path=args.tissue_detection_model_path,
        min_area=args.min_area,
        roi_tree=roi_tree,
        device=devices[0],
    )
    pp = PostProcessor(manager=manager)
    # Wait for slide to be initialized so that the target downsample is known
    while slide.ts.value == 0:
        time.sleep(0.1)
    ts = float(slide.ts.value)

    if len(devices) > 1:
        workers = []
        logger.info("Starting workers on devices: %s", devices)
        for device in devices:
            logger.info("Starting worker on device: %s", device)
            p = tmproc.Process(
                target=worker,
                args=(
                    device,
                    args.model_path,
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
            model_path=args.model_path,
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
    logger.info("Number of detected cells: %s", len(polygons))
    logger.info("Number of invalid cells: %s", pp.n_invalid_cells.value)
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
        logger.info("Number of cells after filtering: %s", len(polygons))

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
        logger.info("Total tissue area: %s", total_tissue_area)
        tissue_cnts_fmt = {
            "type": "FeatureCollection",
            "features": tissue_features,
        }
        logger.info("Number of cells after filtering: %s", len(polygons))
        logger.info(
            "Saving tissue contours to %s/%s",
            output_folder,
            tissue_contours_filename,
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

            logger.info("Found %s artefact contours", len(artefact_cnts))

            if args.filter_artefacts:
                (
                    polygons,
                    artefact_filtered_count,
                    artefact_polygons,
                ) = filter_cells_by_artefacts(polygons, artefact_cnts)

                logger.info(
                    "Removed %s cells in artefact regions",
                    artefact_filtered_count,
                )
                logger.info(
                    "Cells remaining after artefact filtering: %s",
                    len(polygons),
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
        logger.info("Total artefact area: %s", total_artefact_area)

        artefact_contours_fmt = {
            "type": "FeatureCollection",
            "features": artefact_features,
        }
        logger.info(
            "Saving artefact contours to %s/%s",
            output_folder,
            artefact_contours_filename,
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
                "Tissue detection model path must be provided when using --output_type %s"
                % args.output_type
            )

        mpp_x = float(slide.mpp_x.value)
        mpp_y = float(slide.mpp_y.value)

        if args.roi_geojson and roi_class_dict is not None:
            logger.info("Calculating cellular densities per ROI class")

            roi_priority = None
            if args.roi_class_priority:
                roi_priority = [c.strip() for c in args.roi_class_priority]
                logger.info("Using ROI class priority: %s", roi_priority)

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
                labels=["cell"],
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
                labels=["cell"],
            )

    if "csv" in output_types:
        cell_densities_filename = f"{slide_basename}_cell_densities.csv"
        logger.info(
            "Saving cellular densities to %s/%s",
            output_folder,
            cell_densities_filename,
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
            model_config=args.model_path,
            n_cells=len(polygons),
            roi_geojson_path=args.roi_geojson,
        )


def main_with_args():
    parser = argparse.ArgumentParser(
        description="Run CellposeSAM WSI inference."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="cpsam",
        help="Path to the CellposeSAM model (defaults to cpsam if not found).",
    )
    parser.add_argument(
        "--slide_path",
        type=str,
        required=True,
        help="Path to the WSI slide.",
    )
    parser.add_argument(
        "--train_mpp",
        type=float,
        default=DEFAULT_TRAIN_MPP,
        help="Microns per pixel of the training data used for inference scaling.",
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
        "--tile_size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help="Tile size for inference.",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Triggers bf16 inference.",
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
