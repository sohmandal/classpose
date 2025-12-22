import json
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely
import spatialdata as sd
from anndata import AnnData
from spatialdata.models import PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Scale
from tqdm import tqdm

logger = logging.getLogger("classpose")


def create_valid_polygon(
    coordinates: list, holes: list = None, polygon_index: int = None
) -> shapely.Polygon | None:
    """
    Create a valid Shapely polygon from coordinates with optional holes.

    Args:
        coordinates (list): List of coordinates for the exterior ring.
        holes (list, optional): List of hole coordinate arrays. Each hole is a list
            of coordinates representing an interior ring (area excluded from polygon).
        polygon_index (int, optional): Index for logging purposes.

    Returns:
        shapely.Polygon | None: Valid polygon or None if it cannot be fixed.
    """
    try:
        # Create polygon with exterior ring and holes
        polygon = shapely.Polygon(coordinates, holes or [])

        if not polygon.is_valid:
            polygon = polygon.buffer(0)
            if not polygon.is_valid:
                if polygon_index is not None:
                    logger.warning(
                        f"Cannot fix invalid polygon at index {polygon_index}"
                    )
                return None

        return polygon

    except Exception as e:
        if polygon_index is not None:
            logger.warning(
                f"Error creating polygon at index {polygon_index}: {e}"
            )
        else:
            logger.warning(f"Error creating polygon: {e}")
        return None


def map_cells_to_roi_classes(
    cells: list[dict],
    roi_class_dict: dict[str, list[shapely.Polygon]],
    priority_list: list[str] | None = None,
) -> dict[str, list[dict]]:
    """
    Map cells to ROI classes based on containment with optional priority ordering.

    Args:
        cells (list[dict]): List of cell polygons in GeoJSON format.
        roi_class_dict (dict[str, list[shapely.Polygon]]): Dictionary mapping
            ROI class names to lists of polygons.
        priority_list (list[str], optional): Ordered list of ROI class names by priority.
            Classes in this list are checked first in order. Any remaining classes not
            in the priority list are checked after, in their dictionary order.

    Returns:
        dict[str, list[dict]]: Dictionary mapping ROI class names to lists of
            cells contained within those ROI classes.
    """
    logger.info("Mapping cells to ROI classes")

    if priority_list:
        invalid_classes = [c for c in priority_list if c not in roi_class_dict]
        if invalid_classes:
            logger.warning(
                f"Priority list contains classes not found in ROI: {invalid_classes}"
            )

        priority_classes = [c for c in priority_list if c in roi_class_dict]

        remaining_classes = [
            c for c in roi_class_dict.keys() if c not in priority_list
        ]

        ordered_classes = priority_classes + remaining_classes

        logger.info(
            f"ROI class priority order: {priority_classes} (prioritised), "
            f"{remaining_classes} (remaining)"
        )
    else:
        ordered_classes = list(roi_class_dict.keys())
        logger.info("No priority specified, using dictionary order")

    roi_trees = {}
    for class_name, polygons in roi_class_dict.items():
        if polygons:
            roi_trees[class_name] = shapely.STRtree(polygons)

    result = {class_name: [] for class_name in roi_class_dict.keys()}

    for i, cell_data in enumerate(
        tqdm(cells, desc="Mapping cells to ROI classes")
    ):
        cell_polygon = create_valid_polygon(
            cell_data["geometry"]["coordinates"][0], polygon_index=i
        )

        if cell_polygon is None:
            continue

        cell_centroid = cell_polygon.centroid

        for class_name in ordered_classes:
            if class_name not in roi_trees:
                continue

            try:
                if (
                    len(
                        roi_trees[class_name].query(
                            cell_centroid, predicate="within"
                        )
                    )
                    > 0
                ):
                    result[class_name].append(cell_data)
                    break
            except Exception as e:
                logger.warning(
                    f"Error checking cell {i} against ROI class {class_name}: {e}"
                )
                continue

    for class_name, class_cells in result.items():
        logger.info(f"ROI class '{class_name}': {len(class_cells)} cells")

    return result


def calculate_cellular_densities(
    cells: list[dict] | dict[str, list[dict]],
    tissue_area_pixels: float | dict[str, float],
    artefact_area_pixels: float | dict[str, float],
    mpp_x: float,
    mpp_y: float,
    labels: list[str],
) -> pd.DataFrame:
    """
    Calculate cellular densities per class type.

    Args:
        cells (list[dict] | dict[str, list[dict]]): List of cell polygons (global mode)
            or dict mapping ROI class names to cell lists (ROI class mode).
        tissue_area_pixels (float | dict[str, float]): Total tissue area in pixels
            (global mode) or dict mapping ROI class names to tissue areas (ROI class mode).
        artefact_area_pixels (float | dict[str, float]): Total artefact area in pixels
            (global mode) or dict mapping ROI class names to artefact areas (ROI class mode).
        mpp_x (float): Microns per pixel in x direction.
        mpp_y (float): Microns per pixel in y direction.
        labels (list[str]): List of class labels.

    Returns:
        pd.DataFrame: DataFrame with columns: region, cell_class, count, density (cells/mm²).
    """
    logger.info("Computing cellular densities")

    roi_class_mode = isinstance(cells, dict)

    mpp_product = mpp_x * mpp_y
    logger.info(f"MPP (x, y): ({mpp_x}, {mpp_y})")

    if roi_class_mode:
        densities = []

        for roi_class in cells.keys():
            roi_cells = cells[roi_class]
            roi_tissue_area = tissue_area_pixels.get(roi_class, 0)
            roi_artefact_area = artefact_area_pixels.get(roi_class, 0)

            effective_tissue_area_pixels = roi_tissue_area - roi_artefact_area
            effective_tissue_area_mm2 = (
                effective_tissue_area_pixels * mpp_product / 1e6
            )

            logger.info(f"ROI class '{roi_class}':")
            logger.info(f"  Tissue area (pixels): {roi_tissue_area}")
            logger.info(f"  Artefact area (pixels): {roi_artefact_area}")
            logger.info(
                f"  Effective tissue area (pixels): {effective_tissue_area_pixels}"
            )
            logger.info(
                f"  Effective tissue area (mm²): {effective_tissue_area_mm2}"
            )

            class_counts = {label: 0 for label in labels}
            for cell in roi_cells:
                classification = cell["properties"]["classification"]["name"]
                if classification in class_counts:
                    class_counts[classification] += 1

            logger.info(f"  Cell counts per class: {class_counts}")

            for label in labels:
                count = class_counts[label]
                density = (
                    count / effective_tissue_area_mm2
                    if effective_tissue_area_mm2 > 0
                    else 0
                )
                densities.append(
                    {
                        "region": roi_class,
                        "cell_class": label,
                        "count": count,
                        "density": density,
                    }
                )
                logger.info(
                    f"  Cell class '{label}': count={count}, density={density:.2f} cells/mm²"
                )

        return pd.DataFrame(densities)

    else:
        effective_tissue_area_pixels = tissue_area_pixels - artefact_area_pixels
        logger.info(f"Tissue area (pixels): {tissue_area_pixels}")
        logger.info(f"Artefact area (pixels): {artefact_area_pixels}")
        logger.info(
            f"Effective tissue area (pixels): {effective_tissue_area_pixels}"
        )

        effective_tissue_area_mm2 = (
            effective_tissue_area_pixels * mpp_product / 1e6
        )
        logger.info(f"Effective tissue area (mm²): {effective_tissue_area_mm2}")

        class_counts = {label: 0 for label in labels}
        for cell in cells:
            classification = cell["properties"]["classification"]["name"]
            if classification in class_counts:
                class_counts[classification] += 1

        logger.info(f"Cell counts per class: {class_counts}")

        densities = []
        for label in labels:
            count = class_counts[label]
            density = (
                count / effective_tissue_area_mm2
                if effective_tissue_area_mm2 > 0
                else 0
            )
            densities.append(
                {
                    "region": "tissue",
                    "cell_class": label,
                    "count": count,
                    "density": density,
                }
            )
            logger.info(
                f"Class '{label}': count={count}, density={density:.2f} cells/mm²"
            )

        return pd.DataFrame(densities)


def flatten_geojson_properties(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Flatten nested GeoJSON properties for SpatialData compatibility.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with nested JSON string properties.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with flattened properties.
    """
    gdf = gdf.copy()

    if "classification" in gdf.columns:
        parsed_class = gdf["classification"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        gdf["classification_name"] = parsed_class.apply(
            lambda x: x.get("name") if isinstance(x, dict) else None
        )
        gdf["classification_color"] = parsed_class.apply(
            lambda x: x.get("color") if isinstance(x, dict) else None
        )
        gdf = gdf.drop(columns=["classification"])

    if "measurements" in gdf.columns:
        parsed_measurements = gdf["measurements"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )

        for measure in ["area", "perimeter", "centroidX", "centroidY"]:
            gdf[measure] = parsed_measurements.apply(
                lambda x: next(
                    (m["value"] for m in x if m["name"] == measure), None
                )
                if isinstance(x, list)
                else None
            )
        gdf = gdf.drop(columns=["measurements"])

    for col in ["objectType", "isLocked"]:
        if col in gdf.columns:
            gdf = gdf.drop(columns=[col])

    return gdf


def create_spatialdata_output(
    cell_contours_geojson_path: Path,
    cell_centroids_geojson_path: Path,
    tissue_contours_geojson_path: Path | None,
    artefact_contours_geojson_path: Path | None,
    densities_df: pd.DataFrame | None,
    output_path: Path,
    mpp_x: float,
    mpp_y: float,
    slide_basename: str,
    model_config: str,
    n_cells: int,
    roi_geojson_path: str | None = None,
) -> None:
    """
    Create a SpatialData object from pipeline outputs and write to Zarr.

    Args:
        cell_contours_geojson_path (Path): Path to the saved cell contours GeoJSON file.
        cell_centroids_geojson_path (Path): Path to the saved cell centroids GeoJSON file.
        tissue_contours_geojson_path (Path | None): Path to the saved tissue contours GeoJSON file.
        artefact_contours_geojson_path (Path | None): Path to the saved artefact contours GeoJSON file.
        densities_df (pd.DataFrame | None): DataFrame with cellular densities.
        output_path (Path): Path to save the SpatialData Zarr store.
        mpp_x (float): Microns per pixel in x direction.
        mpp_y (float): Microns per pixel in y direction.
        slide_basename (str): Basename of the slide.
        model_config (str): Model configuration used.
        n_cells (int): Number of cells detected.
        roi_geojson_path (str | None): Path to the ROI GeoJSON file if ROI mode was used.
    """
    logger.info("Creating SpatialData object")
    logger.info(f"Using coordinate transformation: Scale({mpp_x}, {mpp_y})")

    pixel_to_microns = Scale([mpp_x, mpp_y], axes=("x", "y"))

    shapes = {}
    points = {}
    tables = {}

    logger.info("Parsing cell contours as shapes for SpatialData")
    cell_contours_gdf = gpd.read_file(cell_contours_geojson_path)
    cell_contours_gdf = flatten_geojson_properties(cell_contours_gdf)
    shapes["cell_contours"] = ShapesModel.parse(
        cell_contours_gdf, transformations={"global": pixel_to_microns}
    )

    logger.info("Parsing cell centroids as points for SpatialData")
    cell_centroids_gdf = gpd.read_file(cell_centroids_geojson_path)
    cell_centroids_gdf = flatten_geojson_properties(cell_centroids_gdf)

    cell_centroids_df = pd.DataFrame(
        cell_centroids_gdf.drop(columns="geometry")
    )
    cell_centroids_df["x"] = cell_centroids_gdf.geometry.x
    cell_centroids_df["y"] = cell_centroids_gdf.geometry.y

    points["cell_centroids"] = PointsModel.parse(
        cell_centroids_df, transformations={"global": pixel_to_microns}
    )

    if tissue_contours_geojson_path is not None:
        logger.info("Parsing tissue contours as shapes for SpatialData")
        tissue_contours_gdf = gpd.read_file(tissue_contours_geojson_path)
        tissue_contours_gdf = flatten_geojson_properties(tissue_contours_gdf)
        shapes["tissue_contours"] = ShapesModel.parse(
            tissue_contours_gdf, transformations={"global": pixel_to_microns}
        )

    if artefact_contours_geojson_path is not None:
        logger.info("Parsing artefact contours as shapes for SpatialData")
        artefact_contours_gdf = gpd.read_file(artefact_contours_geojson_path)
        artefact_contours_gdf = flatten_geojson_properties(
            artefact_contours_gdf
        )
        shapes["artefact_contours"] = ShapesModel.parse(
            artefact_contours_gdf, transformations={"global": pixel_to_microns}
        )

    if roi_geojson_path is not None:
        logger.info("Parsing ROI contours as shapes for SpatialData")
        roi_gdf = gpd.read_file(roi_geojson_path)
        roi_gdf = flatten_geojson_properties(roi_gdf)
        shapes["roi_contours"] = ShapesModel.parse(
            roi_gdf, transformations={"global": pixel_to_microns}
        )

    if densities_df is not None:
        logger.info("Parsing cellular densities as table for SpatialData")
        metadata_cols = ["region", "cell_class"]
        measurement_cols = ["count", "density"]

        adata = AnnData(
            X=densities_df[measurement_cols].values,
            obs=densities_df[metadata_cols].reset_index(drop=True),
        )
        adata.var_names = measurement_cols
        tables["cellular_densities"] = TableModel.parse(adata)

    logger.info("Assembling SpatialData object")
    sdata = sd.SpatialData(
        shapes=shapes if shapes else None,
        points=points if points else None,
        tables=tables if tables else None,
    )

    logger.info("Adding metadata to SpatialData object")
    sdata.attrs["metadata"] = {
        "slide_basename": slide_basename,
        "mpp_x": float(mpp_x),
        "mpp_y": float(mpp_y),
        "model_config": model_config,
        "created_at": pd.Timestamp.now().isoformat(),
        "n_cells_detected": n_cells,
        "coordinate_system": "Pixel coordinates with scale transformation to microns",
    }

    logger.info(f"Writing SpatialData to {output_path}")
    sdata.write(output_path, overwrite=True)
    logger.info("SpatialData object created and saved successfully")
