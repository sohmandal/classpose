import json
import numpy as np
import shapely
import shapely.ops
import polars as pl
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import KDTree
from classpose.log import get_logger

logger = get_logger(__name__)

FEATURES = [
    "area",
    "perimeter",
    "elongation",
    "eccentricity",
    "solidity",
    "formfactor",
    "average_h",
    "average_e",
    "std_h",
    "std_e",
    "entropy_h",
    "entropy_e",
    "dist_to_lymph",
]


def load_geojson_multipolygon(
    geojson_path: str,
) -> list[tuple[shapely.Polygon, str]]:
    """
    Load a geojson file and extract polygons and their classifications.

    Args:
        geojson_path: Path to the geojson file

    Returns:
        List of tuples containing (polygon, classification_name)
    """
    with open(geojson_path) as o:
        geojson = json.load(o)

    features = geojson.get("features", [])
    polygons_with_class = []

    for feature in features:
        coordinates = feature["geometry"]["coordinates"]
        if len(coordinates) > 1:
            shell, holes = coordinates[0], coordinates[1:]
        else:
            shell, holes = coordinates[0], None
        polygon = shapely.Polygon(shell, holes=holes)
        polygon = shapely.make_valid(polygon)
        shapely.prepare(polygon)
        properties = feature.get("properties", {})
        classification = properties.get("classification", {}).get(
            "name", "Tissue"
        )
        polygons_with_class.append((polygon, classification))

    return polygons_with_class


def load_cells(
    cells_path: str,
    cancer_contours_path: str,
    tissue_contours_path: str,
) -> pl.DataFrame:
    """
    Load cell data from a parquet file, calculate distances to lymphocytes,
    and assign cells to tissue regions defined in a geojson file. The tissue
    GeoJSON is used to define the tissue regions and only the union between the
    cancer and tissue regions is considered.


    Args:
        cells_path (str): Path to the parquet file containing cell features.
        cancer_contours_path (str): Path to the geojson file containing region
            contours.
        tissue_contours_path (str): Path to the geojson file containing tissue
            region contours.

    Returns:
        cells (pl.DataFrame): DataFrame of cell features with added distance to
            nearest lymphocyte.
        polygon_areas (pl.DataFrame): DataFrame of tissue region areas.
    """
    cells = pl.read_parquet(cells_path).with_columns(
        (pl.col("major_axis") / pl.col("minor_axis"))
        .alias("elongation")
        .fill_null(0.0)
    )
    centroids = cells.select(["centroid_x", "centroid_y"]).to_numpy()
    points = shapely.points(centroids)
    lymphocyte_centroids = centroids[cells["classification"] == "Lymphocyte"]
    tree = KDTree(lymphocyte_centroids)
    dist_to_lymph, nearest = tree.query(centroids)
    # remove duplicates if any (very rare but can happen)
    """if nearest.shape[1] != cells.shape[0]:
        _, unique_indices = np.unique(nearest[0], return_index=True)
        dist_to_lymph = dist_to_lymph[unique_indices]"""
    cells = cells.with_columns(pl.Series("dist_to_lymph", dist_to_lymph))

    cancer_contours = load_geojson_multipolygon(cancer_contours_path)
    tissue_contours = load_geojson_multipolygon(tissue_contours_path)
    tissue_area = sum([c[0].area for c in tissue_contours])

    if tissue_contours:
        tissue_polys = [p for p, _ in tissue_contours]
        tissue_tree = shapely.STRtree(tissue_polys)
        intersected_contours = []
        for polygon, cl in cancer_contours:
            # Only intersect with tissue polygons whose bounding boxes overlap the cancer contour
            overlapping_idx = tissue_tree.query(polygon)
            if len(overlapping_idx) > 0:
                relevant_tissue = shapely.ops.unary_union(
                    [tissue_polys[i] for i in overlapping_idx]
                )
                intersection = polygon.intersection(relevant_tissue)
                if intersection.area > 0:
                    intersected_contours.append((intersection, cl))
        cancer_contours = intersected_contours

    masks = {}
    polygon_areas = {}
    point_tree = shapely.STRtree(points)
    for polygon, cl in cancer_contours:
        if cl not in masks:
            masks[cl] = np.zeros(cells.shape[0])
            polygon_areas[cl] = 0
        polygon_areas[cl] += polygon.area
        contained_indices = point_tree.query(polygon, predicate="contains")
        masks[cl][contained_indices] = 1
    for cl in masks:
        cells = cells.with_columns(pl.Series(f"{cl}_mask", masks[cl]))
    polygon_areas_df = []
    for k in polygon_areas:
        pa_dict = {"tissue_area": polygon_areas[k], f"{k}_mask": 1}
        polygon_areas_df.append(pa_dict)
    polygon_areas_df = pl.DataFrame(polygon_areas_df)
    return cells, polygon_areas_df, tissue_area


def process_sample(
    parquet_path, geojson_path, tissue_geojson_path, **additional_columns
):
    """
    Process a single sample to calculate summary statistics and cell densities.

    Loads cell and tissue region data, calculates mean and standard deviation
    for morphological and color features per cell classification and tissue
    region. Also calculates cell counts and densities per region.

    Args:
        parquet_path (str or Path): Path to the parquet file containing cell
            features.
        geojson_path (str or Path): Path to the geojson file containing tissue
            region contours.
        tissue_geojson_path (str or Path): Path to the geojson file containing
            tissue region contours.
        **additional_columns: Additional columns to add to the final summary
            (e.g., sample ID).

    Returns:
        overall_summary (pl.DataFrame): DataFrame with aggregated features and
            densities.
        n_cells (int): Total number of cells in the sample.
    """
    cells, polygon_areas, tissue_area = load_cells(
        parquet_path, geojson_path, tissue_geojson_path
    )
    mask_keys = [x for x in polygon_areas.columns if x != "tissue_area"]
    id_vars = [*mask_keys, "classification"]
    summary_feature_cols = [
        *[f"{x}.std" for x in FEATURES],
        *[f"{x}.mean" for x in FEATURES],
        "count",
    ]
    summaries = (
        cells.with_columns(*[pl.col(x).cast(pl.Int64) for x in mask_keys])
        .group_by(id_vars)
        .agg(
            *[pl.col(x).mean().alias(f"{x}.mean") for x in FEATURES],
            *[pl.col(x).std().alias(f"{x}.std") for x in FEATURES],
            pl.col("id").count().alias("count"),
        )
        .unpivot(on=summary_feature_cols, index=id_vars)
        .with_columns(
            pl.concat_str(
                pl.col("classification"),
                pl.col("variable"),
                separator=".",
            ).alias("variable")
        )
        .drop("classification")
        .with_columns(pl.col("value").cast(pl.Float32))
    )
    counts_raw = summaries.filter(pl.col("variable").str.contains("count"))
    n_cells = counts_raw["value"].sum()
    counts = (
        summaries.filter(pl.col("variable").str.contains("count"))
        .join(polygon_areas, on=mask_keys)
        .with_columns(
            (pl.col("value") / pl.col("tissue_area") / 1e-6)
            .alias("value")
            .cast(pl.Float32)
        )
        .with_columns(pl.col("variable").str.replace("count", "density"))
        .drop("tissue_area")
    )
    summaries = summaries.filter(~pl.col("variable").str.contains("count"))

    summaries_wsi = (
        cells.with_columns(*[pl.col(x).cast(pl.Int64) for x in mask_keys])
        .drop(mask_keys)
        .group_by("classification")
        .agg(
            *[pl.col(x).mean().alias(f"{x}.mean") for x in FEATURES],
            *[pl.col(x).std().alias(f"{x}.std") for x in FEATURES],
            pl.col("id").count().alias("count"),
        )
        .unpivot(on=summary_feature_cols, index="classification")
        .with_columns(
            pl.concat_str(
                pl.col("classification"),
                pl.col("variable"),
                separator=".",
            ).alias("variable")
        )
        .drop("classification")
        .with_columns(pl.col("value").cast(pl.Float32))
    )
    counts_wsi = (
        summaries_wsi.filter(pl.col("variable").str.contains("count"))
        .with_columns(
            (pl.col("value") / tissue_area / 1e-6)
            .alias("value")
            .cast(pl.Float32)
        )
        .with_columns(
            pl.concat_str(
                pl.lit("overall"),
                pl.col("variable"),
                separator=".",
            ).str.replace("count", "density")
        )
    )
    summaries_wsi = summaries_wsi.filter(
        ~pl.col("variable").str.contains("count")
    )

    overall_summary = pl.concat(
        [summaries, summaries_wsi, counts, counts_wsi], how="diagonal"
    ).with_columns(
        **{k: pl.lit(v) for k, v in additional_columns.items()},
    )
    return overall_summary, n_cells


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cells_path", type=str, required=True)
    parser.add_argument("--tissue_contours_path", type=str, required=True)
    parser.add_argument("--cancer_contours_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="slide_features.parquet")
    args = parser.parse_args()

    data_dict = {}
    n_no_match = 0
    tissue_contours_path = Path(args.tissue_contours_path)
    cancer_contours_path = Path(args.cancer_contours_path)
    for parquet in Path(args.cells_path).rglob("*parquet"):
        identifier = parquet.name.replace(".parquet", "")
        geojson = Path(f"{cancer_contours_path}/{identifier}.geojson")
        tissue = Path(
            f"{tissue_contours_path}/{identifier}_tissue_contours.geojson"
        )
        if geojson.exists() and tissue.exists():
            data_dict[identifier] = {
                "parquet": parquet,
                "geojson": geojson,
                "tissue": tissue,
            }
        else:
            n_no_match += 1
            logger.warning(f"Sample {identifier} not found in contours paths")

    logger.info(f"Found {len(data_dict)} samples")
    logger.info(f"Found {n_no_match} samples without matching geojson")

    all_summaries = []
    total_n_cells = 0
    with tqdm(data_dict) as pbar:
        for identifier in pbar:
            pbar.set_description(f"Processing {identifier}")
            parquet_path = data_dict[identifier]["parquet"]
            geojson_path = data_dict[identifier]["geojson"]
            tissue_geojson_path = data_dict[identifier]["tissue"]
            overall_summary, n_cells = process_sample(
                parquet_path,
                geojson_path,
                tissue_geojson_path,
                identifier=identifier,
            )
            total_n_cells += n_cells
            all_summaries.append(overall_summary)

    logger.info(f"Total number of cells: {total_n_cells}")
    final_summary = pl.concat(all_summaries, how="diagonal")
    final_summary.write_parquet(args.output)
