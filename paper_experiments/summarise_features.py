"""
Compute per-sample summary statistics and cell densities from cell feature
parquet files and GeoJSON region annotations.

For each tissue sample this script:
1. Loads cell detections (parquet) and matches them to cancer region contours
   (GeoJSON) and tissue contours (GeoJSON).
2. Computes morphological and colour features per cell, including elongation,
   distance to nearest lymphocyte, and the 7 Hu invariant moments.
3. Groups cells by classification and region mask, then calculates per-group
   mean, standard deviation, count, and density (cells per µm²) for all
   features in FEATURES.
4. Repeats the aggregation at WSI level (ignoring region masks) to produce
   overall summary statistics.
5. Concatenates region-level and WSI-level summaries and writes the result
   to a parquet file.

Usage
-----
    python summarise_features.py \\
        --cells_path <dir_with_parquet_files> \\
        --cancer_contours_path <dir_with_geojson_files> \\
        --tissue_contours_path <dir_with_tissue_geojson_files> \\
        --output <output.parquet>
"""

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
    "hu_0",
    "hu_1",
    "hu_2",
    "hu_3",
    "hu_4",
    "hu_5",
    "hu_6",
]


def hu_0(nu20: pl.Expr, nu02: pl.Expr) -> pl.Expr:
    """
    Calculates the 1st Hu invariant moment.

    Args:
        nu20 (pl.Expr): The normalized central moment nu20.
        nu02 (pl.Expr): The normalized central moment nu02.

    Returns:
        pl.Expr: The 1st Hu invariant moment.
    """

    return nu20 + nu02


def hu_1(nu20: pl.Expr, nu02: pl.Expr, nu11: pl.Expr) -> pl.Expr:
    """
    Calculates the 2nd Hu invariant moment.

    Args:
        nu20 (pl.Expr): The normalized central moment nu20.
        nu02 (pl.Expr): The normalized central moment nu02.
        nu11 (pl.Expr): The normalized central moment nu11.

    Returns:
        pl.Expr: The 2nd Hu invariant moment.
    """
    return (nu20 - nu02) ** 2 + 4 * (nu11**2)


def hu_2(nu30: pl.Expr, nu12: pl.Expr, nu21: pl.Expr, nu03: pl.Expr) -> pl.Expr:
    """
    Calculates the 3rd Hu invariant moment.

    Args:
        nu30 (pl.Expr): The normalized central moment nu30.
        nu12 (pl.Expr): The normalized central moment nu12.
        nu21 (pl.Expr): The normalized central moment nu21.
        nu03 (pl.Expr): The normalized central moment nu03.

    Returns:
        pl.Expr: The 3rd Hu invariant moment.
    """
    return (nu30 - 3 * nu12) ** 2 + (3 * nu21 - nu03) ** 2


def hu_3(nu30: pl.Expr, nu12: pl.Expr, nu21: pl.Expr, nu03: pl.Expr) -> pl.Expr:
    """
    Calculates the 4th Hu invariant moment.

    Args:
        nu30 (pl.Expr): The normalized central moment nu30.
        nu12 (pl.Expr): The normalized central moment nu12.
        nu21 (pl.Expr): The normalized central moment nu21.
        nu03 (pl.Expr): The normalized central moment nu03.

    Returns:
        pl.Expr: The 4th Hu invariant moment.
    """
    return (nu30 + nu12) ** 2 + (nu21 + nu03) ** 2


def hu_4(nu30: pl.Expr, nu12: pl.Expr, nu21: pl.Expr, nu03: pl.Expr) -> pl.Expr:
    """
    Calculates the 5th Hu invariant moment.

    Args:
        nu30 (pl.Expr): The normalized central moment nu30.
        nu12 (pl.Expr): The normalized central moment nu12.
        nu21 (pl.Expr): The normalized central moment nu21.
        nu03 (pl.Expr): The normalized central moment nu03.

    Returns:
        pl.Expr: The 5th Hu invariant moment.
    """
    return (nu30 - 3 * nu12) * (nu30 + nu12) * (
        (nu30 + nu12) ** 2 - 3 * (nu21 + nu03) ** 2
    ) + (3 * nu21 - nu03) * (nu21 + nu03) * (
        3 * (nu30 + nu12) ** 2 - (nu21 + nu03) ** 2
    )


def hu_5(
    nu20: pl.Expr,
    nu02: pl.Expr,
    nu11: pl.Expr,
    nu30: pl.Expr,
    nu12: pl.Expr,
    nu21: pl.Expr,
    nu03: pl.Expr,
) -> pl.Expr:
    """
    Calculates the 6th Hu invariant moment.

    Args:
        nu20 (pl.Expr): The normalized central moment nu20.
        nu02 (pl.Expr): The normalized central moment nu02.
        nu11 (pl.Expr): The normalized central moment nu11.
        nu30 (pl.Expr): The normalized central moment nu30.
        nu12 (pl.Expr): The normalized central moment nu12.
        nu21 (pl.Expr): The normalized central moment nu21.
        nu03 (pl.Expr): The normalized central moment nu03.

    Returns:
        pl.Expr: The 6th Hu invariant moment.
    """
    return (nu20 - nu02) * (
        (nu30 + nu12) ** 2 - (nu21 + nu03) ** 2
    ) + 4 * nu11 * (nu30 + nu12) * (nu21 + nu03)


def hu_6(nu30: pl.Expr, nu12: pl.Expr, nu21: pl.Expr, nu03: pl.Expr) -> pl.Expr:
    """
    Calculates the 7th Hu invariant moment.

    Args:
        nu30 (pl.Expr): The normalized central moment nu30.
        nu12 (pl.Expr): The normalized central moment nu12.
        nu21 (pl.Expr): The normalized central moment nu21.
        nu03 (pl.Expr): The normalized central moment nu03.

    Returns:
        pl.Expr: The 7th Hu invariant moment.
    """
    return (3 * nu21 - nu03) * (nu30 + nu12) * (
        (nu30 + nu12) ** 2 - 3 * (nu21 + nu03) ** 2
    ) - (nu30 - 3 * nu12) * (nu21 + nu03) * (
        3 * (nu30 + nu12) ** 2 - (nu21 + nu03) ** 2
    )


def expand_features(cells: pl.DataFrame) -> pl.DataFrame:
    """
    Calculates additional features, particularly elongation (using the major
    and minor axes) and the 7 invariant (Hu) moments.

    Args:
        cells (pl.DataFrame): DataFrame containing cell features with columns
            for major_axis, minor_axis, and normalized central moments (nu20,
            nu02, nu11, nu30, nu12, nu21, nu03).

    Returns:
        pl.DataFrame: DataFrame with additional calculated features including
            elongation and the 7 Hu invariant moments (hu_0 through hu_6).
    """
    cells = cells.with_columns(
        # calculates elongation
        (pl.col("major_axis") / pl.col("minor_axis"))
        .fill_null(0.0)
        .alias("elongation"),
        hu_0(pl.col("nu20"), pl.col("nu02")).alias("hu_0"),
        hu_1(pl.col("nu20"), pl.col("nu02"), pl.col("nu11")).alias("hu_1"),
        hu_2(
            pl.col("nu30"), pl.col("nu12"), pl.col("nu21"), pl.col("nu03")
        ).alias("hu_2"),
        hu_3(
            pl.col("nu30"), pl.col("nu12"), pl.col("nu21"), pl.col("nu03")
        ).alias("hu_3"),
        hu_4(
            pl.col("nu30"), pl.col("nu12"), pl.col("nu21"), pl.col("nu03")
        ).alias("hu_4"),
        hu_5(
            pl.col("nu20"),
            pl.col("nu02"),
            pl.col("nu11"),
            pl.col("nu30"),
            pl.col("nu12"),
            pl.col("nu21"),
            pl.col("nu03"),
        ).alias("hu_5"),
        hu_6(
            pl.col("nu30"), pl.col("nu12"), pl.col("nu21"), pl.col("nu03")
        ).alias("hu_6"),
    )
    return cells


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
    tissue_contours_path: str | None = None,
    calculate_distances: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """
    Load cell data from a parquet file, calculate distances to lymphocytes,
    and assign cells to tissue regions defined in a geojson file. The tissue
    GeoJSON is used to define the tissue regions and only the union between the
    cancer and tissue regions is considered.


    Args:
        cells_path (str): Path to the parquet file containing cell features.
        cancer_contours_path (str): Path to the geojson file containing region
            contours.
        tissue_contours_path (str | None): Path to the geojson file containing
            tissue region contours. Defaults to None.
        calculate_distances (bool): Whether to calculate distances to cancer
            boundaries. Default is False.

    Returns:
        cells (pl.DataFrame): DataFrame of cell features with added distance to
            nearest lymphocyte.
        polygon_areas (pl.DataFrame): DataFrame of tissue region areas.
        tissue_area (int): Total area of tissue regions.
    """
    cells = pl.read_parquet(cells_path)
    cells = expand_features(cells)
    centroids = cells.select(["centroid_x", "centroid_y"]).to_numpy()
    points = shapely.points(centroids)
    lymphocyte_centroids = centroids[cells["classification"] == "Lymphocyte"]
    tree = KDTree(lymphocyte_centroids)
    dist_to_lymph, nearest = tree.query(centroids)
    cells = cells.with_columns(pl.Series("dist_to_lymph", dist_to_lymph))

    cancer_contours = load_geojson_multipolygon(cancer_contours_path)

    tissue_area = 0
    if tissue_contours_path:
        tissue_contours = load_geojson_multipolygon(tissue_contours_path)
        tissue_area = sum([c[0].area for c in tissue_contours])
        tissue_polys = [p for p, _ in tissue_contours]
        tissue_tree = shapely.STRtree(tissue_polys)
        intersected_contours = []
        for polygon, cl in cancer_contours:
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
    if calculate_distances:
        distance_to_boundary = np.ones(cells.shape[0]) * np.inf
        for polygon, cl in cancer_contours:
            distances = polygon.exterior.distance(points)
            distance_to_boundary = np.where(
                distances < distance_to_boundary,
                distances,
                distance_to_boundary,
            )
        cells = cells.with_columns(
            pl.Series("distance_to_boundary", distance_to_boundary)
        )

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

    parser = argparse.ArgumentParser(
        description="Compute per-sample summary statistics and cell densities."
    )
    parser.add_argument(
        "--cells_path",
        type=str,
        required=True,
        help="Path to directory containing parquet files with cell features. "
        "Each file must be named <identifier>.parquet.",
    )
    parser.add_argument(
        "--tissue_contours_path",
        type=str,
        required=True,
        help="Path to directory containing tissue contour annotations as GeoJSON "
        "files. Each file must be named <identifier>_tissue_contours.geojson.",
    )
    parser.add_argument(
        "--cancer_contours_path",
        type=str,
        required=True,
        help="Path to directory containing cancer region annotations as GeoJSON "
        "files. Each file must be named <identifier>.geojson.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="slide_features.parquet",
        help="Path for the output parquet file containing per-sample, per-region "
        "summary statistics and cell densities. Defaults to "
        "'slide_features.parquet'.",
    )
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
