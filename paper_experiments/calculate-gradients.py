"""
Calculate spatial gradients of cell morphological features relative to cancer
boundaries.

For each tissue sample this script:
1. Loads cell detections (parquet) and cancer contour annotations (GeoJSON).
2. Computes each cell's log-transformed distance to the nearest tissue boundary
   and a binary cancer-region mask.
3. Fits an OLS regression (via polars-ols) per sample and morphological feature
   (perimeter, solidity, elongation, entropy_h) using:
   - one-hot dummy columns for each cell type,
   - interaction terms between the dummies and log-distance (gradient inside
     non-cancer regions),
   - interaction terms between the dummies, log-distance, and the cancer mask
     (gradient inside cancer regions).
4. Collects regression coefficients, standard errors, t-values, p-values and
   goodness-of-fit statistics (R², MAE, MSE) across all samples.
5. Writes the joined coefficient/statistics table to a parquet file.

Usage
-----
    python calculate-gradients.py \\
        --cells_path <dir_with_parquet_files> \\
        --cancer_contours_path <dir_with_geojson_files> \\
        --output <output.parquet>
"""

import polars as pl
import polars_ols as pls
from pathlib import Path
from summarise_features import load_cells
from classpose.log import get_logger
from tqdm import tqdm

FEATURES = [
    "perimeter",
    "solidity",
    "elongation",
    "entropy_h",
]
CELL_TYPES = [
    "Epithelial",
    "Neutrophil",
    "Connective",
    "Lymphocyte",
    "Plasma cell",
    "Eosinophil",
]
STATS_COLS = ["r2", "mae", "mse"]
COEFS_COLS = [
    "feature_names",
    "coefficients",
    "standard_errors",
    "t_values",
    "p_values",
]

logger = get_logger(__name__)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate gradients for cell types based on distance to cancer contours."
    )
    parser.add_argument(
        "--cells_path",
        type=str,
        required=True,
        help="Path to directory containing parquet files with cell features. The "
        "parquet files should have at least perimeter, solidity, elongation and "
        "entropy_h, as well as a 'classification' columns.",
    )
    parser.add_argument(
        "--cancer_contours_path",
        type=str,
        required=True,
        help="Path to directory containing cancer contour annotations as GeoJSON "
        "files. Each file must be named <identifier>.geojson to match the "
        "corresponding <identifier>.parquet in --cells_path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for the output parquet file containing OLS coefficients and "
        "fit statistics (R², MAE, MSE) for every sample/feature combination.",
    )
    args = parser.parse_args()

    data_dict = {}
    n_no_match = 0
    cancer_contours_path = Path(args.cancer_contours_path)
    for parquet in Path(args.cells_path).rglob("*parquet"):
        identifier = parquet.name.replace(".parquet", "")
        geojson = Path(f"{cancer_contours_path}/{identifier}.geojson")

        if geojson.exists():
            data_dict[identifier] = {
                "parquet": parquet,
                "geojson": geojson,
            }
        else:
            n_no_match += 1
            logger.warning(f"Sample {identifier} not found in contours paths")

    cell_type_dict = {k: i for i, k in enumerate(CELL_TYPES)}

    all_stats = []
    all_coefficients = []
    for identifier in tqdm(data_dict, desc="Processing samples"):
        cells_path = data_dict[identifier]["parquet"]
        geojson_path = data_dict[identifier]["geojson"]
        cells, polygon_areas, _ = load_cells(
            cells_path,
            geojson_path,
            calculate_distances=True,
        )
        dummy_col_names = [
            f"dummy_classification_{i}" for i in range(len(cell_type_dict))
        ]
        inter_col_names = [f"{k}_log_dist" for k in dummy_col_names]
        inter_col_cancer_names = [
            f"{k}_cancer_log_dist" for k in dummy_col_names
        ]
        cells = cells.with_columns(
            pl.col("classification")
            .replace(cell_type_dict)
            .alias("classification_idx"),
            pl.col("distance_to_boundary").log1p().alias("log_dist"),
        )
        cells = cells.with_columns(
            [
                (pl.col("classification_idx") == str(i))
                .cast(pl.Int8)
                .alias(dummy_col_names[i])
                for i in range(len(CELL_TYPES))
            ]
        )
        cells = cells.with_columns(
            [
                (pl.col(dummy_col_names[i]) * pl.col("log_dist")).alias(
                    inter_col_names[i]
                )
                for i in range(len(CELL_TYPES))
            ]
            + [
                (
                    pl.col(dummy_col_names[i])
                    * pl.col("log_dist")
                    * pl.col("cancer_mask")
                ).alias(inter_col_cancer_names[i])
                for i in range(len(CELL_TYPES))
            ]
        )
        X_col_names = [
            *dummy_col_names,
            *inter_col_names,
            *inter_col_cancer_names,
        ]
        for feature in FEATURES:
            expr = [pl.col(k) for k in X_col_names]
            coefficients = cells.select(
                ((pl.col(feature) - pl.mean(feature)) / pl.std(feature))
                .least_squares.ols(*expr, mode="statistics")
                .alias("coef")
            ).unnest("coef")
            stats = coefficients.select(STATS_COLS).with_columns(
                pl.lit(identifier).alias("identifier"),
                pl.lit(feature).alias("feature"),
            )
            coefficients = coefficients.select(
                pl.col(k).explode() for k in COEFS_COLS
            ).with_columns(
                pl.lit(identifier).alias("identifier"),
                pl.lit(feature).alias("feature"),
            )
            all_stats.append(stats)
            all_coefficients.append(coefficients)

    all_stats = pl.concat(all_stats)
    all_coefficients = pl.concat(all_coefficients)
    final_df = all_coefficients.join(all_stats, on=["identifier", "feature"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    final_df.write_parquet(args.output)
