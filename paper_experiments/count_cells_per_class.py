"""
Count the number of cells belonging to each class from cell feature parquet files.

For each tissue sample this script:
1. Loads cell detections (parquet).
2. Groups cells by classification.
3. Produces per-sample and global counts per class.

Usage
-----
    python count_cells_per_class.py \
        --cells_path <dir_with_parquet_files> \
        --output <output.parquet>
"""

from pathlib import Path

import polars as pl
from tqdm import tqdm

from classpose.log import get_logger


logger = get_logger(__name__)


def count_cells_for_sample(parquet_path: Path, identifier: str) -> pl.DataFrame:
    """Count cells per classification for a single sample.

    Args:
        parquet_path: Path to the parquet file containing cell features.
        identifier: Sample identifier (derived from filename).

    Returns:
        A Polars DataFrame with columns: identifier, classification, count.
    """
    cells = pl.read_parquet(parquet_path)

    if "classification" not in cells.columns:
        raise ValueError(
            f"Parquet file {parquet_path} does not contain a 'classification' column."
        )

    counts = (
        cells
        .group_by("classification")
        .agg(pl.len().alias("count"))
        .with_columns(pl.lit(identifier).alias("identifier"))
        .select("identifier", "classification", "count")
    )
    return counts


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Count cells per classification from parquet cell feature files.",
    )
    parser.add_argument(
        "--cells_path",
        type=str,
        required=True,
        help=(
            "Path to directory containing parquet files with cell features. "
            "Each file must be named <identifier>.parquet."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cell_class_counts.parquet",
        help=(
            "Path for the output parquet file containing per-sample and global "
            "cell counts per class. Defaults to 'cell_class_counts.parquet'."
        ),
    )
    args = parser.parse_args()

    cells_path = Path(args.cells_path)

    all_counts: list[pl.DataFrame] = []
    total_cells = 0

    parquet_files = sorted(cells_path.rglob("*.parquet"))

    if not parquet_files:
        logger.warning(f"No parquet files found under {cells_path}")

    with tqdm(parquet_files) as pbar:
        for parquet in pbar:
            identifier = parquet.name.replace(".parquet", "")
            pbar.set_description(f"Processing {identifier}")

            counts = count_cells_for_sample(parquet, identifier)
            all_counts.append(counts)
            total_cells += int(counts["count"].sum())

    if not all_counts:
        logger.warning("No counts were computed; nothing to write.")
        return

    per_sample_counts = pl.concat(all_counts, how="diagonal")

    # Compute global counts per class across all samples
    global_counts = (
        per_sample_counts
        .group_by("classification")
        .agg(pl.col("count").sum().alias("count"))
        .with_columns(pl.lit("GLOBAL").alias("identifier"))
        .select("identifier", "classification", "count")
    )

    result = pl.concat([per_sample_counts, global_counts], how="diagonal")

    logger.info(f"Total number of cells (all classes, all samples): {total_cells}")
    logger.info(
        "Writing per-sample and global cell class counts to %s", args.output
    )

    result.write_parquet(args.output)


if __name__ == "__main__":
    main()
