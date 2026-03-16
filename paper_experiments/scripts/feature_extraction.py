from __future__ import annotations

import time
import json
import pandas as pd
import numpy as np
import shapely
import cv2
from skimage.color.colorconv import hed_from_rgb
from skimage.measure import regionprops
from multiprocessing import Queue, Process, Lock, Value
from openslide import OpenSlide
from tqdm import tqdm

from classpose.log import get_logger
from classpose import WSIReader

logger = get_logger(__name__)

ADJUSTMENT = 16
BAR_FMT = "{l_bar}{bar:20}{r_bar}{bar:-10b}"
SUPPORTED_WSI_BACKENDS = ["openslide", "czi"]
EPS = 1e-6
LOG_EPS = np.log(EPS)
RGB_ENTROPY_RANGE = (0.0, 1.0)
HED_ENTROPY_RANGE = (0.0, 3.0)
ENTROPY_BINS = 256


class CellLoader:
    def __init__(
        self,
        wsi_path,
        geojson_path,
        n_workers: int = 1,
        n_workers_extraction: int = 1,
        geojson_scale: float = 1.0,
    ):
        self.wsi_path = wsi_path
        self.geojson_path = geojson_path
        self.n_workers = n_workers
        self.n_workers_extraction = n_workers_extraction
        self.geojson_scale = 1.0 if geojson_scale is None else float(geojson_scale)
        
        with open(self.geojson_path) as f:
            self.geojson = json.load(f)

        self.n = len(self.geojson["features"])

        self.lock = Lock()
        self.q_loading = Queue()
        self.loaded_cells = Value("i", 0)
        split_idx = np.array_split(np.arange(self.n), n_workers)
        self.processes = []
        for i, process_idx in enumerate(split_idx):
            logger.info(
                f"Starting cell loading process {i} with {len(process_idx)} cells"
            )
            p = Process(
                target=self.load_cells,
                args=(
                    [self.geojson["features"][idx] for idx in process_idx],
                    self.q_loading,
                    i,
                ),
            )
            p.start()
            self.processes.append(p)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("processes", None)
        return state

    def load_cells(self, cell_features, q: Queue, process_id: int | str):
        wsi = WSIReader(self.wsi_path)
        for cell in cell_features:
            props = cell["properties"]
            classification = props["classification"]["name"]
            coords = np.array(cell["geometry"]["coordinates"][0])
            id = cell["id"]
            centroid = {
                p["name"]: p["value"]
                for p in props["measurements"]
                if "centroid" in p["name"]
            }
            cell = {
                "id": id,
                "classification": classification,
                "coords": shapely.Polygon(coords),
                **centroid,
            }
            with self.lock:
                self.loaded_cells.value += 1
            cell = self.load_single_cell(wsi, cell)
            q.put(cell)
        logger.info(f"Finished loading cells for process {process_id}")
        for _ in range(self.n_workers_extraction):
            q.put(None)

    def load_single_cell(self, wsi: OpenSlide | "CZISlide", cell):
        cell_coords = np.array(cell["coords"].exterior.coords, dtype=np.float64)
        cell_coords = np.rint(cell_coords * self.geojson_scale).astype(np.int32)

        adjustment = max(1, int(round(ADJUSTMENT * self.geojson_scale)))
        min_x = int(cell_coords[:, 0].min() - adjustment)
        min_y = int(cell_coords[:, 1].min() - adjustment)
        max_x = int(cell_coords[:, 0].max() + adjustment)
        max_y = int(cell_coords[:, 1].max() + adjustment)
        size_x = max(1, int(max_x - min_x))
        size_y = max(1, int(max_y - min_y))
        image = np.array(wsi.read_region((min_x, min_y), 0, (size_x, size_y)))
        mask = np.zeros((size_y, size_x), dtype=np.uint8)
        cell_coords = cell_coords - [min_x, min_y]
        cv2.fillPoly(mask, [cell_coords], 1)
        cell["image"] = image[..., :3]
        cell["mask"] = mask
        return cell

    def __del__(self):
        if hasattr(self, "processes"):
            for p in self.processes:
                p.join()
                p.close()


class FeatureExtraction:
    def __init__(self, cell_loader: CellLoader, n_workers: int = 1):
        self.cell_loader = cell_loader
        self.n_workers = n_workers

        self.processes = []

        if self.n_workers > 1:
            self.q_extract = Queue()
            for i in range(self.n_workers):
                logger.info(f"Starting feature extraction process {i}")
                p = Process(
                    target=self.extract,
                    args=(self.q_extract,),
                )
                p.start()
                self.processes.append(p)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("processes", None)
        return state

    def _entropy_from_quantized(self, qvals: np.ndarray, bins: int) -> np.ndarray:
        counts = np.empty((3, bins), dtype=np.int64)
        counts[0] = np.bincount(qvals[:, 0], minlength=bins)
        counts[1] = np.bincount(qvals[:, 1], minlength=bins)
        counts[2] = np.bincount(qvals[:, 2], minlength=bins)
        probs = counts.astype(np.float64)
        probs /= probs.sum(axis=1, keepdims=True) + EPS
        log_probs = np.zeros_like(probs)
        np.log2(probs, out=log_probs, where=probs > 0)
        return -(probs * log_probs).sum(axis=1)

    def extract_3channel_features(self, norm: np.ndarray, channel_names: str) -> dict:
        if norm.shape[0] == 0:
            return {
                f"average_{channel_names[0]}": np.nan,
                f"average_{channel_names[1]}": np.nan,
                f"average_{channel_names[2]}": np.nan,
                f"std_{channel_names[0]}": np.nan,
                f"std_{channel_names[1]}": np.nan,
                f"std_{channel_names[2]}": np.nan,
                f"entropy_{channel_names[0]}": np.nan,
                f"entropy_{channel_names[1]}": np.nan,
                f"entropy_{channel_names[2]}": np.nan,
            }

        average = norm.mean(axis=0)
        std = norm.std(axis=0)
        if channel_names == "rgb":
            qvals = np.clip((norm * 255.0).astype(np.int32), 0, 255)
            entropy = self._entropy_from_quantized(qvals, bins=256)
        else:
            lo, hi = HED_ENTROPY_RANGE
            scale = (ENTROPY_BINS - 1) / max(hi - lo, EPS)
            qvals = np.clip(((norm - lo) * scale).astype(np.int32), 0, ENTROPY_BINS - 1)
            entropy = self._entropy_from_quantized(qvals, bins=ENTROPY_BINS)

        features = {
            f"average_{channel_names[0]}": average[0],
            f"average_{channel_names[1]}": average[1],
            f"average_{channel_names[2]}": average[2],
            f"std_{channel_names[0]}": std[0],
            f"std_{channel_names[1]}": std[1],
            f"std_{channel_names[2]}": std[2],
            f"entropy_{channel_names[0]}": entropy[0],
            f"entropy_{channel_names[1]}": entropy[1],
            f"entropy_{channel_names[2]}": entropy[2],
        }
        return features

    def extract_intensity_features(self, cell):
        rgb = cell["image"][cell["mask"] == 1]
        norm_rgb = rgb.astype(np.float32) / 255.0
        norm_hed = convert_rgb_to_hed(norm_rgb)
        features = {
            **self.extract_3channel_features(norm_rgb, "rgb"),
            **self.extract_3channel_features(norm_hed, "hed"),
        }
        return features

    def extract_shape_features(self, cell: dict):
        # uses shapely to extract shape features
        polygon = cell["coords"]
        polygon = shapely.remove_repeated_points(polygon)
        min_bound_rect = polygon.minimum_rotated_rectangle
        x, y = min_bound_rect.exterior.coords.xy
        a = ((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2) ** 0.5
        b = ((x[2] - x[1]) ** 2 + (y[2] - y[1]) ** 2) ** 0.5
        minor_axes, major_axes = sorted([a, b])
        props = regionprops(cell["mask"].astype(np.uint8))
        if props:
            prop = props[0]
            eccentricity = prop.eccentricity
            orientation = prop.orientation
            solidity = prop.solidity
        else:
            eccentricity = np.nan
            orientation = np.nan
            solidity = np.nan

        formfactor = 4 * np.pi * polygon.area / max(polygon.length**2, EPS)
        features = {
            "area": polygon.area,
            "perimeter": polygon.length,
            "convex_hull_area": polygon.convex_hull.area,
            "minor_axis": minor_axes,
            "major_axis": major_axes,
            "eccentricity": eccentricity,
            "orientation": orientation,
            "solidity": solidity,
            "formfactor": formfactor,
        }

        contour = np.array(polygon.exterior.coords, dtype=np.float32).reshape(-1, 1, 2)
        moments = cv2.moments(contour)
        features.update(moments)

        return features

    def extract_features(self, cell):
        return {
            "id": cell["id"],
            "classification": cell["classification"],
            "centroid_x": cell["centroidX"],
            "centroid_y": cell["centroidY"],
            **self.extract_shape_features(cell),
            **self.extract_intensity_features(cell),
        }

    def extract(self, q):
        n_hits = 0
        while True:
            cell = self.cell_loader.q_loading.get()
            if cell is None:
                n_hits += 1
                time.sleep(0.05)
                if n_hits >= self.cell_loader.n_workers:
                    break
                continue
            features = self.extract_features(cell)
            q.put(features)
        q.put(None)

    def iter_mp(self):
        N = self.cell_loader.n
        fmt = "Extracting (loaded cells: {})"
        n_hits = 0
        with tqdm(
            total=N,
            desc=fmt.format(self.cell_loader.loaded_cells.value),
            bar_format=BAR_FMT,
            mininterval=1.0,
            miniters=500,
        ) as pbar:
            while True:
                features = self.q_extract.get()
                if features is None:
                    n_hits += 1
                    if n_hits >= self.n_workers:
                        break
                    continue
                pbar.update()
                if pbar.n % 1000 == 0:
                    pbar.set_description(fmt.format(self.cell_loader.loaded_cells.value))
                yield features

    def iter_sp(self):
        N = self.cell_loader.n
        fmt = "Extracting (loaded cells: {})"
        n_hits = 0
        with tqdm(
            total=N,
            desc=fmt.format(self.cell_loader.loaded_cells.value),
            bar_format=BAR_FMT,
            mininterval=1.0,
            miniters=500,
        ) as pbar:
            while True:
                cell = self.cell_loader.q_loading.get()
                if cell is None:
                    n_hits += 1
                    time.sleep(0.05)
                    if n_hits >= self.cell_loader.n_workers:
                        break
                    continue
                features = self.extract_features(cell)
                pbar.update()
                if pbar.n % 1000 == 0:
                    pbar.set_description(fmt.format(self.cell_loader.loaded_cells.value))
                yield features

    def __iter__(self):
        if self.processes:
            for features in self.iter_mp():
                yield features
        else:
            for features in self.iter_sp():
                yield features

    def __del__(self):
        if hasattr(self, "processes"):
            if len(self.processes) == 0:
                return
            for p in self.processes:
                p.join()
                p.close()


def convert_rgb_to_hed(rgb_image: np.ndarray) -> np.ndarray:
    stains = np.log(np.maximum(rgb_image, EPS)) / LOG_EPS @ hed_from_rgb
    stains = np.maximum(stains, 0)
    return stains


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wsi_path",
        type=str,
        required=True,
        help="Path to the WSI file",
    )
    parser.add_argument(
        "--geojson_path",
        type=str,
        required=True,
        help="Path to the GeoJSON file",
    )
    parser.add_argument(
        "--n_workers_reading",
        type=int,
        default=1,
        help="Number of workers for the cell loading",
    )
    parser.add_argument(
        "--n_workers_feature",
        type=int,
        default=1,
        help="Number of workers for the feature extraction",
    )
    parser.add_argument(
        "--geojson_scale",
        type=float,
        default=1.0,
        help="Ratio between the geojson scale and the slide scale",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output parquet file",
    )
    args = parser.parse_args()

    start_time = time.time()

    cell_loader = CellLoader(
        args.wsi_path,
        args.geojson_path,
        n_workers=args.n_workers_reading,
        n_workers_extraction=args.n_workers_feature,
        geojson_scale=args.geojson_scale,
    )
    feature_extractor = FeatureExtraction(cell_loader, n_workers=args.n_workers_feature)

    all_features = []
    for features in feature_extractor:
        all_features.append(features)

    logger.info(f"Saving features to {args.output}")
    df = pd.DataFrame(all_features)
    # convert numerical columns to float32
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")

    df.to_parquet(args.output, index=False)

    end_time = time.time()

    logger.info(f"Done! Time: {end_time - start_time:.2f} seconds")