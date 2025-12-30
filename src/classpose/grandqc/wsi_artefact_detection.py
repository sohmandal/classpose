import json
from typing import Any

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from openslide import OpenSlide
from PIL import Image
from tqdm import tqdm

from classpose.grandqc.wsi_qc_helpers import (
    create_geojson_feature,
    extract_slide_info,
    get_preprocessing,
    make_class_map,
    simulate_jpeg_compression,
)
from classpose.grandqc.wsi_tissue_detection import detect_tissue_wsi
from classpose.log import get_logger
from classpose.utils import download_if_unavailable, get_device

grandqc_logger = get_logger(__name__)

MODEL_URL_PATH = "https://zenodo.org/records/14041538/files/GrandQC_MPP1.pth"

# Artefact class colors (from GrandQC) - CORRECTED ORDER
ARTIFACT_COLORS = [
    [0, 0, 0],  # Index 0: Unused (for compatibility with make_class_map)
    [0, 0, 0],  # Class 1: Normal Tissue
    [255, 99, 71],  # Class 2: Folds
    [0, 255, 0],  # Class 3: Dark spots & Foreign Objects
    [255, 0, 0],  # Class 4: Pen marks
    [255, 0, 255],  # Class 5: Edges
    [75, 0, 130],  # Class 6: Out-of-focus areas
    [255, 255, 255],  # Class 7: Background
]

ARTIFACT_CLASS_MAPPING = {
    0: "Unused",  # Not used by GrandQC model
    1: "Normal Tissue",  # GrandQC class 1
    2: "Fold",  # GrandQC class 2
    3: "Darkspot & Foreign Object",  # GrandQC class 3
    4: "PenMarking",  # GrandQC class 4
    5: "Edge & Air Bubble",  # GrandQC class 5
    6: "OOF",  # GrandQC class 6
    7: "Background",  # GrandQC class 7
}


def detect_artefacts_wsi(
    slide: OpenSlide,
    # artefact detection model parameters
    model_art_path: str = "./models/artefact_detection/GrandQC_MPP1.pth",
    mpp_model_art: float = 1.0,
    m_p_s_model_art: int = 512,
    encoder_model_name: str = "timm-efficientnet-b0",
    encoder_model_weights: str = "imagenet",
    device: str = "cuda",
    # tissue detection parameters (passed to tissue detection)
    model_td_path: str = "./models/tissue_detection/Tissue_Detection_MPP10.pth",
    mpp_model_td: int = 10,
    m_p_s_model_td: int = 512,
    min_area: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """
    Detects artefacts in a whole-slide image using GrandQC method.
    First performs tissue detection, then artefact segmentation on tissue areas.
    
    Please note that if you use any part of Classpose which makes use of GrandQC please 
    follow the instructions at https://github.com/cpath-ukk/grandqc/tree/main to cite them 
    appropriately. Similarly to Classpose, GrandQC is under a non-commercial license 
    whose terms can be found at https://github.com/cpath-ukk/grandqc/blob/main/LICENSE.


    Args:
        slide (OpenSlide): slide to detect artefacts in.
        model_art_path (str): path to the artefact detection model.
        mpp_model_art (float): MPP of the artefact detection model (default 1.0).
        m_p_s_model_art (int): patch size of the artefact model.
        encoder_model_name (str): encoder model name.
        encoder_model_weights (str): encoder weights.
        device (str): device to run on.
        model_td_path (str): path to tissue detection model.
        mpp_model_td (int): MPP for tissue detection.
        m_p_s_model_td (int): patch size for tissue detection.
        min_area (int): minimum area for tissue polygons.

    Returns:
        tuple: artefact_mask (np.ndarray), artefact_map (np.ndarray), artefact_cnts (dict), geojson (dict)
    """
    # First, detect tissue
    grandqc_logger.info("Performing tissue detection...")
    _, tissue_mask, _, _, _, _ = detect_tissue_wsi(
        slide,
        model_td_path=model_td_path,
        mpp_model_td=mpp_model_td,
        m_p_s_model_td=m_p_s_model_td,
        encoder_model_name=encoder_model_name,
        encoder_model_weights=encoder_model_weights,
        device=device,
        min_area=min_area,
    )

    # Load artefact model
    model_art_path = download_if_unavailable(model_art_path, MODEL_URL_PATH)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        encoder_model_name, encoder_model_weights
    )

    model = torch.load(model_art_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    # Slide info
    w_l0, h_l0, mpp, thumbnail_dimensions = extract_slide_info(
        slide, mpp_model_art
    )
    grandqc_logger.info(
        "Extracting thumbnail with size %s for artifact detection",
        thumbnail_dimensions,
    )
    image_or = slide.get_thumbnail(thumbnail_dimensions)

    grandqc_logger.info(
        "Simulating jpeg compression to match training preprocessing"
    )

    image = simulate_jpeg_compression(image_or)

    width, height = image.size

    observed_reduction_w = w_l0 / width
    observed_reduction_h = h_l0 / height

    patch_n_w = width // m_p_s_model_art
    patch_n_h = height // m_p_s_model_art

    grandqc_logger.info(
        "Processing %d x %d patches",
        patch_n_w,
        patch_n_h,
    )

    # Resize tissue mask to artefact resolution
    tissue_mask_art = cv2.resize(
        tissue_mask, (width, height), interpolation=cv2.INTER_NEAREST
    )

    grandqc_logger.info(
        "Processing slide thumbnail in patches for artefact detection..."
    )
    grandqc_logger.info(
        "Thumbnail size: %dx%d, processing %d x %d patches",
        width,
        height,
        patch_n_w,
        patch_n_h,
    )

    # Process patches on thumbnail
    for h in tqdm(range(patch_n_h), desc="Processing rows"):
        grandqc_logger.debug("Processing row %d/%d", h, patch_n_h)
        for w in range(patch_n_w):
            h_start = h * m_p_s_model_art
            h_end = (h + 1) * m_p_s_model_art
            w_start = w * m_p_s_model_art
            w_end = (w + 1) * m_p_s_model_art

            image_work = image.crop((w_start, h_start, w_end, h_end))

            td_patch = tissue_mask_art[h_start:h_end, w_start:w_end]

            if (
                np.count_nonzero(td_patch == 1) > 50
            ):  # tissue present (tissue=1, background=0)
                image_pre = get_preprocessing(image_work, preprocessing_fn)
                x_tensor = torch.from_numpy(image_pre).to(device).unsqueeze(0)
                predictions = model.predict(x_tensor)
                predictions = predictions.squeeze().cpu().numpy()
                mask_raw = np.argmax(predictions, axis=0).astype("int8")
                mask = np.where(
                    td_patch == 1, mask_raw, 7
                )  # Keep artefact predictions only in tissue areas
            else:
                # Create mask with the same shape as td_patch
                mask = np.full(td_patch.shape, 7)

            if w == 0:
                temp_mask = mask
            else:
                temp_mask = np.concatenate((temp_mask, mask), axis=1)

        if h == 0:
            artefact_mask = temp_mask
        else:
            artefact_mask = np.concatenate((artefact_mask, temp_mask), axis=0)

    buffer_right = width - (patch_n_w * m_p_s_model_art)
    buffer_bottom = height - (patch_n_h * m_p_s_model_art)

    if buffer_bottom > 0:
        bottom_padding = np.full(
            (buffer_bottom, artefact_mask.shape[1]),
            7,
            dtype=artefact_mask.dtype,
        )
        artefact_mask = np.concatenate((artefact_mask, bottom_padding), axis=0)

    if buffer_right > 0:
        right_padding = np.full(
            (artefact_mask.shape[0], buffer_right), 7, dtype=artefact_mask.dtype
        )
        artefact_mask = np.concatenate((artefact_mask, right_padding), axis=1)

    grandqc_logger.info("Completed processing all patches")

    # Create colored map
    artefact_map = make_class_map(artefact_mask, ARTIFACT_COLORS)
    artefact_map = Image.fromarray(artefact_map)
    artefact_map = artefact_map.resize(
        (int(width * 50 / m_p_s_model_art), int(height * 50 / m_p_s_model_art)),
        Image.Resampling.LANCZOS,
    )
    artefact_map = np.array(artefact_map)

    # Create GeoJSON

    geojson = {"type": "FeatureCollection", "features": []}

    # Initialise artefact_cnts for artefact classes only, probably can be done more efficiently
    artefact_cnts = {}
    scaling_array = np.array([observed_reduction_w, observed_reduction_h])

    total_filtered_artifacts = 0
    min_artifact_area = 10

    for class_value in range(
        1, 7
    ):  # Process classes 1-6, skip unused (0) and background (7)
        class_mask = (artefact_mask == class_value).astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(
            class_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            continue

        holes_idx = []
        if hierarchy is not None:
            holes_idx = np.where(hierarchy[0, :, 3] != -1)[0]

        for i, contour in enumerate(contours):
            if contour.shape[0] < 4:
                continue

            # Filter out small artifacts
            contour_area = cv2.contourArea(contour)
            if class_value >= 2 and contour_area <= min_artifact_area:
                total_filtered_artifacts += 1
                continue

            contour_points = contour.reshape(-1, 2)

            feature = create_geojson_feature(
                contour_points,
                scaling_array,
                ARTIFACT_CLASS_MAPPING.get(class_value, "Unknown"),
                ARTIFACT_COLORS[class_value],
            )
            if feature:
                geojson["features"].append(feature)

            if class_value >= 2 and class_value <= 6 and i not in holes_idx:
                # Scale contour to level 0 coordinates
                scaled_contour = contour[:, 0] * scaling_array
                scaled_contour = np.concatenate(
                    [scaled_contour, scaled_contour[0:1]], 0
                )
                artefact_cnts[f"{class_value}_{i}"] = {
                    "contour": scaled_contour,
                    "holes": [],
                }

        if class_value >= 2 and class_value <= 6:
            for hole_idx in holes_idx:
                if hole_idx < len(contours):
                    parent_idx = (
                        hierarchy[0, hole_idx, 3]
                        if hierarchy is not None
                        else -1
                    )
                    if parent_idx >= 0:
                        parent_key = f"{class_value}_{parent_idx}"
                        if parent_key in artefact_cnts:
                            hole_contour = (
                                contours[hole_idx][:, 0] * scaling_array
                            )
                            artefact_cnts[parent_key]["holes"].append(
                                hole_contour
                            )

    grandqc_logger.info(
        "Filtered %d small artifacts (<= %d pixels)",
        total_filtered_artifacts,
        min_artifact_area,
    )

    del model
    del preprocessing_fn
    del slide

    return artefact_mask, artefact_map, artefact_cnts, geojson


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slide_path",
        type=str,
        help="Path to the slide",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the output",
        required=True,
    )
    parser.add_argument(
        "--mpp_model_art",
        type=float,
        default=1.0,
        help="MPP of the artefact model",
    )
    parser.add_argument(
        "--model_td_path",
        type=str,
        default="./models/tissue_detection/Tissue_Detection_MPP10.pth",
        help="Path to the tissue model",
    )
    parser.add_argument(
        "--model_art_path",
        type=str,
        default="./models/artefact_detection/GrandQC_MPP1.pth",
        help="Path to the artefact model",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=0,
        help="Minimum area for tissue polygons",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the models on",
    )
    args = parser.parse_args()
    args.device = get_device(args.device)[0]

    slide = OpenSlide(args.slide_path)
    artefact_mask, artefact_map, artefact_cnts, geojson = detect_artefacts_wsi(
        slide,
        mpp_model_art=args.mpp_model_art,
        model_art_path=args.model_art_path,
        model_td_path=args.model_td_path,
        min_area=args.min_area,
        device=args.device,
    )

    # Save outputs
    Image.fromarray(artefact_map).save(args.output_path + "_artefact_map.png")
    cv2.imwrite(args.output_path + "_artefact_mask.png", artefact_mask)
    with open(args.output_path + "_artefact_contours.geojson", "w") as f:
        json.dump(geojson, f)
