import json
import uuid
from typing import Any

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from openslide import OpenSlide
from PIL import Image

from classpose.grandqc.wsi_qc_helpers import (
    extract_slide_info,
    get_preprocessing,
    simulate_jpeg_compression,
)
from classpose.log import get_logger
from classpose.utils import download_if_unavailable, get_device

grandqc_logger = get_logger(__name__)

MODEL_URL_PATH = (
    "https://zenodo.org/records/14507273/files/Tissue_Detection_MPP10.pth"
)


def detect_tissue_wsi(
    slide: OpenSlide,
    # tissue detection model parameters
    model_td_path: str = "./models/tissue_detection/Tissue_Detection_MPP10.pth",
    mpp_model_td: int = 10,
    m_p_s_model_td: int = 512,
    encoder_model_name: str = "timm-efficientnet-b0",
    encoder_model_weights: str = "imagenet",
    device: str = "cuda",
    # filtering parameters
    min_area: int = 0,
) -> tuple[Image, np.ndarray, np.ndarray, list[np.ndarray], dict[str, Any],]:
    """
    Detects tissue in a whole-slide image using a U-Net model. This is adapted
    from the GrandQC repository. Please note that if you use any part of Classpose 
    which makes use of GrandQC please follow the instructions at https://github.com/cpath-ukk/grandqc/tree/main 
    to cite them appropriately. Similarly to Classpose, GrandQC is under a non-commercial license 
    whose terms can be found at https://github.com/cpath-ukk/grandqc/blob/main/LICENSE.

    Args:
        slide (OpenSlide): slide to detect tissue in.
        model_td_path (str, optional): path to the tissue detection model.
            Defaults to "./models/tissue_detection/Tissue_Detection_MPP10.pth".
        mpp_model_td (int, optional): MPP of the tissue detection model.
            Defaults to 10.
        m_p_s_model_td (int, optional): microns per square of the tissue
            detection model. Defaults to 512.
        encoder_model_name (str, optional): name of the encoder model. Defaults
            to "timm-efficientnet-b0".
        encoder_model_weights (str, optional): weights of the encoder model.
            Defaults to "imagenet".
        device (str, optional): device to run the model on. Defaults to "cuda".
        min_area (int, optional): minimum area of the tissue polygons. Defaults
            to 0.

    Returns:
        tuple: tuple containing the following elements:
            - image (Image): thumbnail of the slide.
            - mask (np.ndarray): binary mask of the tissue.
            - filled_class_map (np.ndarray): binary mask of the tissue with
                filled polygons.
            - cnts (list[np.ndarray]): list of tissue polygons.
            - geojson (dict[str, Any]): GeoJSON representation of the tissue
                polygons.
    """
    model_td_path = download_if_unavailable(model_td_path, MODEL_URL_PATH)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        encoder_model_name, encoder_model_weights
    )

    model = smp.UnetPlusPlus(
        encoder_name=encoder_model_name,
        encoder_weights=encoder_model_weights,
        classes=2,
        activation=None,
    )

    grandqc_logger.info("Loading model %s", model_td_path)
    model.load_state_dict(torch.load(model_td_path, map_location="cpu"))
    model.to(device)
    model.eval()

    w_l0, h_l0, mpp, thumbnail_dimensions = extract_slide_info(
        slide, mpp_model_td
    )
    grandqc_logger.info(
        "Extracting thumbnail with size %s", thumbnail_dimensions
    )
    image_or = slide.get_thumbnail(thumbnail_dimensions)

    grandqc_logger.info(
        "Simulating jpeg compression to match training preprocessing"
    )

    image = simulate_jpeg_compression(image_or)

    width, height = image.size

    observed_reduction_w = w_l0 / width
    observed_reduction_h = h_l0 / height

    wi_n = width // m_p_s_model_td
    he_n = height // m_p_s_model_td

    overhang_wi = width - wi_n * m_p_s_model_td
    overhang_he = height - he_n * m_p_s_model_td

    grandqc_logger.info(
        "Overhang (< 1 patch) for width and height: %s, %s",
        overhang_wi,
        overhang_he,
    )

    p_s = m_p_s_model_td

    for h in range(he_n + 1):
        for w in range(wi_n + 1):
            if w != wi_n and h != he_n:
                image_work = image.crop(
                    (w * p_s, h * p_s, (w + 1) * p_s, (h + 1) * p_s)
                )
            elif w == wi_n and h != he_n:
                image_work = image.crop(
                    (width - p_s, h * p_s, width, (h + 1) * p_s)
                )
            elif w != wi_n and h == he_n:
                image_work = image.crop(
                    (w * p_s, height - p_s, (w + 1) * p_s, height)
                )
            else:
                image_work = image.crop(
                    (width - p_s, height - p_s, width, height)
                )

            image_pre = get_preprocessing(image_work, preprocessing_fn)
            x_tensor = torch.from_numpy(image_pre).to(device).unsqueeze(0)
            predictions: torch.Tensor = model.predict(x_tensor)
            predictions = predictions.squeeze().cpu().numpy()

            mask: np.ndarray = np.argmax(predictions, axis=0).astype("int8")

            class_mask: np.ndarray = mask

            if w == 0:
                temp_image = mask
                temp_image_class_map = class_mask
            elif w == wi_n:
                mask = mask[:, p_s - overhang_wi : p_s]
                temp_image = np.concatenate((temp_image, mask), axis=1)
                class_mask = class_mask[:, p_s - overhang_wi : p_s]
                temp_image_class_map = np.concatenate(
                    (temp_image_class_map, class_mask), axis=1
                )
            else:
                temp_image = np.concatenate((temp_image, mask), axis=1)
                temp_image_class_map = np.concatenate(
                    (temp_image_class_map, class_mask), axis=1
                )
        if h == 0:
            end_image = temp_image
            end_image_class_map = temp_image_class_map
        elif h == he_n:
            temp_image = temp_image[
                p_s - overhang_he : p_s,
            ]
            end_image = np.concatenate((end_image, temp_image), axis=0)
            temp_image_class_map = temp_image_class_map[
                p_s - overhang_he : p_s, :
            ]
            end_image_class_map = np.concatenate(
                (end_image_class_map, temp_image_class_map), axis=0
            )
        else:
            end_image = np.concatenate((end_image, temp_image), axis=0)
            end_image_class_map = np.concatenate(
                (end_image_class_map, temp_image_class_map), axis=0
            )

    ah, aw = end_image_class_map.shape
    if (ah, aw) != (height, width):
        end_image_class_map = end_image_class_map[
            ah - height : ah, aw - width : aw
        ]

    end_image_class_map = np.uint8(end_image_class_map)
    n_c, cc = cv2.connectedComponents(1 - end_image_class_map)

    filtered_mask = np.zeros_like(end_image_class_map, dtype="uint8")
    sq_size = mpp_model_td**2
    for i in range(1, n_c):
        curr = cc == i
        real_area = sq_size * np.sum(curr)
        if real_area >= min_area:
            filtered_mask[curr] = 1
        else:
            grandqc_logger.debug(
                "Invalid polygon detected: smaller than min area (%s)",
                real_area,
            )

    cnts, hierarchy = cv2.findContours(
        filtered_mask * 255,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if hierarchy is None:
        grandqc_logger.warning("No tissue contours detected in slide.")
        empty_geojson = {"type": "FeatureCollection", "features": []}
        filled_class_map = np.zeros_like(end_image_class_map)
        del model
        del preprocessing_fn
        del slide
        return (image, filtered_mask, filled_class_map, {}, empty_geojson, mpp_model_td)

    holes_idx = np.where(hierarchy[0, :, 3] != -1)[0]
    scaling_array = np.array([observed_reduction_w, observed_reduction_h])
    output_cnts = {}
    filled_class_map = np.zeros_like(end_image_class_map)
    for i, cnt in enumerate(cnts):
        if cnt.shape[0] < 4:
            grandqc_logger.warning(
                "Invalid polygon detected: fewer than 4 points detected (%s)",
                cnt.shape,
            )
            continue
        if i not in holes_idx:
            cv2.drawContours(filled_class_map, [cnt], 0, 255, thickness=10)
            cnt = cnt[:, 0] * scaling_array
            cnt = np.concatenate([cnt, cnt[0:1]], 0)
            output_cnts[i] = {"contour": cnt, "holes": []}
    for cnt_idx in holes_idx:
        parent = hierarchy[0, cnt_idx, 3]
        output_cnts[parent]["holes"].append(cnts[cnt_idx][:, 0] * scaling_array)

    geojson = {
        "type": "FeatureCollection",
        "features": [],
    }
    for idx, cnt in output_cnts.items():
        coords = cnt["contour"].tolist()
        coords.append(coords[0])
        geojson["features"].append(
            {
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
                "properties": {
                    "objectType": "annotation",
                    "isLocked": False,
                    "classification": {
                        "name": "tissue",
                        "color": [0, 0, 0],
                    },
                },
            }
        )

    del model
    del preprocessing_fn
    del slide

    return (
        image,
        filtered_mask,
        filled_class_map,
        output_cnts,
        geojson,
        mpp_model_td,
    )


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
        "--model_path",
        type=str,
        help="Path to the model",
        default="./models/tissue_detection/Tissue_Detection_MPP10.pth",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        help="Minimum area of the polygon",
        default=0,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use",
        default=None,
    )
    args = parser.parse_args()
    args.device = get_device(args.device)[0]

    slide = OpenSlide(args.slide_path)
    image, mask, filled_class_map, _, geojson, mpp_model_td = detect_tissue_wsi(
        slide,
        model_td_path=args.model_path,
        min_area=args.min_area,
        device=args.device,
    )

    image.save(args.output_path + "_image.png")
    cv2.imwrite(args.output_path + "_mask.png", mask * 255)
    cv2.imwrite(args.output_path + "_filled_class_map.png", filled_class_map)
    with open(args.output_path + "_geojson.json", "w") as f:
        json.dump(geojson, f)
