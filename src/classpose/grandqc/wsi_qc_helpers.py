import numpy as np
import torch
from PIL import Image


def simulate_jpeg_compression(image: Image.Image) -> Image.Image:
    """
    Simulates JPEG compression on a PIL image to match training preprocessing.

    Args:
        image (Image.Image): Input image.

    Returns:
        Image.Image: Processed image after JPEG simulation.
    """
    import cv2

    image_np = np.array(image)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    _, image_encoded = cv2.imencode(".jpg", image_np, encode_param)
    image_decoded = cv2.imdecode(image_encoded, 1)
    return Image.fromarray(image_decoded)


def extract_slide_info(
    slide, mpp_model: float
) -> tuple[int, int, float, tuple[int, int]]:
    """
    Extracts slide dimensions, MPP, and thumbnail dimensions.

    Args:
        slide: OpenSlide object.
        mpp_model (float): Model MPP.

    Returns:
        tuple: (w_l0, h_l0, mpp, thumbnail_dimensions)
    """
    w_l0, h_l0 = slide.level_dimensions[0]
    mpp = round(float(slide.properties["openslide.mpp-x"]), 4)
    reduction_factor = mpp_model / mpp
    thumbnail_dimensions = (
        int(w_l0 // reduction_factor),
        int(h_l0 // reduction_factor),
    )
    return w_l0, h_l0, mpp, thumbnail_dimensions


def create_geojson_feature(
    contour_points: np.ndarray,
    scaling_factors: np.ndarray,
    classification_name: str,
    classification_color: list[int],
) -> dict:
    """
    Creates a GeoJSON feature for a polygon.

    Args:
        contour_points (np.ndarray): Contour points.
        scaling_factors (np.ndarray): Scaling factors for coordinates.
        classification_name (str): Name for classification.
        classification_color (list[int]): Color for classification.

    Returns:
        dict: GeoJSON feature.
    """
    import uuid

    scaled_points = contour_points * scaling_factors
    if len(scaled_points) < 4:
        return None
    polygon_points = scaled_points.tolist()
    if not np.array_equal(polygon_points[0], polygon_points[-1]):
        polygon_points.append(polygon_points[0])
    feature = {
        "type": "Feature",
        "id": str(uuid.uuid4()),
        "geometry": {"type": "Polygon", "coordinates": [polygon_points]},
        "properties": {
            "objectType": "annotation",
            "isLocked": False,
            "classification": {
                "name": classification_name,
                "color": classification_color,
            },
        },
    }
    return feature


def to_tensor_x(x: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Transpose the image to channel first and convert to float32.

    Args:
        x (torch.Tensor): Image to be transposed.

    Returns:
        torch.Tensor: Transposed image.
    """
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(
    image: Image.Image, preprocessing_fn: callable
) -> torch.Tensor:
    """
    Convert image to numpy array and apply preprocessing function.

    Args:
        image (Image.Image): Image to be preprocessed.
        preprocessing_fn (callable): Preprocessing function.

    Returns:
        torch.Tensor: Preprocessed image.
    """
    image = np.array(image)
    x = preprocessing_fn(image)
    x = to_tensor_x(x)
    return x


def make_class_map(
    mask: np.ndarray, class_colors: list[list[int]]
) -> np.ndarray:
    """
    Create a color map from the mask.

    Args:
        mask (np.ndarray): Mask to be converted.
        class_colors (list[list[int]]): List of class colors.

    Returns:
        np.ndarray: Color map.
    """
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for class_color in range(0, len(class_colors)):
        idx = mask == class_color
        r[idx] = class_colors[class_color][0]
        g[idx] = class_colors[class_color][1]
        b[idx] = class_colors[class_color][2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb
