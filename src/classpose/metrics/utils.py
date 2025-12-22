import glob
import os
import pathlib
import shutil

import fastremap
import numpy as np
from tqdm import trange


def remap_label(arr):
    """Renumbers labels to be contiguous. Just a wrapper around
    fastremap.renumber.

    Args:
        arr (ndarray): the 2d array contain instances where each instances is marked
            by non-zero integer.

    Returns:
        arr (ndarray): Array with continguous ordering of instances.

    """
    return fastremap.renumber(arr.astype(np.int64))[0]


def cropping_center(x, crop_shape, batch=False):
    """Crop an array at the centre with specified dimensions."""
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x


def rmdir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    return


def recur_find_ext(root_dir, ext_list):
    """Recursively find all files in directories end with the `ext` such as `ext='.png'`.

    Args:
        root_dir (str): Root directory to grab filepaths from.
        ext_list (list): File extensions to consider.

    Returns:
        file_path_list (list): sorted list of filepaths.
    """
    file_path_list = []
    for cur_path, dir_list, file_list in os.walk(root_dir):
        for file_name in file_list:
            file_ext = pathlib.Path(file_name).suffix
            if file_ext in ext_list:
                full_path = os.path.join(cur_path, file_name)
                file_path_list.append(full_path)
    file_path_list.sort()
    return file_path_list


def rm_n_mkdir(dir_path):
    """Remove and then make a new directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def get_bounding_box(img):
    """Get the bounding box coordinates of a binary input- assumes a single object.

    Args:
        img: input binary image.

    Returns:
        bounding box coordinates

    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def load_masks(path: str) -> np.ndarray:
    """
    Load masks from a directory or file in npy or npz format.

    Args:
        path (str): Path to the directory or file.
    Returns:
        np.ndarray: Loaded masks.
    Raises:
        ValueError: If no .npy or .npz files are found in the directory.
    """
    if os.path.isdir(path):
        # If a directory, load all .npy or .npz files
        mask_files = sorted(glob.glob(os.path.join(path, "*.np[yz]")))
        if not mask_files:
            raise ValueError(f"No .npy or .npz files found in {path}")

        # Load the first file to determine format
        first_mask = np.load(mask_files[0], allow_pickle=True)
        if isinstance(first_mask, np.ndarray):
            # Single array per file format
            return [np.load(f, allow_pickle=True) for f in mask_files]
        else:
            # npz format with multiple arrays
            return [np.load(f, allow_pickle=True)["arr_0"] for f in mask_files]
    else:
        # If a single file
        if path.endswith(".npy"):
            return np.load(path, allow_pickle=True)
        elif path.endswith(".npz"):
            return np.load(path, allow_pickle=True)["arr_0"]
        else:
            raise ValueError(f"Unsupported file format: {path}")


def check_and_coherce_if_necessary(masks, expected_shape_length):
    """
    Check if masks have the expected shape and coherce if necessary.

    Args:
        masks (np.ndarray | list[np.ndarray]): Masks to check.
        expected_shape_length (int): Expected number of dimensions for an individual
            mask.

    Returns:
        np.ndarray: Checked and coherced masks.
    Raises:
        ValueError: If masks does not have the expected shape or a shape cohercible
            to that shape.
    """
    if isinstance(masks, np.ndarray) and masks.dtype == "object":
        return list(masks)

    if isinstance(masks, list):
        return masks

    if len(masks.shape) == expected_shape_length:
        masks = masks[None]
    elif len(masks.shape) != (expected_shape_length + 1):
        raise ValueError(
            f"Masks have {len(masks.shape)} dimensions, expected {expected_shape_length}"
        )
    return masks


def filter_out_unlabelled_cells(
    gt_masks: np.ndarray | list[np.ndarray],
    pred_masks: np.ndarray | list[np.ndarray],
    min_iou: float = 0.5,
) -> tuple[np.ndarray | list[np.ndarray], np.ndarray | list[np.ndarray]]:
    """
    Filter out unlabelled cells from the ground truth and predicted masks.

    Args:
        gt_masks (np.ndarray | list[np.ndarray]): Ground truth masks.
        pred_masks (np.ndarray | list[np.ndarray]): Predicted masks.
        min_iou (float, optional): IoU threshold for matching. Defaults to 0.5.

    Returns:
        tuple[np.ndarray | list[np.ndarray], np.ndarray | list[np.ndarray]]: Filtered ground truth and predicted masks.
    """
    for i in trange(len(gt_masks), desc="Filtering out unlabelled cells"):
        gt_mask, pred_mask = gt_masks[i], pred_masks[i]
        gt_instances = remap_label(gt_mask[..., 0])
        pred_instances = remap_label(pred_mask[..., 0])

        # Check if there are any instances to process
        gt_max = gt_instances.max()
        pred_max = pred_instances.max()

        # Skip processing if there are no instances in either mask
        if gt_max <= 0 or pred_max <= 0:
            gt_masks[i] = gt_mask
            pred_masks[i] = pred_mask
            continue

        # Check which GT cells have labels (class > 0)
        gt_has_label = np.unique(gt_instances * (gt_mask[..., 1] > 0))
        gt_has_label = set(gt_has_label[gt_has_label > 0])

        # Skip processing if all instances are labeled
        if len(gt_has_label) == gt_max:
            gt_masks[i] = gt_mask
            pred_masks[i] = pred_mask
            continue

        remove_gt = []
        remove_pred = []

        # Only process unlabelled GT cells
        for gt_id in range(1, gt_max + 1):
            if gt_id in gt_has_label:
                continue  # Skip labeled cells

            gt_mask_lab = gt_instances == gt_id
            rmin1, rmax1, cmin1, cmax1 = get_bounding_box(gt_mask_lab)
            gt_mask_crop = gt_mask_lab[rmin1:rmax1, cmin1:cmax1]
            pred_crop = pred_instances[rmin1:rmax1, cmin1:cmax1]

            pred_overlap = pred_crop[gt_mask_crop > 0]
            pred_overlap_ids = np.unique(pred_overlap)
            pred_overlap_ids = pred_overlap_ids[pred_overlap_ids > 0]

            for pred_id in pred_overlap_ids:
                pred_mask_lab = pred_instances == pred_id

                rmin2, rmax2, cmin2, cmax2 = get_bounding_box(pred_mask_lab)
                r_min = min(rmin1, rmin2)
                r_max = max(rmax1, rmax2)
                c_min = min(cmin1, cmin2)
                c_max = max(cmax1, cmax2)

                gt_crop = gt_mask_lab[r_min:r_max, c_min:c_max]
                pred_crop = pred_mask_lab[r_min:r_max, c_min:c_max]

                total = gt_crop.sum() + pred_crop.sum()
                inter = (gt_crop * pred_crop).sum()
                iou = inter / (total - inter) if total > inter else 0

                if iou > min_iou:
                    remove_gt.append(gt_id)
                    remove_pred.append(pred_id)

        remove_gt = np.unique(remove_gt)
        remove_pred = np.unique(remove_pred)

        gt_mask[np.isin(gt_instances, remove_gt)] = 0
        pred_mask[np.isin(pred_instances, remove_pred)] = 0

        gt_mask[..., 0] = remap_label(gt_mask[..., 0])
        pred_mask[..., 0] = remap_label(pred_mask[..., 0])

        gt_masks[i] = gt_mask
        pred_masks[i] = pred_mask

    return gt_masks, pred_masks
