import json
import numpy as np
import scipy
import tifffile
import pandas as pd
from skimage import io
from skimage import measure
from skimage import draw
from pathlib import Path
from tqdm import tqdm


TEST_FRACTION = 0.1
ROOT_DATA_DIR = Path("./data")

CONIC_ORIGINAL = ROOT_DATA_DIR / "original/conic"
CONSEP_ORIGINAL = ROOT_DATA_DIR / "original/consep"
NUCLS_ORIGINAL = ROOT_DATA_DIR / "original/nucls"
MIDOG_ORIGINAL = ROOT_DATA_DIR / "original/midog"
MONUSAC_ORIGINAL = ROOT_DATA_DIR / "original/monusac"
GLYSAC_ORIGINAL = ROOT_DATA_DIR / "original/glysac"
PUMA_ORIGINAL = ROOT_DATA_DIR / "original/puma"

OUT_DIR_CONIC = ROOT_DATA_DIR / "processed/conic"
OUT_DIR_CONSEP = ROOT_DATA_DIR / "processed/consep"
OUT_DIR_NUCLS = ROOT_DATA_DIR / "processed/nucls"
OUT_DIR_MIDOG = ROOT_DATA_DIR / "processed/midog"
OUT_DIR_MONUSAC = ROOT_DATA_DIR / "processed/monusac"
OUT_DIR_GLYSAC = ROOT_DATA_DIR / "processed/glysac"
OUT_DIR_PUMA = ROOT_DATA_DIR / "processed/puma"

DO_CONIC = False
DO_CONSEP = False
DO_NUCLS = True
DO_MIDOG = False
DO_MONUSAC = False
DO_GLYSAC = False
DO_PUMA = False

CONIC_CONVERSION = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
}

CONSEP_CONVERSION = {
    0: 0,  # background
    1: 1,  # other -> other
    2: 2,  # inflammatory -> inflammatory
    3: 3,  # healthy epithelial -> healthy epithelial
    4: 4,  # dysplastic/malignant epithelial -> malignant epithelial
    5: 5,  # fibroblast -> stroma
    6: 6,  # muscle -> muscle
    7: 5,  # endothelial -> stroma
}

NUCLS_CONVERSION = {
    0: 0,  # background -> bg
    1: 1,  # tumor -> tumor
    2: 2,  # fibroblast -> stroma
    3: 3,  # lymphocyte -> TIL
    4: 4,  # plasma cell -> TIPC
    5: 5,  # macrophage -> TIM
    6: 1,  # mitotic figure -> tumor
    7: 2,  # vascular endothelium -> stroma
    8: 6,  # myoepithelium -> other
    9: 6,  # apoptotic body -> other
    10: 6,  # neutrophil -> other
    11: 6,  # ductal epithelium -> other
    12: 6,  # eosinophil -> other
    99: 0,  # unlabelled -> bg
    253: 0,  # fov -> bg
}

MONUSAC_CONVERSION = {
    0: 0,  # background,
    1: 1,  # epithelial
    2: 2,  # lymphocyte
    3: 3,  # macrophage
    4: 4,  # neutrophil
}

GLYSAC_CONVERSION = {
    0: 0,
    1: 1,  # other
    2: 1,  # other
    3: 3,  # epithelial
    4: 2,  # lymphocyte
    5: 2,  # lymphocyte
    6: 2,  # lymphocyte
    7: 2,  # lymphocyte
    8: 3,  # epithelial
    9: 1,  # other
    10: 1,  # other
}

PUMA_CONVERSION = {
    "nuclei_apoptosis": 1,  # apoptosis
    "nuclei_tumor": 2,  # tumor
    "nuclei_endothelium": 3,  # endothelium
    "nuclei_stroma": 4,  # stroma
    "nuclei_lymphocyte": 5,  # lymphocyte
    "nuclei_histiocyte": 6,  # histiocyte
    "nuclei_epithelium": 7,  # epithelium
    "nuclei_melanophage": 8,  # melanophage -> other
    "nuclei_plasma_cell": 9,  # plasma cell  -> other
    "nuclei_neutrophil": 9,  # neutrophil  -> other
}


def split_dataset(
    counts_df: pd.DataFrame,
    conversion: dict[int | str, int],
    n_small_classes: int = 1,
    n_attempts: int = 250,
    test_fraction: float = TEST_FRACTION,
):
    """
    Split the dataset into train and test sets using a greedy approach.

    Args:
        counts_df (pd.DataFrame): DataFrame containing the counts of each
            class.
        conversion (dict[int | str, int]): Conversion dictionary from original
            class labels to new class labels.
        n_small_classes (int, optional): Number of small classes to include in
            the test set. Defaults to 1.
        n_attempts (int, optional): Number of attempts to make. Defaults to
            250.
        test_fraction (float, optional): Fraction of the dataset to include in
            the test set. Defaults to TEST_FRACTION.

    Returns:
        tuple: A tuple containing the train and test indices.
    """
    all_idx = [x for x in counts_df.index]
    np.random.seed(42)
    best_mae = np.inf
    for _ in range(n_attempts):
        np.random.shuffle(all_idx)
        accumulators = {
            "train": np.zeros(max(conversion.values())),
            "test": np.zeros(max(conversion.values())),
        }
        train_idxs = []
        test_idxs = []
        for idx in all_idx:
            proportions = np.where(
                accumulators["train"] > 0,
                accumulators["test"] / accumulators["train"],
                1.0,
            )
            if np.sum(proportions < test_fraction) > n_small_classes:
                test_idxs.append(idx)
                accumulators["test"] += np.array(counts_df.loc[idx])
            else:
                train_idxs.append(idx)
                accumulators["train"] += np.array(counts_df.loc[idx])

        curr_mae = np.mean(
            np.square(
                accumulators["test"]
                / (accumulators["train"] + accumulators["test"])
                - test_fraction
            )
        )
        if curr_mae < best_mae:
            best_mae = curr_mae
            best_train_idxs = train_idxs
            best_test_idxs = test_idxs
            best_accumulators = accumulators
    print("Train/test proportions:")
    print(
        best_accumulators["test"]
        / (best_accumulators["train"] + best_accumulators["test"])
    )
    return best_train_idxs, best_test_idxs


def tile_image(image: np.ndarray, tile_size: int) -> list[np.ndarray]:
    """Tile an image into overlapping tiles.

    Args:
        image: (array, shape (M, N, [...])): input image.
        tile_size (int): size of the tiles.

    Returns:
        tiles (array, shape (M', N', [..., C])): tiled image.
    """
    M, N = image.shape[:2]
    tiles = []
    n = 0
    for i in range(0, M + 1, tile_size):
        for j in range(0, N + 1, tile_size):
            a, b = i, i + tile_size
            c, d = j, j + tile_size
            if b >= M:
                a, b = M - tile_size, M
            if d >= N:
                c, d = N - tile_size, N
            tiles.append(image[a:b, c:d])
            n += 1
    return tiles


def pad_image_to_size(image: np.ndarray, tile_size: int) -> np.ndarray:
    """
    Pads an image to the nearest multiple of tile_size.

    Args:
        image: (array, shape (M, N, [...])): input image.
        tile_size (int): size of the tiles.

    Returns:
        padded_image (array, shape (M', N', [..., C])): padded image.
    """
    M, N = image.shape[:2]
    pad_height = (tile_size - M) % tile_size
    pad_width = (tile_size - N) % tile_size
    return np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), "constant")


def get_n_tiles(M: int, N: int, tile_size: int) -> int:
    """
    Returns the number of tiles that can be extracted from an image of size (M, N).
    """
    return (M + 1) // tile_size * (N + 1) // tile_size


def tile_dataset(
    images: list[np.ndarray], labels: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tiles an image and its labels into overlapping tiles.

    Args:
        images (list[np.ndarray]): list of images to append to.
        labels (list[np.ndarray]): list of labels to append to.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the tiled images and labels.
    """
    new_images = []
    new_labels = []
    for image, label in zip(images, labels):
        new_images.extend(tile_image(image, 256))
        new_labels.extend(tile_image(label, 256))
    return np.array(new_images), np.array(new_labels)


def save_arrays(
    images: list[np.ndarray],
    labels: list[np.ndarray],
    out_path_root: Path,
    is_object: bool = False,
):
    if is_object:
        images = np.array(images, dtype=np.object_)
        labels = np.array(labels, dtype=np.object_)
    out_path_root.mkdir(exist_ok=True, parents=True)
    np.save(out_path_root / "images.npy", images)
    np.save(out_path_root / "labels.npy", labels)


if DO_CONIC:
    print("conic", CONIC_ORIGINAL)
    images = np.load(CONIC_ORIGINAL / "images.npy")
    labels = np.load(CONIC_ORIGINAL / "labels.npy")
    patch_info = pd.read_csv(CONIC_ORIGINAL / "patch_info.csv")
    patch_info["slide_ids"] = patch_info["patch_info"].str.replace(
        "-[0-9]+$", "", regex=True
    )
    assert patch_info.shape[0] == len(images) == len(labels)
    M = max(CONIC_CONVERSION.values())
    all_counts = {}
    for i, label in enumerate(labels):
        original_image = patch_info["slide_ids"][i]
        if original_image not in all_counts:
            all_counts[original_image] = {i: 0 for i in range(1, M + 1)}
        for c in range(1, M + 1):
            all_counts[original_image][c] += len(
                np.unique(label[:, :, 0][label[:, :, 1] == c])
            )
    counts_df = pd.DataFrame(all_counts).T
    train_slides, test_slides = split_dataset(counts_df, CONIC_CONVERSION)
    train_idxs = patch_info["slide_ids"].str.contains(
        "$|^".join(train_slides), regex=True
    )
    test_idxs = patch_info["slide_ids"].str.contains(
        "$|^".join(test_slides), regex=True
    )

    print(
        "Data leakage:",
        np.any(np.isin(np.where(train_idxs)[0], np.where(test_idxs)[0])),
    )
    print("All patches:", train_idxs.sum() + test_idxs.sum() == images.shape[0])
    images = {
        "train": {"images": images[train_idxs], "labels": labels[train_idxs]},
        "test": {"images": images[test_idxs], "labels": labels[test_idxs]},
    }
    for tt in images:
        for n in images[tt]:
            np.save(OUT_DIR_CONIC / tt / f"{n}.npy", images[tt][n])

# consep
if DO_CONSEP:
    print("consep", CONSEP_ORIGINAL)
    for path_root in ["Train", "Test"]:
        image_dict = {
            p.name.replace(".png", ""): p
            for p in Path(f"{CONSEP_ORIGINAL}/{path_root}/Images").glob("*png")
        }
        label_dict = {
            p.name.replace(".mat", ""): p
            for p in Path(f"{CONSEP_ORIGINAL}/{path_root}/Labels").glob("*mat")
        }

        all_images = []
        all_labels = []
        for k, v in image_dict.items():
            if k not in label_dict:
                print(f"Missing label for {k}")
            else:
                img = io.imread(v)
                label = scipy.io.loadmat(label_dict[k])
                image = img[:, :, :3]
                inst_map = label["inst_map"]
                type_map = label["type_map"]
                type_map = np.vectorize(CONSEP_CONVERSION.get)(type_map)
                all_images.append(image)
                all_labels.append(np.stack([inst_map, type_map], -1))

        out_path_root = OUT_DIR_CONSEP / path_root
        out_path_root.mkdir(exist_ok=True, parents=True)
        if path_root == "Train":
            save_arrays(
                all_images,
                all_labels,
                OUT_DIR_CONSEP / "train_multi_shape",
                True,
            )
            all_images, all_labels = tile_dataset(all_images, all_labels)
            save_arrays(all_images, all_labels, out_path_root)
        else:
            save_arrays(all_images, all_labels, out_path_root, True)


# nucls
if DO_NUCLS:
    print("nucls", NUCLS_ORIGINAL)
    image_dict = {
        p.name.replace(".png", ""): p
        for p in Path(f"{NUCLS_ORIGINAL}/images").glob("*png")
    }
    label_dict = {
        p.name.replace(".png", ""): p
        for p in Path(f"{NUCLS_ORIGINAL}/labels").glob("*png")
    }

    all_slides = sorted(list(set([k.split("_")[0] for k in image_dict.keys()])))

    all_counts = {}
    for k in tqdm(label_dict, desc="Counting labels"):
        label = io.imread(label_dict[k])
        slide = k.split("_")[0]
        type_map = label[:, :, 0]
        inst_map = label[:, :, 2]
        inst_map[inst_map < 3] = 0
        type_map[inst_map < 3] = 0
        type_map = np.vectorize(NUCLS_CONVERSION.get)(type_map)
        us = np.unique(type_map)
        us = us[us > 0]
        if slide not in all_counts:
            all_counts[slide] = {
                i: 0 for i in range(1, max(NUCLS_CONVERSION.values()) + 1)
            }
        for u in us:
            all_counts[slide][u] += len(np.unique(inst_map[type_map == u]))
    all_counts_df = pd.DataFrame(all_counts).T

    train_slides, test_slides = split_dataset(all_counts_df, NUCLS_CONVERSION)

    all_images = {"train": [], "test": []}
    all_labels = {"train": [], "test": []}
    for k, v in tqdm(image_dict.items()):
        if k not in label_dict:
            print(f"Missing label for {k}")
        else:
            img = io.imread(v)
            label = io.imread(label_dict[k])
            im_shape = np.array(img.shape[:2])
            la_shape = np.array(label.shape[:2])
            if not np.array_equal(im_shape, la_shape):
                x_min, y_min = min(im_shape[0], la_shape[0]), min(
                    im_shape[1], la_shape[1]
                )
                d = np.abs(im_shape - la_shape)
                if d.max() > 1:
                    print(f"Image {k} has different shape: {d}")
                img = img[:x_min, :y_min]
                label = label[:x_min, :y_min]
            type_map = label[:, :, 0]
            inst_map = label[:, :, 2]
            type_map[inst_map < 3] = 0
            inst_map[inst_map < 3] = 0
            type_map = np.vectorize(NUCLS_CONVERSION.get)(type_map)
            image = img[:, :, :3]
            slide = k.split("_")[0]
            if slide in train_slides:
                k = "train"
            else:
                k = "test"
            all_images[k].append(image)
            all_labels[k].append(np.stack([inst_map, type_map], -1))

    for k in ["train", "test"]:
        if k == "train":
            save_arrays(
                all_images[k],
                all_labels[k],
                OUT_DIR_NUCLS / "train_multi_shape",
                True,
            )
            all_images[k], all_labels[k] = tile_dataset(
                all_images[k], all_labels[k]
            )
            save_arrays(all_images[k], all_labels[k], OUT_DIR_NUCLS / "train")
        else:
            save_arrays(
                all_images[k], all_labels[k], OUT_DIR_NUCLS / "test", True
            )

    with open(OUT_DIR_NUCLS / "train_slides.txt", "w") as f:
        for slide in train_slides:
            f.write(slide + "\n")

    with open(OUT_DIR_NUCLS / "test_slides.txt", "w") as f:
        for slide in test_slides:
            f.write(slide + "\n")

# midogpp
if DO_MIDOG:
    print("midog", MIDOG_ORIGINAL)
    image_ids = [
        p.name.replace(".tiff", "")
        for p in Path(f"{MIDOG_ORIGINAL}/images").glob("*tiff")
    ]
    np.random.seed(42)
    np.random.shuffle(image_ids)

    split_image_ids = {
        "train": image_ids[: int(len(image_ids) * (1 - TEST_FRACTION))],
        "test": image_ids[int(len(image_ids) * (1 - TEST_FRACTION)) :],
    }

    for split in ["train", "test"]:
        prev_idx = 0
        out_path_root = OUT_DIR_MIDOG / split
        out_path_root.mkdir(exist_ok=True, parents=True)

        n_tiles = 0
        for image_id in tqdm(split_image_ids[split], desc=f"{split}"):
            label = f"{MIDOG_ORIGINAL}/annotations/{image_id}.tiff"
            tiles = tile_image(tifffile.imread(label), 256)
            if split == "train":
                tiles = [t for t in tiles if np.sum(t > 0) > 0]
            n_tiles += len(tiles)

        image_file = np.memmap(
            out_path_root / "images.npy",
            dtype=np.uint8,
            mode="w+",
            shape=(n_tiles, 256, 256, 3),
        )
        label_file = np.memmap(
            out_path_root / "labels.npy",
            dtype=np.uint16,
            mode="w+",
            shape=(n_tiles, 256, 256, 2),
        )
        for image_id in tqdm(split_image_ids[split]):
            image = f"{MIDOG_ORIGINAL}/images/{image_id}.tiff"
            label = f"{MIDOG_ORIGINAL}/annotations/{image_id}.tiff"
            img = tile_image(io.imread(image)[..., :3], 256)
            lab = tile_image(io.imread(label), 256)
            filtered_img, filtered_lab = [], []
            for i in range(len(img)):
                if (split == "train") and (np.sum(lab[i] > 0) > 0):
                    filtered_img.append(img[i])
                    filtered_lab.append(lab[i])
                elif split == "test":
                    filtered_img.append(img[i])
                    filtered_lab.append(lab[i])
            img = np.stack(filtered_img)
            lab = np.stack(filtered_lab)
            image_file[prev_idx : prev_idx + len(img)] = img
            label_file[prev_idx : prev_idx + len(lab)] = lab
            prev_idx += len(img)
        image_file.flush()
        label_file.flush()

if DO_MONUSAC:
    print("monusac", MONUSAC_ORIGINAL)
    for tt in ["train", "test"]:
        input_folders = MONUSAC_ORIGINAL / tt / "images"
        mask_folders = MONUSAC_ORIGINAL / tt / "masks"
        cell_classes = [
            "Epithelial",
            "Lymphocyte",
            "Macrophage",
            "Neutrophil",
            "Ambiguous",
        ]
        all_images = []
        all_labels = []
        all_ambiguous = []
        for i, input_tiff in tqdm(
            enumerate(sorted(input_folders.rglob("*tif")))
        ):
            name = input_tiff.name.replace(".tif", "")
            slide_name = input_tiff.parent.name
            image = io.imread(input_tiff)

            mask_path = mask_folders / f"{name}_masks.tif"
            classes_path = mask_folders / f"{name}_classes.tif"
            ambiguous_path = mask_folders / f"{name}_masks_bad.tif"

            instance_mask = io.imread(mask_path)
            classes_mask = io.imread(classes_path)
            if ambiguous_path.exists():
                ambiguous_mask = io.imread(ambiguous_path)
            else:
                ambiguous_mask = np.zeros(
                    instance_mask.shape, dtype=instance_mask.dtype
                )
            ambiguous_instances = np.unique(ambiguous_mask * instance_mask)
            for i in ambiguous_instances:
                classes_mask[instance_mask == i] = 0
            classes_mask = np.vectorize(MONUSAC_CONVERSION.get)(classes_mask)

            mask = np.stack([instance_mask, classes_mask], -1)
            image = image[..., :3]
            if tt == "train":
                # keep images as they are for testing, pad/tile as necessary for training
                if image.shape[0] < 256 or image.shape[1] < 256:
                    image = pad_image_to_size(image, 256)
                    mask = pad_image_to_size(mask, 256)

            all_images.append(image)
            all_labels.append(mask)
            all_ambiguous.append(ambiguous_mask)

        if tt == "train":
            save_arrays(
                all_images,
                all_labels,
                OUT_DIR_MONUSAC / "train_multi_shape",
                True,
            )
            all_images, all_labels = tile_dataset(all_images, all_labels)
            save_arrays(all_images, all_labels, OUT_DIR_MONUSAC / "train")
        elif tt == "test":
            for i in range(len(all_ambiguous)):
                mask = all_labels[i]
                amb = all_ambiguous[i]
                if amb is None:
                    continue
                u = np.unique(mask[..., 0][amb > 0])
                for j in u:
                    mask[mask[..., 0] == j, 1] = 0
                all_labels[i] = mask
            save_arrays(all_images, all_labels, OUT_DIR_MONUSAC / "test", True)

if DO_GLYSAC:
    print("glysac", GLYSAC_ORIGINAL)
    for tt in ["Train", "Test"]:
        all_images = []
        all_labels = []
        for image_path in tqdm((GLYSAC_ORIGINAL / tt / "Images").glob("*.png")):
            name = image_path.name.replace(".png", "")
            mask_path = GLYSAC_ORIGINAL / tt / "Labels" / f"{name}.mat"
            image = io.imread(image_path)
            mask = scipy.io.loadmat(mask_path)
            inst_map = mask["inst_map"]
            type_map = mask["type_map"]
            type_map = np.vectorize(GLYSAC_CONVERSION.get)(type_map)
            all_images.append(image)
            all_labels.append(np.stack([inst_map, type_map], -1))

        if tt == "Train":
            save_arrays(
                all_images,
                all_labels,
                OUT_DIR_GLYSAC / "train_multi_shape",
                True,
            )
            all_images, all_labels = tile_dataset(all_images, all_labels)
            save_arrays(all_images, all_labels, OUT_DIR_GLYSAC / "train")
        elif tt == "Test":
            save_arrays(all_images, all_labels, OUT_DIR_GLYSAC / "test", True)


if DO_PUMA:
    print("puma", PUMA_ORIGINAL)
    all_images = {
        "_".join(p.name.split(".")[0].split("_")[:5]): p
        for p in PUMA_ORIGINAL.rglob("*.tif")
    }
    all_labels = {
        "_".join(p.name.split(".")[0].split("_")[:5]): p
        for p in PUMA_ORIGINAL.rglob("*.geojson")
    }

    all_keys = list(sorted(all_images.keys()))
    np.random.seed(42)
    np.random.shuffle(all_keys)

    image_arrays = {}
    label_arrays = {}
    label_counts = {}
    for identifier in tqdm(all_keys):
        image = io.imread(all_images[identifier])
        with open(all_labels[identifier]) as f:
            label = json.load(f)
        mask = np.zeros([*image.shape[:-1], 2])
        i = 0
        label_counts[identifier] = {k: 0 for k in PUMA_CONVERSION.values()}
        for feature in label["features"]:
            i += 1
            geometry = feature["geometry"]["coordinates"]
            classification = feature["properties"]["classification"]["name"]
            idx_class = PUMA_CONVERSION[classification]
            label_counts[identifier][idx_class] += 1
            if feature["geometry"]["type"] == "Polygon":
                for g in geometry:
                    g = np.array(g) - 1
                    mask[..., 0][draw.polygon(g[:, 1], g[:, 0])] = i
                    mask[..., 1][draw.polygon(g[:, 1], g[:, 0])] = idx_class
            elif feature["geometry"]["type"] == "MultiPolygon":
                for g in geometry:
                    for h in g:
                        h = np.array(h) - 1
                        mask[..., 0][draw.polygon(h[:, 1], h[:, 0])] = i
                        mask[..., 1][draw.polygon(h[:, 1], h[:, 0])] = idx_class

        image_arrays[identifier] = image
        label_arrays[identifier] = mask

    counts_df = pd.DataFrame(label_counts).T

    train_idxs, test_idxs = split_dataset(counts_df, PUMA_CONVERSION)

    train_arrays, train_labels = [], []
    test_arrays, test_labels = [], []
    for k in train_idxs:
        train_arrays.append(image_arrays[k][..., :3])
        train_labels.append(label_arrays[k])
    for k in test_idxs:
        test_arrays.append(image_arrays[k][..., :3])
        test_labels.append(label_arrays[k])

    train_arrays_tile, train_labels_tile = tile_dataset(
        train_arrays, train_labels
    )
    test_arrays = np.stack(test_arrays)
    test_labels = np.stack(test_labels).astype(np.uint16)

    save_arrays(
        train_arrays, train_labels, OUT_DIR_PUMA / "train_multi_shape", True
    )
    save_arrays(train_arrays_tile, train_labels_tile, OUT_DIR_PUMA / "train")
    save_arrays(test_arrays, test_labels, OUT_DIR_PUMA / "test", True)
