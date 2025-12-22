"""
Adaptation of the code in [1].

[1] https://github.com/MouseLand/cellpose/blob/main/paper/cpsam/semantic.py
"""

import os
import numpy as np
from scipy import ndimage
import torch
from tqdm import trange
from torch import nn
from cellpose import dynamics, vit_sam, models, train
from skimage.transform import resize
import pathlib

import logging

curr_path = str(pathlib.Path(__file__).parent.absolute())

DEVICE = torch.device("cuda:0")

logging.basicConfig(level=logging.INFO)

cl_epithelial = np.array([255, 0, 0])
cl_lymphocyte = np.array([255, 255, 0])
cl_macrophage = np.array([0, 255, 0])
cl_neutrophil = np.array([0, 0, 255])
cl_colors = [cl_epithelial, cl_lymphocyte, cl_macrophage, cl_neutrophil]
cl_names = ["epithelial", "lymphocyte", "macrophage", "neutrophil"]
cl_colors = np.array(cl_colors)


def get_rescale_ratio(training_to_inference_mpp: str):
    ratio = 1.0
    if ":" in training_to_inference_mpp:
        training_mpp, inference_mpp = training_to_inference_mpp.split(":")
        training_mpp = float(training_mpp)
        inference_mpp = float(inference_mpp)
        if training_mpp != inference_mpp:
            ratio = inference_mpp / training_mpp
    else:
        ratio = float(training_to_inference_mpp)
    return ratio


def rescale_if_necessary(image: np.ndarray, training_to_inference_mpp: str):
    """
    Rescale images if necessary.

    Args:
        image (np.ndarray): image to be rescaled.
        training_to_inference_mpp (str): multiplier to rescale images from
            training resolution to inference resolution. For example, if the
            training images are 0.5 microns per pixel and the inference images
            are 1 micron per pixel, then the value should be "0.5". If the
            training images are 0.5 microns per pixel and the inference images
            are 2 microns per pixel, then the value should be "0.5:2".

    Returns:
        np.ndarray: Rescaled images.
    """
    if training_to_inference_mpp is not None:
        ratio = get_rescale_ratio(training_to_inference_mpp)
        if ratio != 1.0:
            sh = image.shape
            new_sh = (int(sh[0] * ratio), int(sh[1] * ratio))
            image = resize(image.astype(np.float64), new_sh)
    return image


def rescale_label_if_necessary(label: np.ndarray, new_sh: tuple[int, int]):
    """
    Rescale labels if necessary.

    Args:
        label (np.ndarray): label to be rescaled.
        new_sh (tuple[int, int]): new shape of the label.

    Returns:
        np.ndarray: Rescaled images.
    """
    sh = label.shape
    if sh != new_sh:
        label = resize(label, new_sh, order=0)
    return label


def rgb_to_masks_classes(rgb):
    masks0 = np.zeros(rgb.shape[:2], "uint16")
    class0 = []
    class0 = np.zeros(rgb.shape[:2], "int")
    j0 = 0
    for ic in range(4):
        c0 = (rgb == cl_colors[ic][np.newaxis, np.newaxis, :]).sum(axis=-1) == 3
        c0 = ndimage.label(c0)[0].astype("uint16")
        class0[c0 > 0] = ic + 1
        masks0[c0 > 0] = c0[c0 > 0] + j0
        j0 += c0.max()
    return masks0, class0


def initialize_class_net(nclasses=5):
    net = vit_sam.Transformer(rdrop=0.4).to(DEVICE)
    # default model
    net.load_model(f"{curr_path}/models/cpsam", device=DEVICE, strict=False)

    # initialize weights for class maps
    ps = 8  # patch size
    nout = 3
    w0 = net.out.weight.data.detach().clone()
    b0 = net.out.bias.data.detach().clone()
    net.out = nn.Conv2d(256, (nout + nclasses) * ps**2, kernel_size=1).to(
        DEVICE
    )
    # set weights for background map
    i = 0
    net.out.weight.data[i * ps**2 : (i + 1) * ps**2] = (
        -0.5 * w0[(nout - 1) * ps**2 : nout * ps**2]
    )
    net.out.bias.data[i * ps**2 : (i + 1) * ps**2] = b0[
        (nout - 1) * ps**2 : nout * ps**2
    ]
    # set weights for maps to 4 nuclei classes
    for i in range(1, nclasses):
        net.out.weight.data[i * ps**2 : (i + 1) * ps**2] = (
            0.5 * w0[(nout - 1) * ps**2 : nout * ps**2]
        )
        net.out.bias.data[i * ps**2 : (i + 1) * ps**2] = b0[
            (nout - 1) * ps**2 : nout * ps**2
        ]
    net.out.weight.data[-(nout * ps**2) :] = w0
    net.out.bias.data[-(nout * ps**2) :] = b0
    net.W2 = nn.Parameter(
        torch.eye((nout + nclasses) * ps**2).reshape(
            (nout + nclasses) * ps**2, nout + nclasses, ps, ps
        ),
        requires_grad=False,
    )
    net.to(DEVICE)
    return net


def train_net(
    train_data,
    train_labels,
    test_data,
    nclasses: int,
    dataset_name: str,
    epochs: int,
):

    train_instances = [x[..., 0] for x in train_labels]
    train_classes = [x[..., 1] for x in train_labels]

    train_data = [np.transpose(x, (2, 0, 1)) for x in train_data]
    train_flows = dynamics.labels_to_flows(train_instances, device=DEVICE)

    train_labels = [
        np.concatenate(
            (
                train_flows[i][:1],
                train_classes[i][None],
                train_flows[i][1:],
            ),
            axis=0,
        )
        for i in range(len(train_data))
    ]

    test_data = [np.swapaxes(x, 0, 2) for x in test_data]

    pclass = np.zeros((nclasses,))
    pclass_img = np.zeros((len(train_data), nclasses))
    for c in range(nclasses):
        pclass_img[:, c] = np.array(
            [(tl[1] == c).mean() for tl in train_labels]
        )
    pclass = pclass_img.mean(axis=0)

    net = initialize_class_net(nclasses=nclasses)

    learning_rate = 5e-5
    weight_decay = 0.1
    batch_size = 4
    n_epochs = epochs
    bsize = 256
    rescale = False
    scale_range = 0.5

    out = train.train_seg(
        net,
        train_data=train_data,
        train_labels=train_labels,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        n_epochs=n_epochs,
        bsize=bsize,
        nimg_per_epoch=len(train_data),
        rescale=rescale,
        scale_range=scale_range,
        min_train_masks=0,
        nimg_test_per_epoch=0,
        model_name="cellpose_" + dataset_name,
        class_weights=1.0 / pclass,
    )


def predict_net(test_data, nclasses: int, dataset_name: str):
    model = models.CellposeModel(
        gpu=True,
        nchan=3,
        pretrained_model=f"{curr_path}/models/cpsam",
    )
    net = initialize_class_net(nclasses=nclasses)
    net.load_model(
        f"{curr_path}/models/" + "cellpose_" + dataset_name,
        device=DEVICE,
        strict=False,
    )
    net.eval()
    model.net = net
    model.net_ortho = None
    masks_pred, flows, styles = model.eval(
        [x for x in test_data],
        diameter=None,
        augment=False,
        bsize=256,
        tile_overlap=0.1,
        batch_size=64,
        flow_threshold=0.4,
        cellprob_threshold=0,
    )
    classes_pred = [s.squeeze().argmax(axis=-1) for s in styles]

    masks_pred = [
        np.stack((x, y), axis=-1) for x, y in zip(masks_pred, classes_pred)
    ]

    return masks_pred


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_path", type=str, default="")
    parser.add_argument("--train_labels_path", type=str, default="")
    parser.add_argument("--test_data_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--out_path", type=str, default=".")
    parser.add_argument("--training_to_inference_mpp", type=str, default="")
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Performs only inference.",
    )
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    train_data = np.load(args.train_data_path, allow_pickle=True)
    train_labels = np.load(args.train_labels_path, allow_pickle=True)
    test_data = np.load(args.test_data_path, allow_pickle=True)

    train_labels = [x.astype(int) for x in train_labels]
    nclasses = int(np.max([x[..., 1].max() + 1 for x in train_labels]))

    if not args.skip_training:
        train_net(
            train_data,
            train_labels,
            test_data,
            nclasses,
            args.dataset_name,
            args.epochs,
        )

    rescaled_test_data = [
        rescale_if_necessary(x, args.training_to_inference_mpp)
        for x in test_data
    ]
    masks_pred = predict_net(rescaled_test_data, nclasses, args.dataset_name)
    for i in trange(len(masks_pred), desc="Post-processing masks"):
        masks_pred[i] = rescale_label_if_necessary(
            masks_pred[i], test_data[i].shape[:2]
        )

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    np.save(args.out_path, np.array(masks_pred, dtype="object"))
