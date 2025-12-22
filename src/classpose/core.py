import numpy as np
import torch
from cellpose import transforms
from cellpose.core import tqdm_out
from PyQt5.QtWidgets import QProgressBar
from tqdm import trange

from classpose.log import get_logger
from classpose.transforms import unaugment_class_tiles

core_logger = get_logger(__name__)


def _to_device(
    x: np.ndarray | torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Converts the input tensor or numpy array to the specified device.

    Args:
        x (torch.Tensor | np.ndarray): The input tensor or numpy array.
        device (torch.device): The target device.
        dtype (torch.dtype, optional): The target data type. Defaults to
            torch.float32.

    Returns:
        torch.Tensor: The converted tensor on the specified device.
    """
    if not isinstance(x, torch.Tensor):
        X = torch.from_numpy(x).to(device, dtype=dtype)
        return X
    else:
        return x


def _from_device(X: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor from the device to a NumPy array on the CPU.

    Args:
        X (torch.Tensor): The input PyTorch tensor.

    Returns:
        numpy.ndarray: The converted NumPy array.
    """
    x = X.detach().to(torch.float32).cpu().numpy()
    return x


def _forward(net: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    """Converts images to torch tensors, runs the network model, and returns numpy arrays.

    Args:
        net (torch.nn.Module): The network model.
        x (numpy.ndarray): The input images.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: The output predictions (flows and cellprob), classes and style features.
    """
    dtype = next(net.parameters()).dtype
    X = _to_device(x, device=net.device, dtype=dtype)
    net.eval()
    with torch.no_grad():
        y, style = net(X.to(dtype=dtype))[:2]
    del X
    y = _from_device(y.float())
    style = _from_device(style.float())
    if net.n_cell_classes > 1:
        y_class = y[:, : net.n_cell_classes]
        y = y[:, net.n_cell_classes :]
    return y, y_class, style


def run_net(
    net: torch.nn.Module,
    imgi: np.ndarray,
    batch_size: int = 8,
    augment: bool = False,
    tile_overlap: float = 0.1,
    bsize: int = 224,
    rsz: float | None = None,
):
    """
    Run network on stack of images.

    (faster if augment is False)

    Args:
        net (torch.nn.Module): cellpose network (model.net)
        imgi (np.ndarray): The input image or stack of images of size
            [Lz x Ly x Lx x nchan].
        batch_size (int, optional): Number of tiles to run in a batch. Defaults
            to 8.
        rsz (float, optional): Resize coefficient(s) for image. Defaults to
            1.0.
        augment (bool, optional): Tiles image with overlapping tiles and flips
            overlapped regions to augment. Defaults to False.
        tile_overlap (float, optional): Fraction of overlap of tiles when
            computing flows. Defaults to 0.1.
        bsize (int, optional): Size of tiles to use in pixels [bsize x bsize].
            Defaults to 224.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
            - y: output of network y. If tiled `y` is averaged in tile
                overlaps. Size of [Ly x Lx x 3] or [Lz x Ly x Lx x 3].
                y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability.
            - y_class: output of network y_class. If tiled `y_class` is averaged
                in tile overlaps. Size of [Ly x Lx x nclasses] or [Lz x Ly x Lx x
                nclasses].
            - style: output of network style. If tiled `style` is averaged over
                tiles. Size of [256].
    """
    # backwards compatibility with standard cellpose model
    if hasattr(net, "n_cell_classes"):
        nclasses = net.n_cell_classes
    else:
        nclasses = None
    # run network
    Lz, Ly0, Lx0, nchan = imgi.shape
    if rsz is not None:
        if not isinstance(rsz, list) and not isinstance(rsz, np.ndarray):
            rsz = [rsz, rsz]
        Lyr, Lxr = int(Ly0 * rsz[0]), int(Lx0 * rsz[1])
    else:
        Lyr, Lxr = Ly0, Lx0

    ly, lx = bsize, bsize
    ypad1, ypad2, xpad1, xpad2 = transforms.get_pad_yx(
        Lyr, Lxr, min_size=(bsize, bsize)
    )
    Ly, Lx = Lyr + ypad1 + ypad2, Lxr + xpad1 + xpad2
    pads = np.array([[0, 0], [ypad1, ypad2], [xpad1, xpad2]])

    if augment:
        ny = max(2, int(np.ceil(2.0 * Ly / bsize)))
        nx = max(2, int(np.ceil(2.0 * Lx / bsize)))
    else:
        ny = (
            1
            if Ly <= bsize
            else int(np.ceil((1.0 + 2 * tile_overlap) * Ly / bsize))
        )
        nx = (
            1
            if Lx <= bsize
            else int(np.ceil((1.0 + 2 * tile_overlap) * Lx / bsize))
        )

    # run multiple slices at the same time
    ntiles = ny * nx
    nimgs = max(
        1, batch_size // ntiles
    )  # number of imgs to run in the same batch
    niter = int(np.ceil(Lz / nimgs))
    ziterator = (
        trange(niter, file=tqdm_out, mininterval=30)
        if niter > 10 or Lz > 1
        else range(niter)
    )
    for k in ziterator:
        inds = np.arange(k * nimgs, min(Lz, (k + 1) * nimgs))
        IMGa = np.zeros((ntiles * len(inds), nchan, ly, lx), "float32")
        for i, b in enumerate(inds):
            # pad image for net so Ly and Lx are divisible by 4
            imgb = (
                transforms.resize_image(imgi[b], rsz=rsz)
                if rsz is not None
                else imgi[b].copy()
            )
            imgb = np.pad(imgb.transpose(2, 0, 1), pads, mode="constant")
            IMG, ysub, xsub, Lyt, Lxt = transforms.make_tiles(
                imgb, bsize=bsize, augment=augment, tile_overlap=tile_overlap
            )
            IMGa[i * ntiles : (i + 1) * ntiles] = np.reshape(
                IMG, (ny * nx, nchan, ly, lx)
            )

        # run network
        for j in range(0, IMGa.shape[0], batch_size):
            bslc = slice(j, min(j + batch_size, IMGa.shape[0]))
            ya0, y_class0, stylea0 = _forward(net, IMGa[bslc])
            if j == 0:
                nout = ya0.shape[1]
                ya = np.zeros((IMGa.shape[0], nout, ly, lx), "float32")
                if nclasses:
                    y_classa = np.zeros(
                        (IMGa.shape[0], nclasses, ly, lx), "float32"
                    )
                stylea = np.zeros((IMGa.shape[0], 256), "float32")
            ya[bslc] = ya0
            if nclasses:
                y_classa[bslc] = y_class0
            stylea[bslc] = stylea0

        # average tiles
        for i, b in enumerate(inds):
            if i == 0 and k == 0:
                yf = np.zeros((Lz, nout, Ly, Lx), "float32")
                if nclasses:
                    y_classf = np.zeros((Lz, nclasses, Ly, Lx), "float32")
                styles = np.zeros((Lz, 256), "float32")
            y = ya[i * ntiles : (i + 1) * ntiles]
            if nclasses:
                y_class = y_classa[i * ntiles : (i + 1) * ntiles]
            if augment:
                y = np.reshape(y, (ny, nx, 3, ly, lx))
                y = transforms.unaugment_tiles(y)
                y = np.reshape(y, (-1, 3, ly, lx))
                if nclasses:
                    y_class = np.reshape(y_class, (ny, nx, nclasses, ly, lx))
                    y_class = unaugment_class_tiles(y_class)
                    y_class = np.reshape(y_class, (-1, nclasses, ly, lx))
            yfi = transforms.average_tiles(y, ysub, xsub, Lyt, Lxt)
            yf[b] = yfi[:, : imgb.shape[-2], : imgb.shape[-1]]
            if nclasses:
                y_classfi = transforms.average_tiles(
                    y_class, ysub, xsub, Lyt, Lxt
                )
                y_classf[b] = y_classfi[:, : imgb.shape[-2], : imgb.shape[-1]]
            stylei = stylea[i * ntiles : (i + 1) * ntiles].sum(axis=0)
            stylei /= (stylei**2).sum() ** 0.5
            styles[b] = stylei
    # slices from padding
    yf = yf[:, :, ypad1 : Ly - ypad2, xpad1 : Lx - xpad2]
    yf = yf.transpose(0, 2, 3, 1)
    if nclasses:
        y_classf = y_classf[:, :, ypad1 : Ly - ypad2, xpad1 : Lx - xpad2]
        y_classf = y_classf.transpose(0, 2, 3, 1)
    return yf, y_classf, np.array(styles)


def run_3D(
    net: torch.nn.Module,
    imgs: np.ndarray,
    batch_size: int = 8,
    augment: bool = False,
    tile_overlap: float = 0.1,
    bsize: int = 224,
    net_ortho: torch.nn.Module | None = None,
    progress: QProgressBar | None = None,
):
    """
    Run network on image z-stack. Runs faster if augment is False.

    Args:
        imgs (np.ndarray): The input image stack of size [Lz x Ly x Lx x nchan].
        batch_size (int, optional): Number of tiles to run in a batch. Defaults to 8.
        rsz (float, optional): Resize coefficient(s) for image. Defaults to 1.0.
        anisotropy (float, optional): for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y). Defaults to None.
        augment (bool, optional): Tiles image with overlapping tiles and flips overlapped regions to augment. Defaults to False.
        tile_overlap (float, optional): Fraction of overlap of tiles when computing flows. Defaults to 0.1.
        bsize (int, optional): Size of tiles to use in pixels [bsize x bsize]. Defaults to 224.
        net_ortho (class, optional): cellpose network for orthogonal ZY and ZX planes. Defaults to None.
        progress (QProgressBar, optional): pyqt progress bar. Defaults to None.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
            - y is a 4D array of size [Lz x Ly x Lx x 3].
            - y_classf is a 4D array of size [Lz x Ly x Lx x nclasses]
            - style is a 2D array of size [Lz x 256] summarizing the style of the image.
    """
    # backwards compatibility with standard cellpose model
    if hasattr(net, "n_cell_classes"):
        nclasses = net.n_cell_classes
    else:
        nclasses = None

    sstr = ["YX", "ZY", "ZX"]
    pm = [(0, 1, 2, 3), (1, 0, 2, 3), (2, 0, 1, 3)]
    ipm = [(0, 1, 2), (1, 0, 2), (1, 2, 0)]
    cp = [(1, 2), (0, 2), (0, 1)]
    cpy = [(0, 1), (0, 1), (0, 1)]
    shape = imgs.shape[:-1]
    yf = np.zeros((*shape, 4), "float32")
    if nclasses:
        y_classf = np.zeros((*shape, nclasses), "float32")
    for p in range(3):
        xsl = imgs.transpose(pm[p])
        # per image
        core_logger.info(
            "running %s: %d planes of size (%d, %d)"
            % (sstr[p], shape[pm[p][0]], shape[pm[p][1]], shape[pm[p][2]])
        )
        y, y_class, style = run_net(
            net,
            xsl,
            batch_size=batch_size,
            augment=augment,
            bsize=bsize,
            tile_overlap=tile_overlap,
            rsz=None,
        )
        yf[..., -1] += y[..., -1].transpose(ipm[p])
        for j in range(2):
            yf[..., cp[p][j]] += y[..., cpy[p][j]].transpose(ipm[p])

        if nclasses:
            y_classf[..., -1] += y_class[..., -1].transpose(ipm[p])
            for j in range(2):
                y_classf[..., cp[p][j]] += y_class[..., cpy[p][j]].transpose(
                    ipm[p]
                )

        y = None
        del y

        if progress is not None:
            progress.setValue(25 + 15 * p)

    return yf, y_classf, style
