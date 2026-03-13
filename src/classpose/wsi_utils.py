import numpy as np
from PIL import Image
from pylibCZIrw import czi as pyczi

from classpose.log import get_logger

logger = get_logger(__name__)


class CZISlide:
    """
    OpenSlide-like interface for CZI files using pylibCZIrw.
    """

    def __init__(self, path: str):
        """
        Initialize a CZI slide.

        Args:
            path (str): path to the CZI file.
        """
        self._path = path
        # open_czi returns a context manager, so we have to consume it
        self._czi = pyczi.CziReader(path)
        metadata = self._czi.metadata["ImageDocument"]["Metadata"]
        distance = metadata["Scaling"]["Items"]["Distance"]
        mpp_x = (
            float([x for x in distance if x["@Id"] == "X"][0]["Value"]) / 1e-6
        )
        mpp_y = (
            float([x for x in distance if x["@Id"] == "Y"][0]["Value"]) / 1e-6
        )
        logger.info(f"MPP X: {mpp_x}, MPP Y: {mpp_y}")

        # Build level dimensions (scene 0, channel 0 by default)
        bbp = self._czi.total_bounding_box_no_pyramid
        bbp = {
            "x": bbp["X"][0],
            "y": bbp["Y"][0],
            "w": bbp["X"][1] - bbp["X"][0],
            "h": bbp["Y"][1] - bbp["Y"][0],
        }
        bb = self._czi.total_bounding_box
        bounding_rect = self._czi.scenes_bounding_rectangle[0]
        logger.info(f"Total bounding box: {bb}")
        logger.info(f"Scenes bounding rectangle: {bounding_rect}")
        logger.info(f"Bounding box (layer 0): {bbp}")
        self._x_min = bbp["x"]
        self._y_min = bbp["y"]
        w = bbp["w"]
        h = bbp["h"]

        self.level_downsamples = [1, 2, 4, 8, 16]
        self.level_dimensions = [
            (w // ds, h // ds) for ds in self.level_downsamples
        ]
        self.level_count = len(self.level_dimensions)

        # Expose basic properties
        self.properties = {
            "openslide.vendor": "zeiss",
            "openslide.mpp-x": mpp_x,
            "openslide.mpp-y": mpp_y,
            # we are currently processing bounds internally, so this is set to 0
            # to avoid double offset
            "openslide.bounds-x": 0,
            "openslide.bounds-y": 0,
            "openslide.real-bounds-x": -self._x_min,
            "openslide.real-bounds-y": -self._y_min,
        }

    def read_region(
        self, location: tuple, level: int, size: tuple
    ) -> np.ndarray:
        """
        Read a region from the CZI file.

        Args:
            location (tuple): (x, y) in level-0 pixel coordinates (top-left corner).
            level (int): Pyramid level (0 = full resolution).
            size (tuple): (width, height) of the region to read.

        Returns:
            np.ndarray: Array of shape (height, width, channels), dtype uint8.
        """
        x, y = location
        w, h = size
        downsample = self.level_downsamples[level]

        # Scale location to level-0 coordinates then offset by CZI origin
        roi = (
            int(x) + self._x_min,
            int(y) + self._y_min,
            int(w * downsample),
            int(h * downsample),
        )

        # read() returns a numpy array: (H, W, C) in BGR or BGRA
        tile = self._czi.read(roi=roi, zoom=1.0 / downsample)

        # Resize if the returned shape doesn't exactly match (can happen at edges)
        if tile.shape[:2] != (h, w):
            tile = np.array(Image.fromarray(tile).resize((w, h)))

        # data is BGR, convert to RGB
        tile = np.stack([tile[:, :, 2], tile[:, :, 1], tile[:, :, 0]], -1)

        return tile

    def get_thumbnail(self, size: tuple) -> np.ndarray:
        """
        Return a thumbnail of the whole slide.
        """
        w, h = self.level_dimensions[0]
        thumb_w, thumb_h = size
        zoom = min(thumb_w / w, thumb_h / h)
        return self._czi.read(zoom=zoom)

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """
        Get the best level for a given downsample factor.

        Args:
            downsample (float): Downsample factor.

        Returns:
            int: Best level.
        """
        for i, ds in enumerate(self.level_downsamples):
            if downsample <= ds:
                return i
        return i

    def close(self):
        self._czi.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
