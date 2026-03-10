import os
from classpose.log import get_logger

logger = get_logger(__name__)

WSI_READERS = ["czi-zeiss", "openslide"]


def get_wsi_reader(reader_str: str):
    """
    Get a WSI reader for the given path.

    Args:
        reader_str (str): path to the WSI file.

    Returns:
        WSI reader class (either OpenSlide or CZISlide).
    """
    if reader_str not in WSI_READERS:
        readers = list(WSI_READERS.keys())
        err = f"Reader {reader_str} not supported. "
        err += f"Should be one of {readers}"
        logger.error(err)
        raise ValueError(err)
    if reader_str == "czi-zeiss":
        try:
            from classpose.wsi_utils import CZISlide
        except ImportError:
            err = "czi-zeiss reader requires pylibCZIrw"
            logger.error(err)
            raise ImportError(err)
        return CZISlide
    elif reader_str == "openslide":
        from openslide import OpenSlide

        return OpenSlide


def WSIReader(*args, **kwargs):
    reader_str = os.environ.get("WSI_READER", "openslide")
    return get_wsi_reader(reader_str)(*args, **kwargs)


__all__ = ["WSIReader"]
