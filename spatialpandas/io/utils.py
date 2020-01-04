import pathlib

import fsspec


def validate_coerce_filesystem(path, filesystem=None):
    """
    Validate filesystem argument and return an fsspec file system object

    Args:
        path: Path as a string
        filesystem: Optional fsspec filesystem object to use to open the file. If not
            provided, filesystem type is inferred from path

    Returns:
        fsspec file system
    """
    if filesystem is None:
        return fsspec.open(path).fs
    else:
        if isinstance(filesystem, (str, pathlib.Path)):
            return fsspec.filesystem(str(filesystem))
        elif isinstance(filesystem, fsspec.AbstractFileSystem):
            return filesystem
        else:
            raise ValueError(
                "Received invalid filesystem value with type: {typ}".format(
                    typ=type(filesystem)
                )
            )
