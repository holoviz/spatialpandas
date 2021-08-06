from os import PathLike
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import fsspec

PathType = Union[PathLike, str, Path]


def validate_coerce_filesystem(
    path: PathType,
    filesystem: Optional[Union[str, fsspec.AbstractFileSystem]] = None,
    storage_options: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> fsspec.AbstractFileSystem:
    """
    Validate filesystem argument and return an fsspec file system object

    Args:
        path: Path as a string
        filesystem: Optional fsspec filesystem object to use to open the file. If not
            provided, filesystem type is inferred from path
        storage_options: Key/value pairs to be passed on to the file-system backend, if any.

    Returns:
        fsspec file system
    """
    if isinstance(filesystem, fsspec.AbstractFileSystem):
        return filesystem
    fsspec_opts = kwargs.copy()
    if storage_options:
        fsspec_opts.update(storage_options)
    if filesystem is None:
        return fsspec.open(path, **fsspec_opts).fs
    else:
        try:
            return fsspec.filesystem(filesystem, **fsspec_opts)
        except ValueError:
            raise ValueError(
                "Received invalid filesystem value with type: {typ}".format(
                    typ=type(filesystem)
                )
            )


def _maybe_prepend_protocol(
    paths: Iterable[PathType],
    filesystem: fsspec.AbstractFileSystem,
) -> Iterable[PathType]:
    protocol = filesystem.protocol if isinstance(
        filesystem.protocol, str) else filesystem.protocol[0]
    if protocol not in ("file", "abstract"):
        # Add back prefix (e.g. s3://)
        paths = ["{proto}://{p}".format(proto=protocol, p=p) for p in paths]
    return paths
