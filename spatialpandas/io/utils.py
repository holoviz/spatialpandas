import math
import time
from functools import wraps
from inspect import signature
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import fsspec

try:
    from tenacity import (
        retry as tenacity_retry,
        stop_after_attempt,
        wait_exponential,
    )
except ImportError:
    tenacity_retry = stop_after_attempt = wait_exponential = None

try:
    from retrying import retry as retrying_retry
except ImportError:
    retrying_retry = None

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
    if protocol and protocol not in ("file", "abstract"):
        # Add back prefix (e.g. s3://)
        paths = ["{proto}://{p}".format(proto=protocol, p=p) for p in paths]
    return paths


def retry(tries, delay=3, backoff=2, max_delay=120, exceptions=Exception):
    """Retry decorator with exponential backoff.

    Retries a function or method until it returns True.
    Based on https://wiki.python.org/moin/PythonDecoratorLibrary#Retry

    Parameters
    ----------
    delay:
        sets the initial delay in seconds, and backoff sets the factor by which
        the delay should lengthen after each failure.
    backoff:
        must be greater than 1, or else it isn't really a backoff.
    tries: must be at least 0, and delay greater than 0.
    exceptions:
        Single or multiple exceptions to allow.
    """
    if backoff <= 1:
        raise ValueError("backoff must be greater than 1")

    tries = math.floor(tries)
    if tries < 0:
        raise ValueError("tries must be 0 or greater")

    if delay <= 0:
        raise ValueError("delay must be greater than 0")

    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay  # make mutable

            while mtries > 0:
                try:
                    rv = f(*args, **kwargs)
                    return rv
                except exceptions as e:
                    mtries -= 1  # consume an attempt
                    time.sleep(mdelay)  # wait...
                    mdelay *= backoff  # make future wait longer
                    mdelay = min(mdelay, max_delay)
                    if mtries <= 0:
                        raise e

        return f_retry  # true decorator -> decorated function

    return deco_retry  # @retry(arg[, ...]) -> true decorator


def _make_retry_decorator(
    *args,
    retry_lib=None,
    stop_max_attempt_number=24,
    wait_exponential_max=120000,
    wait_exponential_multiplier=100,
    **kwargs,
):
    if tenacity_retry and stop_after_attempt and wait_exponential and (retry_lib is None or retry_lib == "tenacity"):
        stop = kwargs.pop("stop", stop_after_attempt(stop_max_attempt_number))
        wait = kwargs.pop("wait", wait_exponential(
            multiplier=wait_exponential_multiplier,
            max=wait_exponential_max / 1000,
        ))
        reraise = kwargs.pop("reraise", True)
        retryer = tenacity_retry(
            *args,
            reraise=reraise,
            stop=stop,
            wait=wait,
            **kwargs,
        )
    elif retrying_retry and (retry_lib is None or retry_lib == "retrying"):
        retryer = retrying_retry(
            *args,
            wait_exponential_multiplier=wait_exponential_multiplier,
            wait_exponential_max=wait_exponential_max,
            stop_max_attempt_number=stop_max_attempt_number,
            **kwargs,
        )
    else:
        delay = kwargs.pop("delay", 1)
        retryer = retry(
            stop_max_attempt_number,
            *args,
            delay=delay,
            backoff=wait_exponential_multiplier,
            max_delay=wait_exponential_max / 1000,
            **kwargs,
        )
    return retryer


def _make_fs_retry(filesystem, retryit=None):
    retryit = retryit or _make_retry_decorator()

    # For filesystems that provide a "refresh" argument, set it to True
    if 'refresh' in signature(filesystem.ls).parameters:
        ls_kwargs = {'refresh': True}
    else:
        ls_kwargs = {}

    class FSretry:

        @staticmethod
        @retryit
        def rm_retry(file_path):
            filesystem.invalidate_cache()
            if filesystem.exists(file_path):
                filesystem.rm(file_path, recursive=True)
                if filesystem.exists(file_path):
                    # Make sure we keep retrying until file does not exist
                    raise ValueError(
                        "Deletion of {path} not yet complete".format(
                            path=file_path))

        @staticmethod
        @retryit
        def mkdirs_retry(dir_path):
            filesystem.makedirs(dir_path, exist_ok=True)

        @staticmethod
        @retryit
        def ls_retry(dir_path):
            filesystem.invalidate_cache()
            return filesystem.ls(dir_path, **ls_kwargs)

        @staticmethod
        @retryit
        def move_retry(p1, p2):
            if filesystem.exists(p1):
                filesystem.move(p1, p2)

    return FSretry
