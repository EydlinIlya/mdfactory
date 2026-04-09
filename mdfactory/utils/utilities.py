# ABOUTME: General-purpose utility functions for mdfactory
# ABOUTME: Provides working directory management, YAML loading, and file locking
"""General-purpose utility functions for mdfactory."""

import contextlib
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import yaml


@contextlib.contextmanager
def working_directory(path, create=False, cleanup=False, exists_ok=True):
    """Change working directory and return to previous on exit.

    Parameters
    ----------
    path : str or Path
        Target directory to change into.
    create : bool, optional
        Create the directory if it does not exist. Default is False.
    cleanup : bool, optional
        Remove the directory before creating it. Implies ``create=True``.
        Default is False.
    exists_ok : bool, optional
        If False, raise ``FileExistsError`` when the directory already exists.
        Default is True.

    Yields
    ------
    Path
        Resolved path of the working directory.

    Raises
    ------
    FileExistsError
        If the directory exists and ``exists_ok`` is False.

    """
    path = Path(path).resolve()
    if path.is_dir() and not exists_ok:
        raise FileExistsError("Path already exists.")
    if cleanup:
        create = True
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(prev_cwd)


@contextlib.contextmanager
def temporary_working_directory(prefix="temp_"):
    """Create a temporary directory and change into it.

    Parameters
    ----------
    prefix : str, optional
        Prefix for the temporary directory name. Default is ``"temp_"``.

    Yields
    ------
    Path
        Resolved path of the temporary directory.

    """
    with tempfile.TemporaryDirectory(prefix=prefix) as temp_dir:
        with working_directory(temp_dir) as tmp:
            yield tmp


def load_yaml_file(yaml_file_path: str) -> Dict[str, Any]:
    """Load data from a YAML file.

    Parameters
    ----------
    yaml_file_path : str
        The path to the YAML file.

    Returns
    -------
    Dict[str, Any]
        The loaded data from the YAML file.

    """
    with open(yaml_file_path, "r") as file:
        return yaml.safe_load(file)


@contextlib.contextmanager
def lock_local_folder(folder, retries: int = 120, wait: float = 2.0, message: str = ""):
    """Acquire a file-based lock on a folder, blocking until available.

    Create a ``<folder>.lock`` sentinel file and yield. If the lock file
    already exists, retry up to *retries* times with *wait* seconds between
    attempts. The lock file is removed on exit.

    Parameters
    ----------
    folder : str or Path
        Path to the folder to lock.
    retries : int, optional
        Maximum number of acquisition attempts. Default is 120.
    wait : float, optional
        Seconds to sleep between retries. Default is 2.0.
    message : str, optional
        Informational message (currently unused, reserved for logging).

    Yields
    ------
    None

    Raises
    ------
    TimeoutError
        If the lock cannot be acquired within the retry limit.

    """
    lockfile = Path(f"{folder}.lock")
    was_locked = False
    created_lockfile = False
    other_exception = None
    for _ in range(retries):
        try:
            lockfile.touch(exist_ok=False)
            created_lockfile = True
            was_locked = False
            # print(f"Acquired lock for folder {folder}.", message)
            yield
        except FileExistsError as e:
            if created_lockfile:
                other_exception = e
            # print(f"Folder {folder} is already locked.", message, e)
            was_locked = True
            time.sleep(wait)
        except Exception as e:
            other_exception = e
        finally:
            if other_exception:
                lockfile.unlink()
                raise other_exception
            if was_locked:
                # print(f"Folder {folder} was locked.", message)
                continue
            lockfile.unlink()
            return
    raise TimeoutError(f"Could not acquire lock for folder {folder} after {retries} retries.")
