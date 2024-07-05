import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator, Tuple

import dask.array as da
import numpy as np
from dask import delayed
from skimage.io.collection import alphanumeric_key
from tifffile import imread as tiff_imread


def get_fov_names(
    folder_name: str, project_path: str = ".", pattern: str | None = None
) -> Generator[Tuple[str, str, str], None, None]:
    r"""
    Get the names of the fields of view in a folder.

    Arguments
    ---------
    folder_name: str
        The name of the folder containing the images.
    project_path: str
        The path to the project.
    pattern: str
        The pattern to match the filenames.
        Default is r"^(?P<FOV>\w+?)_(?P<Time>\w+?)?(?:_.*?)?\.(?P<Extension>\w+)$"
    Returns:
    fov_names: Generator[Tuple[str, str, str], None, None]
        Generator yielding the names of the fields of view,
        their frame number and the filename.
    """
    if pattern is None:
        pattern = r"^(?P<FOV>\w+?)_(?P<Time>\w+?)?(?:_.*?)?\.(?P<Extension>\w+)$"
    # check if the folder exists
    if not os.path.exists(os.path.join(project_path, folder_name)):
        print(f"Folder {folder_name} not found")
        return
    file_names = os.listdir(os.path.join(project_path, folder_name))
    file_names = sorted(file_names, key=alphanumeric_key)
    # proceed only if files are found
    if len(file_names) == 0:
        print(f"No files found in {folder_name}")
        return
    for filename in file_names:
        match = re.match(pattern, filename)
        if match:
            fov = match.group("FOV")
            time = match.group("Time")
            yield fov, time, filename
        else:
            print(f"No match for: {filename}")


def single_tiff_to_array(
    path: str,
    folder: str,
    field_of_view: str,
    pattern: str,
    dask: bool = True,
    num_workers: int = 1,
) -> np.ndarray:
    r"""
    Load a folder containing single tiff images into a dask array.

    Arguments
    ---------
        path (str): The path to the project.
        folder (str): The name of the folder containing the images.
        field_of_view (str): The name of the field of view.
        pattern (str): The pattern to match the filenames.
        dask (bool): Whether to use dask for lazy loading.
        num_workers (int): The number of workers to use for concurrent loading.

    Returns
    -------
        stack (np.ndarray): The stack of images.
    """
    fov_names = get_fov_names(folder, path, pattern)
    filenames = [
        os.path.join(path, folder, filename)
        for fov, _, filename in fov_names
        if str(fov) == str(field_of_view)
    ]
    if len(filenames) > 1:
        filenames = sorted(filenames, key=alphanumeric_key)

    if dask:
        # open first image to get the shape
        first_image = tiff_imread(filenames[0])
        shape_ = first_image.shape
        dtype_ = first_image.dtype
        # Using Dask's delayed execution model for lazy loading
        lazy_imread = delayed(tiff_imread)  # Using tifffile for lazy reading
        lazy_arrays = [lazy_imread(fn) for fn in filenames]
        stack = da.stack(
            [da.from_delayed(la, shape=shape_, dtype=dtype_) for la in lazy_arrays],
            axis=0,
        )
    else:
        # Use ThreadPoolExecutor to load images concurrently
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Mapping each future to its index to preserve order
            future_to_index = {
                executor.submit(tiff_imread, fn): i for i, fn in enumerate(filenames)
            }
            results = [None] * len(
                filenames
            )  # Pre-allocate the result list to preserve order
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                # Place the result in the correct order
                results[index] = future.result()
            # Stack images into a single array
            stack = np.stack(results, axis=0)

    return stack
