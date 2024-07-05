"""A collection of utils to process images with a focus on timeseries analysis."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bioimage_utils")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Benjamin Graedel, Lucien Hinderling"
__email__ = "benjamin.graedel@unibe.ch, lucien.hinderling@unibe.ch"
