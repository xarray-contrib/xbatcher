from importlib.metadata import (
    PackageNotFoundError as _PackageNotFoundError,
    version as _version,
)

from . import testing  # noqa: F401
from .accessors import BatchAccessor  # noqa: F401
from .generators import BatchGenerator, BatchSchema  # noqa: F401
from .util.print_versions import show_versions  # noqa: F401

try:
    __version__ = _version(__name__)
except _PackageNotFoundError:
    # package is not installed
    __version__ = 'unknown'
