from importlib.metadata import PackageNotFoundError, version

from .accessors import BatchAccessor  # noqa: F401
from .generators import BatchGenerator  # noqa: F401
from .util.print_versions import show_versions  # noqa: F401

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = 'unknown'
