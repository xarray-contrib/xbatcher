from pkg_resources import DistributionNotFound, get_distribution

from .accessors import BatchAccessor  # noqa: F401
from .generators import BatchGenerator  # noqa: F401
from .util.print_versions import show_versions  # noqa: F401

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:  # noqa: F401; pragma: no cover
    # package is not installed
    pass
