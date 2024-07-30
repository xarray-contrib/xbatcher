import importlib
import platform
import sys


def show_versions(file=sys.stdout):
    """
    Print various dependency versions,  including information about:

    - xbatcher
    - System information (Python version, Operating System)
    - Dependency versions (Xarray, etc)

    Based on https://github.com/GenericMappingTools/pygmt/blob/09f9e65ebebfa929f9ddc2af90e05f3302c2239d/pygmt/__init__.py#L95
    """

    def _get_module_version(modname):
        """
        Get version information of a Python module.

        Copied from https://github.com/GenericMappingTools/pygmt/blob/09f9e65ebebfa929f9ddc2af90e05f3302c2239d/pygmt/__init__.py#L111
        """
        try:
            if modname in sys.modules:
                module = sys.modules[modname]
            else:
                module = importlib.import_module(modname)

            return getattr(module, '__version__', 'installed')
        except ImportError:
            return None

    sys_info = {
        'python': sys.version.replace('\n', ' '),
        'executable': sys.executable,
        'machine': platform.platform(),
    }

    deps = [
        # Required
        'dask',
        'numpy',
        'xarray',
        # Optional
        'torch',
        # Setup/test
        'pip',
        'conda',
        'pytest',
        # Misc.
        'IPython',
        'sphinx',
    ]
    __version__ = f'v{importlib.metadata.version("xbatcher")}'

    print('xbatcher information:', file=file)
    print(f'  version: {__version__}', file=file)

    print('System information:', file=file)
    for key, val in sys_info.items():
        print(f'  {key}: {val}', file=file)

    print('Dependency information:', file=file)
    for modname in deps:
        print(f'  {modname}: {_get_module_version(modname)}', file=file)


if __name__ == '__main__':  # pragma: no cover
    show_versions()
