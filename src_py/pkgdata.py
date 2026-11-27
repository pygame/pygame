"""
pkgdata is a simple, extensible way for a package to acquire data file
resources.

The getResource function is equivalent to the standard idioms, such as
the following minimal implementation:

    import sys, os

    def getResource(identifier, pkgname=__name__):
        pkgpath = os.path.dirname(sys.modules[pkgname].__file__)
        path = os.path.join(pkgpath, identifier)
        return file(os.path.normpath(path), mode='rb')

When a __loader__ is present on the module given by __name__, it will defer
getResource to its get_data implementation and return it as a file-like
object (such as StringIO).
"""

__all__ = ["getResource"]
import io
import os
import posixpath
import sys

try:
    import importlib.resources as _importlib_resources
except ImportError:
    _importlib_resources = None


def getResource(identifier, pkgname=__name__):
    """
    Acquire a readable object for a given package name and identifier.
    An IOError will be raised if the resource can not be found.

    For example:
        mydata = getResource('mypkgdata.jpg').read()

    Note that the package name must be fully qualified, if given, such
    that it would be found in sys.modules.

    In some cases, getResource will return a real file object.  In that
    case, it may be useful to use its name attribute to get the path
    rather than use it as a file-like object.  For example, you may
    be handing data off to a C API.
    """
    mod = sys.modules[pkgname]

    if _importlib_resources is not None and not os.path.isabs(identifier):
        anchor = mod.__package__ or pkgname
        norm_id = posixpath.normpath(identifier.replace("\\", "/"))
        parts = [p for p in norm_id.split("/") if p and p != "."]
        if parts and ".." not in parts and not norm_id.startswith("/"):
            if hasattr(_importlib_resources, "files"):
                try:
                    res = _importlib_resources.files(anchor).joinpath(*parts)
                    if res.is_file():
                        if isinstance(res, os.PathLike):
                            return res.open("rb")
                        return io.BytesIO(res.read_bytes())
                except (
                    ImportError,
                    NotImplementedError,
                    AttributeError,
                    TypeError,
                    ValueError,
                ):
                    pass
            elif (
                len(parts) == 1
                and hasattr(_importlib_resources, "is_resource")
                and hasattr(_importlib_resources, "open_binary")
            ):
                try:
                    if _importlib_resources.is_resource(anchor, parts[0]):
                        return _importlib_resources.open_binary(anchor, parts[0])
                except (
                    ImportError,
                    NotImplementedError,
                    AttributeError,
                    TypeError,
                    ValueError,
                ):
                    pass

    path_to_file = getattr(mod, "__file__", None)
    if path_to_file is None:
        raise OSError(f"{mod!r} has no __file__!")
    path = os.path.join(os.path.dirname(path_to_file), identifier)

    # pylint: disable=consider-using-with
    return open(os.path.normpath(path), "rb")
