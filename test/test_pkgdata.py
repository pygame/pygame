import importlib.util
import os
import sys
import types
import tempfile

# Load the `pkgdata.py` module directly from `src_py` so the tests can run
# without installing the package into the environment.
_pkgdata_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src_py", "pkgdata.py")
)
spec = importlib.util.spec_from_file_location("pkgdata_mod", _pkgdata_path)
pkgdata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pkgdata)


def test_get_resource_reads_file():
    """getResource should open a file next to a module's __file__.

    This test creates a fake package module with a __file__ pointing to a
    temporary directory containing a data file, and ensures `getResource`
    returns a readable file-like object with the expected contents.
    """
    tmpdir = tempfile.mkdtemp(prefix="pkgdata_test_")
    pkg_name = "fake_pkg_for_pkgdata_test"
    try:
        # create a dummy module and place it in sys.modules
        mod = types.ModuleType(pkg_name)
        # point __file__ to a file inside the tempdir so getResource can
        # locate the data file relative to it
        mod.__file__ = os.path.join(tmpdir, "__init__.py")
        sys.modules[pkg_name] = mod

        data_path = os.path.join(tmpdir, "example.dat")
        with open(data_path, "wb") as f:
            f.write(b"pygame-pkgdata-test")

        fobj = pkgdata.getResource("example.dat", pkgname=pkg_name)
        try:
            content = fobj.read()
        finally:
            fobj.close()

        assert content == b"pygame-pkgdata-test"
    finally:
        # cleanup
        sys.modules.pop(pkg_name, None)
        try:
            import shutil

            shutil.rmtree(tmpdir)
        except Exception:
            pass



if __name__ == "__main__":
    # allow running this test file directly without pytest
    test_get_resource_reads_file()
    print("TEST-PASSED")
