import gc
import importlib
import importlib.util
import io
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile

try:
    from pygame import pkgdata

    _PKGDATA_PATH = pathlib.Path(pkgdata.__file__).resolve()
except ModuleNotFoundError:
    _PKGDATA_PATH = (
        pathlib.Path(__file__).resolve().parent.parent / "src_py" / "pkgdata.py"
    )
    _spec = importlib.util.spec_from_file_location("pkgdata", _PKGDATA_PATH)
    pkgdata = importlib.util.module_from_spec(_spec)
    sys.modules[pkgdata.__name__] = pkgdata
    _spec.loader.exec_module(pkgdata)


class PkgDataTest(unittest.TestCase):
    def test_no_pkg_resources_import(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_res_path = pathlib.Path(tmpdir) / "pkg_resources.py"
            pkg_res_path.write_text(
                "import warnings\n"
                'warnings.warn("pkg_resources is deprecated as an API.", UserWarning)\n'
                "def resource_exists(*args, **kwargs): return False\n"
                "def resource_stream(*args, **kwargs): raise NotImplementedError\n",
                encoding="utf-8",
            )
            pkgdata_path_str = str(_PKGDATA_PATH)
            script = (
                "import importlib.util, sys, warnings\n"
                f"sys.path.insert(0, {tmpdir!r})\n"
                "spec = importlib.util.spec_from_file_location(\n"
                f"    'pkgdata_check', {pkgdata_path_str!r}\n"
                ")\n"
                "mod = importlib.util.module_from_spec(spec)\n"
                "with warnings.catch_warnings(record=True) as caught:\n"
                "    warnings.simplefilter('always')\n"
                "    spec.loader.exec_module(mod)\n"
                "msgs = [str(w.message) for w in caught]\n"
                "assert not caught, f'Unexpected warnings: {msgs}'\n"
                "assert 'pkg_resources' not in sys.modules\n"
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = tmpdir
            proc = subprocess.run(
                [sys.executable, "-W", "default", "-c", script],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            self.assertEqual(
                proc.returncode,
                0,
                msg=f"stdout={proc.stdout}\nstderr={proc.stderr}",
            )

    def test_get_resource_default_icon(self):
        with pkgdata.getResource("pygame_icon.bmp") as f:
            self.assertEqual(f.mode, "rb")
            self.assertTrue(f.name.endswith("pygame_icon.bmp"))
            data = f.read(2)
            self.assertEqual(data, b"BM")

    def test_get_resource_package_and_submodule(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_name = "pg_test_fs_pkg"
            pkg_dir = pathlib.Path(tmpdir) / pkg_name
            nested_dir = pkg_dir / "nested"
            nested_dir.mkdir(parents=True)
            (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
            (pkg_dir / "submod.py").write_text("", encoding="utf-8")
            (pkg_dir / "data.bin").write_bytes(b"fs-root")
            (nested_dir / "item.bin").write_bytes(b"fs-nested")

            sys.path.insert(0, tmpdir)
            try:
                importlib.import_module(f"{pkg_name}.submod")
                with pkgdata.getResource("data.bin", pkgname=pkg_name) as f:
                    self.assertEqual(f.mode, "rb")
                    self.assertEqual(f.read(), b"fs-root")
                with pkgdata.getResource("data.bin", pkgname=f"{pkg_name}.submod") as f:
                    self.assertEqual(f.mode, "rb")
                    self.assertEqual(f.read(), b"fs-root")
                with pkgdata.getResource(
                    "nested/item.bin", pkgname=f"{pkg_name}.submod"
                ) as f:
                    self.assertEqual(f.mode, "rb")
                    self.assertEqual(f.read(), b"fs-nested")
            finally:
                sys.path.remove(tmpdir)
                sys.path_importer_cache.pop(tmpdir, None)
                for name in list(sys.modules):
                    if name == pkg_name or name.startswith(f"{pkg_name}."):
                        sys.modules.pop(name, None)

    def test_get_resource_zipimport(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_name = "pg_test_zip_pkg"
            zip_path = pathlib.Path(tmpdir) / "pkg.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(f"{pkg_name}/__init__.py", "")
                zf.writestr(f"{pkg_name}/submod.py", "")
                zf.writestr(f"{pkg_name}/data.bin", b"zip-root")
                zf.writestr(f"{pkg_name}/nested/item.bin", b"zip-nested")

            zip_str = str(zip_path)
            sys.path.insert(0, zip_str)
            try:
                importlib.import_module(f"{pkg_name}.submod")
                with pkgdata.getResource("data.bin", pkgname=f"{pkg_name}.submod") as f:
                    self.assertEqual(f.read(), b"zip-root")
                    self.assertFalse(hasattr(f, "name"))
                with pkgdata.getResource(
                    "./data.bin", pkgname=f"{pkg_name}.submod"
                ) as f:
                    self.assertEqual(f.read(), b"zip-root")
                with pkgdata.getResource(
                    "nested/item.bin", pkgname=f"{pkg_name}.submod"
                ) as f:
                    self.assertEqual(f.read(), b"zip-nested")
                    self.assertFalse(hasattr(f, "name"))
                with pkgdata.getResource(
                    r"nested\item.bin", pkgname=f"{pkg_name}.submod"
                ) as f:
                    self.assertEqual(f.read(), b"zip-nested")
            finally:
                sys.path.remove(zip_str)
                sys.path_importer_cache.pop(zip_str, None)
                for name in list(sys.modules):
                    if name == pkg_name or name.startswith(f"{pkg_name}."):
                        sys.modules.pop(name, None)
                gc.collect()

    def test_get_resource_synthetic_and_unimported_parent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod_name = "pg_test_synthetic_mod"
            target = pathlib.Path(tmpdir) / "synthetic.bin"
            target.write_bytes(b"synthetic-bytes")
            mod = types.ModuleType(mod_name)
            mod.__file__ = str(pathlib.Path(tmpdir) / "synthetic.py")
            sys.modules[mod_name] = mod

            orphan_name = "pg_unimported_parent_xyz.submod"
            orphan_py = pathlib.Path(tmpdir) / "submod.py"
            orphan_py.write_text("", encoding="utf-8")
            spec = importlib.util.spec_from_file_location(orphan_name, orphan_py)
            orphan_mod = importlib.util.module_from_spec(spec)
            sys.modules[orphan_name] = orphan_mod
            spec.loader.exec_module(orphan_mod)
            try:
                with pkgdata.getResource("synthetic.bin", pkgname=mod_name) as f:
                    self.assertEqual(f.mode, "rb")
                    self.assertEqual(f.read(), b"synthetic-bytes")
                with pkgdata.getResource("synthetic.bin", pkgname=orphan_name) as f:
                    self.assertEqual(f.mode, "rb")
                    self.assertEqual(f.read(), b"synthetic-bytes")
            finally:
                sys.modules.pop(mod_name, None)
                sys.modules.pop(orphan_name, None)

    def test_get_resource_legacy_importlib_fallback(self):
        class LegacyResourcesStub:
            @staticmethod
            def is_resource(anchor, name):
                return anchor == "pg_legacy_pkg" and name == "legacy.bin"

            @staticmethod
            def open_binary(_anchor, _name):
                return io.BytesIO(b"legacy-bytes")

        mod_name = "pg_legacy_pkg.sub"
        mod = types.ModuleType(mod_name)
        mod.__package__ = "pg_legacy_pkg"
        sys.modules[mod_name] = mod
        orig_res = pkgdata._importlib_resources
        pkgdata._importlib_resources = LegacyResourcesStub
        try:
            with pkgdata.getResource("./legacy.bin", pkgname=mod_name) as f:
                self.assertEqual(f.read(), b"legacy-bytes")
        finally:
            pkgdata._importlib_resources = orig_res
            sys.modules.pop(mod_name, None)

    def test_get_resource_errors(self):
        with self.assertRaises(KeyError):
            pkgdata.getResource("anything.bin", pkgname="nonexistent_pygame_mod_xyz")

        mod_name = "pg_test_no_file_mod"
        sys.modules[mod_name] = types.ModuleType(mod_name)
        try:
            with self.assertRaises(OSError):
                pkgdata.getResource("missing.bin", pkgname=mod_name)
        finally:
            sys.modules.pop(mod_name, None)

        with self.assertRaises(FileNotFoundError):
            pkgdata.getResource("definitely_missing_resource_xyz.bin")


if __name__ == "__main__":
    unittest.main()
