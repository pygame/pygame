#!/usr/bin/env python3
"""Run the pygame pixelarray_test module against the in-tree build.

This helper mirrors the manual harness described in docs/issue-4163-pixelarray.md
so contributors can execute the regression suite without installing pygame first.
"""

import importlib
import importlib.util
import os
import sys
import unittest
from pathlib import Path


def _find_build_package_path() -> str:
    """Return the pygame package directory under build/ if present."""
    override = os.environ.get("PYGAME_BUILD_PACKAGE_PATH")
    if override:
        override_path = Path(override)
        if not override_path.is_dir():
            raise FileNotFoundError(
                f"Environment override PYGAME_BUILD_PACKAGE_PATH={override} "
                "does not point to a directory"
            )
        return str(override_path)

    build_root = Path("build")
    if not build_root.exists():
        raise FileNotFoundError("No build/ directory found. Did you run build_ext -i?")

    candidates = sorted(
        (
            candidate
            for candidate in build_root.glob("lib*")
            if (candidate / "pygame").is_dir()
        ),
        key=lambda path: len(str(path)),
    )
    if not candidates:
        raise FileNotFoundError(
            "Could not locate a built pygame package under build/. "
            "Ensure setup.py build_ext -i has completed successfully."
        )
    return str(candidates[0] / "pygame")


def load_pygame_from_source():
    """Load pygame directly from src_py plus the locally built extensions."""
    build_module_path = _find_build_package_path()
    spec = importlib.util.spec_from_file_location(
        "pygame",
        "src_py/__init__.py",
        submodule_search_locations=["src_py", build_module_path],
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to locate pygame package in source tree")

    module = importlib.util.module_from_spec(spec)
    sys.modules["pygame"] = module
    spec.loader.exec_module(module)
    module.__path__.extend(["src_py", build_module_path])
    return module


def expose_tests_package():
    """Expose pygame.tests by aliasing the repository's test/ package."""
    test_pkg_path = os.path.abspath("test")
    spec = importlib.util.spec_from_file_location(
        "pygame.tests",
        os.path.join(test_pkg_path, "__init__.py"),
        submodule_search_locations=[test_pkg_path],
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to locate pygame.tests package in test/")

    module = importlib.util.module_from_spec(spec)
    sys.modules["pygame.tests"] = module
    spec.loader.exec_module(module)
    module.__path__ = [test_pkg_path]
    return module


def main():
    load_pygame_from_source()
    expose_tests_package()

    test_root = os.path.abspath("test")
    sys.path.insert(0, test_root)

    module = importlib.import_module("pixelarray_test")
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
