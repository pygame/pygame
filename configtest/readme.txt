The unit tests for the Windows configuration scripts.

These must be run from their containing directory. That directory
must be a subdirectory of the python scripts they test.

test_congif_msys.py:
    Test dependency search. Requires testdir directory.

test_config_win.py
    Test dependency search. Requires testdir directroy.

test_dll.py
    Test the shared DLL information.

test_msys.py
    Test the MSYS support module. Verifies that path name converion
    and MSYS bash shell can be calls. Requires MSYS to be installed
    and configured for MinGW.

