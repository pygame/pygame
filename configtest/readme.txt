The unit tests for the Windows configuration scripts.

These must be run from their containing directory. That directory
must be a subdirectory of the python scripts they test.

test_congif_msys.py:
    Test dependency search. Requires testdir directory. Must
    be run from the MSYS console.

test_config_msys_i.py:
    Internals test. Check MSYS to path conversion.

test_config_win.py
    Test dependency search. Requires testdir directroy.

test_dll.py
    Test the shared DLL information.


