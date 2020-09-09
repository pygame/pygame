"""
binaries hook for pygame seems to be required for pygame 2.0 Windows.
Otherwise some essential DLLs will not be transfered to the exe.
"""

import platform

if platform.system() == "Windows":
    
    from PyInstaller.utils.hooks import collect_dynamic_libs

    pre_binaries = collect_dynamic_libs('pygame')
    binaries = []

    for b in pre_binaries:
        binary, location = b
        # settles all the DLLs into the top level folder, which prevents duplication
        # with the DLLs already being put there.
        binaries.append((binary, "."))
