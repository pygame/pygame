"""
binaries hook for pygame seems to be required for pygame 2.0 Windows.
Otherwise some essential DLLs will not be transfered to the exe.
"""

import os
import platform

from pygame import __file__ as pygame_main_file

# Any resources that pygame depends upon must be put into this list,
# so that pyinstaller includes it while running
pygame_res = ["freesansbold.ttf", "pygame.ico", "pygame_icon.bmp",
              "pygame_icon.icns", "pygame_icon.svg", "pygame_icon.tiff"]

# Get pygame's folder
pygame_folder = os.path.dirname(os.path.abspath(pygame_main_file))

# datas is the variable that pyinstaller looks for while processing hooks
datas = []
for res in pygame_res:
    res_path = os.path.join(pygame_folder, res)
    if os.path.exists(res_path):
        datas.append((res_path, "pygame"))

if platform.system() == "Windows":
    
    from PyInstaller.utils.hooks import collect_dynamic_libs

    pre_binaries = collect_dynamic_libs('pygame')
    binaries = []

    for b in pre_binaries:
        binary, location = b
        # settles all the DLLs into the top level folder, which prevents duplication
        # with the DLLs already being put there.
        binaries.append((binary, "."))
