from briefcase.bootstraps.base import BaseGuiBootstrap


class PygameCEGuiBootstrap(BaseGuiBootstrap):
    display_name_annotation = "does not support iOS/Android deployment"

    def app_source(self):
        return """\
import importlib.metadata
import os
import sys

import pygame


SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
BGCOLOR = (255, 255, 255)


def main():
    # Linux desktop environments use an app's .desktop file to integrate the app
    # in to their application menus. The .desktop file of this app will include
    # the StartupWMClass key, set to app's formal name. This helps associate the
    # app's windows to its menu item.
    #
    # For association to work, any windows of the app must have WMCLASS property
    # set to match the value set in app's desktop file. For pygame, this is
    # set using the SDL_VIDEO_X11_WMCLASS environment variable.

    # Find the name of the module that was used to start the app
    app_module = sys.modules["__main__"].__package__
    # Retrieve the app's metadata
    metadata = importlib.metadata.metadata(app_module)

    os.environ["SDL_VIDEO_X11_WMCLASS"] = metadata["Formal-Name"]

    pygame.init()
    pygame.display.set_caption(metadata["Formal-Name"])
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        screen.fill(BGCOLOR)
        pygame.display.flip()

    pygame.quit()
"""

    def pyproject_table_briefcase_app_extra_content(self):
        return """
requires = [
    "pygame",
]
"""

    def pyproject_table_macOS(self):
        return """\
universal_build = true
requires = [
    "std-nslog~=1.0.0",
]
"""

    def pyproject_table_linux(self):
        return """\
requires = [
]
"""

    def pyproject_table_linux_system_debian(self):
        return """\
system_requires = [
]

system_runtime_requires = [
]
"""

    def pyproject_table_linux_system_rhel(self):
        return """\
system_requires = [
]

system_runtime_requires = [
]
"""

    def pyproject_table_linux_system_suse(self):
        return """\
system_requires = [
]

system_runtime_requires = [
]
"""

    def pyproject_table_linux_system_arch(self):
        return """\
system_requires = [
]

system_runtime_requires = [
]
"""

    def pyproject_table_linux_appimage(self):
        return """\
manylinux = "manylinux_2_28"

system_requires = [
]

linuxdeploy_plugins = [
]
"""

    def pyproject_table_linux_flatpak(self):
        return """\
flatpak_runtime = "org.freedesktop.Platform"
flatpak_runtime_version = "23.08"
flatpak_sdk = "org.freedesktop.Sdk"
"""

    def pyproject_table_windows(self):
        return """\
requires = [
]
"""

    def pyproject_table_iOS(self):
        return """\
supported = false
"""

    def pyproject_table_android(self):
        return """\
supported = false
"""

    def pyproject_table_web(self):
        return """\
supported = false
"""
