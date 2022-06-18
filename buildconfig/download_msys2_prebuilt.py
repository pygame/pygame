import os
import sys
import logging
import subprocess


def install_pacman_package(pkg_name):
    """ This installs a package in the current MSYS2 environment

    Does not download again if the package is already installed
    and if the version is the latest available in MSYS2
    """
    output = subprocess.run(['pacman', '-S', '--noconfirm', pkg_name],
                            capture_output=True, text=True)
    if output.returncode != 0:
        logging.error(
            "Error {} while downloading package {}: \n{}".
            format(output.returncode, pkg_name, output.stderr))

    return output.returncode != 0


def get_packages(MINGW_ARCH):
    deps = [
        '{}-SDL2',
        '{}-SDL2_ttf',
        '{}-SDL2_image',
        '{}-SDL2_mixer',
        '{}-portmidi',
        '{}-libpng',
        '{}-libjpeg-turbo',
        '{}-libtiff',
        '{}-zlib',
        '{}-libwebp',
        '{}-libvorbis',
        '{}-libogg',
        '{}-flac',
        '{}-libmodplug',
        '{}-mpg123',
        '{}-opus',
        '{}-opusfile',
        '{}-freetype'
    ]

    MINGW_PREFIX_DICT = {
        "mingw64":"mingw-w64-x86_64",
        "mingw32":"mingw-w64-i686",
        "ucrt64":"mingw-w64-ucrt-x86_64",
        "clang64":"mingw-w64-clang-x86_64",
        "clang32":"mingw-w64-clang-i686",
    }

    packages = [x.format(MINGW_PREFIX_DICT[MINGW_ARCH]) for x in deps]
    return packages


def install_prebuilts(MINGW_ARCH):
    """ For installing prebuilt dependencies.
    """
    errors = False
    print("Installing pre-built dependencies")
    for pkg in get_packages(MINGW_ARCH):
        print(f"Installing {pkg}")
        error = install_pacman_package(pkg)
        errors = errors or error
    if errors:
        raise Exception("Some dependencies could not be installed")


def update(MINGW_ARCH=None):
    if MINGW_ARCH:
        print("The MSYS2 environment is now set to \"{}\"".format(MINGW_ARCH))
        install_prebuilts(MINGW_ARCH)
    else:
        # Set fallback MSYS2 environment
        if "MINGW_ARCH" not in os.environ:
            if sys.maxsize > 2**32:
                MINGW_ARCH = "mingw64"
            else:
                MINGW_ARCH = "mingw32"
        else:
            MINGW_ARCH = os.environ["MINGW_ARCH"]
        print("The MSYS2 environment is now set to \"{}\"".format(MINGW_ARCH))
        install_prebuilts(MINGW_ARCH)


if __name__ == '__main__':
    update()
