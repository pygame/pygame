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


def get_packages(x86=True, x64=True):
    deps = [
        'mingw-w64-{}-SDL2',
        'mingw-w64-{}-SDL2_ttf',
        'mingw-w64-{}-SDL2_image',
        'mingw-w64-{}-SDL2_mixer',
        'mingw-w64-{}-portmidi',
        'mingw-w64-{}-libpng',
        'mingw-w64-{}-libjpeg-turbo',
        'mingw-w64-{}-libtiff',
        'mingw-w64-{}-zlib',
        'mingw-w64-{}-libwebp',
        'mingw-w64-{}-libvorbis',
        'mingw-w64-{}-libogg',
        'mingw-w64-{}-flac',
        'mingw-w64-{}-libmodplug',
        'mingw-w64-{}-mpg123',
        'mingw-w64-{}-opus',
        'mingw-w64-{}-opusfile',
        'mingw-w64-{}-freetype'
    ]

    packages = []
    if x86:
        packages.extend([x.format('i686') for x in deps])
    if x64:
        packages.extend([x.format('x86_64') for x in deps])
    return packages


def install_prebuilts(x86=True, x64=True):
    """ For installing prebuilt dependencies.
    """
    errors = False
    print("Installing pre-built dependencies")
    for pkg in get_packages(x86=x86, x64=x64):
        print(f"Installing {pkg}")
        error = install_pacman_package(pkg)
        errors = errors or error
    if errors:
        raise Exception("Some dependencies could not be installed")


def update(x86=True, x64=True):
    install_prebuilts(x86=x86, x64=x64)


if __name__ == '__main__':
    update()
