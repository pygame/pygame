"""Config with conan package manager.
"""

import os
import sys
import subprocess
import json


class Dependency:
    def __init__(self, conanbuildinfo, name, conan_name, extra_libs=None):
        self.name = name
        self.inc_dir = None
        self.lib_dir = None
        self.libs = None
        self.found = 0
        self.cflags = ''
        infos = [info for info in conanbuildinfo['dependencies'] if info['name'] == conan_name]

        if infos:
            info = infos[0]
            self.found = 1
            self.lib_dir = info['lib_paths'][:]
            self.libs = info['libs'][:]
            self.inc_dir = info['include_paths'][:]

            if info["frameworks"]:
                for n in info["frameworks"]:
                    # -Xlinker is a weird thing for distutils.extension.read_setup_file
                    # so that it can pass things through to the linker from the Setup file.
                    self.cflags += (' -Xlinker "-framework" -Xlinker "' + n + '"')

        if not extra_libs is None:
            self.libs.extend(extra_libs)


def conan_install(force_build=True):
    """
    """
    build_dir = os.path.join('build', 'conan')

    if not os.path.exists(build_dir):
        if not os.path.exists('build'):
            os.mkdir('build')
        os.mkdir(build_dir)

    os.chdir(build_dir)

    cmd = [
        "conan",
        "install",
        os.path.join('..', '..', 'buildconfig', 'conanconf'),
    ]
    if force_build:
        cmd.append("--build")

    if '-conan' in sys.argv:
        other_args = sys.argv[sys.argv.index('-conan')+1:]
        cmd.extend(other_args)

    print(cmd)
    try:
        return subprocess.call(cmd)
    finally:
        os.chdir(os.path.join('..', '..'))

def main(sdl2=True):
    # conan_install(force_build=True)
    # Reuse the previous conan build with this setting:
    conan_install(force_build=False)

    conanbuildinfo_json = os.path.join('build', 'conan', 'conanbuildinfo.json')
    conanbuildinfo = json.load(open(conanbuildinfo_json))

    DEPS = [
        Dependency(conanbuildinfo, "SDL", "sdl2"),
        Dependency(conanbuildinfo, "FONT", "sdl2_ttf"),
        Dependency(conanbuildinfo, "IMAGE", "sdl2_image"),
        Dependency(conanbuildinfo, "MIXER", "sdl2_mixer"),
        Dependency(conanbuildinfo, "PNG", "libpng"),
        Dependency(conanbuildinfo, "JPEG", "libjpeg"),
        Dependency(conanbuildinfo, "FREETYPE", "freetype"),
        Dependency(conanbuildinfo, "PORTMIDI", "portmidi"),
        Dependency(conanbuildinfo, "PORTTIME", "portmidi"),
    ]

    return DEPS

if __name__ == '__main__':
    print ("""This is the configuration subscript for the Conan package manager.
Please run "config.py" for full configuration.""")
