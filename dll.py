# dll.py module

"""DLL specifics"""

import re

#re file patterns and dependencies of Pygame DLL dependencies
regexs = {
    'MIXER': r'(lib){0,1}SDL_mixer\.dll$',
    'VORBISFILE': r'(lib){0,1}vorbisfile(-3){0,1}\.dll$',
    'VORBIS': r'(lib){0,1}vorbis(-0){0,1}\.dll$',
    'OGG': r'(lib){0,1}ogg(-0){0,1}\.dll$',
    'SMPEG': r'(lib){0,1}smpeg\.dll$',
    'IMAGE': r'(lib){0,1}SDL_image\.dll$',
    'TIFF': r'(lib){0,1}tiff\.dll$',
    'JPEG': r'(lib){0,1}jpeg\.dll$',
    'PNG': r'(lib){0,1}png(1[23])\.dll$',
    'FONT': r'(lib){0,1}SDL_ttf\.dll$',
    'Z': r'zlib1\.dll$',
    'SDL': r'(lib){0,1}SDL\.dll$',
    }

lib_dependencies = {
    'SDL_mixer': ['SDL', 'vorbisfile', 'smpeg'],
    'vorbisfile': ['vorbis'],
    'vorbis': ['ogg'],
    'ogg': [],
    'smpeg': ['SDL'],
    'SDL_image': ['SDL', 'jpeg', 'png', 'tiff'],
    'tiff': ['jpeg', 'z'],
    'jpeg': [],
    'png': ['z'],
    'SDL_ttf': ['SDL', 'z'],
    'z': [],
    'SDL': [],
    }

def tester(name):
    def test(f):
        return match(f) is not None
    match =  re.compile(regexs[name], re.I).match
    return test

def dependencies(libs):
    r = {}
    for lib in libs:
        try:
            deps = lib_dependencies[lib]
        except KeyError:
            pass
        else:
            r[lib] = 1
            r.update(dependencies(deps))
    return r
