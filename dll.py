# dll.py module

"""DLL specifics"""

import re

# Table of dependencies.
# Name, File root name, File regex, Dependency list of file root names
dependencies = [
    ('MIXER', 'SDL_mixer',  r'(lib){0,1}SDL_mixer\.dll$', ['SDL', 'vorbisfile', 'smpeg']),
    ('VORBISFILE', 'vorbisfile',  r'(lib){0,1}vorbisfile(-3){0,1}\.dll$',  ['vorbis']),
    ('VORBIS', 'vorbis', r'(lib){0,1}vorbis(-0){0,1}\.dll$', ['ogg']),
    ('OGG', 'ogg', r'(lib){0,1}ogg(-0){0,1}\.dll$', []),
    ('SMPEG', 'smpeg', r'(lib){0,1}smpeg\.dll$', ['SDL']),
    ('IMAGE', 'SDL_image', r'(lib){0,1}SDL_image\.dll$', ['SDL', 'jpeg', 'png', 'tiff']),
    ('TIFF', 'tiff', r'(lib){0,1}tiff\.dll$',  ['jpeg', 'z']),
    ('JPEG', 'jpeg', r'(lib){0,1}jpeg\.dll$', []),
    ('PNG', 'png', r'(lib){0,1}png(1[23])\.dll$', ['z']),
    ('FONT', 'SDL_ttf', r'(lib){0,1}SDL_ttf\.dll$', ['SDL', 'z']),
    ('Z', 'z', r'zlib1\.dll$', []),
    ('SDL', 'SDL', r'(lib){0,1}SDL\.dll$', [])
]

# regexs: Maps name to regex.
# lib_dependencies: Maps file root name to list of dependencies.
# file_root_name: Maps name to root name.

regexs = {}
lib_dependencies = {}
file_root_names = {}
for name, root, regex, deps in dependencies:
    regexs[name] = regex
    lib_dependencies[root] = deps
    file_root_names[name] = root

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

def name_to_root(name):
    return file_root_names[name]
