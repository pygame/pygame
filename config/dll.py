# dll.py module

"""DLL specifics"""

# Some definitions:

#   Library file name (name): The library name used in linker -l options. It
#   also represents the key identifier.
#       e.g. SDL_mixer
   
import re

# Table of dependencies.
# name, File regex, Dependency list of file root names
libraries = [
    ('SDL', r'(lib){0,1}SDL\.dll$', []),
    ('SDL_mixer', r'(lib){0,1}SDL_mixer\.dll$', ['SDL', 'vorbisenc',
                                                 'vorbisfile', 'smpeg']),
    ('SDL_image', r'(lib){0,1}SDL_image\.dll$', ['SDL', 'jpeg', 'png', 'tiff']),
    ('SDL_ttf', r'(lib){0,1}SDL_ttf\.dll$', ['SDL', 'z', 'freetype']),
    ('SDL_gfx', r'(lib){0,1}SDL_gfx\.dll$', ['SDL']),
    ('freetype', r'(lib){0,1}freetype(-6){0,1}\.dll$', []),
    ('vorbisenc', r'(lib){0,1}vorbisenc(-3){0,1}\.dll$', ['vorbis']),
    ('vorbisfile', r'(lib){0,1}vorbisfile(-3){0,1}\.dll$', ['vorbis']),
    ('vorbis', r'(lib){0,1}vorbis(-0){0,1}\.dll$', ['ogg']),
    ('ogg', r'(lib){0,1}ogg(-0){0,1}\.dll$', []),
    ('smpeg', r'(lib){0,1}smpeg\.dll$', ['SDL']),
    ('tiff', r'(lib){0,1}tiff(-3){0,1}\.dll$',  ['jpeg', 'z']),
    ('jpeg', r'(lib){0,1}jpeg\.dll$', []),
    ('png', r'(lib){0,1}png(1[23])(-0){0,1}\.dll$', ['z']),
    ('z', r'zlib1\.dll$', []),
]

# regexs: Maps name to DLL file name regex.
# lib_dependencies: Maps name to list of dependencies.

regexs = {}
lib_dependencies = {}
for name, regex, deps in libraries:
    regexs[name] = regex
    lib_dependencies[name] = deps
del name, regex, deps

def tester (name):
    """For a library name return a function which tests dll file names"""
    
    def test(file_name):
        """Return true if file name f is a valid DLL name"""
        
        return match(file_name) is not None

    match = re.compile(regexs[name], re.I).match
    test.library_name = name  # Available for debugging.
    return test

def dependencies(*names):
    """Return a set of dependencies for the list of library file roots

    The return set is a dictionary keyed on library root name with values of 1.
    """

    root_set = {}
    for root in names:
        try:
            deps = lib_dependencies[root]
        except KeyError:
            pass
        else:
            root_set[root] = 1
            root_set.update(dependencies(*deps))
    return root_set
