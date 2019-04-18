import unittest
import ctypes
import pygame
import os.path
import re


pgheader = os.path.join('src_c', '_pygame.h')
local_entry = '_PYGAME_C_API'

ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
ctypes.pythonapi.PyCapsule_GetPointer.restype = \
    ctypes.POINTER(ctypes.c_void_p)

def moduleToNUMSLOTS(module):
    return 'PYGAMEAPI_{}_NUMSLOTS'.format(
        module.__name__.split('.')[-1].upper())


class ModuleEntryTest(unittest.TestCase):
    """ensure that C API slots have been initialized"""

    @staticmethod
    def numslots_for(module, macroname=None,
                     header=pgheader):
        """number of C API slots or -1 if it cannot be found"""
        if not macroname:
            macroname = 'PYGAMEAPI_{}_NUMSLOTS'.format(
                module.__name__.split('.')[-1].upper())
        capture_numslots_expr = \
            re.compile('\s*#define.*{}.*(\d+)'
                .format(macroname))
        capture_numslots = capture_numslots_expr.match

        numslots = -1
        with open(header) as f:
            for line in f:
                match = capture_numslots(line)
                if match:
                    numslots = int(match.group(1))
                    break
        return numslots

    def check_module(self, module, header=pgheader, macronameFunc=moduleToNUMSLOTS):
        """NULL check module's (zero-initialized) API slot entries"""
        capsule = getattr(module, local_entry)
        capsuleobj = ctypes.py_object(capsule)
        name = ctypes.pythonapi.PyCapsule_GetName(capsuleobj)

        # find number of slots
        macroname = macronameFunc(module)
        numslots = self.numslots_for(module, macroname=macroname,
                                     header=header)
        self.assertNotEqual(numslots, -1,
            "cannot find {} in {}".format(macroname, header))

        # void *moduletable[] = { ... };
        moduletable = ctypes.pythonapi.PyCapsule_GetPointer(
            capsuleobj, name)
        for entryind in range(numslots):
            self.assertIsNotNone(moduletable[entryind],
                'API slot {} is NULL'.format(entryind))

    def check_module_str(self, modulestr, header=pgheader,
                         macronameFunc=moduleToNUMSLOTS):
        try:
            mod = __import__('.'.join(('pygame', modulestr)))
            self.check_module(getattr(pygame, modulestr),
                              header, macronameFunc)
        except ImportError:
            unittest.skip('module unavailable')


class ExtModuleEntryTest(ModuleEntryTest):
    freetype_header = os.path.join('src_c', 'freetype.h')

    @unittest.skipIf(not os.path.isfile(freetype_header), "cannot find the header")
    def test_freetype(self):
        self.check_module_str('_freetype',
                              header=self.freetype_header,
                              macronameFunc=lambda m: 'PYGAMEAPI_FREETYPE_NUMSLOTS')

    bufferproxy_header = os.path.join('src_c', 'pgbufferproxy.h')

    @unittest.skipIf(not os.path.isfile(bufferproxy_header), "cannot find the header")
    def test_bufferproxy(self):
        self.check_module_str('bufferproxy', header=self.bufferproxy_header,
                              macronameFunc=lambda m: 'PYGAMEAPI_BUFPROXY_NUMSLOTS')

    font_header = os.path.join('src_c', 'font.h')

    @unittest.skipIf(not os.path.isfile(font_header), "cannot find the header")
    def test_font(self):
        self.check_module_str('font', header=self.font_header)

    mask_header = os.path.join('src_c', 'mask.h')

    @unittest.skipIf(not os.path.isfile(mask_header), "cannot find the header")
    def test_mask(self):
        self.check_module_str('mask', header=self.mask_header)

    mixer_header = os.path.join('src_c', 'mixer.h')

    @unittest.skipIf(not os.path.isfile(mixer_header), "cannot find the header")
    def test_mixer(self):
        self.check_module_str('mixer', header=self.mixer_header)


@unittest.skipIf(not os.path.isfile(pgheader),
                 "Skipping because we cannot find _pygame.h")
class BaseModuleEntryTest(ModuleEntryTest):
    pass

#
# add tests for the base modules
#
base_modules = [
    'base',
    'rect',
    'cdrom',
    'joystick',
    'display',
    'surface',
    'surflock',
    'event',
    'rwobject',
    'pixelarray',
    'color',
    'math',
    ]
if pygame.version.vernum[0] == 2:
    # display has no API with SDL 2
    base_modules.remove('display')
base_modules = [getattr(pygame, modname) for modname in base_modules if modname in pygame.__dict__]

def _makechecker(test, mod):
    def ret(test):
        BaseModuleEntryTest.check_module(test, mod)
    return ret

for mod in base_modules:
    modname = mod.__name__.split('.')[-1]
    setattr(BaseModuleEntryTest,
            "test_{}".format(modname),
            _makechecker(BaseModuleEntryTest, mod))
