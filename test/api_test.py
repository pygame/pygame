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
                    numslots = int(match[1])
                    break
        return numslots

    def check_module(self, module, header=pgheader):
        """NULL check module's (zero-initialized) API slot entries"""
        capsule = getattr(module, local_entry)
        capsuleobj = ctypes.py_object(capsule)
        name = ctypes.pythonapi.PyCapsule_GetName(capsuleobj)

        # find number of slots
        macroname = 'PYGAMEAPI_{}_NUMSLOTS'.format(
            module.__name__.split('.')[-1].upper())
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
